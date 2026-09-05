"""LoRA support for the LLaDA-Image ComfyUI components.

Design constraints
------------------
Both supported base components hold lazy/quantized weights that must never be
materialized just to merge a LoRA delta:

* the LLaDA2 text encoder is made of ``LazyGGUFLinear`` layers (mmap-backed
  GGUF, dequantized per call);
* the denoiser transformer may be a ``LazyINT8Linear`` model (int8_tensorwise,
  dequantized per call) or plain BF16 ``nn.Linear`` layers.

Merging would dequantize the whole model into BF16 (several GB for the 8B
text encoder), so LoRA is injected as a *bypass*: every target Linear is
replaced by a wrapper that first calls the untouched base layer and then adds
``B(A(x)) * scale``. This keeps quantization intact and only keeps the small
A/B matrices resident.

The wrapper tree is built by shallow-cloning only the module chain that leads
to each patched layer. All other modules (and all parameters/buffers) are
shared with the original component, so applying a LoRA never mutates the
component that the loaders cached; several different LoRA branches can
therefore coexist and chaining multiple LoRA loaders just wraps deeper.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-LLaDA-Image")


# Prefixes that common export tools prepend to the model-internal paths.
# The longest matching prefix is stripped before path resolution.
_STRIP_PREFIXES = (
    "base_model.model.",
    "base_model.",
    "diffusion_model.",
    "transformer.",
    "text_encoder.",
    "unet.",
    "model.",
    "",
)


def _lora_pair_keys(key: str):
    """Return ("A"|"B", module_path) for a PEFT-style LoRA key, else None."""
    if key.endswith(".lora_A.weight"):
        return "A", key[: -len(".lora_A.weight")]
    if key.endswith(".lora_B.weight"):
        return "B", key[: -len(".lora_B.weight")]
    return None


class _LoraPair(nn.Module):
    """Small container registering one A/B pair so device/dtype moves apply."""

    def __init__(self, a: torch.Tensor, b: torch.Tensor, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.a = nn.Parameter(a.detach().clone().contiguous(), requires_grad=False)
        self.b = nn.Parameter(b.detach().clone().contiguous(), requires_grad=False)


class LoraWrappedLinear(nn.Module):
    """Linear-compatible leaf that runs the base layer plus a LoRA bypass.

    The base module is kept untouched (quantized weights stay quantized); one
    or more ``(A, B, scale)`` pairs can be stacked, which is what chained LoRA
    loaders produce.
    """

    def __init__(self, base: nn.Module, pairs: list[_LoraPair]):
        super().__init__()
        self.base = base
        self.pairs = nn.ModuleList(pairs)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.pairs:
            return y

        compute_dtype = x.dtype
        for pair in self.pairs:
            a = pair.a.to(device=x.device, dtype=compute_dtype)
            b = pair.b.to(device=x.device, dtype=compute_dtype)
            try:
                h = F.linear(x, a)
                h = F.linear(h, b)
            except RuntimeError as e:
                raise RuntimeError(
                    f"LoRA shape mismatch on {self._target_name()}: {e} "
                    f"(lora_A={tuple(pair.a.shape)}, lora_B={tuple(pair.b.shape)}, "
                    f"input={tuple(x.shape)})"
                ) from e
            y = y + h * float(pair.scale)
        return y

    def _target_name(self) -> str:
        return type(self.base).__name__


def _base_like_target(base: nn.Module) -> bool:
    """Whitelist of layers that can host a LoRA bypass."""
    if isinstance(base, (nn.Linear,)):
        return True
    # Lazy/quantized siblings defined by this project's adapter.
    name = type(base).__name__
    return name in {"LazyGGUFLinear", "LazyINT8Linear"}


def _resolve_child(module: nn.Module, seg: str) -> nn.Module:
    """Resolve one path segment through the registered module tree.

    Covers plain attributes, ``nn.ModuleList`` integer keys and
    ``nn.ModuleDict`` string keys such as ``all_x_embedder.1-1``.
    """
    children = module._modules
    if seg not in children:
        raise KeyError(seg)
    return children[seg]


def _resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    cur = root
    for seg in dotted.split("."):
        cur = _resolve_child(cur, seg)
    return cur


def _resolve_with_prefixes(root: nn.Module, dotted: str):
    """Resolve ``dotted`` inside the module tree, stripping export prefixes.

    PEFT-style adapters usually carry a ``base_model.model.`` prefix on every
    key; trainers that exported from a submodule may not. Try the exact path
    first, then each prefix-stripped candidate until one resolves.

    Returns ``(module, canonical)`` where ``canonical`` is the path that
    actually resolved inside the tree (prefix already removed). Callers that
    later walk the tree again (e.g. the clone step) MUST use ``canonical``,
    never the raw ``dotted`` input.
    """
    candidates = [dotted] + [
        dotted[len(prefix):] for prefix in _STRIP_PREFIXES
        if prefix and dotted.startswith(prefix)
    ]
    last_error = None
    for candidate in candidates:
        try:
            return _resolve_module(root, candidate), candidate
        except (KeyError, AttributeError) as e:
            last_error = e
    raise last_error or KeyError(dotted)


def _in_out_features(base: nn.Module):
    """Best-effort (in_features, out_features) of the base layer.

    Returns (None, None) when the quantized storage hides the true shape.
    """
    try:
        if isinstance(base, nn.Linear):
            out, in_ = base.weight.shape
            return int(in_), int(out)
        raw = getattr(base, "_qweight", None)
        if raw is None:
            raw = getattr(base, "weight", None)
        if raw is not None:
            shape = tuple(int(v) for v in getattr(raw, "tensor_shape", raw.shape))
            if len(shape) == 2:
                return int(shape[1]), int(shape[0])
    except Exception:
        pass
    return None, None


def _oriented_pair(a: torch.Tensor, b: torch.Tensor, base: nn.Module):
    """Normalize A/B to the PEFT layout used by the bypass math.

    Bypass math expects ``A`` shaped (r, in) and ``B`` shaped (out, r).
    PEFT stores exactly that; some exporters store the transposes, so when the
    base layer dimensions are known the matrices are flipped accordingly.
    """
    in_feat, out_feat = _in_out_features(base)
    if in_feat is not None:
        if a.dim() == 2 and a.shape[1] != in_feat and a.shape[0] == in_feat:
            a = a.transpose(0, 1).contiguous()
    if out_feat is not None:
        if b.dim() == 2 and b.shape[0] != out_feat and b.shape[1] == out_feat:
            b = b.transpose(0, 1).contiguous()
    if a.dim() != 2 or b.dim() != 2:
        raise RuntimeError(
            "LoRA matrices must be 2-D, "
            f"got lora_A={tuple(a.shape)}, lora_B={tuple(b.shape)}."
        )
    return a, b


def _clone_chain_with_leaves(root: nn.Module, wrappers: dict[str, nn.Module]) -> nn.Module:
    """Shallow-clone every module chain leading to a replaced leaf.

    Only the containers on the way from ``root`` to each wrapper are copied;
    their parameters, buffers and non-target children stay shared with the
    original tree, so the operation is cheap and never mutates the input
    component. Hooks dicts are duplicated so offload hooks installed later on
    the clone cannot leak into the original component (or vice versa).
    """
    if not wrappers:
        return root

    _HOOK_KEYS = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_hooks",
        "_non_persistent_buffers_set",
    )

    def _copy_module(module: nn.Module) -> nn.Module:
        clone = copy.copy(module)
        # copy.copy shallow-copies __dict__; make sure mutable registries are
        # independent so hooks installed later on the clone (e.g. offload)
        # cannot leak back into the original component tree.
        clone._modules = dict(module._modules)
        for key in _HOOK_KEYS:
            value = module.__dict__.get(key)
            if isinstance(value, dict):
                clone.__dict__[key] = dict(value)
            elif isinstance(value, set):
                clone.__dict__[key] = set(value)
        return clone

    new_root = _copy_module(root)
    clones = {id(root): new_root}

    for dotted, wrapper in wrappers.items():
        segments = dotted.split(".")
        orig = root
        cur = new_root

        for index, seg in enumerate(segments):
            if index == len(segments) - 1:
                if seg not in cur._modules:
                    raise RuntimeError(
                        f"LoRA target does not resolve inside the model: {dotted!r}"
                    )
                cur._modules[seg] = wrapper
                continue

            next_orig = orig._modules[seg]
            next_clone = clones.get(id(next_orig))
            if next_clone is None:
                next_clone = _copy_module(next_orig)
                clones[id(next_orig)] = next_clone
            cur._modules[seg] = next_clone
            orig = next_orig
            cur = next_clone

    return new_root


def apply_lora(model: nn.Module, lora_sd: dict, strength: float, *, strict: bool = True) -> nn.Module:
    """Return a shallow clone of ``model`` with the LoRA bypasses applied.

    ``lora_sd`` must map PEFT-style keys (``<path>.lora_A.weight`` /
    ``<path>.lora_B.weight``) to tensors. Keys that cannot be resolved inside
    this model are ignored; when nothing resolves and ``strict`` is true an
    error explains the expected format, otherwise the model is returned
    untouched so callers can try the same file against another component.
    """
    pairs_by_path: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
    extra_keys: list[str] = []

    for key, tensor in lora_sd.items():
        parsed = _lora_pair_keys(key)
        if parsed is None:
            if key.endswith((".lora_A", ".lora_B", "lora_embedding_A", "lora_embedding_B")):
                extra_keys.append(key)
            continue
        kind, path = parsed
        if kind == "A":
            pairs_by_path.setdefault(path, [None, None])[0] = tensor
        else:
            pairs_by_path.setdefault(path, [None, None])[1] = tensor

    incomplete = [p for p, (a, b) in pairs_by_path.items() if a is None or b is None]
    if incomplete:
        raise RuntimeError(
            "LoRA file is missing lora_A or lora_B for: "
            + ", ".join(incomplete[:8])
        )

    resolved = {}
    for path, (a, b) in pairs_by_path.items():
        try:
            leaf, canonical = _resolve_with_prefixes(model, path)
        except (KeyError, AttributeError) as e:
            extra_keys.append(path)
            continue

        if isinstance(leaf, LoraWrappedLinear):
            base = leaf.base
            existing = list(leaf.pairs)
        else:
            base = leaf
            existing = []

        if not _base_like_target(base):
            raise RuntimeError(
                f"LoRA targets unsupported module type {type(base).__name__} at "
                f"{path!r}; only Linear layers can host a LoRA bypass."
            )

        a, b = _oriented_pair(a, b, base)
        scale = float(strength)
        pair = _LoraPair(a, b, scale)
        resolved[canonical] = LoraWrappedLinear(base, existing + [pair])

    if not resolved:
        if not strict:
            log.info("LoRA: no keys matched %s; leaving it untouched", type(model).__name__)
            return model
        sample = ", ".join(repr(k) for k in list(lora_sd)[:8])
        raise RuntimeError(
            "No LoRA key matched any layer of the connected model. Expected "
            "PEFT-style keys ending in '<module path>.lora_A.weight' / "
            "'.lora_B.weight'. First keys in file: " + sample
        )

    if extra_keys:
        log.info(
            "LoRA: %d key(s) did not map to a layer and were ignored (%s)",
            len(extra_keys),
            ", ".join(repr(k) for k in extra_keys[:6]),
        )

    patched = _clone_chain_with_leaves(model, resolved)
    n_targets = len(resolved)
    log.info(
        "LoRA applied to %s: %d target layers, strength=%.4f",
        type(model).__name__,
        n_targets,
        strength,
    )
    return patched


def load_lora_tensors(lora_path: str | Path):
    """Read a LoRA .safetensors file into a plain state-dict mapping."""
    from safetensors.torch import load_file

    path = Path(lora_path)
    sd = load_file(str(path), device="cpu")

    has_lora_keys = any(key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight") for key in sd)
    if not has_lora_keys:
        sample = ", ".join(repr(k) for k in list(sd)[:8])
        raise RuntimeError(
            f"{path.name} does not look like a PEFT-style LoRA file "
            f"(no '<path>.lora_A/.lora_B.weight' keys). First keys: {sample}"
        )
    return sd


class LoraHandle:
    """One LoRA file at a fixed strength, produced by the LoRA loader node.

    This is a pure *intervention* handle: it carries no model. The sampling
    node applies it around the loaded components right before the pipeline is
    assembled. Instances are cached by (file, strength) in nodes.py so that
    repeated runs of the same workflow reuse the handle and the applied
    clones, keeping the assembled-pipeline cache effective.

    Tensors are loaded lazily and kept for the lifetime of the handle;
    ``Path``/file name are kept for diagnostics only.
    """

    def __init__(self, name: str, path: str | Path, strength: float):
        self.name = name
        self.path = Path(path)
        self.strength = float(strength)
        self._tensors = None

    def tensors(self) -> dict:
        if self._tensors is None:
            self._tensors = load_lora_tensors(self.path)
        return self._tensors
