"""LLaDA-Image ComfyUI nodes (component-level loaders + sampling nodes).

Node topology
-------------
The old monolithic loader is split into independent component loaders, one
per model file the user must own:

    LLaDA Image Diffusion Model Loader   -> LLADA_DIFFUSION_MODEL
    LLaDA Image Text Encoder Loader      -> LLADA_TEXT_ENCODER   (.gguf only)
    LLaDA Image VAE Loader               -> LLADA_VAE
    LLaDA Image QueryFormer Loader       -> LLADA_QUERYFORMER
    LLaDA Image Text Projection Loader   -> LLADA_TEXT_PROJECTION
    LLaDA Image SigVQ Loader             -> LLADA_SIGVQ
    LLaDA Image LoRA Loader              -> LLADA_LORA (optional intervention)

The sampling nodes (Text to Image / Edit) receive the six components directly
and assemble the internal Diffusers pipeline there (tokenizer/scheduler are
bundled). A LoRA loader output handle can be attached to the sampling node's
``lora`` input; the adapter is applied around the transformer/text encoder
right before assembly and never becomes part of the component chain. There is
no visible LLADA_PIPELINE type anymore; unloading is implicit: every
component loader keeps only the most recently loaded object, so selecting a
different file drops the previous one, and the pipeline cache is rebuilt when
any component identity changes.
"""

import gc
import logging
import time
import weakref
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import folder_paths

log = logging.getLogger("ComfyUI-LLaDA-Image")

from . import llada_lora
from . import llada_gguf_adapter as _gguf_adapter
from .llada_gguf_adapter import (
    SOURCE_ATTR,
    DTYPE_ATTR,
    assemble_llada_pipeline,
    load_llada2_gguf_encoder,
    load_llada_aux_model,
    load_llada_transformer,
    load_llada_vae,
)


NODE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = NODE_DIR / "configs"

# One cached component object per kind. The cached value is the tuple
# (identity, object) and a new selection simply overwrites the entry, which
# makes model replacement the unloading mechanism (there is no Unload node).
_COMPONENT_CACHE = {}

# Assembled pipelines keyed by (variant, component ids). Only the latest
# assembly is retained; evicting happens when a sampling node misses.
_PIPELINE_CACHE = {}

# LoRA results keyed by (input component id, lora file, strength). Bounded.
_LORA_CACHE = OrderedDict()
_LORA_CACHE_LIMIT = 16

# LoRA handles keyed by (file, strength). A handle is a pure intervention
# object (no model reference) produced by the LoRA loader node; keeping it
# stable lets the sampling nodes reuse applied clones and hit the pipeline
# cache on repeated runs.
_LORA_HANDLE_CACHE = {}

# Prompt embeddings keyed by (encoding-chain ids, prompt, negative prompt).
# A hit skips the full text-encoder forward on repeated runs; entries only
# hold CPU tensors. Bounded LRU like the LoRA caches above.
_PROMPT_EMBEDS_CACHE = OrderedDict()
_PROMPT_EMBEDS_CACHE_LIMIT = 12


MODEL_EXTENSIONS = {".safetensors", ".gguf", ".bin", ".pt", ".pth"}


def _folder_roots(*keys):
    """Return every registered Comfy model root for the requested categories."""
    roots = []
    seen = set()

    for key in keys:
        try:
            paths = folder_paths.get_folder_paths(key)
        except Exception:
            paths = []

        for p in paths or []:
            p = str(Path(p))
            if p not in seen:
                seen.add(p)
                roots.append(Path(p))

    return roots


def _scan_model_files(keys, extensions=MODEL_EXTENSIONS):
    """Scan the actual registered model directories recursively.

    This intentionally does not depend only on get_filename_list(), because
    some Comfy installs/plugins filter extensions and can omit .gguf files.
    """
    found = {}

    # First include anything Comfy itself already registered.
    for key in keys:
        try:
            for name in folder_paths.get_filename_list(key):
                try:
                    full = folder_paths.get_full_path(key, name)
                except Exception:
                    full = None
                if full:
                    found[name.replace("\\", "/")] = Path(full)
        except Exception:
            pass

    # Then scan every registered directory directly so GGUF is always visible.
    for root in _folder_roots(*keys):
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in extensions:
                continue
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = p.name
            found.setdefault(rel, p)

    return dict(sorted(found.items(), key=lambda kv: kv[0].lower()))


def _is_llada_transformer_file(path: Path) -> bool:
    """Header-only architecture check; never loads model tensors into RAM."""
    if path.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
    except Exception:
        return False

    # Full/BF16/FP8 LLaDA transformer signatures.
    full_signatures = (
        "layers.0.attention.to_q.weight",
        "layers.0.attention.to_k.weight",
        "all_x_embedder.1-1.weight",
    )
    if all(k in keys for k in full_signatures):
        return True

    # Native int8_tensorwise files produced from the same LLaDA architecture.
    int8_signatures = (
        "layers.0.attention.to_q.comfy_quant",
        "layers.0.attention.to_k.comfy_quant",
        "all_x_embedder.1-1.weight",
    )
    if all(k in keys for k in int8_signatures):
        return True

    return False


def _diffusion_files():
    files = _scan_model_files(("diffusion_models", "unet"), {".safetensors"})
    # Only expose actual LLaDA transformer checkpoints. This prevents Flux/Fibo
    # checkpoints such as single_transformer_blocks.* from ever entering the loader.
    return {name: path for name, path in files.items() if _is_llada_transformer_file(path)}


def _text_encoder_files():
    # The standalone text-encoder path requires the LLaDA2 GGUF encoder, so
    # only GGUF files are offered (scanning physical roots keeps .gguf visible).
    return _scan_model_files(("text_encoders", "clip"), {".gguf"})


def _vae_files():
    return _scan_model_files(("vae",), {".safetensors", ".bin", ".pt", ".pth"})


def _resolve_scanned_file(mapping, name, kind):
    path = mapping.get(name)
    if path and path.is_file():
        return path
    raise FileNotFoundError(f"Could not resolve selected {kind}: {name!r}")


def _check_bundled_assets():
    """Architecture/config code belongs to the node itself; no 'support pack'."""
    expected_configs = (
        "transformer_config.json",
        "text_encoder_config.json",
        "queryformer_config.json",
        "sigvq_config.json",
        "text_projection_config.json",
        "vae_config.json",
        "model_index.json",
        "configuration_llada2uni_moe.py",
        "modeling_llada2uni_moe.py",
        "fused_moe_ops.py",
    )
    missing_cfg = [name for name in expected_configs if not (CONFIG_DIR / name).is_file()]
    if missing_cfg:
        raise RuntimeError(
            "Custom-node config bundle is incomplete. Missing: " + ", ".join(missing_cfg)
        )


def _torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def _comfy_image_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    image = image.detach().cpu().clamp(0, 1)
    arr = (image.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pil_to_comfy(images):
    if isinstance(images, Image.Image):
        images = [images]

    batch = []
    for image in images:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        arr = np.asarray(image).astype(np.float32) / 255.0
        batch.append(torch.from_numpy(arr))
    return torch.stack(batch, dim=0)


# ---------------------------------------------------------------------------
# Auxiliary component discovery (SigVQ / text_projection / QueryFormer)
#
# Each small LLaDA aux component is bound to ONE dedicated directory under the
# ComfyUI models tree: models/query_former, models/text_projection and
# models/sigvq. Only *.safetensors inside that component's own directory are
# offered by its loader — full-tree scanning made the dropdowns cross over
# into unrelated model directories. The directories are registered as Comfy
# model folders (and created when missing) so folder_paths / extra_model_paths
# lookups behave like every other model category.
# ---------------------------------------------------------------------------

_AUX_KIND_CACHE = {}


def _comfy_models_root():
    try:
        root = getattr(folder_paths, "models_dir", None)
        if root:
            return Path(root)
    except Exception:
        pass
    return Path(folder_paths.__file__).resolve().parent / "models"


# Dedicated models subdirectory per aux component: (folder key, dirnames).
# The primary dirname is created when missing; the "queryformer" spelling is
# only adopted when the user already created such a directory.
_AUX_DIRS = {
    "queryformer": ("llada_query_former", ("query_former", "queryformer")),
    "text_projection": ("llada_text_projection", ("text_projection",)),
    "sigvq": ("llada_sigvq", ("sigvq",)),
}

# Canonical models-relative directory used in messages/placeholders.
_AUX_DIR_LABEL = {
    "queryformer": "query_former",
    "text_projection": "text_projection",
    "sigvq": "sigvq",
}

# Human names for the redirect error when a file is picked from the wrong
# component directory.
_AUX_LOADER_LABEL = {
    "queryformer": "QueryFormer",
    "text_projection": "Text Projection",
    "sigvq": "SigVQ",
}


def _ensure_aux_folder_registered(kind: str):
    """Register (and create) the dedicated models subdirectory for ``kind``."""
    folder_key, dirnames = _AUX_DIRS[kind]
    base = _comfy_models_root()
    for dirname in dirnames:
        root = base / dirname
        if dirname != dirnames[0] and not root.is_dir():
            continue  # secondary spelling: only adopt pre-existing folders
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("Cannot create LLaDA aux folder %s: %s", root, e)
            continue
        try:
            paths = folder_paths.folder_names_and_paths.get(folder_key)
            registered = paths is not None and any(str(p) == str(root) for p in paths[0])
        except Exception:
            registered = False
        if not registered:
            folder_paths.add_model_folder_path(folder_key, str(root))


def _classify_safetensors(path: Path):
    """Return the LLaDA aux component kind a file's tensor keys belong to.

    Signatures follow the bundled modeling code:
    - QueryFormer stores the learnable ``meta_queries`` parameter;
    - SigVQ owns ``visual.patch_embed`` / ``vqmodel.*`` / ``prior_token_embedding``;
    - Text Projection ends with the root-level ``projector`` linear layer.
    """
    try:
        stat = path.stat()
        cached = _AUX_KIND_CACHE.get(str(path))
        if cached is not None and cached[0] == (stat.st_mtime_ns, stat.st_size):
            return cached[1]
    except OSError:
        return None

    kind = None
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
    except Exception:
        keys = None

    if keys is not None:
        if "meta_queries" in keys:
            kind = "queryformer"
        elif any(
            k.startswith(("visual.patch_embed", "vqmodel.", "prior_token_embedding"))
            for k in keys
        ):
            kind = "sigvq"
        elif "projector.weight" in keys:
            kind = "text_projection"

    try:
        stat = path.stat()
        _AUX_KIND_CACHE[str(path)] = ((stat.st_mtime_ns, stat.st_size), kind)
    except OSError:
        pass
    return kind


def _scan_aux_component_files(kind: str):
    """*.safetensors inside this component's dedicated models directory only.

    Files whose tensor keys classify as a different LLaDA aux component are
    hidden (they landed in the wrong directory); unknown signatures stay
    visible so the loader error can report the concrete key mismatch.
    """
    _ensure_aux_folder_registered(kind)
    found = _scan_model_files((_AUX_DIRS[kind][0],), {".safetensors"})
    misplaced = []
    visible = {}
    for name, path in found.items():
        classified = _classify_safetensors(path)
        if classified in (None, kind):
            visible[name] = path
        else:
            misplaced.append(f"{name} ({classified})")
    if misplaced:
        log.warning(
            "LLaDA %s directory contains misplaced files, ignoring: %s",
            _AUX_DIR_LABEL[kind],
            ", ".join(misplaced),
        )
    return visible


# ---------------------------------------------------------------------------
# Component caching helpers
# ---------------------------------------------------------------------------

def _load_component(kind: str, identity: tuple, factory):
    """Return the cached component for ``kind`` or build and cache a new one.

    Only the latest object per kind is cached; replacing it is how models get
    released after the pipeline cache has been evicted.
    """
    entry = _COMPONENT_CACHE.get(kind)
    if entry is not None and entry[0] == identity:
        return entry[1]
    obj = factory()
    _COMPONENT_CACHE[kind] = (identity, obj)
    return obj


def _load_component_logged(kind, identity, factory):
    """_load_component plus a cache-hit / wall-time log line per loader run."""
    entry = _COMPONENT_CACHE.get(kind)
    cached = entry is not None and entry[0] == identity
    started = time.monotonic()
    result = _load_component(kind, identity, factory)
    log.info(
        "LLaDA %s: %s in %.1fs",
        kind,
        "component cache hit" if cached else "loaded",
        time.monotonic() - started,
    )
    return result


def _component_cache_key(*parts):
    return parts


def _probe_install_encode_stages(components):
    """Wrap each encode-chain stage forward with a one-shot wall-time log.

    Temporary diagnostics for the slow GGUF text encode path: the wrappers
    record per-stage seconds and the output device, then get removed right
    after the encode call. Returns the restore list (owner, original).
    Failures degrade silently so probing never blocks generation.
    """
    restores = []
    try:
        encoder = components["text_encoder"]
        stages = [
            ("embedding", getattr(encoder, "get_input_embeddings", lambda: None)()),
            ("backbone", getattr(encoder, "model", None)),
            ("queryformer", components.get("queryformer")),
            ("text_projection", components.get("text_projection")),
        ]
        for label, obj in stages:
            if obj is None:
                continue
            first = next(iter(obj.parameters()), None)
            log.info(
                "LLaDA encode probe %s: first param on %s",
                label,
                getattr(first, "device", None),
            )
            original = getattr(obj, "forward", None)
            if original is None:
                continue

            def timed(*args, _label=label, _original=original, **kwargs):
                started = time.monotonic()
                result = _original(*args, **kwargs)
                log.info(
                    "LLaDA encode probe %s: %.1fs (out on %s)",
                    _label,
                    time.monotonic() - started,
                    getattr(result, "device", None),
                )
                return result

            obj.forward = timed
            restores.append((obj, original))
        try:
            stop_profile = _gguf_adapter._encode_profile_enter(encoder)
        except Exception as e:
            log.warning("LLaDA encode profile setup failed, skipping: %s", e)
            stop_profile = None
        if stop_profile is not None:
            restores.append(stop_profile)
    except Exception as e:
        log.warning("LLaDA encode probe setup failed, skipping: %s", e)
    return restores


def _probe_restore_encode_stages(restores):
    """Put every wrapped forward back to its original."""
    for item in restores:
        try:
            if callable(item):
                item()
            else:
                item[0].forward = item[1]
        except Exception:
            pass


def _prompt_embeds_for(pipe, components, prompt, negative_prompt, do_cfg):
    """Encode through ``pipe.encode_prompt`` or reuse a cached copy.

    Returns ``(pipe kwargs, hit)``: call ``pipe(prompt=None, **kwargs)`` with
    the returned kwargs. Encoding happens exactly like the pipeline internals
    would (same method, same defaults), so a cached result is identical to a
    fresh one; misses are plain first-run encodes.
    """
    chain = (
        components["text_encoder"],
        components["queryformer"],
        components["text_projection"],
    )
    key = (tuple(id(c) for c in chain), prompt, negative_prompt)
    entry = _PROMPT_EMBEDS_CACHE.get(key)
    if entry is not None:
        refs, payload = entry
        if all(ref() is comp for ref, comp in zip(refs, chain)):
            _PROMPT_EMBEDS_CACHE.move_to_end(key)
            return payload, True
        del _PROMPT_EMBEDS_CACHE[key]

    restores = _probe_install_encode_stages(components)
    encode_started = time.monotonic()
    try:
        encoded = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if do_cfg else None,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            max_sequence_length=2048,
        )
    finally:
        _probe_restore_encode_stages(restores)
    log.info("LLaDA prompt embeds encoded in %.1fs", time.monotonic() - encode_started)
    embeds, embeds_mask, neg_embeds, neg_mask = encoded
    payload = {
        "prompt_embeds": _embed_to_cpu(embeds),
        "prompt_attention_mask": _embed_to_cpu(embeds_mask),
        "negative_prompt_embeds": _embed_to_cpu(neg_embeds),
        "negative_prompt_attention_mask": _embed_to_cpu(neg_mask),
    }
    _PROMPT_EMBEDS_CACHE[key] = (
        tuple(weakref.ref(c) for c in chain),
        payload,
    )
    while len(_PROMPT_EMBEDS_CACHE) > _PROMPT_EMBEDS_CACHE_LIMIT:
        _PROMPT_EMBEDS_CACHE.popitem(last=False)
    return payload, False


def _embed_to_cpu(value):
    """Move one encoded embed tensor (or None) to the CPU for cheap storage."""
    if value is None:
        return None
    return value.detach().to("cpu")


def _log_generation_stats(label, started, embeds_hit):
    """Wall time plus this run's CUDA peak; helps VRAM/throughput tuning."""
    seconds = time.monotonic() - started
    vram = ""
    if torch.cuda.is_available():
        try:
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            vram = f", peak VRAM {peak:.2f} GiB"
        except Exception:
            vram = ""
    log.info(
        "LLaDA %s done in %.1fs (prompt embeds %s)%s",
        label,
        seconds,
        "cache hit" if embeds_hit else "encoded",
        vram,
    )


def _release_pipeline_cache():
    """Drop every assembled pipeline so stale components can be collected."""
    _PROMPT_EMBEDS_CACHE.clear()
    if _PIPELINE_CACHE:
        _PIPELINE_CACHE.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _assembled_pipeline(components: dict, variant: str, weights_on_gpu: bool = False):
    """Return the pipeline for a component set, assembling it on a cache miss.

    Components are identified by object identity: loader caches and the LoRA
    cache return stable objects, so repeated executions hit the cache and only
    a real workflow change triggers a rebuild. ``weights_on_gpu`` is part of
    the cache key (it changes weight placement, not outputs).
    """
    key = (
        variant,
        tuple(id(components[name]) for name in (
            "transformer", "text_encoder", "vae", "queryformer",
            "text_projection", "sigvq",
        )),
        bool(weights_on_gpu),
    )
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    _release_pipeline_cache()
    started = time.monotonic()
    pipe = assemble_llada_pipeline(
        CONFIG_DIR,
        transformer=components["transformer"],
        text_encoder=components["text_encoder"],
        vae=components["vae"],
        queryformer=components["queryformer"],
        text_projection=components["text_projection"],
        sigvq=components["sigvq"],
        variant=variant,
        weights_on_gpu=bool(weights_on_gpu),
    )
    log.info("LLaDA pipeline assembled in %.1fs", time.monotonic() - started)
    _PIPELINE_CACHE[key] = pipe
    return pipe


# ---------------------------------------------------------------------------
# Component loader nodes
# ---------------------------------------------------------------------------

class LLaDAImageDiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        diffusion = list(_diffusion_files().keys()) or ["<no LLaDA diffusion models found>"]
        return {
            "required": {
                "diffusion_model": (diffusion,),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("LLADA_DIFFUSION_MODEL",)
    RETURN_NAMES = ("diffusion_model",)
    FUNCTION = "load"
    CATEGORY = "LLaDA Image"

    def load(self, diffusion_model, dtype):
        _check_bundled_assets()
        name = diffusion_model
        path = _resolve_scanned_file(_diffusion_files(), name, "diffusion model")

        if path.suffix.lower() != ".safetensors":
            raise RuntimeError(
                "LLaDA transformer must be a .safetensors file. "
                f"Selected: {path.name}"
            )
        if not _is_llada_transformer_file(path):
            raise RuntimeError(
                "Wrong diffusion model selected. This loader only accepts LLaDA-Image "
                "transformer checkpoints. The selected file does not have LLaDA keys "
                "(layers.* / all_x_embedder.*). "
                f"Selected: {path.name}"
            )

        compute_dtype = _torch_dtype(dtype)

        def factory():
            model = load_llada_transformer(path, CONFIG_DIR, compute_dtype)
            setattr(model, SOURCE_ATTR, name)
            setattr(model, DTYPE_ATTR, compute_dtype)
            return model

        identity = _component_cache_key(name, dtype)
        return (_load_component_logged("diffusion", identity, factory),)


class LLaDAImageTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        encoders = list(_text_encoder_files().keys()) or ["<no LLaDA GGUF text encoders found>"]
        return {
            "required": {
                "text_encoder": (encoders,),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("LLADA_TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load"
    CATEGORY = "LLaDA Image"

    def load(self, text_encoder, dtype):
        _check_bundled_assets()
        name = text_encoder
        path = _resolve_scanned_file(_text_encoder_files(), name, "text encoder")

        if path.suffix.lower() != ".gguf":
            raise RuntimeError(
                "This text-encoder loader expects the LLaDA2 GGUF encoder "
                f"(.gguf). Selected: {path.name}"
            )

        compute_dtype = _torch_dtype(dtype)

        def factory():
            model = load_llada2_gguf_encoder(path, CONFIG_DIR, dtype=compute_dtype)
            setattr(model, SOURCE_ATTR, name)
            setattr(model, DTYPE_ATTR, compute_dtype)
            return model

        identity = _component_cache_key(name, dtype)
        return (_load_component_logged("text_encoder", identity, factory),)


class LLaDAImageVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        vaes = list(_vae_files().keys()) or ["<no VAEs found>"]
        return {
            "required": {
                "vae": (vaes,),
            }
        }

    RETURN_TYPES = ("LLADA_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "LLaDA Image"

    def load(self, vae):
        _check_bundled_assets()
        name = vae
        path = _resolve_scanned_file(_vae_files(), name, "VAE")

        def factory():
            model = load_llada_vae(path, CONFIG_DIR, torch.bfloat16)
            setattr(model, SOURCE_ATTR, name)
            return model

        identity = _component_cache_key(name)
        return (_load_component_logged("vae", identity, factory),)


class _LLaDAAuxLoaderBase:
    """Shared dropdown logic for the three small diffusers components.

    Subclasses MUST declare their own RETURN_TYPES/RETURN_NAMES: the base
    class cannot compute them at class-body evaluation time.
    """

    KIND = None  # "queryformer" | "text_projection" | "sigvq"
    INPUT_LABEL = None

    @classmethod
    def INPUT_TYPES(cls):
        files = list(_scan_aux_component_files(cls.KIND).keys())
        placeholder = (
            f"<no LLaDA {cls.INPUT_LABEL} files under "
            f"models/{_AUX_DIR_LABEL[cls.KIND]}>"
        )
        return {
            "required": {
                cls.INPUT_LABEL: (files or [placeholder],),
            }
        }

    FUNCTION = "load"
    CATEGORY = "LLaDA Image"

    def load(self, **kwargs):
        _check_bundled_assets()
        name = kwargs[self.INPUT_LABEL]
        mapping = _scan_aux_component_files(self.KIND)
        path = _resolve_scanned_file(mapping, name, self.KIND)

        classified = _classify_safetensors(path)
        if classified not in (None, self.KIND):
            raise RuntimeError(
                f"{path.name!r} classifies as a LLaDA "
                f"{_AUX_LOADER_LABEL[classified]} checkpoint. Move it to "
                f"models/{_AUX_DIR_LABEL[classified]}/ and select it in the "
                f"LLaDA Image {_AUX_LOADER_LABEL[classified]} Loader instead."
            )

        def factory():
            model = load_llada_aux_model(self.KIND, path, CONFIG_DIR)
            setattr(model, SOURCE_ATTR, name)
            return model

        identity = _component_cache_key(name)
        return (_load_component_logged(self.KIND, identity, factory),)


class LLaDAImageQueryFormerLoader(_LLaDAAuxLoaderBase):
    KIND = "queryformer"
    INPUT_LABEL = "queryformer"
    RETURN_TYPES = ("LLADA_QUERYFORMER",)
    RETURN_NAMES = ("queryformer",)


class LLaDAImageTextProjectionLoader(_LLaDAAuxLoaderBase):
    KIND = "text_projection"
    INPUT_LABEL = "text_projection"
    RETURN_TYPES = ("LLADA_TEXT_PROJECTION",)
    RETURN_NAMES = ("text_projection",)


class LLaDAImageSigVQLoader(_LLaDAAuxLoaderBase):
    KIND = "sigvq"
    INPUT_LABEL = "sigvq"
    RETURN_TYPES = ("LLADA_SIGVQ",)
    RETURN_NAMES = ("sigvq",)


# ---------------------------------------------------------------------------
# LoRA node
# ---------------------------------------------------------------------------

def _lora_files():
    names = []
    try:
        names = list(folder_paths.get_filename_list("loras"))
    except Exception:
        names = []
    return sorted(set(names), key=str.lower)


def _lora_full_path(name: str) -> Path:
    try:
        full = folder_paths.get_full_path("loras", name)
    except Exception:
        full = None
    if full:
        return Path(full)
    for root in _folder_roots("loras"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve LoRA file: {name!r}")


def _components_with_lora(components: dict, lora):
    """Return ``components`` with the LoRA handle applied as bypasses.

    The handle's keys are matched against both the transformer and the text
    encoder (keys that resolve in neither model are simply not part of that
    adapter's scope). Applied clones are cached per component identity so
    repeated queue runs keep the assembled-pipeline cache effective. Raises
    when nothing matched at all — a silently ineffective LoRA is treated as
    an error, never as a no-op.
    """
    if lora is None:
        return components

    started = time.monotonic()
    sd = lora.tensors()
    patched = {}
    for name in ("transformer", "text_encoder"):
        comp = components[name]
        if comp is None:
            continue
        key = (id(comp), lora.name, lora.strength)
        cached = _LORA_CACHE.get(key)
        if cached is not None:
            ref, patched_model = cached
            if ref() is comp:
                _LORA_CACHE.move_to_end(key)
                patched[name] = patched_model
                continue
            del _LORA_CACHE[key]

        patched_model = llada_lora.apply_lora(comp, sd, lora.strength, strict=False)
        if patched_model is not comp:
            _LORA_CACHE[key] = (weakref.ref(comp), patched_model)
            while len(_LORA_CACHE) > _LORA_CACHE_LIMIT:
                _LORA_CACHE.popitem(last=False)
            patched[name] = patched_model

    if not patched:
        raise RuntimeError(
            f"LoRA {lora.name!r} matched no layer of the connected transformer "
            "or text encoder. Expected PEFT-style keys ending in "
            "'<module path>.lora_A.weight' / '.lora_B.weight'."
        )
    log.info("LLaDA LoRA intervention ready in %.1fs", time.monotonic() - started)
    return {**components, **patched}


class LLaDAImageLoRALoader:
    """Load one PEFT-style LoRA file as an independent intervention handle.

    This node does NOT wrap the base components: it only produces a
    lightweight ``LLADA_LORA`` handle. Connect that handle to the ``lora``
    input of the sampling node (Text to Image / Edit); the adapter is applied
    around the transformer / text encoder right before the pipeline is
    assembled. Leave the input unconnected (or toggle ``enabled`` off) to run
    without any intervention.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = list(_lora_files()) or ["<no LoRAs found>"]
        return {
            "required": {
                "lora_name": (loras,),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LLADA_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load"
    CATEGORY = "LLaDA Image"
    DESCRIPTION = (
        "Loads a PEFT-style LoRA (.safetensors with '<path>.lora_A/.lora_B.weight' "
        "keys) as an intervention handle. Connect it to the lora input of "
        "LLaDA Image Text to Image / Edit; toggle enabled off (or bypass the "
        "node) to run without it."
    )

    def load(self, lora_name, strength, enabled=True):
        if not enabled:
            log.info("LLaDA LoRA %s: disabled, no intervention", lora_name)
            return (None,)

        if lora_name == "<no LoRAs found>":
            raise RuntimeError(
                "No LoRA file found under ComfyUI/models/loras. Drop a "
                "PEFT-style .safetensors LoRA file there, refresh the node, "
                "then select it."
            )

        path = _lora_full_path(lora_name)
        strength = float(strength)
        key = (lora_name, strength)
        handle = _LORA_HANDLE_CACHE.get(key)
        if handle is None:
            handle = llada_lora.LoraHandle(lora_name, path, strength)
            _LORA_HANDLE_CACHE[key] = handle
        return (handle,)


# ---------------------------------------------------------------------------
# Sampling nodes (component inputs, assembly happens here)
# ---------------------------------------------------------------------------

_VARIANT_CHOICES = (["Auto", "Base", "Turbo"], {"default": "Auto"})


class LLaDAImageTextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_model": ("LLADA_DIFFUSION_MODEL",),
                "text_encoder": ("LLADA_TEXT_ENCODER",),
                "vae": ("LLADA_VAE",),
                "queryformer": ("LLADA_QUERYFORMER",),
                "text_projection": ("LLADA_TEXT_PROJECTION",),
                "sigvq": ("LLADA_SIGVQ",),
                "variant": _VARIANT_CHOICES,
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 640, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "lora": ("LLADA_LORA",),
                # Optional speed/VRAM trade: keep INT8 denoiser weights on GPU.
                "weights_on_gpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "LLaDA Image"
    DESCRIPTION = (
        "Text-to-image generation. Receives the six LLaDA component models "
        "from their loaders and assembles the pipeline here (shared through "
        "the component cache). The optional lora handle from the LoRA loader "
        "is applied around the transformer/text encoder before assembly."
    )

    def generate(
        self,
        diffusion_model,
        text_encoder,
        vae,
        queryformer,
        text_projection,
        sigvq,
        variant,
        prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
        negative_prompt="",
        lora=None,
        weights_on_gpu=False,
    ):
        components = _components_with_lora(
            {
                "transformer": diffusion_model,
                "text_encoder": text_encoder,
                "vae": vae,
                "queryformer": queryformer,
                "text_projection": text_projection,
                "sigvq": sigvq,
            },
            lora,
        )
        pipe = _assembled_pipeline(components, variant, weights_on_gpu)

        do_cfg = float(guidance_scale) > 1.0
        neg_prompt = negative_prompt if negative_prompt.strip() else None
        embeds_kwargs, embeds_hit = _prompt_embeds_for(
            pipe, components, prompt, neg_prompt, do_cfg
        )

        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        started = time.monotonic()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        try:
            result = pipe(
                prompt=None,
                generation_mode="text",
                height=int(height),
                width=int(width),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
                output_type="pil",
                **embeds_kwargs,
            )
        finally:
            _log_generation_stats("T2I", started, embeds_hit)

        return (_pil_to_comfy(result.images),)


class LLaDAImageEdit:
    """Native LLaDA-Image editing.

    This is NOT img2img/denoise-strength emulation. The source IMAGE is sent
    through the model's official generation_mode="editing" path, which uses
    LLaDA-Image's SigVQ/image-conditioning components.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_model": ("LLADA_DIFFUSION_MODEL",),
                "text_encoder": ("LLADA_TEXT_ENCODER",),
                "vae": ("LLADA_VAE",),
                "queryformer": ("LLADA_QUERYFORMER",),
                "text_projection": ("LLADA_TEXT_PROJECTION",),
                "sigvq": ("LLADA_SIGVQ",),
                "variant": _VARIANT_CHOICES,
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe the edit you want to make",
                    },
                ),
                "width": ("INT", {"default": 640, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "lora": ("LLADA_LORA",),
                # Optional speed/VRAM trade: keep INT8 denoiser weights on GPU.
                "weights_on_gpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "LLaDA Image"
    DESCRIPTION = (
        "Native LLaDA-Image editing using generation_mode='editing' and the "
        "model's SigVQ image-conditioning path. The optional lora handle "
        "from the LoRA loader is applied before assembly."
    )

    def edit(
        self,
        diffusion_model,
        text_encoder,
        vae,
        queryformer,
        text_projection,
        sigvq,
        variant,
        image,
        prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
        negative_prompt="",
        lora=None,
        weights_on_gpu=False,
    ):
        width = int(width)
        height = int(height)

        # Official editing path requires dimensions divisible by 32.
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError(
                f"LLaDA native editing requires width/height divisible by 32; "
                f"got {width}x{height}."
            )

        if image is None:
            raise ValueError("LLaDA Image Edit requires a source IMAGE.")

        components = _components_with_lora(
            {
                "transformer": diffusion_model,
                "text_encoder": text_encoder,
                "vae": vae,
                "queryformer": queryformer,
                "text_projection": text_projection,
                "sigvq": sigvq,
            },
            lora,
        )
        pipe = _assembled_pipeline(components, variant, weights_on_gpu)

        do_cfg = float(guidance_scale) > 1.0
        neg_prompt = negative_prompt if negative_prompt.strip() else None
        embeds_kwargs, embeds_hit = _prompt_embeds_for(
            pipe, components, prompt, neg_prompt, do_cfg
        )

        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        source_image = _comfy_image_to_pil(image)

        started = time.monotonic()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        try:
            result = pipe(
                prompt=None,
                image=source_image,
                generation_mode="editing",
                height=height,
                width=width,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
                output_type="pil",
                **embeds_kwargs,
            )
        finally:
            _log_generation_stats("Edit", started, embeds_hit)

        if not hasattr(result, "images") or not result.images:
            raise RuntimeError("LLaDA native editing returned no images.")

        return (_pil_to_comfy(result.images),)


NODE_CLASS_MAPPINGS = {
    "LLaDAImageDiffusionModelLoader": LLaDAImageDiffusionModelLoader,
    "LLaDAImageTextEncoderLoader": LLaDAImageTextEncoderLoader,
    "LLaDAImageVAELoader": LLaDAImageVAELoader,
    "LLaDAImageQueryFormerLoader": LLaDAImageQueryFormerLoader,
    "LLaDAImageTextProjectionLoader": LLaDAImageTextProjectionLoader,
    "LLaDAImageSigVQLoader": LLaDAImageSigVQLoader,
    "LLaDAImageLoRALoader": LLaDAImageLoRALoader,
    "LLaDAImageTextToImage": LLaDAImageTextToImage,
    "LLaDAImageEdit": LLaDAImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLaDAImageDiffusionModelLoader": "LLaDA Image Diffusion Model Loader",
    "LLaDAImageTextEncoderLoader": "LLaDA Image Text Encoder Loader",
    "LLaDAImageVAELoader": "LLaDA Image VAE Loader",
    "LLaDAImageQueryFormerLoader": "LLaDA Image QueryFormer Loader",
    "LLaDAImageTextProjectionLoader": "LLaDA Image Text Projection Loader",
    "LLaDAImageSigVQLoader": "LLaDA Image SigVQ Loader",
    "LLaDAImageLoRALoader": "LLaDA Image LoRA Loader",
    "LLaDAImageTextToImage": "LLaDA Image Text to Image",
    "LLaDAImageEdit": "LLaDA Image Edit",
}
