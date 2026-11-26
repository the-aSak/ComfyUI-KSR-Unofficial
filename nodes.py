"""Native ComfyUI nodes for Kandinsky ImageSR 2.0.

Pipeline per image, in three phases so each model is loaded once instead of once per tile:
  1. VAE encode all bicubic-upsampled 1024 px tiles (ComfyUI VAE, e.g. FLUX.1 ae.safetensors)
  2. one transformer step per tile: corrected = lq - t * v(lq, t)
  3. VAE decode, Hanning-window blending into an output buffer in system RAM
Tiling, window and timestep presets follow the reference pipeline
(https://github.com/kandinskylab/kandinsky-image-sr, src/kandinsky_image_sr/pipeline.py).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths

from .kandinsky_vendor.tiling import boundary_aware_window, pad_to_min_tile, tile_positions
from .native.model import KandinskySRTransformer

log = logging.getLogger("KandinskyImageSR")

# From the reference pipeline (kandinsky_image_sr/pipeline.py, MIT).
CREATIVITY_TIMESTEPS = {
    2: (0.07, 0.086, 0.118, 0.134, 0.15, 0.18, 0.21, 0.27, 0.30),
    4: (0.17, 0.1885, 0.207, 0.2255, 0.244, 0.278, 0.312, 0.346, 0.38),
}
FLUX_VAE_SCALING = 0.3611  # pipeline scales latents by vae.config.scaling_factor only (no shift)
WEIGHT_DTYPES = {"default": None, "fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}
# Large matmul weights only; embeddings, out layer, modulation (fp32) and norms stay as stored.
QUANTIZABLE = ("self_attention.to_query.weight", "self_attention.to_key.weight", "self_attention.to_value.weight",
               "self_attention.out_layer.weight", "feed_forward.in_layer.weight", "feed_forward.out_layer.weight")
ACTIVATION_BYTES_PER_TILE = 768 * 1024**2
VAE_CHUNK = 8  # tiles per VAE encode/decode call; ComfyUI splits further by free VRAM


class KandinskySRModel:
    """What travels between the nodes: the patcher plus a few constants."""

    def __init__(self, patcher, compute_dtype, vae_scaling):
        self.patcher = patcher
        self.compute_dtype = compute_dtype
        self.vae_scaling = vae_scaling

    @property
    def transformer(self) -> KandinskySRTransformer:
        return self.patcher.model


def _read_metadata(path: str) -> dict:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return f.metadata() or {}


def load_native(path: str, weight_dtype: str = "default") -> KandinskySRModel:
    meta = _read_metadata(path)
    if "transformer_config" not in meta:
        raise ValueError("file has no Kandinsky ImageSR metadata; create it with tools/merge_kandinsky_sr.py or download it (see README)")
    config = json.loads(meta["transformer_config"])
    vae_scaling = json.loads(meta["vae_config"]).get("scaling_factor", FLUX_VAE_SCALING) if "vae_config" in meta else FLUX_VAE_SCALING

    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if any(k.startswith("transformer.") for k in sd):  # all-in-one file: drop the bundled VAE
        sd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
    target = WEIGHT_DTYPES[weight_dtype]
    if target is not None:
        sd = {k: v.to(target) if k.endswith(QUANTIZABLE) else v for k, v in sd.items()}

    load_device = mm.get_torch_device()
    compute_dtype = torch.bfloat16 if mm.should_use_bf16(load_device) else torch.float32
    if load_device.type == "cuda" and compute_dtype == torch.bfloat16:
        # The checkpoint keeps the modulation layers in fp32, but the reference runs them under CUDA bf16 autocast,
        # which rounds weight and bias to bf16 on every call. Rounding once here gives the same result and saves
        # ~1.2 GB of VRAM.
        sd = {k: v.to(torch.bfloat16) if "modulation.out_layer." in k and v.dtype == torch.float32 else v
              for k, v in sd.items()}
    model = KandinskySRTransformer(config, comfy.ops.manual_cast, device="meta", dtype=torch.bfloat16)
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval().requires_grad_(False)

    patcher_cls = getattr(comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher)
    patcher = patcher_cls(model, load_device=load_device, offload_device=mm.unet_offload_device())
    return KandinskySRModel(patcher, compute_dtype, vae_scaling)


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class _Progress:
    """ComfyUI progress bar (UI) plus a tqdm bar in the console; samples device VRAM use."""

    def __init__(self, total, device):
        self.ui = comfy.utils.ProgressBar(total)
        self.console = tqdm(total=total, unit="tile", dynamic_ncols=True)
        self.device = device
        self.peak_device_used = 0

    def sample_vram(self):
        # Device-wide (all processes). ComfyUI's dynamic VRAM manager allocates weights outside
        # torch's caching allocator, so torch.cuda.max_memory_allocated would miss them.
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            self.peak_device_used = max(self.peak_device_used, total - free)

    def phase(self, name):
        self.console.set_description(f"Kandinsky SR {name}")

    def step(self, n=1):
        mm.throw_exception_if_processing_interrupted()
        self.sample_vram()
        self.ui.update(n)
        self.console.update(n)

    def close(self):
        self.console.close()


@torch.no_grad()
def upscale_image(sr: KandinskySRModel, vae, image_hwc: torch.Tensor, factor: int, t: float,
                  overlap: float, tile_batch: int, progress: _Progress, timings: dict) -> torch.Tensor:
    model = sr.transformer
    tile = model.baked_latent_size * 8
    lq_tile = tile // factor
    stride = max(1, int(lq_tile * (1.0 - overlap)))
    image = image_hwc.permute(2, 0, 1).unsqueeze(0).float().clamp(0.0, 1.0).cpu()
    padded, height, width = pad_to_min_tile(image, lq_tile)
    tops = tile_positions(padded.shape[-2], lq_tile, stride)
    lefts = tile_positions(padded.shape[-1], lq_tile, stride)
    cells = [(r, c) for r in range(len(tops)) for c in range(len(lefts))]

    # 1. encode
    start = time.perf_counter()
    progress.phase("encode")
    latents = []
    for chunk in _chunks(cells, VAE_CHUNK):
        tiles = torch.cat([
            F.interpolate(padded[:, :, tops[r]:tops[r] + lq_tile, lefts[c]:lefts[c] + lq_tile],
                          size=(tile, tile), mode="bicubic", align_corners=False)
            for r, c in chunk
        ]).clamp(0.0, 1.0)
        latents.append(vae.encode(tiles.movedim(1, -1)).float().cpu() * sr.vae_scaling)
        progress.step(len(chunk))
    lq = torch.cat(latents)
    timings["encode"] += time.perf_counter() - start

    # 2. one flow step per tile
    start = time.perf_counter()
    progress.phase("transformer")
    mm.load_models_gpu([sr.patcher], memory_required=ACTIVATION_BYTES_PER_TILE * tile_batch)
    device = sr.patcher.load_device
    t_index = model.timestep_index(t)
    autocast = (torch.autocast(device_type=device.type, dtype=sr.compute_dtype)
                if device.type == "cuda" and sr.compute_dtype != torch.float32 else nullcontext())
    corrected = []
    for batch in torch.split(lq, tile_batch):
        x = batch.to(device)
        index = torch.full((x.shape[0],), t_index, device=device, dtype=torch.long)
        with autocast:
            v = model(x.permute(0, 2, 3, 1).unsqueeze(1), index, sr.compute_dtype)
        v = v.squeeze(1).permute(0, 3, 1, 2)
        corrected.append((x - t * v).cpu())
        progress.sample_vram()  # before the step: the transformer is still resident here
        progress.step(x.shape[0])
    del lq
    timings["transformer"] += time.perf_counter() - start

    # 3. decode + blend (output buffer in RAM)
    start = time.perf_counter()
    progress.phase("decode")
    out_h, out_w = padded.shape[-2] * factor, padded.shape[-1] * factor
    acc = torch.zeros(1, 3, out_h, out_w)
    weights = torch.zeros(1, 1, out_h, out_w)
    for chunk, batch in zip(_chunks(cells, VAE_CHUNK), torch.split(torch.cat(corrected), VAE_CHUNK)):
        decoded = vae.decode(batch / sr.vae_scaling).float().cpu().movedim(-1, 1)  # B 3 H W in [0, 1]
        for i, (r, c) in enumerate(chunk):
            window = boundary_aware_window(tile, tile, top=r == 0, bottom=r == len(tops) - 1,
                                           left=c == 0, right=c == len(lefts) - 1, device=torch.device("cpu"))[None, None]
            y, x = tops[r] * factor, lefts[c] * factor
            acc[:, :, y:y + tile, x:x + tile] += decoded[i:i + 1] * window
            weights[:, :, y:y + tile, x:x + tile] += window
        progress.step(len(chunk))
    timings["decode"] += time.perf_counter() - start

    out = (acc / weights.clamp_min(1e-6))[:, :, :height * factor, :width * factor]
    return out.permute(0, 2, 3, 1)


class KandinskySRModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (list(WEIGHT_DTYPES), {"default": "default",
                                 "tooltip": "fp8 halves the transformer's memory (attention/FF weights only). Slight quality cost."}),
            }
        }

    RETURN_TYPES = ("KANDINSKY_SR_MODEL",)
    RETURN_NAMES = ("sr_model",)
    FUNCTION = "load"
    CATEGORY = "image/upscaling/KSR (unofficial)"

    def load(self, model_name, weight_dtype):
        return (load_native(folder_paths.get_full_path_or_raise("diffusion_models", model_name), weight_dtype),)


class KandinskySRUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sr_model": ("KANDINSKY_SR_MODEL",),
                "vae": ("VAE", {"tooltip": "FLUX.1 VAE (ae.safetensors)"}),
                "image": ("IMAGE",),
                "upscale_factor": (["4", "2"], {"default": "4"}),
                "creativity": ("INT", {"default": 4, "min": 0, "max": 8,
                               "tooltip": "Calibrated presets; 4 = model default. Higher = more invented detail."}),
                "overlap": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.75, "step": 0.05,
                            "tooltip": "Tile overlap (fraction of the low-res tile). 0.25 = reference."}),
                "tile_batch_size": ("INT", {"default": 1, "min": 1, "max": 16,
                                    "tooltip": "Tiles per transformer forward pass. Higher = slightly faster, more VRAM."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling/KSR (unofficial)"

    def upscale(self, sr_model, vae, image, upscale_factor, creativity, overlap, tile_batch_size):
        factor = int(upscale_factor)
        t = CREATIVITY_TIMESTEPS[factor][creativity]
        model = sr_model.transformer
        lq_tile = model.baked_latent_size * 8 // factor
        stride = max(1, int(lq_tile * (1.0 - overlap)))
        _, h, w, _ = image.shape
        n_tiles = len(tile_positions(max(h, lq_tile), lq_tile, stride)) * len(tile_positions(max(w, lq_tile), lq_tile, stride))

        device = sr_model.patcher.load_device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings = {"encode": 0.0, "transformer": 0.0, "decode": 0.0}
        progress = _Progress(3 * n_tiles * image.shape[0], device)
        start = time.perf_counter()
        try:
            outputs = [upscale_image(sr_model, vae, img[..., :3], factor, t, overlap, tile_batch_size, progress, timings)
                       for img in image]
        finally:
            progress.close()

        total = time.perf_counter() - start
        peak = (f", VRAM in use on device {progress.peak_device_used / 1024**3:.2f} GB (all processes, sampled)"
                f", torch peak allocated {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB (activations)"
                if device.type == "cuda" else "")
        log.info("Kandinsky SR x%d c%d (t=%s): %d image(s) %dx%d, %d tiles each, %.1f s total "
                 "(encode %.1f / transformer %.1f / decode %.1f)%s",
                 factor, creativity, t, image.shape[0], w, h, n_tiles, total,
                 timings["encode"], timings["transformer"], timings["decode"], peak)
        return (torch.cat(outputs).to(mm.intermediate_dtype() if hasattr(mm, "intermediate_dtype") else torch.float32),)


NODE_CLASS_MAPPINGS = {
    "KandinskySRModelLoader": KandinskySRModelLoader,
    "KandinskySRUpscale": KandinskySRUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KandinskySRModelLoader": "Kandinsky ImageSR Model Loader (unofficial)",
    "KandinskySRUpscale": "Kandinsky ImageSR Upscale (unofficial)",
}
