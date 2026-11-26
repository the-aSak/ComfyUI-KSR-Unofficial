#!/usr/bin/env python3
"""Check whether the VAE bundled with Kandinsky ImageSR 2.0 equals the FLUX.1 VAE (ae.safetensors).

The Kandinsky copy is in Diffusers format (and possibly lower precision); ae.safetensors is in the
original BFL/LDM format. Keys are converted with ComfyUI's own converter, then every tensor is
compared after casting the FLUX tensor to the Kandinsky tensor's dtype.

Run with ComfyUI's Python (needs torch + safetensors):
  python compare_vae.py <ComfyUI dir> <Kandinsky snapshot>/vae/diffusion_pytorch_model.safetensors <path>/ae.safetensors
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_converter(comfy_root: Path):
    # comfy/diffusers_convert.py only imports re/torch/logging, so it can be loaded on its own
    spec = importlib.util.spec_from_file_location("diffusers_convert", comfy_root / "comfy" / "diffusers_convert.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_vae_state_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comfyui", type=Path)
    parser.add_argument("kandinsky_vae", type=Path)
    parser.add_argument("flux_ae", type=Path)
    args = parser.parse_args()

    kandinsky = load_converter(args.comfyui)(load_file(args.kandinsky_vae))
    flux = load_file(args.flux_ae)

    only_k, only_f = sorted(kandinsky.keys() - flux.keys()), sorted(flux.keys() - kandinsky.keys())
    if only_k or only_f:
        print(f"key mismatch: {len(only_k)} only in Kandinsky {only_k[:5]}, {len(only_f)} only in FLUX {only_f[:5]}")
        return 1

    dtypes_k = {str(v.dtype) for v in kandinsky.values()}
    dtypes_f = {str(v.dtype) for v in flux.values()}
    worst, worst_key, differing = 0.0, None, 0
    for key, k in kandinsky.items():
        f = flux[key]
        if k.shape != f.shape:
            print(f"shape mismatch {key}: {tuple(k.shape)} vs {tuple(f.shape)}")
            return 1
        diff = (k.float() - f.to(k.dtype).float()).abs().max().item()
        if diff > 0:
            differing += 1
            if diff > worst:
                worst, worst_key = diff, key

    print(f"{len(kandinsky)} tensors, dtypes Kandinsky {dtypes_k}, FLUX {dtypes_f}")
    if differing == 0:
        same_dtype = dtypes_k == dtypes_f
        print("IDENTICAL" + ("" if same_dtype else " after casting FLUX to the Kandinsky dtype (Kandinsky ships a downcast copy)"))
        return 0
    print(f"DIFFERENT: {differing} tensors differ, largest |diff| = {worst:.3e} in {worst_key}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
