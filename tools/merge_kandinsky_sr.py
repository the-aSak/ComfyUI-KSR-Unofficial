#!/usr/bin/env python3
"""Merge the transformer of a Kandinsky ImageSR 2.0 Diffusers snapshot into ONE .safetensors file
for ComfyUI (models/diffusion_models/).

  tensors         all transformer shards (incl. the baked time/xattn tables), names unchanged
  __metadata__    transformer config, VAE config (latent scaling factor) and model_index.json as JSON strings

The VAE is not included: Kandinsky ImageSR uses the FLUX.1 VAE (ae.safetensors), see tools/compare_vae.py.

Pure standard library: no torch, no safetensors package, constant RAM use.
Tensor bytes are copied 1:1 (no dtype conversion); --verify re-reads every tensor
from source and target and compares the raw bytes.

Usage:
  python merge_kandinsky_sr.py <snapshot_dir> <output.safetensors> [--verify]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

FORMAT_VERSION = "1"
CHUNK = 64 * 1024 * 1024
DTYPE_SIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4,
              "F16": 2, "BF16": 2, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1,
              "F8_E4M3": 1, "F8_E5M2": 1}


def read_header(path: Path) -> tuple[dict, int]:
    """Return (tensor entries, absolute offset where the data buffer starts)."""
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def component_files(component_dir: Path) -> list[Path]:
    index = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [component_dir / name for name in sorted(set(weight_map.values()))]
    single = component_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors weights found in {component_dir}")


def collect(snapshot: Path, transformer_only: bool = False) -> list[dict]:
    """One record per tensor: output name, dtype, shape, source file and byte range."""
    records = []
    components = {"transformer": ""} if transformer_only else {"transformer": "transformer.", "vae": "vae."}
    for component, prefix in components.items():
        for file in component_files(snapshot / component):
            header, data_start = read_header(file)
            for name, info in header.items():
                begin, end = info["data_offsets"]
                records.append({
                    "name": f"{prefix}{name}",
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                    "file": file,
                    "start": data_start + begin,
                    "length": end - begin,
                })
    names = [r["name"] for r in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate tensor names across shards")
    # Larger element size first: every tensor's byte length is a multiple of its element
    # size and the header is padded to 8, so this ordering keeps all tensors aligned.
    records.sort(key=lambda r: (-DTYPE_SIZE.get(r["dtype"], 1), r["name"]))
    return records


def build_metadata(snapshot: Path) -> dict[str, str]:
    def compact(path: Path) -> str:
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), separators=(",", ":"))

    return {
        "format": "kandinsky_image_sr_single_file",
        "format_version": FORMAT_VERSION,
        "transformer_config": compact(snapshot / "transformer" / "config.json"),
        "vae_config": compact(snapshot / "vae" / "config.json"),
        "model_index": compact(snapshot / "model_index.json"),
    }


def copy_range(src, dst, length: int) -> None:
    while length:
        block = src.read(min(CHUNK, length))
        if not block:
            raise OSError("unexpected end of source file")
        dst.write(block)
        length -= len(block)


def merge(snapshot: Path, output: Path, transformer_only: bool = False) -> list[dict]:
    records = collect(snapshot, transformer_only)
    header: dict = {"__metadata__": build_metadata(snapshot)}
    offset = 0
    for r in records:
        header[r["name"]] = {"dtype": r["dtype"], "shape": r["shape"],
                             "data_offsets": [offset, offset + r["length"]]}
        r["out_start"] = offset
        offset += r["length"]

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    data_start = 8 + len(header_bytes)

    tmp = output.with_suffix(output.suffix + ".part")
    handles: dict[Path, object] = {}
    try:
        with tmp.open("wb") as dst:
            dst.write(struct.pack("<Q", len(header_bytes)))
            dst.write(header_bytes)
            total = len(records)
            for i, r in enumerate(records, 1):
                src = handles.get(r["file"]) or handles.setdefault(r["file"], r["file"].open("rb"))
                src.seek(r["start"])
                copy_range(src, dst, r["length"])
                if i % 50 == 0 or i == total:
                    print(f"\r  {i}/{total} tensors written", end="", flush=True)
        print()
    finally:
        for h in handles.values():
            h.close()
    tmp.replace(output)
    for r in records:
        r["out_start"] += data_start
    return records


def verify(records: list[dict], output: Path) -> None:
    handles: dict[Path, object] = {}
    try:
        with output.open("rb") as out:
            for i, r in enumerate(records, 1):
                src = handles.get(r["file"]) or handles.setdefault(r["file"], r["file"].open("rb"))
                src.seek(r["start"])
                out.seek(r["out_start"])
                remaining = r["length"]
                while remaining:
                    n = min(CHUNK, remaining)
                    if src.read(n) != out.read(n):
                        raise ValueError(f"byte mismatch in tensor {r['name']}")
                    remaining -= n
                if i % 50 == 0 or i == len(records):
                    print(f"\r  {i}/{len(records)} tensors verified", end="", flush=True)
        print()
    finally:
        for h in handles.values():
            h.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", type=Path, help="local copy of kandinskylab/Kandinsky-ImageSR-2.0")
    parser.add_argument("output", type=Path, help="target .safetensors file")
    parser.add_argument("--verify", action="store_true", help="compare every tensor byte-for-byte afterwards")
    args = parser.parse_args()

    if args.output.exists():
        print(f"refusing to overwrite existing file: {args.output}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"merging {args.snapshot} -> {args.output}")
    records = merge(args.snapshot, args.output, transformer_only=True)
    size_gb = args.output.stat().st_size / 1e9
    print(f"done: {len(records)} tensors, {size_gb:.2f} GB")
    if args.verify:
        verify(records, args.output)
        print("verify: all tensors identical to source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
