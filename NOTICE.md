# Third-party components and modifications

## 1. Vendored code: `kandinsky_vendor/tiling.py`

- **Source:** https://github.com/kandinskylab/kandinsky-image-sr, `src/kandinsky_image_sr/tiling.py`,
  commit `6c2405796cf7576b0cffff27acd103d667b91ca8` (2026-11-25)
- **License:** MIT, © 2026 kandinskylab (copy: `kandinsky_vendor/LICENSE`)
- **Modifications:** none. `kandinsky_vendor/__init__.py` is new and only holds this attribution.

`nodes.py` also takes the 18 calibrated timesteps (`CREATIVITY_TIMESTEPS`), the per-tile procedure (bicubic upsampling,
`lq - t * v`, latent scaling) and the blending logic from that repository's `pipeline.py` (MIT).

## 2. Re-implemented code: `native/model.py`

An independent re-implementation of the Kandinsky ImageSR baked transformer on ComfyUI primitives.
It reproduces the computation of
- `src/kandinsky_image_sr/baked_transformer.py` of the repository above (MIT, © 2026 kandinskylab), and
- the Kandinsky 5 building blocks in Diffusers, `src/diffusers/models/transformers/transformer_kandinsky.py`
  (Apache License 2.0, © The HuggingFace Team; full text: `LICENSE-APACHE-2.0`).
  `SelfAttention._rotary` follows that file's `apply_rotary` helper almost line by line.

State-dict key names are kept identical so that the original weights load unchanged.

## 3. Weights (not part of this repository)

The transformer file on Hugging Face ([hmdlsk/Kandinsky-ImageSR-2.0-ComfyUI](https://huggingface.co/hmdlsk/Kandinsky-ImageSR-2.0-ComfyUI)) is produced by `tools/merge_kandinsky_sr.py`.
Tensor values are **not** modified: no retraining, no requantization, no dtype conversion. Byte-identity can be
checked with `--verify`.

- **Source:** https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0, folder `transformer/` (main branch as of 2026-11-26)
- **License:** MIT
- **Changes:** the 3 shards are concatenated into one file (tensor names unchanged). Stored as safetensors metadata:
  `transformer/config.json` (`transformer_config`), `vae/config.json` (`vae_config`, for the latent scaling factor
  only) and `model_index.json` (`model_index`), plus `format = kandinsky_image_sr_single_file` and `format_version = 1`.

**VAE: not redistributed.** Kandinsky ImageSR bundles the FLUX.1 VAE (Black Forest Labs, Apache-2.0) as bf16. Users load
their own copy (`ae.safetensors` from black-forest-labs/FLUX.1-schnell). `tools/compare_vae.py` verifies that both are
the same weights.

At load time the nodes can convert attention/feed-forward weights to fp8 (`weight_dtype`) and, on CUDA, store the
modulation layers as bf16 (numerically identical under autocast, see `nodes.py`). The file itself is not changed.
