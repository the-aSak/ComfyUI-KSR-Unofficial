"""Kandinsky ImageSR 2.0 baked transformer, re-implemented on ComfyUI primitives.

The math mirrors `baked_transformer.py` of https://github.com/kandinskylab/kandinsky-image-sr (which builds on the
Diffusers Kandinsky 5 modules) operation by operation, including where it computes in fp32. Differences:
  * Linear/RMSNorm come from `comfy.ops`, so weights can be stored in fp8 and the model can be
    partially offloaded by ComfyUI (weights are cast to the compute dtype on use).
  * Attention goes through ComfyUI's `optimized_attention` (SDPA, Sage, Flash, ... per CLI flags).
  * RoPE is computed for the actual grid instead of slicing a precomputed 128-entry table
    (same values: outer(position, frequency) in fp32).
State-dict keys are identical to the Diffusers model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from comfy.ldm.modules.attention import optimized_attention


def _layer_norm_f32(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],))


class Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params, ops, device):
        super().__init__()
        # fp32 in the checkpoint (Diffusers `_keep_in_fp32_modules`); never fp8 (bf16 on CUDA, see load_native)
        self.out_layer = ops.Linear(time_dim, num_params * model_dim, device=device, dtype=torch.float32)

    def forward(self, x):
        return self.out_layer(F.silu(x))


class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim, ops, device, dtype):
        super().__init__()
        self.num_heads = dim // head_dim
        kw = {"device": device, "dtype": dtype}
        self.to_query = ops.Linear(dim, dim, bias=True, **kw)
        self.to_key = ops.Linear(dim, dim, bias=True, **kw)
        self.to_value = ops.Linear(dim, dim, bias=True, **kw)
        self.query_norm = ops.RMSNorm(head_dim, device=device, dtype=dtype)
        self.key_norm = ops.RMSNorm(head_dim, device=device, dtype=dtype)
        self.out_layer = ops.Linear(dim, dim, bias=True, **kw)

    @staticmethod
    def _rotary(x, rope):
        x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
        out = (rope * x_).sum(dim=-1)
        return out.reshape(*x.shape).to(torch.bfloat16)  # as upstream (hard-coded bf16)

    def forward(self, x, rope):
        b, s, _ = x.shape
        q = self.to_query(x).reshape(b, s, self.num_heads, -1)
        k = self.to_key(x).reshape(b, s, self.num_heads, -1)
        v = self.to_value(x).reshape(b, s, self.num_heads, -1)
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        q = self._rotary(q, rope).type_as(q)
        k = self._rotary(k, rope).type_as(k)
        out = optimized_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.num_heads, skip_reshape=True)
        return self.out_layer(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, ops, device, dtype):
        super().__init__()
        self.in_layer = ops.Linear(dim, ff_dim, bias=False, device=device, dtype=dtype)
        self.out_layer = ops.Linear(ff_dim, dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(x)))


class BakedBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, num_timesteps, ops, device, dtype):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9, ops, device)
        self.self_attention = SelfAttention(model_dim, head_dim, ops, device, dtype)
        self.feed_forward = FeedForward(model_dim, ff_dim, ops, device, dtype)
        self.register_buffer("xattn_bias_table", torch.empty(num_timesteps, model_dim, dtype=torch.float32, device=device))

    def forward(self, x, time_embed, rope, t_index):
        attn_params, _, ff_params = torch.chunk(self.visual_modulation(time_embed).unsqueeze(1), 3, dim=-1)

        shift, scale, gate = torch.chunk(attn_params, 3, dim=-1)
        out = (_layer_norm_f32(x) * (scale.float() + 1.0) + shift.float()).type_as(x)
        out = self.self_attention(out, rope)
        x = (x.float() + gate.float() * out.float()).type_as(x)

        bias = self.xattn_bias_table.to(x.device).index_select(0, t_index).unsqueeze(1)
        x = (x.float() + bias).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = (_layer_norm_f32(x) * (scale.float() + 1.0) + shift.float()).type_as(x)
        out = self.feed_forward(out)
        return (x.float() + gate.float() * out.float()).type_as(x)


class VisualEmbeddings(nn.Module):
    def __init__(self, in_dim, model_dim, patch_size, ops, device, dtype):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = ops.Linear(math.prod(patch_size) * in_dim, model_dim, device=device, dtype=dtype)

    def forward(self, x):  # x: B T H W C
        b, t, h, w, c = x.shape
        pt, ph, pw = self.patch_size
        x = x.view(b, t // pt, pt, h // ph, ph, w // pw, pw, c).permute(0, 1, 3, 5, 2, 4, 6, 7).flatten(4, 7)
        return self.in_layer(x)


class OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, out_dim, patch_size, ops, device, dtype):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2, ops, device)
        self.out_layer = ops.Linear(model_dim, math.prod(patch_size) * out_dim, bias=True, device=device, dtype=dtype)

    def forward(self, x, time_embed):  # x: B T H W D
        shift, scale = torch.chunk(self.modulation(time_embed).unsqueeze(1), 2, dim=-1)
        x = (_layer_norm_f32(x) * (scale.float()[:, None, None] + 1.0) + shift.float()[:, None, None]).type_as(x)
        x = self.out_layer(x)
        b, t, h, w, _ = x.shape
        pt, ph, pw = self.patch_size
        return (
            x.view(b, t, h, w, -1, pt, ph, pw)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2).flatten(2, 3).flatten(3, 4)
        )


def _rope_3d(axes_dims, grid, device, max_period=10000.0):
    """Kandinsky5RoPE3D for positions arange(T), arange(H), arange(W) and scale factor 1."""
    t, h, w = grid
    parts = []
    for axes_dim, n in zip(axes_dims, (t, h, w)):
        half = axes_dim // 2
        freq = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        parts.append(torch.outer(torch.arange(n, dtype=torch.float32), freq).to(device))
    args = torch.cat(
        [
            parts[0].view(1, t, 1, 1, -1).repeat(1, 1, h, w, 1),
            parts[1].view(1, 1, h, 1, -1).repeat(1, t, 1, w, 1),
            parts[2].view(1, 1, 1, w, -1).repeat(1, t, h, 1, 1),
        ],
        dim=-1,
    )
    cos, sin = torch.cos(args), torch.sin(args)
    rope = torch.stack([cos, -sin, sin, cos], dim=-1).view(*args.shape, 2, 2)
    return rope.unsqueeze(-4).flatten(1, 3)  # 1, S, 1, D/2, 2, 2


class KandinskySRTransformer(nn.Module):
    def __init__(self, config: dict, ops, device=None, dtype=None):
        super().__init__()
        if config.get("visual_cond", False):
            raise ValueError("visual_cond checkpoints are not supported")
        if config.get("attention_type", "regular") != "regular":
            raise ValueError("only regular attention is supported")
        model_dim = config.get("model_dim", 2560)
        time_dim = config.get("time_dim", 512)
        ff_dim = config.get("ff_dim", 10240)
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.axes_dims = tuple(config.get("axes_dims", (16, 24, 24)))
        self.baked_timesteps = tuple(float(v) for v in config["baked_timesteps"])
        self.baked_latent_size = int(config.get("baked_latent_size", 128))
        n_t = len(self.baked_timesteps)

        self.visual_embeddings = VisualEmbeddings(config.get("in_visual_dim", 16), model_dim, self.patch_size, ops, device, dtype)
        self.visual_transformer_blocks = nn.ModuleList(
            BakedBlock(model_dim, time_dim, ff_dim, sum(self.axes_dims), n_t, ops, device, dtype)
            for _ in range(config.get("num_visual_blocks", 50))
        )
        self.out_layer = OutLayer(model_dim, time_dim, config.get("out_visual_dim", 16), self.patch_size, ops, device, dtype)
        self.register_buffer("time_embed_table", torch.empty(n_t, time_dim, dtype=torch.float32, device=device))

    def timestep_index(self, t: float) -> int:
        for i, value in enumerate(self.baked_timesteps):
            if abs(value - t) <= 1e-6:
                return i
        raise ValueError(f"t={t} is not baked into this checkpoint; available: {list(self.baked_timesteps)}")

    def forward(self, latents: torch.Tensor, t_index: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        """latents: B T H W C (scaled VAE latents). Returns the flow prediction, same layout, fp32."""
        _, t, h, w, _ = latents.shape
        if (h, w) != (self.baked_latent_size, self.baked_latent_size):
            raise ValueError(f"expected {self.baked_latent_size}x{self.baked_latent_size} latents, got {h}x{w}")
        time_embed = self.time_embed_table.to(latents.device).index_select(0, t_index)
        x = self.visual_embeddings(latents.to(compute_dtype))
        grid = x.shape[1:4]
        rope = _rope_3d(self.axes_dims, grid, x.device)
        x = x.flatten(1, 3)
        for block in self.visual_transformer_blocks:
            x = block(x, time_embed, rope, t_index)
        x = x.reshape(x.shape[0], *grid, -1)
        return self.out_layer(x, time_embed).float()
