"""MLA, SwiGLU MLP and the sigmoid-routed MoE (Ling-3.0 / BailingMoeV3 style)."""
from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kda import RMSNorm


# ----------------------------------------------------------------------------- RoPE (interleaved)
@lru_cache(maxsize=16)
def _inv_freq(dim: int, theta: float, device: str) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))


def rope_cos_sin(T: int, dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.outer(torch.arange(T, dtype=torch.float32, device=device), _inv_freq(dim, theta, str(device)))
    return freqs.cos(), freqs.sin()  # [T, dim/2]


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B,T,h,d] (or [B,T,1,d]); pairs (x0,x1),(x2,x3)... are rotated."""
    xf = x.float()
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)
    return out.to(x.dtype)


# ----------------------------------------------------------------------------- MLA
class MultiLatentAttention(nn.Module):
    """MLA with low-rank q / kv, decoupled shared RoPE key and a head-wise sigmoid output gate.

    Training-time form: K/V are re-expanded from the latent (the latent-only cache trick matters for
    decode and is not needed for this training stack)."""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.nh = cfg.num_attention_heads
        self.dn, self.dr, self.dv = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        self.dqk = self.dn + self.dr
        self.kv_rank = cfg.kv_lora_rank
        H = cfg.hidden_size
        # q path and kv-latent path read the same input -> one fused down projection
        self.q_rank = cfg.q_lora_rank
        self.down = nn.Linear(H, cfg.q_lora_rank + cfg.kv_lora_rank + self.dr, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.q_b = nn.Linear(cfg.q_lora_rank, self.nh * self.dqk, bias=False)
        self.kv_b = nn.Linear(cfg.kv_lora_rank, self.nh * (self.dn + self.dv), bias=False)
        self.g_proj = nn.Linear(H, self.nh, bias=False) if cfg.gated_attention else None
        self.dense = nn.Linear(self.nh * self.dv, H, bias=False)
        self.theta = cfg.rope_theta

    def forward(self, x):
        B, T, _ = x.shape
        qa, kv_a, k_rot = self.down(x).split([self.q_rank, self.kv_rank, self.dr], dim=-1)
        q = self.q_b(self.q_norm(qa)).view(B, T, self.nh, self.dqk)
        q_nope, q_rot = q.split([self.dn, self.dr], dim=-1)
        kv = self.kv_b(self.kv_norm(kv_a)).view(B, T, self.nh, self.dn + self.dv)
        k_nope, v = kv.split([self.dn, self.dv], dim=-1)

        cos, sin = rope_cos_sin(T, self.dr, self.theta, x.device)
        q_rot = apply_rope_interleaved(q_rot, cos, sin)
        k_rot = apply_rope_interleaved(k_rot.view(B, T, 1, self.dr), cos, sin).expand(B, T, self.nh, self.dr)
        q = torch.cat([q_nope, q_rot], dim=-1).transpose(1, 2)  # [B,h,T,dqk]
        k = torch.cat([k_nope, k_rot], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.dv < self.dqk:  # flash SDPA needs equal head dims: zero-pad v, slice the output
            v = F.pad(v, (0, self.dqk - self.dv))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.dqk**-0.5)
        o = o[..., : self.dv].transpose(1, 2)  # [B,T,h,dv]
        if self.g_proj is not None:
            o = o * torch.sigmoid(self.g_proj(x)).unsqueeze(-1)
        return self.dense(o.reshape(B, T, self.nh * self.dv))


# ----------------------------------------------------------------------------- FFN
class SwiGLU(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_up = nn.Linear(hidden, 2 * inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


_grouped_mm = getattr(F, "grouped_mm", None) or getattr(torch, "_grouped_mm", None)


class MoE(nn.Module):
    """Fine-grained MoE: sigmoid scores, (optionally group-limited) top-k on (score + bias), normalised top-k
    weights, ``routed_scaling_factor``, one shared expert. Load balancing is aux-loss-free (a per-expert bias
    that ``update_bias()`` nudges after each step) plus a tiny sequence-level auxiliary loss.

    Experts are stacked tensors; tokens are dispatched by sorting on expert id (no ModuleList, no
    ``repeat_interleave`` copy).  Two expert-compute paths:

    * ``grouped``: two grouped GEMMs (``F.grouped_mm``) over the expert-sorted tokens, offsets computed on
      device -> **no Python per-expert loop and no host sync** (CUDA bf16, SM>=80; also runs on CPU fp32).
    * ``loop``: per-expert matmuls; needs one ``counts.tolist()`` host sync per layer (fallback).

    Shared expert: ``moe_shared_expert_intermediate_size`` wide, optionally multiplied by a learned gate
    (``shared_gate``: "scalar" = exp(g) with g a per-layer scalar; "token" = 2*sigmoid(w.x)), both start at
    exactly 1.0 so the gate only moves the shared/routed balance if the data asks for it."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        E, H, I = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.E, self.H, self.I = E, H, I
        self.k, self.n_group, self.topk_group = cfg.num_experts_per_tok, cfg.n_group, cfg.topk_group
        self.scale = cfg.routed_scaling_factor
        self.router = nn.Parameter(torch.empty(E, H))
        nn.init.kaiming_uniform_(self.router, a=math.sqrt(5))
        self.register_buffer("expert_bias", torch.zeros(E), persistent=True)
        self.register_buffer("load", torch.zeros(E), persistent=False)
        self.w13 = nn.Parameter(torch.empty(E, H, 2 * I))
        self.w2 = nn.Parameter(torch.empty(E, I, H))
        for w in (self.w13, self.w2):
            nn.init.normal_(w, std=cfg.initializer_range)
        self.shared = SwiGLU(H, cfg.moe_shared_expert_intermediate_size) if cfg.moe_shared_expert_intermediate_size else None
        self.gate_mode = cfg.shared_gate if self.shared is not None else "none"
        if self.gate_mode == "scalar":
            self.shared_gate = nn.Parameter(torch.zeros(()))  # 0-dim -> no weight decay
        elif self.gate_mode == "token":
            self.shared_gate = nn.Parameter(torch.zeros(H))
        elif self.gate_mode != "none":
            raise ValueError(f"shared_gate={self.gate_mode!r}")

    # ------------------------------------------------------------------ routing
    def route(self, flat, use_groups: bool = True):
        logits = F.linear(flat.float(), self.router.float())
        scores = torch.sigmoid(logits)
        biased = scores + self.expert_bias
        N = flat.shape[0]
        if use_groups and self.n_group > 1:
            gs = biased.view(N, self.n_group, -1).topk(2, dim=-1)[0].sum(-1)
            top_g = gs.topk(self.topk_group, dim=-1)[1]
            mask = torch.zeros_like(gs).scatter_(1, top_g, 1.0)
            mask = mask.unsqueeze(-1).expand(N, self.n_group, self.E // self.n_group).reshape(N, self.E)
            biased = biased.masked_fill(mask == 0, float("-inf"))
        idx = biased.topk(self.k, dim=-1)[1]
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.scale
        return idx, w, scores

    # ------------------------------------------------------------------ experts
    def _use_grouped(self, xs) -> bool:
        mode = self.cfg.moe_dispatch
        if mode == "loop" or _grouped_mm is None or self.H % 8 or self.I % 8:
            return False
        return True if mode == "grouped" else xs.is_cuda

    def _experts(self, xs, counts):
        dev = xs.device.type
        dt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else xs.dtype
        xs = xs.to(dt)
        w13, w2 = self.w13.to(dt), self.w2.to(dt)
        if self._use_grouped(xs):
            offs = counts.cumsum(0).to(torch.int32)
            g, u = _grouped_mm(xs, w13, offs=offs).chunk(2, dim=-1)
            return _grouped_mm(F.silu(g) * u, w2, offs=offs)
        outs = []
        for e, xe in enumerate(xs.split(counts.tolist())):  # the single host sync of the fallback path
            if xe.shape[0]:
                g, u = (xe @ w13[e]).chunk(2, dim=-1)
                outs.append((F.silu(g) * u) @ w2[e])
        return torch.cat(outs, 0)

    def forward(self, x):
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        idx, w, scores = self.route(flat)
        N, k = idx.shape

        flat_idx = idx.reshape(-1)
        order = torch.argsort(flat_idx, stable=True)  # slot (token*k + j) sorted by expert
        counts = torch.zeros(self.E, dtype=torch.long, device=x.device).scatter_add_(
            0, flat_idx, torch.ones_like(flat_idx)
        )  # (bincount would sync on CUDA)
        if self.training:
            self.load += counts.float()
        ys = self._experts(flat[order // k], counts)  # [N*k, H] in expert-sorted order
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        # un-sort + weighted sum over the k slots: deterministic, no atomics, no zeros_like + index_add_
        y = (ys[inv].view(N, k, H) * w.unsqueeze(-1).to(ys.dtype)).sum(1)
        if self.shared is not None:
            sh = self.shared(flat)
            if self.gate_mode == "scalar":
                sh = sh * self.shared_gate.exp().to(sh.dtype)
            elif self.gate_mode == "token":
                sh = sh * (2 * torch.sigmoid(flat.float() @ self.shared_gate.float())).unsqueeze(-1).to(sh.dtype)
            y = y.to(sh.dtype) + sh

        aux = x.new_zeros((), dtype=torch.float32)
        if self.training and self.cfg.seq_aux_coef > 0:
            # sequence-level balance loss: sum_e f_e * P_e, per sequence (DeepSeek-V3 seq_aux)
            f = torch.zeros(B, self.E, device=x.device).scatter_add_(
                1, idx.view(B, T * k), torch.ones(B, T * k, device=x.device)
            ) * (self.E / (T * k))
            P = (scores / (scores.sum(-1, keepdim=True) + 1e-20)).view(B, T, self.E).mean(1)
            aux = self.cfg.seq_aux_coef * (f * P).sum(-1).mean()
        return y.view(B, T, H).to(x.dtype), aux

    @torch.no_grad()
    def update_bias(self):
        """aux-loss-free balancing: raise bias of under-used experts, lower the over-used ones.
        Branch-free (all-zero load -> sign(0) = 0), so there is no ``.sum() > 0`` host sync."""
        self.expert_bias += self.cfg.expert_bias_update_rate * torch.sign(self.load.mean() - self.load)
        self.load.zero_()
