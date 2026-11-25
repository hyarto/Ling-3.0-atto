"""Kimi Delta Attention (KDA) in pure PyTorch.

Recurrence (per head, state S in R^{dk x dv}, per-channel decay alpha_t = exp(g_t) in (0,1]^{dk}):

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

``kda_recurrent`` is the O(T) reference / decode path. ``chunk_kda`` is the training path: the WY /
UT-transform used by gated DeltaNet, extended to per-channel decay. Inside a chunk of C tokens with
cumulative log-decay G_i:

    A_ti   = beta_t * sum_d k_td exp(G_td - G_id) k_id         (i < t)
    U      = (I + A)^-1 (beta*V - beta*(k*exp(G)) S_0)          # "pseudo values"
    o_t    = (q_t*exp(G_t))^T S_0 + sum_{i<=t} (q_t exp(G_t-G_i) . k_i) U_i
    S_next = Diag(exp(G_C)) S_0 + (k * exp(G_C - G))^T U

Everything that does not depend on S_0 (A, I+A inverse, masks) is computed for *all* chunks in parallel;
only a few small matmuls remain inside the sequential chunk loop. exp(+x) factors are re-centred on the
chunk mid-point and clamped, which together with the bounded gate (g >= kda_lower_bound) keeps fp32 safe
for C <= 32.

VRAM: the chunk loop would store one [B,H,dk,dv] state per chunk for backward. We wrap the loop in
activation checkpointing (``scan_checkpoint``), recomputing the cheap scan instead of keeping the states.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_EXP_CLAMP = 60.0


# ----------------------------------------------------------------------------- reference
def kda_recurrent(q, k, v, g, beta, initial_state=None):
    """Naive recurrence. q,k:[B,T,H,dk] v:[B,T,H,dv] g:[B,T,H,dk] (log decay<=0) beta:[B,T,H]."""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    S = q.new_zeros(B, H, dk, dv) if initial_state is None else initial_state.float().clone()
    outs = []
    for t in range(T):
        S = S * g[:, t].exp().unsqueeze(-1)  # Diag(alpha) S
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        pred = torch.einsum("bhd,bhdv->bhv", kt, S)
        S = S + torch.einsum("bhd,bhv->bhdv", kt, bt.unsqueeze(-1) * (vt - pred))
        outs.append(torch.einsum("bhd,bhdv->bhv", q[:, t], S))
    return torch.stack(outs, 1), S


# ----------------------------------------------------------------------------- chunked
def _scan(U0, W, qG, Aqk, decayL, kdecT, S0):
    """Sequential inter-chunk part. All inputs are [B,H,N,...]; returns outputs [B,H,N,C,dv], final S."""
    N = U0.shape[2]
    S = S0
    outs = []
    for n in range(N):
        U = U0[:, :, n] - W[:, :, n] @ S  # [B,H,C,dv]
        outs.append(qG[:, :, n] @ S + Aqk[:, :, n] @ U)
        S = decayL[:, :, n].unsqueeze(-1) * S + kdecT[:, :, n] @ U
    return torch.stack(outs, 2), S


def chunk_kda(q, k, v, g, beta, chunk_size: int = 16, initial_state=None, scan_checkpoint: bool = True):
    """Chunked KDA. Same I/O as :func:`kda_recurrent`. Returns (o [B,T,H,dv] in q.dtype, final_state fp32)."""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    C = chunk_size
    out_dtype = q.dtype
    pad = (-T) % C
    if pad:  # zero k/v/beta and zero log-decay => the state passes through padding untouched
        q, k, v, g = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, 0, 0, pad))
    Tp = T + pad
    N = Tp // C

    def prep(x):  # [B,T,H,d] -> [B,H,N,C,d] fp32
        return x.float().reshape(B, N, C, H, -1).permute(0, 3, 1, 2, 4)

    q, k, v, g = prep(q), prep(k), prep(v), prep(g)
    beta = beta.float().reshape(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)  # [B,H,N,C,1]

    G = g.cumsum(dim=3)  # cumulative log decay inside each chunk (non-increasing)
    ref = G[:, :, :, C // 2 : C // 2 + 1]  # re-centre exponentials around the chunk mid-point
    e_pos = (G - ref).clamp(max=_EXP_CLAMP).exp()
    e_neg = (ref - G).clamp(max=_EXP_CLAMP).exp()
    k_neg = k * e_neg

    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    strict = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), -1)
    lower = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
    A = ((k * e_pos) @ k_neg.transpose(-1, -2)) * beta
    A = A.masked_fill(~strict, 0.0)
    Aqk = ((q * e_pos) @ k_neg.transpose(-1, -2)).masked_fill(~lower, 0.0)

    Gamma = G.exp()
    rhs = torch.cat([beta * v, beta * (k * Gamma)], dim=-1)  # [.., dv+dk]
    sol = torch.linalg.solve_triangular(A + eye, rhs, upper=False, unitriangular=True)
    U0, W = sol[..., :dv], sol[..., dv:]

    GL = G[:, :, :, -1:]  # [B,H,N,1,dk]
    kdecT = (k * (GL - G).exp()).transpose(-1, -2)  # [B,H,N,dk,C]
    decayL = GL.squeeze(3).exp()  # [B,H,N,dk]
    qG = q * Gamma

    S0 = (
        q.new_zeros(B, H, dk, dv)
        if initial_state is None
        else initial_state.float().to(q.device)
    )
    if scan_checkpoint and torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
        o, S = checkpoint(_scan, U0, W, qG, Aqk, decayL, kdecT, S0, use_reentrant=False)
    else:
        o, S = _scan(U0, W, qG, Aqk, decayL, kdecT, S0)
    o = o.permute(0, 2, 3, 1, 4).reshape(B, Tp, H, dv)[:, :T]
    return o.to(out_dtype), S


# ----------------------------------------------------------------------------- module
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class CausalShortConv(nn.Module):
    """Depthwise causal conv + SiLU over the channel dim (shared by q,k,v through concatenation)."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(kernel))

    def forward(self, x):  # x: [B,T,C]
        y = F.conv1d(x.transpose(1, 2), self.weight.to(x.dtype), padding=self.kernel - 1, groups=x.shape[-1])
        return F.silu(y[..., : x.shape[1]]).transpose(1, 2)


_FLA_STATE = {"tried": False, "fn": None}


def _try_fla():
    if not _FLA_STATE["tried"]:
        _FLA_STATE["tried"] = True
        try:  # optional Triton kernels (CUDA only)
            from fla.ops.kda import chunk_kda as fla_chunk_kda  # type: ignore

            _FLA_STATE["fn"] = fla_chunk_kda
        except Exception:
            _FLA_STATE["fn"] = None
    return _FLA_STATE["fn"]


class KimiDeltaAttention(nn.Module):
    """KDA token mixer. One fused input projection feeds q,k,v,f(decay),g(out-gate),beta; q/k/v share one
    depthwise conv. (Ling-3.0-tiny: ``no_kda_lora=True`` -> full-rank f/g projections, as here.)"""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, dk = cfg.kda_num_heads, cfg.kda_head_dim
        self.H, self.dk = H, dk
        P = H * dk
        self.P = P
        self.in_proj = nn.Linear(cfg.hidden_size, 3 * P + 2 * P + H, bias=False)  # q k v | f g | beta
        self.conv = CausalShortConv(3 * P, cfg.short_conv_kernel_size)
        self.A_log = nn.Parameter(torch.log(torch.empty(H, dtype=torch.float32).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.empty(P, dtype=torch.float32).uniform_(-6.0, -2.0))
        self.o_norm = RMSNorm(dk, cfg.rms_norm_eps)
        self.o_proj = nn.Linear(P, cfg.hidden_size, bias=False)
        self.lower_bound = cfg.kda_lower_bound
        self.chunk = cfg.kda_chunk_size
        self.use_fla = cfg.use_fla

    def gate(self, f):
        """Bounded per-channel log-decay in (lower_bound, 0): the 'safe gate' of Kimi Linear / Ling."""
        B, T, _ = f.shape
        f = f.float() + self.dt_bias
        a = self.A_log.float().exp().repeat_interleave(self.dk)  # per-head temperature -> per channel
        g = self.lower_bound * torch.sigmoid(a * f)
        return g.view(B, T, self.H, self.dk)

    def forward(self, x):
        B, T, _ = x.shape
        H, dk, P = self.H, self.dk, self.P
        proj = self.in_proj(x)
        qkv, f, gate_out, b = proj.split([3 * P, P, P, H], dim=-1)
        qkv = self.conv(qkv)
        q, k, v = (t.reshape(B, T, H, dk) for t in qkv.split(P, dim=-1))
        q = F.normalize(q.float(), dim=-1) * dk**-0.5
        k = F.normalize(k.float(), dim=-1)
        g = self.gate(f)
        beta = torch.sigmoid(b.float())

        o = None
        fla_fn = _try_fla() if (self.use_fla and x.is_cuda) else None
        if fla_fn is not None:
            try:
                o, _ = fla_fn(
                    q=q.to(x.dtype), k=k.to(x.dtype), v=v, g=g, beta=beta,
                    initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
                    use_gate_in_kernel=False,
                )
            except Exception as e:  # signature drift between fla versions -> permanent fallback
                print(f"[kda] fla kernel failed ({e!r}); falling back to torch chunk_kda")
                _FLA_STATE["fn"] = None
        if o is None:
            o, _ = chunk_kda(q, k, v, g, beta, chunk_size=self.chunk)
        o = self.o_norm(o) * torch.sigmoid(gate_out).view(B, T, H, dk).to(o.dtype)
        return self.o_proj(o.reshape(B, T, P).to(x.dtype))
