"""Optimisers.

* Dense parameters  -> fused AdamW (decay only on >=2-D weights).
* Engram tables     -> ``RowwiseAdagrad``: sparse-gradient, row-wise second-moment (one fp32 scalar per
  row), bf16 weights with stochastic rounding.

Memory for the 1.2B-parameter Engram (rank 32, 37.5M rows):
    bf16 weights            2.40 GB
    AdamW fp32 m+v          9.60 GB   (+ dense fp32 grads 4.8 GB)   <- avoided
    RowwiseAdagrad state    0.15 GB   (+ sparse grads: only touched rows)
"""
from __future__ import annotations

import torch


def stochastic_round_bf16(x32: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 with stochastic rounding (unbiased): tiny updates are not lost to round-to-nearest."""
    bits = x32.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


class RowwiseAdagrad(torch.optim.Optimizer):
    """Adagrad with one accumulator per embedding *row* (as in DLRM / FBGEMM `rowwise_adagrad`).

    Works on parameters whose ``.grad`` is a sparse COO tensor (``F.embedding(..., sparse=True)``)."""

    def __init__(self, params, lr: float = 0.05, eps: float = 1e-8, init_acc: float = 0.0):
        super().__init__(params, dict(lr=lr, eps=eps, init_acc=init_acc))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if not g.is_sparse:  # dense fallback (e.g. hook-less debugging)
                    g = g.to_sparse()
                g = g.coalesce()
                idx = g.indices()[0]
                vals = g.values().float()  # [U, rank]
                st = self.state[p]
                if "acc" not in st:
                    st["acc"] = torch.full((p.shape[0],), group["init_acc"], device=p.device, dtype=torch.float32)
                acc = st["acc"]
                acc[idx] += vals.square().mean(-1)
                step = group["lr"] * vals / (acc[idx].sqrt().unsqueeze(-1) + group["eps"])
                new = p[idx].float() - step
                p[idx] = stochastic_round_bf16(new) if p.dtype == torch.bfloat16 else new.to(p.dtype)
        return None

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None


class MultiOptimizer:
    """Tiny wrapper so the train loop sees one optimiser (step / zero_grad / state_dict)."""

    def __init__(self, opts: dict[str, torch.optim.Optimizer]):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts.values() for g in o.param_groups]

    def step(self):
        for o in self.opts.values():
            o.step()

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts.values():
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {k: o.state_dict() for k, o in self.opts.items()}

    def load_state_dict(self, sd):
        for k, o in self.opts.items():
            o.load_state_dict(sd[k])


def build_optimizer(model, lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95), engram_lr=0.05, fused=None):
    decay, no_decay, tables = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if getattr(p, "is_engram_table", False):
            tables.append(p)
        elif p.ndim >= 2 and "embed" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    if fused is None:
        fused = torch.cuda.is_available()
    opts = {
        "dense": torch.optim.AdamW(
            [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)],
            lr=lr, betas=betas, fused=fused,
        )
    }
    if tables:
        opts["engram"] = RowwiseAdagrad(tables, lr=engram_lr)
    return MultiOptimizer(opts)
