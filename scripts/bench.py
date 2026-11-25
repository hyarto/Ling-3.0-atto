"""Speed / memory benchmark of the training step with the optimisation switches toggled.

    python scripts/bench.py --config nano --batch 8 --seq 256
    python scripts/bench.py --config atto_full --batch 4 --seq 2048 --device cuda   # needs ~40 GB w/o ckpt

Memory metric (device independent): bytes of *unique* tensor storages that autograd keeps alive between
forward and backward ("saved activations") + CUDA peak when on GPU.
"""
from __future__ import annotations

import argparse
import copy
import time

import torch

from ling_atto.config import AttoConfig
from ling_atto.model import build_model
from ling_atto.optim import build_optimizer


def saved_bytes(model, x):
    seen: dict[int, int] = {}

    def pack(t):
        try:
            st = t.untyped_storage()
            seen[st.data_ptr()] = st.nbytes()
        except Exception:
            pass
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(x, labels=x)
    out["loss"].backward()
    model.zero_grad(set_to_none=True)
    return sum(seen.values())


def run(cfg, args, label):
    torch.manual_seed(0)
    dev = args.device
    model = build_model(cfg, device=dev)
    model.train()
    opt = build_optimizer(model)
    x = torch.randint(2, min(cfg.vocab_size, 500), (args.batch, args.seq), device=dev)
    x[:, 0] = 1
    ctx = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda"))
    with ctx():
        act = saved_bytes(model, x)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def step():
        with ctx():
            o = model(x, labels=x)
        o["loss"].backward()
        opt.step(); opt.zero_grad(set_to_none=True); model.update_expert_bias()

    for _ in range(2):
        step()
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(args.iters):
        step()
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t) / args.iters
    peak = f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if dev == "cuda" else "n/a"
    print(f"{label:34s} saved-act {act / 2**20:9.1f} MB | peak {peak:>8s} | {dt * 1e3:8.1f} ms/step | "
          f"{args.batch * args.seq / dt:10,.0f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nano")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    base = AttoConfig.load(args.config)
    variants = {
        "no memory opts": dict(kda_core_checkpoint=False, gradient_checkpointing=False),
        "kda core ckpt (default)": dict(kda_core_checkpoint=True, gradient_checkpointing=False),
        "kda core ckpt + layer ckpt": dict(kda_core_checkpoint=True, gradient_checkpointing=True),
    }
    for label, kv in variants.items():
        cfg = copy.deepcopy(base)
        for k, v in kv.items():
            setattr(cfg, k, v)
        run(cfg, args, label)
    cfg = copy.deepcopy(base)
    cfg.engram.enabled = False
    run(cfg, args, "default, engram off")


if __name__ == "__main__":
    main()
