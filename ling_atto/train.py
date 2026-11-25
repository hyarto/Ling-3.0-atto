"""Training entry point.

    python -m ling_atto.train --config nano --data synthetic --steps 300
    python -m ling_atto.train --config nano --data tinystories --steps 600 --no_engram   # ablation

Speed / VRAM switches (all default to the cheap-and-fast choice):
  bf16 autocast, TF32 matmuls, fused AdamW, flash SDPA, parallel chunk precompute in KDA, scan-level
  activation checkpointing in KDA, chunked + checkpointed LM-head CE, sorted MoE dispatch, sparse
  row-wise Engram optimiser; optional --grad_ckpt (layer-level recompute) when VRAM is the bottleneck.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import AttoConfig
from .data import StreamLoader, build_stream, load_docs
from .model import build_model, param_breakdown
from .optim import build_optimizer
from .tokenizer import LFMTokenizer, ReducedVocab


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="nano")
    p.add_argument("--data", default="synthetic")
    p.add_argument("--tokenizer_dir", default="assets/lfm3")
    p.add_argument("--out", default="runs/nano")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--engram_lr", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--no_engram", action="store_true")
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, loader, autocast_ctx, n=8):
    model.eval()
    tot = 0.0
    for _ in range(n):
        x = loader.next()
        with autocast_ctx():
            tot += float(model(x, labels=x)["lm_loss"])
    model.train()
    return tot / n


def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    use_bf16 = (a.dtype == "bf16") or (a.dtype == "auto" and device == "cuda")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def autocast_ctx():
        return torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16)

    cfg = AttoConfig.load(a.config)
    if a.no_engram:
        cfg.engram.enabled = False
    cfg.gradient_checkpointing = a.grad_ckpt
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data + vocab
    base = LFMTokenizer(a.tokenizer_dir)
    docs = load_docs(a.data, a.max_docs)
    if cfg.vocab_size < len(base):
        vocab = ReducedVocab.build(base, docs, cfg.vocab_size)
        vocab.save(out / "vocab_map.json")
    else:
        vocab = ReducedVocab.identity(base)
    stream = build_stream(vocab, docs)
    split = int(len(stream) * 0.95)
    train_loader = StreamLoader(stream[:split], a.batch_size, a.seq_len, device, a.seed)
    val_loader = StreamLoader(stream[split:], a.batch_size, a.seq_len, device, a.seed + 1)
    print(f"[data] docs={len(docs)} tokens={len(stream):,} vocab={vocab.vocab_size} "
          f"({'reduced' if vocab.vocab_size < len(base) else 'full LFM3'})")

    # ---- model
    model = build_model(cfg, vocab, device=device)
    cfg.save(out / "config.json")
    b = param_breakdown(model)
    print("[model] " + " | ".join(f"{k}={v:,}" for k, v in b.items() if k.endswith(("total", "backbone", "tables", "embedding", "token"))))
    print(f"[model] layers: {cfg.layer_types()}  (KDA:MLA = {b['kda_layers']}:{b['mla_layers']})")
    opt = build_optimizer(model, a.lr, a.weight_decay, engram_lr=a.engram_lr)
    dense_params = [p for n, p in model.named_parameters() if not getattr(p, "is_engram_table", False)]

    # ---- loop
    model.train()
    hist = []
    t_last, tok_last = time.time(), 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        for g in opt.opts["dense"].param_groups:
            g["lr"] = lr_at(step, a)
        x = train_loader.next()
        with autocast_ctx():
            o = model(x, labels=x)
        o["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(dense_params, a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.update_expert_bias()
        tok_last += x.numel()
        if (step + 1) % a.log_every == 0 or step == 0:
            dt = time.time() - t_last
            mem = f" vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB" if device == "cuda" else ""
            print(f"step {step + 1:5d} loss {float(o['lm_loss']):.4f} aux {float(o['aux_loss']):.5f} "
                  f"gn {float(gn):.2f} lr {lr_at(step, a):.2e} tok/s {tok_last / max(dt, 1e-9):,.0f}{mem}", flush=True)
            t_last, tok_last = time.time(), 0
            hist.append({"step": step + 1, "loss": float(o["lm_loss"])})
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            vl = evaluate(model, val_loader, autocast_ctx)
            print(f"  [eval] step {step + 1} val_loss {vl:.4f} ppl {math.exp(vl):.2f}", flush=True)
            hist.append({"step": step + 1, "val_loss": vl})

    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out / "ckpt.pt")
    (out / "history.json").write_text(json.dumps(hist))
    # ---- sample
    prompt = "The capital of France is" if a.data == "synthetic" else "Once upon a time"
    ids = torch.tensor([[vocab.BOS] + vocab.encode(prompt)], device=device)
    with autocast_ctx():
        gen = model.generate(ids, max_new_tokens=24, temperature=0.7)
    print("[sample]", vocab.decode(gen[0].tolist()[1:]))
    return hist


if __name__ == "__main__":
    main()
