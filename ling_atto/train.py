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
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .config import AttoConfig
from .data import (
    IndexedSequenceLoader,
    StreamLoader,
    build_stream,
    is_tokenized_source,
    load_docs,
    open_token_stream,
    split_sequence_indices,
)
from .model import build_model, param_breakdown
from .optim import build_optimizer
from .tokenizer import LFMTokenizer, ReducedVocab


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="nano")
    p.add_argument("--data", default="synthetic")
    p.add_argument("--tokenizer_dir", default="assets/lfm3")
    p.add_argument("--out", default="runs/nano")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--epochs", type=float, default=None, help="If set, iterate sequentially over shuffled corpus for N epochs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
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
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--val_seqs", type=int, default=4096)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=8)
    p.add_argument("--save_every", type=int, default=0, help="Save resumable checkpoint every N optimizer steps (0 disables)")
    p.add_argument("--resume", default=None, help="Path to checkpoint or 'auto' to resume from <out>/ckpt_latest.pt")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / max(1, a.warmup)
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    t = min(max(t, 0.0), 1.0)
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


def _save_checkpoint(path: Path, model, opt, cfg: AttoConfig, step: int, best_val: float, hist: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": cfg.to_dict(),
            "step": step,
            "best_val": best_val,
            "history": hist,
        },
        tmp,
    )
    os.replace(tmp, path)


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
    if is_tokenized_source(a.data):
        if cfg.vocab_size < len(base):
            raise ValueError(f"Pre-tokenized stream uses full LFM3 vocab ({len(base)}), "
                             f"cannot run with smaller config vocab_size={cfg.vocab_size}")
        vocab = ReducedVocab.identity(base)
        stream = open_token_stream(a.data, max_tokens=a.max_tokens)
        tag = "full LFM3"
        src_desc = f"pretokenized={a.data}"
    else:
        docs = load_docs(a.data, a.max_docs)
        if cfg.vocab_size < len(base):
            vocab = ReducedVocab.build(base, docs, cfg.vocab_size)
            vocab.save(out / "vocab_map.json")
            tag = "reduced"
        else:
            vocab = ReducedVocab.identity(base)
            tag = "full LFM3"
        stream = build_stream(vocab, docs)
        if a.max_tokens is not None:
            stream = stream[: a.max_tokens]
        src_desc = f"docs={len(docs)}"

    if a.epochs is not None:
        train_ids, val_ids = split_sequence_indices(len(stream), a.seq_len, val_seqs=a.val_seqs, seed=a.seed)
        train_loader = IndexedSequenceLoader(stream, train_ids, a.batch_size, a.seq_len, device)
        val_loader = IndexedSequenceLoader(stream, val_ids, a.batch_size, a.seq_len, device)
        epoch_steps = max(1, int(round(a.epochs * len(train_ids) / (a.batch_size * a.grad_accum))))
        if a.steps is None or a.steps <= 0:
            a.steps = epoch_steps
        print(f"[data] {src_desc} total_tokens={len(stream):,} train_seqs={len(train_ids):,} "
              f"val_seqs={len(val_ids):,} epoch_steps={epoch_steps:,} target_steps={a.steps:,} vocab={vocab.vocab_size} ({tag})")
    else:
        if a.steps is None:
            a.steps = 300
        split = int(len(stream) * 0.95)
        if isinstance(stream, np.ndarray) and not isinstance(stream, np.memmap):
            train_loader = StreamLoader(stream[:split], a.batch_size, a.seq_len, device, a.seed)
            val_loader = StreamLoader(stream[split:], a.batch_size, a.seq_len, device, a.seed + 1)
        else:
            train_loader = StreamLoader(stream, a.batch_size, a.seq_len, device, a.seed, start=0, end=split)
            val_loader = StreamLoader(stream, a.batch_size, a.seq_len, device, a.seed + 1, start=split, end=len(stream))
        print(f"[data] {src_desc} tokens={len(stream):,} vocab={vocab.vocab_size} ({tag})")

    # ---- model
    model = build_model(cfg, vocab, device=device)
    cfg.save(out / "config.json")
    b = param_breakdown(model)
    print("[model] " + " | ".join(f"{k}={v:,}" for k, v in b.items() if k.endswith(("total", "backbone", "tables", "embedding", "token"))))
    print(f"[model] layers: {cfg.layer_types()}  (KDA:MLA = {b['kda_layers']}:{b['mla_layers']})")
    opt = build_optimizer(model, a.lr, a.weight_decay, engram_lr=a.engram_lr)
    dense_params = [p for _, p in model.named_parameters() if not getattr(p, "is_engram_table", False)]

    # ---- optional resume
    start_step = 0
    best_val = float("inf")
    hist: list[dict] = []
    resume_path: Path | None = None
    if a.resume == "auto":
        cand = out / "ckpt_latest.pt"
        if cand.exists():
            resume_path = cand
    elif a.resume:
        resume_path = Path(a.resume)
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        hist = list(ckpt.get("history", []))
        if isinstance(train_loader, IndexedSequenceLoader):
            train_loader.set_cursor(start_step * a.batch_size * a.grad_accum)
        print(f"[resume] loaded {resume_path} at step={start_step} best_val={best_val:.4f}", flush=True)

    stop_requested = {"flag": False}

    def _handle_signal(signum, _frame):
        stop_requested["flag"] = True
        print(f"\n[signal] received signal {signum}, will save checkpoint and exit after current step...", flush=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ---- loop
    model.train()
    t_start = time.time()
    t_last, tok_last = t_start, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok_per_step = a.batch_size * a.seq_len * a.grad_accum
    for step in range(start_step, a.steps):
        cur_lr = lr_at(step, a)
        for g in opt.opts["dense"].param_groups:
            g["lr"] = cur_lr
        lm_loss_acc = 0.0
        aux_loss_acc = 0.0
        for _ in range(a.grad_accum):
            x = train_loader.next()
            with autocast_ctx():
                o = model(x, labels=x)
                loss = o["loss"] / a.grad_accum
            loss.backward()
            lm_loss_acc += float(o["lm_loss"]) / a.grad_accum
            aux_loss_acc += float(o["aux_loss"]) / a.grad_accum
            tok_last += x.numel()

        gn = torch.nn.utils.clip_grad_norm_(dense_params, a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.update_expert_bias()

        done_step = step + 1
        if done_step % a.log_every == 0 or step == start_step:
            now = time.time()
            dt = now - t_last
            tok_s = tok_last / max(dt, 1e-9)
            avg_tok_s = ((done_step - start_step) * tok_per_step) / max(now - t_start, 1e-9)
            rem_sec = (a.steps - done_step) * tok_per_step / max(avg_tok_s, 1e-9)
            ep_str = f" ep {done_step * tok_per_step / max(1, len(train_loader) * a.seq_len):.3f}" if isinstance(train_loader, IndexedSequenceLoader) else ""
            mem = f" vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB" if device == "cuda" else ""
            print(f"step {done_step:6d}/{a.steps}{ep_str} loss {lm_loss_acc:.4f} aux {aux_loss_acc:.5f} "
                  f"gn {float(gn):.2f} lr {cur_lr:.2e} tok/s {tok_s:,.0f}{mem} eta {rem_sec/3600:.1f}h", flush=True)
            t_last, tok_last = now, 0
            hist.append({"step": done_step, "loss": lm_loss_acc})

        if done_step % a.eval_every == 0 or done_step == a.steps:
            if isinstance(val_loader, IndexedSequenceLoader):
                val_loader.set_cursor(0)
            vl = evaluate(model, val_loader, autocast_ctx, n=a.eval_batches)
            print(f"  [eval] step {done_step} val_loss {vl:.4f} ppl {math.exp(vl):.2f}", flush=True)
            hist.append({"step": done_step, "val_loss": vl})
            if vl < best_val:
                best_val = vl
                if a.save_every > 0:
                    _save_checkpoint(out / "ckpt_best.pt", model, opt, cfg, done_step, best_val, hist)

        if (a.save_every > 0 and done_step % a.save_every == 0) or stop_requested["flag"]:
            _save_checkpoint(out / "ckpt_latest.pt", model, opt, cfg, done_step, best_val, hist)
            (out / "history.json").write_text(json.dumps(hist))
            print(f"  [ckpt] saved {out / 'ckpt_latest.pt'} at step={done_step}", flush=True)
            if stop_requested["flag"]:
                return hist

    _save_checkpoint(out / "ckpt.pt", model, opt, cfg, a.steps, best_val, hist)
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
