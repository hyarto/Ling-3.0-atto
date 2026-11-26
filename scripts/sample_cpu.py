"""Run a quick CPU text-generation sample from a run's latest checkpoint.

Forces ``CUDA_VISIBLE_DEVICES=""`` so it never touches GPUs (safe to run alongside GPU training).

Examples:
    PYTHONPATH=. python scripts/sample_cpu.py runs/atto_full_1epoch
    PYTHONPATH=. python scripts/sample_cpu.py --run runs/atto_full_1epoch --prompt "The capital of France is"
    PYTHONPATH=. python scripts/sample_cpu.py --ckpt runs/atto_full_1epoch/ckpt_latest.pt --max_new_tokens 48
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import time
from pathlib import Path

import torch

from ling_atto.config import AttoConfig
from ling_atto.model import build_model
from ling_atto.tokenizer import LFMTokenizer, ReducedVocab


def resolve_ckpt(run_or_ckpt: Path) -> tuple[Path, Path]:
    if run_or_ckpt.is_file():
        return run_or_ckpt, run_or_ckpt.parent
    if not run_or_ckpt.is_dir():
        raise FileNotFoundError(f"Run/checkpoint path not found: {run_or_ckpt}")
    for name in ("ckpt_latest.pt", "ckpt.pt", "ckpt_best.pt"):
        cand = run_or_ckpt / name
        if cand.exists():
            return cand, run_or_ckpt
    pts = sorted(run_or_ckpt.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoint found in {run_or_ckpt}")
    return pts[0], run_or_ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample from a Ling-3.0-atto checkpoint on CPU.")
    ap.add_argument("target", nargs="?", help="Run directory or .pt checkpoint path")
    ap.add_argument("--run", help="Run directory (e.g. runs/atto_full_1epoch); picks ckpt_latest.pt / ckpt.pt")
    ap.add_argument("--ckpt", help="Explicit path to a .pt checkpoint file")
    ap.add_argument("--tokenizer_dir", default="assets/lfm3")
    ap.add_argument("--prompt", nargs="+", default=["Once upon a time", "The capital of France is", "def fibonacci(n):"])
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16, help="torch CPU threads (default: 16)")
    args = ap.parse_args()

    raw_target = args.ckpt or args.run or args.target
    if not raw_target:
        ap.error("Provide a run directory or checkpoint via positional arg, --run, or --ckpt")

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    ckpt_path, run_dir = resolve_ckpt(Path(raw_target))

    t0 = time.perf_counter()
    print(f"[load] reading {ckpt_path} on CPU (threads={torch.get_num_threads()})...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config")
    if cfg_dict is not None:
        cfg = AttoConfig.from_dict(cfg_dict)
    elif (run_dir / "config.json").exists():
        cfg = AttoConfig.load(run_dir / "config.json")
    else:
        raise RuntimeError("Checkpoint has no 'config' and no config.json in run directory")

    base = LFMTokenizer(args.tokenizer_dir)
    vocab_map = run_dir / "vocab_map.json"
    vocab = ReducedVocab.load(base, vocab_map) if vocab_map.exists() else ReducedVocab.identity(base)

    model = build_model(cfg, vocab, device="cpu")
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step", "?")
    best_val = ckpt.get("best_val")
    del ckpt
    model.eval()
    t_load = time.perf_counter() - t0
    val_str = f" best_val={best_val:.4f}" if isinstance(best_val, (int, float)) and best_val < float("inf") else ""
    print(f"[model] loaded step={step}{val_str} vocab={vocab.vocab_size} in {t_load:.2f}s", flush=True)

    torch.manual_seed(args.seed)
    for idx, prompt in enumerate(args.prompt):
        ids = torch.tensor([[vocab.BOS] + vocab.encode(prompt)], dtype=torch.long, device="cpu")
        tg = time.perf_counter()
        gen = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        dt = time.perf_counter() - tg
        text = vocab.decode(gen[0].tolist()[1:])
        print(f"\n--- sample {idx + 1} ({args.max_new_tokens} tok in {dt:.2f}s, {args.max_new_tokens / max(dt, 1e-9):.1f} tok/s) ---")
        print(text, flush=True)


if __name__ == "__main__":
    main()
