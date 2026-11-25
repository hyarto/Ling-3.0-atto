"""Tiny paired ablation of the MoE changes (nano config, TinyStories head, CPU, 4 procs x 1 thread).

    python scripts/moe_ablation.py --steps 300 --seeds 0 1

Every arm keeps the *activated* FFN width/token and the total FFN parameter count (nearly) constant:
  nano base : E=8,  k=2, I=32, shared=32  (active 2*32+32 = 96,  total 8*32+32 = 288)
  nano wide : E=10, k=2, I=24, shared=48  (active 2*24+48 = 96,  total 10*24+48 = 288)
Seeds are paired across arms (same init RNG stream start and same data order).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = json.loads((ROOT / "configs" / "nano.json").read_text())
WIDE = dict(num_experts=10, moe_intermediate_size=24, moe_shared_expert_intermediate_size=48)
ARMS = {
    "A base(n_group=2)": {},
    "B wide-shared": dict(WIDE),
    "C base+n_group=1": dict(n_group=1, topk_group=1),
    "D wide+n_group=1": dict(WIDE, n_group=1, topk_group=1),
    "E D+scalar-gate": dict(WIDE, n_group=1, topk_group=1, shared_gate="scalar"),
    "F D+token-gate": dict(WIDE, n_group=1, topk_group=1, shared_gate="token"),
}


def run(arm, seed, a, tmp):
    cfg = dict(BASE, **ARMS[arm])
    name = re.sub(r"\W+", "_", arm) + f"_s{seed}"
    cp = tmp / f"{name}.json"
    cp.write_text(json.dumps(cfg))
    out = tmp / name
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    cmd = [sys.executable, "-m", "ling_atto.train", "--config", str(cp), "--data", "tinystories", "--steps", str(a.steps),
           "--max_docs", str(a.max_docs), "--seed", str(seed), "--out", str(out), "--eval_every", str(a.steps),
           "--eval_batches", "40", "--log_every", str(a.steps), "--warmup", "20"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    (tmp / f"{name}.log").write_text(r.stdout + r.stderr)
    m = re.findall(r"val_loss ([\d.]+)", r.stdout)
    pm = re.search(r"active_per_token=([\d,]+)", r.stdout)
    return arm, seed, float(m[-1]) if m else float("nan"), pm.group(1) if pm else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max_docs", type=int, default=2500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--tmp", default="/tmp/ablation")
    ap.add_argument("--out", default="logs/moe_ablation.md")
    a = ap.parse_args()
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = [(arm, s) for s in a.seeds for arm in a.arms]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda j: run(*j, a, tmp), jobs))
    lines = [f"MoE ablation: nano, TinyStories head ({a.max_docs} docs), {a.steps} steps, B16 T128, seeds {a.seeds}",
             "", "| arm | active params/token | " + " | ".join(f"val s{s}" for s in a.seeds) + " | mean val loss | Δ vs first |",
             "|---|---|" + "---|" * (len(a.seeds) + 2)]
    means = {arm: st.mean(r[2] for r in res if r[0] == arm) for arm in a.arms}
    base = means[a.arms[0]]
    for arm in a.arms:
        v = [r[2] for r in res if r[0] == arm]
        ap_ = next(r[3] for r in res if r[0] == arm)
        lines.append(f"| {arm} | {ap_} | " + " | ".join(f"{x:.4f}" for x in v)
                     + f" | **{means[arm]:.4f}** | {means[arm] - base:+.4f} |")
    txt = "\n".join(lines)
    print(txt)
    Path(ROOT / a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
