"""Audit of the four MoE concerns (static numbers from a config + dynamic numbers from a checkpoint).

    python scripts/moe_audit.py --config configs/atto_full.json
    python scripts/moe_audit.py --config nano --ckpt runs/x/ckpt.pt --data tinystories

1. shared expert width vs routed width / share of the activated FFN
2. is there a scale on the shared path (routed has routed_scaling_factor)
3. group-limited routing: fraction of k-subsets that are *reachable*, and (with a ckpt) how often the
   group limit actually changes a token's expert set
4. expert-loop cost: python-loop iterations + host syncs per step, and aten-op count / time of one MoE layer
   fwd+bwd for dispatch="loop" vs "grouped"
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ling_atto.config import AttoConfig  # noqa: E402
from ling_atto.layers import MoE  # noqa: E402


def reachable_fraction(E, k, n_group, topk_group):
    if n_group <= 1:
        return 1.0
    per = E // n_group
    ok = tot = 0
    for c in itertools.combinations(range(E), k):
        tot += 1
        ok += len({e // per for e in c}) <= topk_group
    return ok / tot


def static_report(cfg: AttoConfig):
    I, S, k = cfg.moe_intermediate_size, cfg.moe_shared_expert_intermediate_size, cfg.num_experts_per_tok
    n_moe = sum(cfg.is_moe_layer(i) for i in range(cfg.num_hidden_layers))
    print(f"[1] shared width {S} vs routed width {I} (ratio {S / I:.2f}x); activated FFN width/token = {k}*{I}+{S} = {k * I + S};"
          f" shared share = {S / (k * I + S):.1%}")
    print(f"[2] shared gate = {cfg.shared_gate!r}; routed path scale = {cfg.routed_scaling_factor} (shared has "
          f"{'a learned gate' if cfg.shared_gate != 'none' else 'NO scale'})")
    r = reachable_fraction(cfg.num_experts, k, cfg.n_group, cfg.topk_group)
    print(f"[3] E={cfg.num_experts} k={k} n_group={cfg.n_group} topk_group={cfg.topk_group}: "
          f"{r:.1%} of the C(E,k) expert subsets are reachable by the router")
    print(f"[4] dispatch={cfg.moe_dispatch!r}: loop path = {cfg.num_experts} python iters + 1 host sync per MoE layer "
          f"-> {cfg.num_experts * n_moe} iters, {n_moe} syncs per fwd (grouped path: 0 / 0)")


def op_stats(cfg, dispatch, tokens=1024, reps=3):
    c = replace(cfg, moe_dispatch=dispatch)
    torch.manual_seed(0)
    m = MoE(c)
    x = torch.randn(tokens // 128, 128, c.hidden_size, requires_grad=True)
    m(x)[0].sum().backward()  # warm
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        m(x)[0].sum().backward()
    n_ops = sum(e.count for e in prof.key_averages())
    t = time.perf_counter()
    for _ in range(reps):
        m(x)[0].sum().backward()
    return n_ops, (time.perf_counter() - t) / reps * 1e3


@torch.no_grad()
def dynamic_report(cfg_path, ckpt, data, max_docs):
    from ling_atto.data import StreamLoader, build_stream, load_docs
    from ling_atto.model import build_model
    from ling_atto.tokenizer import LFMTokenizer, ReducedVocab

    sd = torch.load(ckpt, map_location="cpu")
    cfg = AttoConfig.from_dict(sd["config"])
    base = LFMTokenizer("assets/lfm3")
    docs = load_docs(data, max_docs)
    vocab = ReducedVocab.build(base, docs, cfg.vocab_size) if cfg.vocab_size < len(base) else ReducedVocab.identity(base)
    stream = build_stream(vocab, docs)
    model = build_model(cfg, vocab, device="cpu")
    model.load_state_dict(sd["model"])
    model.eval()
    x = StreamLoader(stream, 16, 128, "cpu", 0).next()
    rows = []

    def hook(name):
        def f(m, inp, out):
            h = inp[0].reshape(-1, inp[0].shape[-1])
            idx, _, _ = m.route(h)
            idx_free, _, _ = m.route(h, use_groups=False)
            changed = (idx.sort(-1)[0] != idx_free.sort(-1)[0]).any(-1).float().mean().item()
            sh = m.shared(h)
            if m.gate_mode == "scalar":
                sh = sh * m.shared_gate.exp()
            routed = out[0].reshape(-1, h.shape[-1]) - sh
            load = torch.bincount(idx.reshape(-1), minlength=m.E).float()
            p = load / load.sum()
            ent = -(p * (p + 1e-12).log()).sum().item() / math.log(m.E)
            gate = "-" if m.gate_mode == "none" else (f"{m.shared_gate.exp().item():.2f}" if m.gate_mode == "scalar" else "tok")
            rows.append((name, changed, sh.norm(dim=-1).mean().item() / routed.norm(dim=-1).mean().item(),
                         load.max().item() / load.mean().item(), ent, gate))
        return f

    for i, l in enumerate(model.layers):
        if l.is_moe:
            l.ffn.register_forward_hook(hook(f"L{i}"))
    model(x)
    print(f"[dynamic] {ckpt}")
    print("  layer  group-limit-changes-set  |shared|/|routed|  maxload/mean  load-entropy  shared-gate")
    for r in rows:
        print(f"  {r[0]:5s}  {r[1]:>22.1%}  {r[2]:>16.2f}  {r[3]:>12.2f}  {r[4]:>12.3f}  {r[5]:>10s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="atto_full")
    ap.add_argument("--ckpt")
    ap.add_argument("--data", default="tinystories")
    ap.add_argument("--max_docs", type=int, default=2500)
    ap.add_argument("--bench_config", default="atto_full")
    a = ap.parse_args()
    cfg = AttoConfig.load(a.config)
    print(f"== static audit: {a.config}")
    static_report(cfg)
    b = AttoConfig.load(a.bench_config)
    print(f"== op count / time of one MoE layer fwd+bwd on CPU ({a.bench_config}, 1024 tokens; host-sync cost is invisible on CPU)")
    for d in ("loop", "grouped"):
        n, ms = op_stats(b, d)
        print(f"  dispatch={d:8s} aten ops={n:5d}  {ms:7.2f} ms")
    if a.ckpt:
        dynamic_report(a.config, a.ckpt, a.data, a.max_docs)


if __name__ == "__main__":
    main()
