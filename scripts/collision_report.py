"""Compare Engram layouts at an *equal parameter budget* on real tokens.

    python scripts/collision_report.py --data tinystories --vocab 2048 --budget 300000 --hidden 64

Layouts
  wide-1head      DeepSeek-style: one hash head, bucket row as wide as the model (rank=hidden)
  wide-4head      4 hash heads, still full-width rows
  lowrank-1head   rank=8 rows, 1 head           (more buckets, no multi-hash)
  lowrank-4head   rank=8 rows, 4 heads          (ours: low rank + multi-hash [+ signs + 25/75 budget split])
Reported per n-gram order: distinct n-grams, buckets per head, load factor (distinct n-grams per bucket =
how much noise each single head's row carries; low-rank rows cut this ~dim/rank x), and the % of n-grams
that are *indistinguishable* (same bucket in ALL heads; an idealised identifiability bound -- the model
still has to decode the K noisy codes, which is what the gate + wkv are for).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ling_atto.config import AttoConfig
from ling_atto.data import build_stream, load_docs
from ling_atto.engram import Engram, EngramLayout, build_ngram_tokens, collision_report
from ling_atto.tokenizer import LFMTokenizer, ReducedVocab, build_compressed_token_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--budget", type=int, default=300_000, help="engram params per layer (tables+proj)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=300_000)
    ap.add_argument("--tokenizer_dir", default="assets/lfm3")
    a = ap.parse_args()

    base = LFMTokenizer(a.tokenizer_dir)
    docs = load_docs(a.data)
    vocab = ReducedVocab.build(base, docs, a.vocab) if a.vocab < len(base) else ReducedVocab.identity(base)
    stream = torch.from_numpy(build_stream(vocab, docs)[: a.tokens].astype(np.int64))
    tmap, csize = build_compressed_token_map(vocab)
    print(f"tokens={len(stream):,} vocab={vocab.vocab_size} compressed={csize} budget/layer={a.budget:,} hidden={a.hidden}")
    comp = torch.as_tensor(tmap)[stream][None]
    ng = build_ngram_tokens(comp, 3, int(tmap[vocab.PAD]), int(tmap[vocab.BOS]))

    layouts = {
        "wide-1head": dict(rank=a.hidden, n_heads=1, order_shares=(0.5, 0.5), signed=False),
        "wide-4head": dict(rank=a.hidden, n_heads=4, order_shares=(0.5, 0.5), signed=False),
        "lowrank-1head": dict(rank=8, n_heads=1, order_shares=(0.5, 0.5), signed=False),
        "lowrank-4head (ours)": dict(rank=8, n_heads=4, order_shares=(0.25, 0.75), signed=True),
    }
    for name, kw in layouts.items():
        cfg = AttoConfig.load("nano")
        cfg.hidden_size = a.hidden
        for k, v in kw.items():
            setattr(cfg.engram, k, v)
        cfg.engram.layer_ids = (1,)
        cfg.engram.total_params = a.budget
        lay = EngramLayout.build(cfg.engram, a.hidden)
        eng = Engram(cfg, 1, lay, csize)
        rep = collision_report(comp, lay, ng, eng)
        print(f"\n{name}: rows={sum(lay.num_rows):,}  table={lay.table_params():,} params  proj={lay.wkv_params:,}")
        for order, r in rep.items():
            print(f"  {order}: distinct={r['distinct_ngrams']:>9,} buckets/head={r['rows_per_head'][0]:>8,} "
                  f"n-grams/bucket={r['distinct_ngrams'] / r['rows_per_head'][0]:6.1f}  "
                  f"indistinguishable={100 * r['all_heads_collision_rate']:6.3f}%")


if __name__ == "__main__":
    main()
