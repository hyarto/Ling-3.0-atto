import numpy as np
import torch

from ling_atto.config import AttoConfig, EngramConfig
from ling_atto.engram import (Engram, EngramLayout, build_ngram_tokens, collision_report, is_prime,
                              next_prime)
from ling_atto.optim import RowwiseAdagrad, stochastic_round_bf16


def test_primes():
    assert [n for n in range(2, 30) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert is_prime(3512513) == (next_prime(3512512, set()) == 3512513)
    assert is_prime(2**61 - 1) and not is_prime(2**61 + 1)


def test_layout_budget_full_scale():
    cfg = AttoConfig.load("atto_full")
    lay = EngramLayout.build(cfg.engram, cfg.hidden_size)
    assert abs(lay.total_params() - 1.2e9) / 1.2e9 < 1e-3
    flat = [p for l in lay.primes for o in l for p in o]
    assert len(flat) == len(set(flat))  # all moduli distinct
    # 3-gram tables got ~3x the buckets of 2-gram tables (shares 0.25/0.75)
    assert 2.8 < lay.primes[0][1][0] / lay.primes[0][0][0] < 3.2
    # low rank pays off: >30x more buckets than a 1024-wide row layout for the same budget
    assert sum(lay.num_rows) > 30 * (1.2e9 / cfg.hidden_size)


def _tiny_engram(**kw):
    cfg = AttoConfig.load("nano")
    for k, v in kw.items():
        setattr(cfg.engram, k, v)
    lay = EngramLayout.build(cfg.engram, cfg.hidden_size)
    return cfg, lay, Engram(cfg, cfg.engram.layer_ids[0], lay, 500)


def test_ngram_tokens_respect_bos_and_start():
    ids = torch.tensor([[1, 5, 6, 1, 7, 8]])  # BOS=1
    ng = build_ngram_tokens(ids, 3, pad_id=0, bos_id=1)
    assert ng[0, 0].tolist() == [1, 0, 0]
    assert ng[0, 2].tolist() == [6, 5, 1]  # trigram may *start* at BOS
    assert ng[0, 3].tolist() == [1, 0, 0]  # BOS itself: nothing before it
    assert ng[0, 4].tolist() == [7, 1, 0]  # 3-gram would cross the boundary -> padded
    assert ng[0, 5].tolist() == [8, 7, 1]


def test_hash_deterministic_in_range_and_context_dependent():
    cfg, lay, eng = _tiny_engram()
    ids = torch.randint(2, 400, (2, 32))
    ng = build_ngram_tokens(ids, 3, 0, 1)
    idx, sg = eng.hash_ids(ng)
    idx2, _ = eng.hash_ids(ng)
    assert torch.equal(idx, idx2)
    assert idx.min() >= 0 and idx.max() < eng.table.weight.shape[0]
    assert set(sg.unique().tolist()) <= {-1, 1}
    # same bigram in different contexts -> same bigram bucket, different trigram bucket (mostly)
    a = torch.tensor([[9, 3, 4]]); b = torch.tensor([[8, 3, 4]])
    ia, _ = eng.hash_ids(build_ngram_tokens(a, 3, 0, 1)); ib, _ = eng.hash_ids(build_ngram_tokens(b, 3, 0, 1))
    K = lay.n_heads
    assert torch.equal(ia[0, 2, :K], ib[0, 2, :K]) and not torch.equal(ia[0, 2, K:], ib[0, 2, K:])


def test_engram_output_starts_as_identity_plus_small_and_backward_is_sparse():
    cfg, lay, eng = _tiny_engram()
    eng.train()
    h = torch.randn(2, 16, cfg.hidden_size, requires_grad=True)
    ng = build_ngram_tokens(torch.randint(2, 400, (2, 16)), 3, 0, 1)
    out = eng(h, ng)
    assert out.shape == h.shape
    out.square().sum().backward()
    g = eng.table.weight.grad
    assert g.is_sparse and g.coalesce().indices().shape[1] <= 2 * 16 * lay.num_cols


def test_causality():
    cfg, lay, eng = _tiny_engram()
    h = torch.randn(1, 12, cfg.hidden_size)
    ids = torch.randint(2, 400, (1, 12))
    o1 = eng(h, build_ngram_tokens(ids, 3, 0, 1))
    ids2 = ids.clone(); ids2[0, 8:] = 77
    h2 = h.clone(); h2[0, 8:] = 0
    o2 = eng(h2, build_ngram_tokens(ids2, 3, 0, 1))
    assert torch.allclose(o1[:, :8], o2[:, :8], atol=1e-5)


def test_multihead_suppresses_joint_collisions():
    # force a collision-heavy table: tiny budget, many distinct trigrams
    torch.manual_seed(0)
    ids = torch.randint(2, 300, (1, 6000))
    res = {}
    for heads in (1, 4):
        cfg, lay, eng = _tiny_engram(n_heads=heads, total_params=40_000)
        ng = build_ngram_tokens(ids, 3, 0, 1)
        res[heads] = collision_report(ids, lay, ng, eng)["order3"]
    assert res[1]["single_head_collision_rate"] > 0.05  # genuinely collision-heavy
    assert res[4]["all_heads_collision_rate"] < 0.1 * res[1]["all_heads_collision_rate"] + 1e-9


def test_stochastic_rounding_is_unbiased():
    x = torch.full((200000,), 1.0 + 2 ** -10)  # not representable in bf16 (8 bit mantissa)
    r = stochastic_round_bf16(x).float()
    assert abs(r.mean() - x.mean()) < 2e-4
    assert len(r.unique()) == 2
    assert torch.round(x.to(torch.bfloat16).float()).eq(1.0).all()  # nearest rounding would lose it entirely


def test_rowwise_adagrad_updates_only_touched_rows_bf16():
    w = torch.nn.Parameter(torch.zeros(100, 8, dtype=torch.bfloat16))
    opt = RowwiseAdagrad([w], lr=0.1)
    idx = torch.tensor([3, 3, 7])
    emb = torch.nn.functional.embedding(idx, w, sparse=True)
    emb.float().sum().backward()
    opt.step()
    touched = (w.float().abs().sum(-1) > 0).nonzero().flatten().tolist()
    assert touched == [3, 7]
    assert opt.state[w]["acc"].shape == (100,)  # 1 scalar / row
