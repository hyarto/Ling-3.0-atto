import pytest
import torch

from ling_atto.config import AttoConfig
from ling_atto.model import AttoModel, build_model, param_breakdown


def test_ratio_is_4_to_1():
    cfg = AttoConfig.load("atto_full")
    t = cfg.layer_types()
    assert t.count("kda") == 4 * t.count("mla")
    assert all(t[i] == "mla" for i in range(4, len(t), 5))


def test_full_scale_budgets_on_meta_device():
    m = build_model(AttoConfig.load("atto_full"), meta=True)
    b = param_breakdown(m)
    assert 0.68e9 < b["backbone"] < 0.74e9  # ~0.7B backbone (incl. 128k x 1024 embeddings)
    assert abs(b["engram_total"] - 1.2e9) / 1.2e9 < 0.01
    assert b["kda_layers"] / b["mla_layers"] == 4


def test_nano_budgets_are_hundreds_of_k():
    b = param_breakdown(build_model(AttoConfig.load("nano"), meta=True))
    assert 1e5 < b["backbone"] < 1e6 and 1e5 < b["engram_total"] < 1e6


@pytest.mark.parametrize("ckpt", [False, True])
def test_forward_backward_and_chunked_ce_matches_full(ckpt):
    cfg = AttoConfig.load("nano")
    cfg.gradient_checkpointing = ckpt
    cfg.ce_chunk_tokens = 37  # odd chunk size on purpose
    torch.manual_seed(0)
    m = build_model(cfg)
    m.train()
    x = torch.randint(2, 500, (3, 40)); x[:, 0] = 1
    out = m(x, labels=x)
    out["loss"].backward()
    assert all(p.grad is not None for p in m.parameters())
    m.eval()
    h, _ = m.hidden_states(x)
    ref = torch.nn.functional.cross_entropy(
        (h[:, :-1] @ m.head_weight.T).reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
    assert torch.allclose(m(x, labels=x)["lm_loss"], ref, atol=1e-4)


def test_model_is_causal():
    cfg = AttoConfig.load("nano")
    m = build_model(cfg).eval()
    x = torch.randint(2, 500, (1, 48))
    y = x.clone(); y[0, 30:] = (y[0, 30:] + 7) % 500 + 2
    a, b = m(x)["logits"], m(y)["logits"]
    assert torch.allclose(a[:, :30], b[:, :30], atol=1e-4)
    assert not torch.allclose(a[:, 30:], b[:, 30:], atol=1e-4)


def test_moe_bias_balancing_moves_toward_uniform():
    cfg = AttoConfig.load("nano")
    m = build_model(cfg)
    moe = m.layers[1].ffn
    x = torch.randn(4, 64, cfg.hidden_size)
    moe.train()
    with torch.no_grad():
        moe.expert_bias[0] = -5.0  # starve expert 0
        moe(x)
        before = moe.load.clone()
        for _ in range(30):
            moe(x)
            moe.update_bias()
        moe.load.zero_()
        moe(x)
    assert before[0] == 0  # the starved expert really received nothing at the start
    assert moe.expert_bias[0] > -5.0  # pushed up
    assert moe.load.std() <= before.std() + 1e-6 or moe.load[0] > 0


def test_kda_core_checkpoint_gives_identical_grads():
    grads = []
    for ck in (False, True):
        cfg = AttoConfig.load("nano")
        cfg.kda_core_checkpoint = ck
        torch.manual_seed(0)
        m = build_model(cfg)
        m.train()
        x = torch.randint(2, 500, (2, 40)); x[:, 0] = 1
        m(x, labels=x)["loss"].backward()
        grads.append({n: p.grad.to_dense() if p.grad.is_sparse else p.grad for n, p in m.named_parameters()})
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-5, rtol=1e-4), n


def test_kda_rejects_unsafe_chunk_size():
    cfg = AttoConfig.load("nano")
    cfg.kda_chunk_size = 32
    with pytest.raises(ValueError):
        build_model(cfg)
