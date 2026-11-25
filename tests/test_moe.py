import pytest
import torch
import torch.nn.functional as F

from ling_atto.config import AttoConfig
from ling_atto.layers import MoE


def _cfg(**kw):
    c = AttoConfig.load("nano")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _reference(m: MoE, x):
    """Naive per-token reference: no sorting, no grouping."""
    flat = x.reshape(-1, x.shape[-1])
    idx, w, _ = m.route(flat)
    out = torch.zeros_like(flat)
    for t in range(flat.shape[0]):
        for j in range(idx.shape[1]):
            e = idx[t, j]
            g, u = (flat[t] @ m.w13[e]).chunk(2)
            out[t] = out[t] + w[t, j] * ((F.silu(g) * u) @ m.w2[e])
    if m.shared is not None:
        sh = m.shared(flat)
        if m.gate_mode == "scalar":
            sh = sh * m.shared_gate.exp()
        elif m.gate_mode == "token":
            sh = sh * 2 * torch.sigmoid(flat @ m.shared_gate).unsqueeze(-1)
        out = out + sh
    return out.view_as(x)


@pytest.mark.parametrize("dispatch", ["loop", "grouped"])
@pytest.mark.parametrize("gate", ["none", "scalar", "token"])
@pytest.mark.parametrize("n_group", [1, 2])
def test_moe_matches_reference_fwd_bwd(dispatch, gate, n_group):
    torch.manual_seed(0)
    cfg = _cfg(moe_dispatch=dispatch, shared_gate=gate, n_group=n_group, hidden_size=64)
    m = MoE(cfg)
    with torch.no_grad():  # make the gates non-trivial
        if gate != "none":
            m.shared_gate.normal_(0, 0.3)
    x = torch.randn(2, 12, cfg.hidden_size, requires_grad=True)
    y, _ = m(x)
    yr = _reference(m, x)
    assert torch.allclose(y, yr, atol=1e-5, rtol=1e-4)
    g1 = torch.autograd.grad((y**2).sum(), [x, m.w13, m.w2])
    g2 = torch.autograd.grad((yr**2).sum(), [x, m.w13, m.w2])
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-3)


def test_gates_start_as_identity():
    cfg0, cfg1 = _cfg(shared_gate="none"), _cfg(shared_gate="scalar")
    m0, m1 = MoE(cfg0), MoE(cfg1)
    m1.load_state_dict(m0.state_dict(), strict=False)
    x = torch.randn(1, 8, cfg0.hidden_size)
    assert torch.allclose(m0(x)[0], m1(x)[0], atol=1e-6)
    cfg2 = _cfg(shared_gate="token")
    m2 = MoE(cfg2)
    m2.load_state_dict(m0.state_dict(), strict=False)
    assert torch.allclose(m0(x)[0], m2(x)[0], atol=1e-6)


def test_scalar_gate_gets_grad_and_no_decay():
    m = MoE(_cfg(shared_gate="scalar"))
    m(torch.randn(1, 8, m.H))[0].sum().backward()
    assert m.shared_gate.grad is not None and m.shared_gate.ndim == 0


def test_update_bias_balances_without_sync():
    m = MoE(_cfg())
    m.load.copy_(torch.tensor([10.0, 0, 0, 0, 0, 0, 0, 0]))
    m.update_bias()
    assert m.expert_bias[0] < 0 < m.expert_bias[1] and m.load.sum() == 0
    before = m.expert_bias.clone()
    m.update_bias()  # zero load -> unchanged
    assert torch.equal(before, m.expert_bias)


def test_group_routing_restricts_choices():
    m = MoE(_cfg(n_group=2, topk_group=1))
    x = torch.randn(64, m.H)
    idx, _, _ = m.route(x)
    grp = idx // (m.E // 2)
    assert (grp == grp[:, :1]).all()  # every token's experts live in one group
    idx1, _, _ = m.route(x, use_groups=False)
    assert not ((idx1 // (m.E // 2)) == (idx1 // (m.E // 2))[:, :1]).all()
