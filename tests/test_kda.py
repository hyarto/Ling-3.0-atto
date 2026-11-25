import pytest
import torch

from ling_atto.kda import chunk_kda, kda_recurrent


def _inputs(B=2, T=50, H=3, dk=16, dv=24, seed=0, strong_decay=False):
    g_ = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, dk, generator=g_)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, dk, generator=g_), dim=-1)
    v = torch.randn(B, T, H, dv, generator=g_)
    lo = -5.0 if strong_decay else -0.5
    g = lo * torch.sigmoid(torch.randn(B, T, H, dk, generator=g_) * 2)
    beta = torch.sigmoid(torch.randn(B, T, H, generator=g_))
    return q, k, v, g, beta


@pytest.mark.parametrize("chunk", [8, 16])
@pytest.mark.parametrize("strong", [False, True])
def test_chunk_matches_recurrent(chunk, strong):
    q, k, v, g, beta = _inputs(strong_decay=strong)  # T=50 is not a multiple of chunk -> exercises padding
    o_ref, S_ref = kda_recurrent(q, k, v, g, beta)
    o, S = chunk_kda(q, k, v, g, beta, chunk_size=chunk)
    assert torch.allclose(o, o_ref, atol=2e-4, rtol=2e-4), (o - o_ref).abs().max()
    assert torch.allclose(S, S_ref, atol=2e-4, rtol=2e-4)


def test_initial_state_and_split():
    q, k, v, g, beta = _inputs(T=32)
    o_full, S_full = chunk_kda(q, k, v, g, beta, chunk_size=16)
    o1, S1 = chunk_kda(q[:, :16], k[:, :16], v[:, :16], g[:, :16], beta[:, :16], chunk_size=16)
    o2, S2 = chunk_kda(q[:, 16:], k[:, 16:], v[:, 16:], g[:, 16:], beta[:, 16:], chunk_size=16, initial_state=S1)
    assert torch.allclose(torch.cat([o1, o2], 1), o_full, atol=2e-4, rtol=2e-4)
    assert torch.allclose(S2, S_full, atol=2e-4, rtol=2e-4)


def test_backward_matches_reference_and_checkpoint():
    q, k, v, g, beta = _inputs(T=32, strong_decay=True)
    outs = []
    for fn in ("ref", "ckpt", "plain"):
        xs = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        if fn == "ref":
            o, _ = kda_recurrent(*xs)
        else:
            o, _ = chunk_kda(*xs, chunk_size=16, scan_checkpoint=(fn == "ckpt"))
        (o.square().sum()).backward()
        outs.append([x.grad for x in xs])
    for a, b in zip(outs[0], outs[1]):
        assert torch.allclose(a, b, atol=2e-3, rtol=2e-3), (a - b).abs().max()
    for a, b in zip(outs[1], outs[2]):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
