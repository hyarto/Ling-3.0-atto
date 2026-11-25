"""Engram: hashed n-gram memory written into the residual stream (after DeepSeek-V4.1 `engram.py`),
re-parameterised for a *collision-dominated* regime (1.2B parameters vs. an astronomically large n-gram
space).

Why plain Engram collides at this size
--------------------------------------
With the compressed LFM3 alphabet |V_c| ~ 9e4 there are ~8e9 bigrams and ~7e14 trigrams of which a few
hundred million occur in a real corpus. A DeepSeek-style table with 1024-wide rows buys only ~1.2M rows
for 1.2B parameters, i.e. >100 distinct n-grams per bucket.  What we do about it:

1. **Low-rank bucket codes.**  A row stores a ``rank``-dim code (32) instead of a model-width vector;
   a *grouped linear* ``wkv`` (acting on the concatenation of all heads' codes) up-projects to
   key/value of width ``hidden``.  Same parameter budget -> ``hidden/rank`` x more buckets.
   (Equivalent to a factorisation  E[bucket] = U[bucket] @ W  with a shared W.)
2. **K independent hash heads per n-gram order** (``n_heads`` distinct prime moduli, hash-embedding
   style).  Two n-grams are confused only if they collide in (almost) all K heads; partial collisions
   look like small noise on a few coordinates of the concatenated code, which ``wkv`` averages out.
3. **Signed hashing (count-sketch).**  Each (head, n-gram) draws a +-1 sign, so colliding rows add up
   with random signs: the colliding noise has zero mean instead of a bias.
4. **Per-order budgets.**  Bigrams are far fewer and far more frequent than trigrams, so the budget is
   split e.g. 25% / 75% instead of uniformly (``order_shares``).
5. **Context gate** (from DeepSeek's design): the retrieved value is scaled by a sigmoid gate computed
   from the *current hidden state* against the retrieved key -- a colliding, irrelevant memory has a low
   match score and is switched off.  This is what ultimately makes collisions tolerable.
6. Token normalisation (``build_compressed_token_map``) shrinks the alphabet before hashing.

N-grams never extend across a BOS token (document boundary in packed batches).

Training memory: tables are stored in bf16 and updated with a *row-wise sparse optimiser* with
stochastic rounding (see ``optim.py``): optimiser state is 4 bytes per row instead of 8 bytes per
parameter, and gradients are sparse (only touched rows).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kda import RMSNorm

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:  # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(start: int, seen: set[int]) -> int:
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout. ``primes[layer][order_idx][head]`` is a table's modulus; the tables of a layer are
    laid out back-to-back in one parameter of ``num_rows[layer]`` rows."""

    layer_ids: tuple[int, ...]
    orders: tuple[int, ...]  # e.g. (2, 3)
    n_heads: int
    rank: int
    primes: tuple  # [L][O][H]
    num_rows: tuple[int, ...]  # per layer
    wkv_params: int

    @property
    def num_cols(self) -> int:
        return len(self.orders) * self.n_heads

    def table_params(self) -> int:
        return sum(self.num_rows) * self.rank

    def total_params(self) -> int:
        return self.table_params() + self.wkv_params

    @classmethod
    def build(cls, ecfg, hidden: int) -> "EngramLayout":
        L = len(ecfg.layer_ids)
        orders = tuple(range(2, ecfg.max_ngram_size + 1))
        O, K = len(orders), ecfg.n_heads
        shares = np.asarray(ecfg.order_shares[:O], dtype=np.float64)
        if len(shares) < O:
            shares = np.concatenate([shares, np.full(O - len(shares), shares[-1] if len(shares) else 1.0)])
        shares = shares / shares.sum()
        cols = O * K
        # projection params per layer: wkv (cols*rank -> 2*hidden) + gate/conv/norm vectors
        per_layer_proj = cols * ecfg.rank * 2 * hidden + 2 * hidden + hidden * max(ecfg.conv_kernel, 0) + hidden
        wkv_params = L * per_layer_proj
        table_budget = max(ecfg.total_params - wkv_params, 0)
        rows_per_layer = table_budget // (L * ecfg.rank)
        primes, seen, num_rows = [], set(), []
        for _ in range(L):
            per_order, total = [], 0
            for o in range(O):
                start = max(int(rows_per_layer * shares[o] / K), 3)
                ps = []
                for _ in range(K):
                    p = next_prime(start, seen)
                    seen.add(p)
                    ps.append(p)
                    start = p
                per_order.append(tuple(ps))
                total += sum(ps)
            primes.append(tuple(per_order))
            num_rows.append(total)
        return cls(tuple(ecfg.layer_ids), orders, K, ecfg.rank, tuple(primes), tuple(num_rows), wkv_params)


def hash_multipliers(layer_idx: int, n: int, vocab: int, seed: int) -> torch.Tensor:
    """Odd multipliers, bounded so ``token * multiplier`` cannot overflow int64 (as in DeepSeek's code)."""
    bound = max(1, (np.iinfo(np.int64).max // max(vocab, 1)) // 2)
    gen = np.random.default_rng(seed * (layer_idx + 1))
    return torch.tensor(gen.integers(0, bound, size=(n,), dtype=np.int64) * 2 + 1)


def build_ngram_tokens(comp_ids: torch.Tensor, max_n: int, pad_id: int, bos_id: int) -> torch.Tensor:
    """comp_ids [B,T] -> [B,T,max_n] holding tokens t, t-1, ..., t-(max_n-1), with positions before the
    sequence start (or before a BOS) replaced by ``pad_id``."""
    B, T = comp_ids.shape
    cols = [comp_ids]
    blocked = torch.zeros_like(comp_ids, dtype=torch.bool)
    pos = torch.arange(T, device=comp_ids.device)
    for s in range(1, max_n):
        # if the token at t-(s-1) is BOS, the n-gram reaching back to t-s would cross a document boundary
        raw_prev = F.pad(comp_ids, (s - 1, 0), value=pad_id)[:, :T]
        blocked = blocked | (raw_prev == bos_id)
        shifted = F.pad(comp_ids, (s, 0), value=pad_id)[:, :T]  # token at t-s
        cols.append(torch.where(blocked | (pos < s), torch.full_like(shifted, pad_id), shifted))
    return torch.stack(cols, dim=-1)


class EngramTable(nn.Module):
    """One layer's pile of bucket rows. Kept as a separate module so the optimiser can find it."""

    def __init__(self, rows: int, rank: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rows, rank, dtype=dtype))
        self.weight.is_engram_table = True  # type: ignore[attr-defined]  (routed to the sparse optimiser)
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, idx):
        return F.embedding(idx, self.weight, sparse=self.training and torch.is_grad_enabled())


class Engram(nn.Module):
    def __init__(self, cfg, layer_id: int, layout: EngramLayout, hash_vocab: int):
        super().__init__()
        e = cfg.engram
        self.layer_id = layer_id
        self.li = layout.layer_ids.index(layer_id)
        self.layout = layout
        self.hidden = cfg.hidden_size
        self.signed = e.signed
        self.eps = cfg.rms_norm_eps
        dtype = getattr(torch, e.table_dtype)
        self.table = EngramTable(layout.num_rows[self.li], layout.rank, dtype)
        cols = layout.num_cols
        self.wkv = nn.Linear(cols * layout.rank, 2 * cfg.hidden_size, bias=False)
        nn.init.normal_(self.wkv.weight, std=cfg.initializer_range)
        self.q_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.v_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.conv = None
        if e.conv_kernel > 0:
            self.dilation = e.max_ngram_size
            self.conv = nn.Conv1d(cfg.hidden_size, cfg.hidden_size, e.conv_kernel, groups=cfg.hidden_size,
                                  dilation=self.dilation, padding=0, bias=False)
            nn.init.zeros_(self.conv.weight)  # y = v at init (Engram paper)
        # hashing constants
        flat_primes = [p for per_order in layout.primes[self.li] for p in per_order]  # order-major
        offsets = np.cumsum([0] + flat_primes[:-1])
        self.register_buffer("primes", torch.tensor(flat_primes, dtype=torch.int64), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False)
        self.register_buffer("mult", hash_multipliers(self.li, e.max_ngram_size, hash_vocab, e.seed), persistent=False)
        self.register_buffer("head_shift", torch.arange(layout.n_heads, dtype=torch.int64) * 3 + 29, persistent=False)
        self.max_n = e.max_ngram_size
        self.orders = layout.orders
        self.K = layout.n_heads

    # ---- hashing
    @torch.no_grad()
    def hash_ids(self, ng_tokens: torch.Tensor):
        """ng_tokens [B,T,max_n] -> (bucket ids [B,T,O*K], signs [B,T,O*K] in {+1,-1})."""
        prod = ng_tokens * self.mult  # [B,T,max_n]
        rolling = prod[..., 0]
        ids, signs = [], []
        pi = 0
        for n in range(2, self.max_n + 1):
            rolling = torch.bitwise_xor(rolling, prod[..., n - 1])
            for h in range(self.K):
                ids.append(rolling % self.primes[pi] + self.offsets[pi])
                if self.signed:
                    signs.append(((rolling >> self.head_shift[h]) & 1) * 2 - 1)
                pi += 1
        idx = torch.stack(ids, dim=-1)
        sg = torch.stack(signs, dim=-1) if self.signed else None
        return idx, sg

    def forward(self, h: torch.Tensor, ng_tokens: torch.Tensor):
        B, T, D = h.shape
        idx, sg = self.hash_ids(ng_tokens)
        emb = self.table(idx)  # [B,T,cols,rank]
        if sg is not None:
            emb = emb * sg.unsqueeze(-1).to(emb.dtype)
        kv = self.wkv(emb.flatten(-2).to(h.dtype))
        key, value = kv.split(D, dim=-1)

        hf, kf = h.float(), key.float()
        rstd = torch.rsqrt(hf.square().mean(-1) + self.eps) * torch.rsqrt(kf.square().mean(-1) + self.eps)
        dot = (hf * (self.q_weight.float() * self.k_weight.float()) * kf).sum(-1) * rstd * D**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))  # signed sqrt (DeepSeek)
        v = (gate.unsqueeze(-1) * value.float()).to(h.dtype)
        if self.conv is not None:
            c = self.v_norm(v).transpose(1, 2)
            c = F.pad(c, ((self.conv.kernel_size[0] - 1) * self.dilation, 0))
            v = v + F.silu(self.conv(c)).transpose(1, 2)
        return h + v

    # ---- diagnostics
    def table_params(self) -> int:
        return self.table.weight.numel()


# ----------------------------------------------------------------------------- collision analysis
@torch.no_grad()
def collision_report(ids: torch.Tensor, layout: EngramLayout, ng_tokens: torch.Tensor, engram: Engram) -> dict:
    """Empirical collision statistics on a token stream.

    For every (order, head) table: number of distinct n-grams vs distinct occupied buckets, and the
    fraction of distinct n-grams that share their bucket with another n-gram ('single-head collision').
    Also the fraction confused in *all* K heads at once (what the multi-head design suppresses)."""
    idx, _ = engram.hash_ids(ng_tokens)  # [B,T,cols]
    B, T, cols = idx.shape
    out = {}
    pi = 0
    all_heads_conflict = {}
    for oi, n in enumerate(layout.orders):
        # key of the n-gram = its raw token tuple
        toks = ng_tokens[..., :n].reshape(-1, n)
        uniq, inv = torch.unique(toks, dim=0, return_inverse=True)
        n_unique = uniq.shape[0]
        per_head_coll = []
        confused = torch.zeros(n_unique, dtype=torch.bool)
        for h in range(layout.n_heads):
            b = idx[..., pi].reshape(-1)
            first = torch.zeros(n_unique, dtype=torch.int64)
            first[inv] = b  # bucket per distinct n-gram (identical for identical n-grams)
            ub, counts = torch.unique(first, return_counts=True)
            sizes = torch.zeros(int(first.max()) + 1, dtype=torch.int64)
            sizes[ub] = counts
            shared = sizes[first] > 1
            per_head_coll.append(float(shared.float().mean()))
            if h == 0:
                combo = first.clone()
            else:
                combo = combo * 1000003 + first  # joint bucket signature over heads
            pi += 1
        # confusion across all heads = identical joint signature with another n-gram
        _, c2 = torch.unique(combo, return_counts=True)
        sig_counts = c2[torch.unique(combo, return_inverse=True)[1]]
        out[f"order{n}"] = {
            "distinct_ngrams": int(n_unique),
            "rows_per_head": [int(p) for p in layout.primes[engram.li][oi]],
            "single_head_collision_rate": float(np.mean(per_head_coll)),
            "all_heads_collision_rate": float((sig_counts > 1).float().mean()),
        }
    return out
