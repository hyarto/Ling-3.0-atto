"""Ling-3.0-atto: hybrid KDA/MLA (4:1) + MoE backbone with Engram memory layers."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import AttoConfig
from .engram import Engram, EngramLayout, build_ngram_tokens
from .kda import KimiDeltaAttention, RMSNorm
from .layers import MoE, MultiLatentAttention, SwiGLU


class DecoderLayer(nn.Module):
    def __init__(self, cfg: AttoConfig, i: int, layout: EngramLayout | None, hash_vocab: int):
        super().__init__()
        self.kind = cfg.layer_type(i)
        self.mixer = KimiDeltaAttention(cfg, i) if self.kind == "kda" else MultiLatentAttention(cfg, i)
        self.is_moe = cfg.is_moe_layer(i)
        self.ffn = MoE(cfg) if self.is_moe else SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.engram = (
            Engram(cfg, i, layout, hash_vocab) if (layout is not None and i in layout.layer_ids) else None
        )

    def forward(self, x, ng_tokens):
        if self.engram is not None:  # memory is written in *before* the mixer so attention can read it
            x = self.engram(x, ng_tokens)
        x = x + self.mixer(self.norm1(x))
        h = self.norm2(x)
        if self.is_moe:
            y, aux = self.ffn(h)
        else:
            y, aux = self.ffn(h), x.new_zeros((), dtype=torch.float32)
        return x + y, aux


def _ce_chunk(h, w, y):
    return F.cross_entropy(F.linear(h, w).float(), y, ignore_index=-100, reduction="sum")


class AttoModel(nn.Module):
    def __init__(self, cfg: AttoConfig, token_map: np.ndarray | None = None, pad_id: int = 0, bos_id: int = 1):
        super().__init__()
        self.cfg = cfg
        e = cfg.engram
        self.layout = EngramLayout.build(e, cfg.hidden_size) if (e.enabled and e.layer_ids) else None
        if self.layout is not None:
            if token_map is None:  # no normalisation: identity map (compressed vocab == vocab)
                token_map = np.arange(cfg.vocab_size, dtype=np.int64)
            if len(token_map) < cfg.vocab_size:  # model vocab padded beyond tokenizer (e.g. 128000 vs 125033)
                extra = np.full(cfg.vocab_size - len(token_map), token_map[pad_id], dtype=np.int64)
                token_map = np.concatenate([token_map, extra])
            e.compressed_vocab_size = int(token_map.max()) + 1
            self.register_buffer("token_map", torch.as_tensor(token_map, dtype=torch.int64), persistent=False)
            self.pad_c, self.bos_c = int(token_map[pad_id]), int(token_map[bos_id])
        hv = e.compressed_vocab_size or cfg.vocab_size
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i, self.layout, hv) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = None if cfg.tie_word_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init)
        # scaled init of residual-branch output projections
        s = cfg.initializer_range / math.sqrt(2 * cfg.num_hidden_layers)
        for l in self.layers:
            for lin in (getattr(l.mixer, "o_proj", None), getattr(l.mixer, "dense", None)):
                if lin is not None:
                    nn.init.normal_(lin.weight, std=s)
            if l.is_moe:
                nn.init.normal_(l.ffn.w2, std=s)
                if l.ffn.shared is not None:
                    nn.init.normal_(l.ffn.shared.down.weight, std=s)
            else:
                nn.init.normal_(l.ffn.down.weight, std=s)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.initializer_range)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.initializer_range)

    @property
    def head_weight(self):
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    # ------------------------------------------------------------------ forward
    def hidden_states(self, input_ids):
        ng = None
        if self.layout is not None:
            comp = self.token_map[input_ids]
            ng = build_ngram_tokens(comp, self.cfg.engram.max_ngram_size, self.pad_c, self.bos_c)
        x = self.embed(input_ids)
        aux_total = x.new_zeros((), dtype=torch.float32)
        for layer in self.layers:
            if self.cfg.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x, aux = checkpoint(layer, x, ng, use_reentrant=False)
            else:
                x, aux = layer(x, ng)
            aux_total = aux_total + aux
        return self.norm(x), aux_total

    def forward(self, input_ids, labels=None):
        h, aux = self.hidden_states(input_ids)
        if labels is None:
            return {"logits": F.linear(h, self.head_weight)}
        # next-token loss with a chunked, checkpointed LM head: [B*T, V] logits never exist at once
        h = h[:, :-1].reshape(-1, h.shape[-1])
        y = labels[:, 1:].reshape(-1)
        n_valid = (y != -100).sum().clamp_min(1)
        total = h.new_zeros((), dtype=torch.float32)
        ck = self.cfg.ce_chunk_tokens
        w = self.head_weight
        for s in range(0, h.shape[0], ck):
            if self.training and torch.is_grad_enabled():
                total = total + checkpoint(_ce_chunk, h[s : s + ck], w, y[s : s + ck], use_reentrant=False)
            else:
                total = total + _ce_chunk(h[s : s + ck], w, y[s : s + ck])
        lm = total / n_valid
        return {"loss": lm + aux, "lm_loss": lm.detach(), "aux_loss": aux.detach()}

    @torch.no_grad()
    def update_expert_bias(self):
        for l in self.layers:
            if l.is_moe:
                l.ffn.update_bias()

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens=32, temperature=0.8, top_k=40):
        """Cache-free sampling (recomputes the prefix; fine for smoke tests)."""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.cfg.max_position_embeddings :])["logits"][:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = logits.topk(min(top_k, logits.shape[-1]))
                    logits[logits < v[:, -1:]] = float("-inf")
                nxt = torch.multinomial(logits.softmax(-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


# ----------------------------------------------------------------------------- accounting
def param_breakdown(model: AttoModel) -> dict[str, int]:
    cfg = model.cfg
    emb = model.embed.weight.numel() + (0 if model.lm_head is None else model.lm_head.weight.numel())
    tab = sum(p.numel() for n, p in model.named_parameters() if getattr(p, "is_engram_table", False))
    eng_total = sum(p.numel() for n, p in model.named_parameters() if ".engram." in n or n.startswith("engram"))
    total = sum(p.numel() for p in model.parameters())
    kda = sum(p.numel() for l in model.layers if l.kind == "kda" for p in l.mixer.parameters())
    mla = sum(p.numel() for l in model.layers if l.kind == "mla" for p in l.mixer.parameters())
    ffn = sum(p.numel() for l in model.layers for p in l.ffn.parameters())
    # activated params / token: backbone minus the inactive routed experts (+ embeddings counted once)
    per_expert = cfg.hidden_size * cfg.moe_intermediate_size * 3
    n_moe = sum(1 for l in model.layers if l.is_moe)
    inactive = n_moe * (cfg.num_experts - cfg.num_experts_per_tok) * per_expert if cfg.num_experts else 0
    backbone = total - eng_total
    return {
        "total": total,
        "backbone": backbone,  # incl. embeddings -> the '0.7B' number
        "backbone_non_embedding": backbone - emb,
        "embedding": emb,
        "kda_mixers": kda,
        "mla_mixers": mla,
        "ffn": ffn,
        "engram_total": eng_total,
        "engram_tables": tab,
        "engram_proj": eng_total - tab,
        "active_per_token": backbone - inactive + (0),
        "kda_layers": sum(1 for l in model.layers if l.kind == "kda"),
        "mla_layers": sum(1 for l in model.layers if l.kind == "mla"),
    }


def build_model(cfg: AttoConfig, tokenizer=None, device="cpu", meta=False) -> AttoModel:
    """Build a model; `tokenizer` (ReducedVocab) supplies Engram's token-normalising compression map."""
    token_map, pad_c, bos_c = None, 0, 1
    if tokenizer is not None and cfg.engram.enabled:
        from .tokenizer import build_compressed_token_map

        token_map, _ = build_compressed_token_map(tokenizer)
        pad_c, bos_c = tokenizer.PAD, tokenizer.BOS
    if meta:
        with torch.device("meta"):
            return AttoModel(cfg, token_map, pad_c, bos_c)
    return AttoModel(cfg, token_map, pad_c, bos_c).to(device)
