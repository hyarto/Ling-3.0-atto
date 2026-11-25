"""Configuration for Ling-3.0-atto.

Design recap (see README for the long version)
----------------------------------------------
* Backbone follows Ling-3.0-tiny (BailingMoeV3): hybrid token mixers + sigmoid-routed fine-grained MoE.
  - token mixers: Kimi Delta Attention (KDA, linear attention) and Multi-head Latent Attention (MLA)
  - Ling-3.0-tiny uses ``layer_group_size=4``  -> KDA:MLA = 3:1.
    Atto uses ``layer_group_size=5``           -> KDA:MLA = 4:1 (the last layer of every group is MLA).
* Vocabulary: LiquidAI LFM3 tokenizer (128k).  For smoke-scale runs the vocab can be shrunk with
  ``ling_atto.tokenizer.ReducedVocab``.
* Engram: extra hashed n-gram memory (DeepSeek-V4.1 style), re-parameterised with
    - a low-rank bucket code (rank r << model dim, up-projected by a grouped linear),
    - K independent signed hash heads per n-gram order (count-sketch / hash-embedding style),
    - a per-order bucket budget, so rare 3-grams do not starve frequent 2-grams,
  to survive the heavy collision regime at 1.2B parameters.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class EngramConfig:
    enabled: bool = True
    layer_ids: tuple[int, ...] = (1, 11)
    max_ngram_size: int = 3  # orders 2..max_ngram_size are hashed (like DeepSeek: max_ngram_size - 1 n-grams)
    n_heads: int = 4  # independent hash functions per n-gram order
    rank: int = 32  # width of one bucket row (the low-rank code); projected up by `wkv`
    total_params: int = 1_200_000_000  # budget for ALL engram tables (sum over layers)
    order_shares: tuple[float, ...] = (0.25, 0.75)  # bucket budget share per order (2-gram, 3-gram, ...)
    signed: bool = True  # count-sketch signs: colliding rows cancel in expectation instead of adding up
    table_dtype: str = "bfloat16"  # storage dtype of the tables (stochastic rounding on update)
    conv_kernel: int = 4  # causal depthwise conv on the gated value (Engram paper); 0 disables
    pad_token: str = "<|pad|>"
    seed: int = 10007
    # runtime-filled
    compressed_vocab_size: int = 0

    def num_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads


@dataclass
class AttoConfig:
    # --- sizes ---
    vocab_size: int = 128000
    hidden_size: int = 1024
    num_hidden_layers: int = 20
    layer_group_size: int = 5  # 4 KDA + 1 MLA  => 4:1
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    # --- KDA (linear attention) ---
    kda_num_heads: int = 8
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    kda_lower_bound: float = -5.0
    kda_chunk_size: int = 16
    # --- MLA ---
    num_attention_heads: int = 8
    q_lora_rank: int = 256
    kv_lora_rank: int = 192
    qk_nope_head_dim: int = 96
    qk_rope_head_dim: int = 32
    v_head_dim: int = 96
    rope_theta: float = 6_000_000.0
    gated_attention: bool = True  # head-wise sigmoid output gate
    # --- FFN / MoE ---
    intermediate_size: int = 3072  # dense FFN (first_k_dense_replace layers)
    first_k_dense_replace: int = 1
    num_experts: int = 32
    num_experts_per_tok: int = 4
    n_group: int = 4
    topk_group: int = 2
    moe_intermediate_size: int = 256
    moe_shared_expert_intermediate_size: int = 256
    routed_scaling_factor: float = 2.5
    expert_bias_update_rate: float = 1e-3  # aux-loss-free balancing (noaux_tc)
    seq_aux_coef: float = 1e-3
    # --- engram ---
    engram: EngramConfig = field(default_factory=EngramConfig)
    # --- training-time memory/speed knobs ---
    gradient_checkpointing: bool = False
    ce_chunk_tokens: int = 4096  # chunked fused LM-head + CE so [B*T, V] logits are never materialised
    use_fla: bool = True  # use flash-linear-attention kernels for KDA if installed (CUDA only)

    # ------------------------------------------------------------------ helpers
    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def layer_type(self, i: int) -> str:
        """'mla' for the last layer in each group, else 'kda'. With group=5 this is exactly 4:1."""
        return "mla" if (i + 1) % self.layer_group_size == 0 else "kda"

    def layer_types(self) -> list[str]:
        return [self.layer_type(i) for i in range(self.num_hidden_layers)]

    def is_moe_layer(self, i: int) -> bool:
        return self.num_experts > 0 and i >= self.first_k_dense_replace

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "AttoConfig":
        d = dict(d)
        eng = d.pop("engram", {}) or {}
        eng_names = {f.name for f in fields(EngramConfig)}
        eng = {k: (tuple(v) if isinstance(v, list) else v) for k, v in eng.items() if k in eng_names}
        names = {f.name for f in fields(cls)}
        return cls(engram=EngramConfig(**eng), **{k: v for k, v in d.items() if k in names})

    @classmethod
    def load(cls, path: str | Path) -> "AttoConfig":
        p = Path(path)
        if not p.exists():  # allow bare names like "nano"
            p = Path(__file__).resolve().parent.parent / "configs" / f"{path}.json"
        return cls.from_dict(json.loads(p.read_text()))
