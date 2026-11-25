"""LFM3 tokenizer wrapper + reduced-vocabulary view.

`LiquidAI/LFM3-1.7B` tokenizer.json is downloaded on demand (``fetch_lfm3_tokenizer``).

``ReducedVocab`` keeps the ``N`` most frequent LFM3 tokens of a corpus (+ specials) and maps the rest to
``<unk>``; this lets us run the *same architecture* end-to-end with a 1-4k vocab (embedding of a few
hundred KB) while still using the real LFM3 BPE for text -> ids.
"""
from __future__ import annotations

import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

LFM3_REPO = "LiquidAI/LFM3-1.7B"
_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def fetch_lfm3_tokenizer(dest: str | Path = "assets/lfm3", repo: str = LFM3_REPO) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        out = dest / name
        if out.exists() and out.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        print(f"[tokenizer] downloading {url}")
        urllib.request.urlretrieve(url, out)
    return dest


class LFMTokenizer:
    """Thin wrapper over the raw Rust tokenizer (no transformers dependency)."""

    def __init__(self, path: str | Path = "assets/lfm3"):
        from tokenizers import Tokenizer

        path = Path(path)
        if not (path / "tokenizer.json").exists():
            fetch_lfm3_tokenizer(path)
        self.backend = Tokenizer.from_file(str(path / "tokenizer.json"))
        cfg = json.loads((path / "tokenizer_config.json").read_text())
        self.bos_token = cfg.get("bos_token", "<|startoftext|>")
        self.eos_token = cfg.get("eos_token", "<|im_end|>")
        self.pad_token = cfg.get("pad_token", "<|pad|>")
        self.vocab_size = self.backend.get_vocab_size(with_added_tokens=True)
        self.bos_id = self.token_to_id(self.bos_token)
        self.eos_id = self.token_to_id(self.eos_token)
        self.pad_id = self.token_to_id(self.pad_token)

    def token_to_id(self, tok: str) -> int:
        i = self.backend.token_to_id(tok)
        if i is None:
            raise KeyError(tok)
        return i

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self.backend.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return self.backend.decode(list(ids), skip_special_tokens=False)

    def __len__(self) -> int:
        return self.vocab_size


class ReducedVocab:
    """Compact id space on top of :class:`LFMTokenizer`.

    Layout: ``0 pad | 1 bos | 2 eos | 3 unk | 4.. kept LFM3 ids (by frequency)``.
    ``orig_ids[new_id]`` gives the original LFM3 id (used for Engram's token-normalising compression
    map and for decoding).
    """

    N_SPECIAL = 4
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, base: LFMTokenizer, orig_ids: Sequence[int]):
        self.base = base
        self.orig_ids = np.asarray(orig_ids, dtype=np.int64)  # len == vocab_size
        self.old2new = np.full(len(base), self.UNK, dtype=np.int64)
        self.old2new[self.orig_ids[self.N_SPECIAL:]] = np.arange(self.N_SPECIAL, len(self.orig_ids))
        self.old2new[base.pad_id] = self.PAD
        self.old2new[base.bos_id] = self.BOS
        self.old2new[base.eos_id] = self.EOS

    # ---- construction
    @classmethod
    def build(cls, base: LFMTokenizer, texts: Iterable[str], vocab_size: int) -> "ReducedVocab":
        counts: Counter[int] = Counter()
        for t in texts:
            counts.update(base.encode(t))
        for sp in (base.pad_id, base.bos_id, base.eos_id):
            counts.pop(sp, None)
        keep = [i for i, _ in counts.most_common(vocab_size - cls.N_SPECIAL)]
        # slots 0..3 are specials; their "orig id" is the real special id (unk -> pad id)
        orig = [base.pad_id, base.bos_id, base.eos_id, base.pad_id] + keep
        return cls(base, orig)

    @classmethod
    def identity(cls, base: LFMTokenizer) -> "ReducedVocab":
        """Full vocab, ids unchanged (specials are not relocated)."""
        r = cls.__new__(cls)
        r.base = base
        r.orig_ids = np.arange(len(base), dtype=np.int64)
        r.old2new = r.orig_ids.copy()
        r.PAD, r.BOS, r.EOS, r.UNK = base.pad_id, base.bos_id, base.eos_id, base.pad_id
        return r

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"orig_ids": self.orig_ids.tolist()}))

    @classmethod
    def load(cls, base: LFMTokenizer, path: str | Path) -> "ReducedVocab":
        return cls(base, json.loads(Path(path).read_text())["orig_ids"])

    # ---- api
    @property
    def vocab_size(self) -> int:
        return len(self.orig_ids)

    def encode(self, text: str) -> list[int]:
        return self.old2new[np.asarray(self.base.encode(text), dtype=np.int64)].tolist()

    def encode_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self.old2new[np.asarray(x, dtype=np.int64)] for x in self.base.encode_batch(texts)]

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        n = len(self.orig_ids)
        for i in ids:
            idx = int(i)
            if idx < 0 or idx >= n:
                out.append("<pad>")
                continue
            o = int(self.orig_ids[idx])
            out.append("<unk>" if idx == self.UNK and self.vocab_size < len(self.base) else self.base.decode([o]))
        return "".join(out)


# --------------------------------------------------------------------------------------
# Engram: token-normalising compression map (DeepSeek-V4.1 `build_compressed_token_map`)
# --------------------------------------------------------------------------------------
def build_compressed_token_map(tok: ReducedVocab) -> tuple[np.ndarray, int]:
    """Map every (reduced) token id to a smaller id space where tokens that normalise alike collapse
    (" The", "the", "THE" -> same id). N-grams are hashed over these ids, which shrinks the effective
    alphabet and therefore the collision pressure. Returns (lookup[vocab], compressed_vocab_size)."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tok.base.backend
    key_to_new: dict[str, int] = {}
    lookup = np.zeros(tok.vocab_size, dtype=np.int64)
    for new_id in range(tok.vocab_size):
        orig = int(tok.orig_ids[new_id])
        if new_id < ReducedVocab.N_SPECIAL and tok.vocab_size < len(tok.base):
            key = f"\ue001special{new_id}"  # specials never merge with text tokens
        else:
            text = backend.decode([orig], skip_special_tokens=False)
            if "\ufffd" in text:
                key = backend.id_to_token(orig)  # partial UTF-8 byte token: keep raw form
            else:
                n = normalizer.normalize_str(text)
                key = n if n else text
        j = key_to_new.setdefault(key, len(key_to_new))
        lookup[new_id] = j
    return lookup, len(key_to_new)
