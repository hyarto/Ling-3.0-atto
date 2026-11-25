"""Corpus loading + token-stream batching.

Sources
  synthetic     offline generator: grammar noise + deterministic entity->attribute 'facts' that repeat in
                fixed 2/3-gram contexts (exactly what an n-gram memory should pick up)
  tinystories   first ~N MB of TinyStories-valid (downloaded via HTTP range request, cached)
  <path>        any UTF-8 text file; documents are separated by blank lines or <|endoftext|>
"""
from __future__ import annotations

import os
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch

TS_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"


def synthetic_docs(n_docs: int = 4000, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    countries = ["France", "Spain", "Italy", "Japan", "Brazil", "Egypt", "Kenya", "Norway", "Chile", "India",
                 "Peru", "Cuba", "Iran", "Ghana", "Nepal", "Laos"]
    cities = ["Paris", "Madrid", "Rome", "Tokyo", "Brasilia", "Cairo", "Nairobi", "Oslo", "Santiago", "Delhi",
              "Lima", "Havana", "Tehran", "Accra", "Kathmandu", "Vientiane"]
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]
    animals = ["cat", "dog", "fox", "owl", "bear", "wolf", "deer", "hare"]
    verbs = ["runs", "sleeps", "jumps", "waits", "sings", "hides", "walks", "plays"]
    capital = dict(zip(countries, cities))
    fav = {a: rng.choice(colors) for a in animals}
    docs = []
    for _ in range(n_docs):
        s = []
        for _ in range(rng.randint(3, 8)):
            r = rng.random()
            if r < 0.35:
                c = rng.choice(countries)
                s.append(f"The capital of {c} is {capital[c]}.")
            elif r < 0.6:
                a = rng.choice(animals)
                s.append(f"The {a} likes the color {fav[a]}.")
            else:
                s.append(f"A {rng.choice(colors)} {rng.choice(animals)} {rng.choice(verbs)} near the {rng.choice(colors)} {rng.choice(animals)}.")
        docs.append(" ".join(s))
    return docs


def tinystories_docs(cache_dir="assets/data", max_mb: float = 8.0) -> list[str]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / "tinystories_valid_head.txt"
    if not f.exists():
        req = urllib.request.Request(TS_URL, headers={"Range": f"bytes=0-{int(max_mb * 2**20)}"})
        with urllib.request.urlopen(req) as r:
            f.write_bytes(r.read())
    txt = f.read_text(encoding="utf-8", errors="ignore")
    docs = [d.strip() for d in txt.split("<|endoftext|>")]
    return [d for d in docs[:-1] if d]  # last doc is truncated


def load_docs(source: str, max_docs: int | None = None) -> list[str]:
    if source == "synthetic":
        docs = synthetic_docs()
    elif source == "tinystories":
        docs = tinystories_docs()
    else:
        txt = Path(source).read_text(encoding="utf-8", errors="ignore")
        docs = [d.strip() for d in (txt.split("<|endoftext|>") if "<|endoftext|>" in txt else txt.split("\n\n"))]
        docs = [d for d in docs if d]
    return docs[:max_docs] if max_docs else docs


def build_stream(vocab, docs: list[str]) -> np.ndarray:
    """BOS-delimited token stream (BOS also acts as the Engram n-gram barrier)."""
    parts = []
    for ids in vocab.encode_batch(docs):
        parts.append(np.concatenate([[vocab.BOS], ids]).astype(np.int32))
    return np.concatenate(parts)


class ShardedTokenStream:
    """Read-only memmapped view over one or more raw int32 ``.bin`` token shards."""

    def __init__(self, paths: list[Path], max_tokens: int | None = None):
        if not paths:
            raise FileNotFoundError("No .bin token shards found")
        self.shards: list[np.memmap] = []
        self.starts: list[int] = []
        total = 0
        for p in paths:
            n_items = p.stat().st_size // 4
            if n_items <= 0:
                continue
            if max_tokens is not None and total >= max_tokens:
                break
            take = n_items if max_tokens is None else min(n_items, max_tokens - total)
            mm = np.memmap(p, dtype=np.int32, mode="r", shape=(n_items,))[:take]
            self.shards.append(mm)
            self.starts.append(total)
            total += take
        if total == 0:
            raise ValueError("Token shards are empty")
        self.starts_arr = np.asarray(self.starts, dtype=np.int64)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def read_window(self, pos: int, length: int) -> np.ndarray:
        idx = int(np.searchsorted(self.starts_arr, pos, side="right")) - 1
        off = pos - self.starts[idx]
        shard = self.shards[idx]
        if off + length <= len(shard):
            return np.asarray(shard[off : off + length], dtype=np.int64)
        chunks = [np.asarray(shard[off:], dtype=np.int64)]
        rem = length - len(chunks[0])
        s_idx = idx + 1
        while rem > 0:
            nxt = self.shards[s_idx]
            take = min(rem, len(nxt))
            chunks.append(np.asarray(nxt[:take], dtype=np.int64))
            rem -= take
            s_idx += 1
        return np.concatenate(chunks)


def _iter_bin_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and ".tmp" not in d)
        for fn in sorted(filenames):
            if fn.endswith(".bin") and not fn.startswith(".") and ".tmp" not in fn:
                out.append(Path(dirpath) / fn)
    return sorted(out)


def is_tokenized_source(source: str) -> bool:
    p = Path(source)
    if p.is_file() and p.suffix in {".bin", ".npy"}:
        return True
    if p.is_dir():
        return len(_iter_bin_files(p)) > 0
    return False


def open_token_stream(source: str | Path, max_tokens: int | None = None):
    p = Path(source)
    if p.is_file() and p.suffix == ".npy":
        arr = np.load(p, mmap_mode="r")
        return arr[:max_tokens] if max_tokens is not None else arr
    if p.is_file() and p.suffix == ".bin":
        return ShardedTokenStream([p], max_tokens=max_tokens)
    if p.is_dir():
        return ShardedTokenStream(_iter_bin_files(p), max_tokens=max_tokens)
    raise ValueError(f"Unsupported tokenized source: {source}")


class StreamLoader:
    def __init__(
        self,
        stream: np.ndarray | ShardedTokenStream,
        batch_size: int,
        seq_len: int,
        device,
        seed: int = 0,
        start: int = 0,
        end: int | None = None,
    ):
        self.B, self.T, self.device = batch_size, seq_len, device
        self.g = torch.Generator().manual_seed(seed)
        self.start = start
        self.end = len(stream) if end is None else end
        assert self.end - self.start > seq_len + 1, "corpus shorter than one sequence"
        if isinstance(stream, np.ndarray) and not isinstance(stream, np.memmap):
            self.s = torch.from_numpy(stream[self.start : self.end].astype(np.int64))
            self.stream = None
        else:
            self.s = None
            self.stream = stream

    def next(self) -> torch.Tensor:
        if self.s is not None:
            ix = torch.randint(0, len(self.s) - self.T, (self.B,), generator=self.g)
            x = torch.stack([self.s[i : i + self.T] for i in ix.tolist()])
            return x.to(self.device, non_blocking=True)
        ix = torch.randint(self.start, self.end - self.T, (self.B,), generator=self.g)
        if isinstance(self.stream, ShardedTokenStream):
            rows = [self.stream.read_window(i, self.T) for i in ix.tolist()]
        else:
            rows = [np.asarray(self.stream[i : i + self.T], dtype=np.int64) for i in ix.tolist()]
        x = torch.from_numpy(np.stack(rows))
        return x.to(self.device, non_blocking=True)


def split_sequence_indices(total_tokens: int, seq_len: int, val_seqs: int = 4096, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split non-overlapping sequences uniformly into shuffled train indices and evenly-spaced val indices."""
    n_seqs = total_tokens // seq_len
    if n_seqs < 2:
        raise ValueError(f"Corpus too short ({total_tokens} tokens) for seq_len={seq_len}")
    n_val = min(val_seqs, max(1, n_seqs // 20))
    val_idx = np.unique(np.linspace(0, n_seqs - 1, n_val, dtype=np.int64))
    mask = np.ones(n_seqs, dtype=bool)
    mask[val_idx] = False
    train_idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    rng.shuffle(train_idx)
    return train_idx, val_idx


class IndexedSequenceLoader:
    """Deterministic sequential loader over shuffled non-overlapping sequence indices (for exact-epoch runs)."""

    def __init__(
        self,
        stream: np.ndarray | ShardedTokenStream,
        seq_ids: np.ndarray,
        batch_size: int,
        seq_len: int,
        device,
        cursor: int = 0,
    ):
        self.stream = stream
        self.seq_ids = seq_ids
        self.B, self.T, self.device = batch_size, seq_len, device
        self.cursor = cursor

    def __len__(self) -> int:
        return len(self.seq_ids)

    def set_cursor(self, cursor: int) -> None:
        self.cursor = cursor

    def next(self) -> torch.Tensor:
        n = len(self.seq_ids)
        rows = []
        for _ in range(self.B):
            sid = int(self.seq_ids[self.cursor % n])
            self.cursor += 1
            pos = sid * self.T
            if isinstance(self.stream, ShardedTokenStream):
                rows.append(self.stream.read_window(pos, self.T))
            else:
                rows.append(np.asarray(self.stream[pos : pos + self.T], dtype=np.int64))
        x = torch.from_numpy(np.stack(rows))
        return x.to(self.device, non_blocking=True)
