"""Corpus loading + token-stream batching.

Sources
  synthetic     offline generator: grammar noise + deterministic entity->attribute 'facts' that repeat in
                fixed 2/3-gram contexts (exactly what an n-gram memory should pick up)
  tinystories   first ~N MB of TinyStories-valid (downloaded via HTTP range request, cached)
  <path>        any UTF-8 text file; documents are separated by blank lines or <|endoftext|>
"""
from __future__ import annotations

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


class StreamLoader:
    def __init__(self, stream: np.ndarray, batch_size: int, seq_len: int, device, seed: int = 0):
        self.s = torch.from_numpy(stream.astype(np.int64))
        self.B, self.T, self.device = batch_size, seq_len, device
        self.g = torch.Generator().manual_seed(seed)
        assert len(self.s) > seq_len + 1, "corpus shorter than one sequence"

    def next(self) -> torch.Tensor:
        ix = torch.randint(0, len(self.s) - self.T, (self.B,), generator=self.g)
        x = torch.stack([self.s[i : i + self.T] for i in ix.tolist()])
        return x.to(self.device, non_blocking=True)
