"""Pre-tokenize smollm-corpus parquet shards with Ling-3.0-atto's LFM3 tokenizer.

Writes BOS-delimited int32 binary shards (``[BOS] + token_ids`` per document) matching
``ling_atto.data.build_stream`` so training can memmap shards directly without tokenization overhead.

Example:
    PYTHONPATH=. python scripts/tokenize_smollm.py \
        --input_dir /workspace/smollm-corpus/data \
        --output_dir /workspace/smollm-corpus/tokenized_lfm3 \
        --tokenizer_dir assets/lfm3 \
        --workers 64
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ling_atto.tokenizer import LFMTokenizer

_WORKER_TOK: LFMTokenizer | None = None
_WORKER_BOS: int = 0


def _init_worker(tokenizer_dir: str) -> None:
    global _WORKER_TOK, _WORKER_BOS
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOK = LFMTokenizer(tokenizer_dir)
    _WORKER_BOS = _WORKER_TOK.bos_id


def _tokenize_row_group(task: tuple[str, int, str, int]) -> tuple[str, int, int, int]:
    """Tokenize one parquet row group to a temp .bin file; return (parquet_path, rg_idx, docs, tokens)."""
    parquet_path, rg_idx, tmp_path, batch_size = task
    assert _WORKER_TOK is not None
    tok = _WORKER_TOK
    bos = np.int32(_WORKER_BOS)

    pf = pq.ParquetFile(parquet_path)
    table = pf.read_row_group(rg_idx, columns=["text"])
    texts = table.column("text").to_pylist()

    n_docs = 0
    n_tokens = 0
    tmp_file = Path(tmp_path)
    tmp_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_file, "wb") as fp:
        for i in range(0, len(texts), batch_size):
            chunk = [t for t in texts[i : i + batch_size] if isinstance(t, str) and t]
            if not chunk:
                continue
            encoded = tok.encode_batch(chunk)
            parts = []
            for ids in encoded:
                if not ids:
                    continue
                arr = np.empty(len(ids) + 1, dtype=np.int32)
                arr[0] = bos
                arr[1:] = ids
                parts.append(arr)
                n_docs += 1
                n_tokens += len(arr)
            if parts:
                np.concatenate(parts).tofile(fp)
    return parquet_path, rg_idx, n_docs, n_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="/workspace/smollm-corpus/data")
    ap.add_argument("--output_dir", default="/workspace/smollm-corpus/tokenized_lfm3")
    ap.add_argument("--tokenizer_dir", default="assets/lfm3")
    ap.add_argument("--subsets", nargs="+", default=["python-edu", "cosmopedia-v2", "fineweb-edu-dedup"])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2048)
    args = ap.parse_args()

    t0 = time.time()
    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tok = LFMTokenizer(args.tokenizer_dir)
    print(f"[tok] loaded LFMTokenizer vocab={len(tok)} bos={tok.bos_id} eos={tok.eos_id} pad={tok.pad_id}", flush=True)

    manifest_all: dict = {
        "tokenizer_dir": str(Path(args.tokenizer_dir).resolve()),
        "vocab_size": len(tok),
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "pad_id": tok.pad_id,
        "dtype": "int32",
        "layout": "bos_delimited_docs",
        "subsets": {},
    }

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.tokenizer_dir,)) as ex:
        for subset in args.subsets:
            sub_in = in_root / subset
            sub_out = out_root / subset
            sub_out.mkdir(parents=True, exist_ok=True)
            manifest_path = sub_out / "manifest.json"

            pq_files = sorted(sub_in.glob("*.parquet"))
            if not pq_files:
                print(f"[skip] {subset}: no parquet files in {sub_in}", flush=True)
                continue

            existing_meta: dict[str, dict] = {}
            if manifest_path.exists():
                try:
                    prev = json.loads(manifest_path.read_text())
                    existing_meta = {s["file"]: s for s in prev.get("shards", [])}
                except Exception:
                    existing_meta = {}

            todo_files: list[tuple[Path, Path, int]] = []
            done_shards: dict[str, dict] = {}
            for pq_f in pq_files:
                bin_name = pq_f.stem + ".bin"
                bin_path = sub_out / bin_name
                if bin_path.exists() and bin_path.stat().st_size > 0 and bin_name in existing_meta:
                    done_shards[bin_name] = existing_meta[bin_name]
                else:
                    n_rg = pq.ParquetFile(pq_f).num_row_groups
                    todo_files.append((pq_f, bin_path, n_rg))

            if todo_files:
                ts = time.time()
                tmp_dir = sub_out / ".tmp_rg"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tasks = []
                file_rgs: dict[str, int] = {}
                file_bins: dict[str, Path] = {}
                for pq_f, bin_path, n_rg in todo_files:
                    file_rgs[str(pq_f)] = n_rg
                    file_bins[str(pq_f)] = bin_path
                    for rg_idx in range(n_rg):
                        tmp_rg = tmp_dir / f"{pq_f.stem}.rg{rg_idx:04d}.bin"
                        tasks.append((str(pq_f), rg_idx, str(tmp_rg), args.batch_size))

                print(f"[{subset}] tokenizing {len(todo_files)} files ({len(tasks)} row-groups) with {args.workers} workers...", flush=True)
                rg_results: dict[str, dict[int, tuple[int, int, str]]] = {str(p): {} for p, _, _ in todo_files}
                futs = [ex.submit(_tokenize_row_group, t) for t in tasks]
                completed_rg = 0
                for fut in as_completed(futs):
                    pq_str, rg_idx, n_docs, n_toks = fut.result()
                    completed_rg += 1
                    rg_path = tmp_dir / f"{Path(pq_str).stem}.rg{rg_idx:04d}.bin"
                    rg_results[pq_str][rg_idx] = (n_docs, n_toks, str(rg_path))

                    if len(rg_results[pq_str]) == file_rgs[pq_str]:
                        bin_path = file_bins[pq_str]
                        tmp_out = bin_path.with_suffix(".bin.tmp")
                        total_docs = 0
                        total_toks = 0
                        with open(tmp_out, "wb") as dst:
                            for r in range(file_rgs[pq_str]):
                                d_cnt, t_cnt, r_path = rg_results[pq_str][r]
                                total_docs += d_cnt
                                total_toks += t_cnt
                                with open(r_path, "rb") as src:
                                    while True:
                                        buf = src.read(16 * 1024 * 1024)
                                        if not buf:
                                            break
                                        dst.write(buf)
                                os.remove(r_path)
                        os.replace(tmp_out, bin_path)
                        done_shards[bin_path.name] = {
                            "file": bin_path.name,
                            "source": Path(pq_str).name,
                            "docs": total_docs,
                            "tokens": total_toks,
                            "bytes": bin_path.stat().st_size,
                        }
                        print(f"  [{subset}] {bin_path.name} done: {total_docs:,} docs, {total_toks:,} tokens "
                              f"({len(done_shards)}/{len(pq_files)} files, {time.time()-ts:.1f}s)", flush=True)

                try:
                    tmp_dir.rmdir()
                except OSError:
                    pass

            shards_sorted = [done_shards[f.stem + ".bin"] for f in pq_files if (f.stem + ".bin") in done_shards]
            sub_docs = sum(s["docs"] for s in shards_sorted)
            sub_toks = sum(s["tokens"] for s in shards_sorted)
            sub_bytes = sum(s["bytes"] for s in shards_sorted)
            sub_manifest = {
                "subset": subset,
                "vocab_size": len(tok),
                "bos_id": tok.bos_id,
                "dtype": "int32",
                "total_docs": sub_docs,
                "total_tokens": sub_toks,
                "total_bytes": sub_bytes,
                "shards": shards_sorted,
            }
            manifest_path.write_text(json.dumps(sub_manifest, indent=2))
            manifest_all["subsets"][subset] = {
                "total_docs": sub_docs,
                "total_tokens": sub_toks,
                "total_bytes": sub_bytes,
                "num_shards": len(shards_sorted),
            }
            print(f"[{subset}] COMPLETE: {len(shards_sorted)} shards, {sub_docs:,} docs, "
                  f"{sub_toks:,} tokens ({sub_bytes / 2**30:.2f} GiB)", flush=True)

    manifest_all["total_docs"] = sum(v["total_docs"] for v in manifest_all["subsets"].values())
    manifest_all["total_tokens"] = sum(v["total_tokens"] for v in manifest_all["subsets"].values())
    manifest_all["total_bytes"] = sum(v["total_bytes"] for v in manifest_all["subsets"].values())
    manifest_all["elapsed_sec"] = round(time.time() - t0, 2)
    (out_root / "manifest.json").write_text(json.dumps(manifest_all, indent=2))
    print(f"[ALL DONE] total_docs={manifest_all['total_docs']:,} total_tokens={manifest_all['total_tokens']:,} "
          f"size={manifest_all['total_bytes'] / 2**30:.2f} GiB in {manifest_all['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
