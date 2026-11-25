import numpy as np
import torch

from ling_atto.data import IndexedSequenceLoader, ShardedTokenStream, split_sequence_indices
from ling_atto.tokenizer import LFMTokenizer, ReducedVocab
from ling_atto.train import main as train_main


def test_sharded_stream_and_epoch_loader_cross_boundary(tmp_path):
    s1 = np.arange(0, 15, dtype=np.int32)
    s2 = np.arange(15, 32, dtype=np.int32)
    p1 = tmp_path / "real" / "a.bin"
    p2 = tmp_path / "real" / "b.bin"
    p1.parent.mkdir()
    s1.tofile(p1)
    s2.tofile(p2)
    link_dir = tmp_path / "link"
    link_dir.symlink_to(p1.parent)
    from ling_atto.data import open_token_stream
    stream = open_token_stream(link_dir)
    assert len(stream) == 32
    assert stream.read_window(12, 6).tolist() == [12, 13, 14, 15, 16, 17]

    train_ids, val_ids = split_sequence_indices(len(stream), seq_len=4, val_seqs=2, seed=0)
    assert len(train_ids) + len(val_ids) == 8
    assert set(train_ids.tolist()).isdisjoint(set(val_ids.tolist()))

    loader = IndexedSequenceLoader(stream, train_ids, batch_size=2, seq_len=4, device="cpu")
    b0 = loader.next()
    assert b0.shape == (2, 4)
    loader.set_cursor(0)
    b0_again = loader.next()
    assert torch.equal(b0, b0_again)


def test_epoch_train_save_and_resume(tmp_path):
    out = tmp_path / "run"
    train_main([
        "--config", "nano",
        "--data", "synthetic",
        "--device", "cpu",
        "--max_docs", "120",
        "--epochs", "1",
        "--steps", "4",
        "--batch_size", "4",
        "--grad_accum", "2",
        "--seq_len", "32",
        "--save_every", "2",
        "--eval_every", "2",
        "--eval_batches", "2",
        "--log_every", "2",
        "--out", str(out),
    ])
    assert (out / "ckpt_latest.pt").exists()
    assert (out / "ckpt.pt").exists()
    # Resume from ckpt_latest.pt (saved at step 4 -> run up to step 6)
    hist = train_main([
        "--config", "nano",
        "--data", "synthetic",
        "--device", "cpu",
        "--max_docs", "120",
        "--epochs", "1",
        "--steps", "6",
        "--batch_size", "4",
        "--grad_accum", "2",
        "--seq_len", "32",
        "--save_every", "2",
        "--eval_every", "2",
        "--eval_batches", "2",
        "--log_every", "1",
        "--resume", "auto",
        "--out", str(out),
    ])
    assert any(h["step"] == 6 for h in hist)


def test_reduced_vocab_decode_out_of_range(tmp_path):
    # Minimal mock via ReducedVocab.__new__
    r = ReducedVocab.__new__(ReducedVocab)
    r.orig_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    r.UNK = 3
    r.base = None
    assert r.decode([999999, -1]) == "<pad><pad>"
