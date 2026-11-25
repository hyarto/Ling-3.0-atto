#!/usr/bin/env bash
# One-click full 1-epoch training script for Ling-3.0-atto on pre-tokenized smollm-corpus.
#
# Usage:
#   bash scripts/train_1epoch.sh
#
# Optional environment overrides:
#   BATCH_SIZE=8 SEQ_LEN=2048 GRAD_ACCUM=8 LR=3e-4 ENGRAM_LR=0.05 \
#   SAVE_EVERY=500 EVAL_EVERY=500 OUT_DIR=runs/atto_full_1epoch bash scripts/train_1epoch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

CONFIG="${CONFIG:-atto_full}"
DATA="${DATA:-data/smollm_lfm3}"
TOKENIZER_DIR="${TOKENIZER_DIR:-assets/lfm3}"
OUT_DIR="${OUT_DIR:-runs/atto_full_1epoch}"
LOG_FILE="${LOG_FILE:-logs/train_1epoch.log}"

BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"       # 8 * 2048 * 8 = 131,072 tokens / optimizer step
EPOCHS="${EPOCHS:-1}"
STEPS="${STEPS:-0}"                 # 0 => auto-computed from 1 full epoch
LR="${LR:-3e-4}"
ENGRAM_LR="${ENGRAM_LR:-0.05}"
WARMUP="${WARMUP:-1000}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
VAL_SEQS="${VAL_SEQS:-4096}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-2000}"    # ~3.8-4.0h at 19k tok/s (131,072 tok/step)
EVAL_BATCHES="${EVAL_BATCHES:-32}"
SAVE_EVERY="${SAVE_EVERY:-2000}"    # ~3.8-4.0h per checkpoint (~262M tokens)
RESUME="${RESUME:-auto}"
SEED="${SEED:-42}"

if [[ ! -d "$DATA" && ! -f "$DATA" ]]; then
  echo "[error] tokenized dataset not found at: $ROOT/$DATA" >&2
  echo "Run scripts/tokenize_smollm.py first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "Ling-3.0-atto Full 1-Epoch Training"
echo "  root          : $ROOT"
echo "  python        : $PYTHON_BIN"
echo "  config        : $CONFIG"
echo "  data          : $DATA"
echo "  out_dir       : $OUT_DIR (resume=$RESUME)"
echo "  log_file      : $LOG_FILE"
echo "  batch / seq   : B=$BATCH_SIZE, T=$SEQ_LEN, grad_accum=$GRAD_ACCUM ($((BATCH_SIZE * SEQ_LEN * GRAD_ACCUM)) tok/step)"
echo "  epochs        : $EPOCHS (steps=$STEPS)"
echo "  lr / engram   : lr=$LR, engram_lr=$ENGRAM_LR, warmup=$WARMUP"
echo "  save / eval   : save_every=$SAVE_EVERY, eval_every=$EVAL_EVERY"
echo "============================================================"

"$PYTHON_BIN" -u -m ling_atto.train \
  --config "$CONFIG" \
  --data "$DATA" \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --out "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --seq_len "$SEQ_LEN" \
  --lr "$LR" \
  --engram_lr "$ENGRAM_LR" \
  --warmup "$WARMUP" \
  --min_lr_ratio "$MIN_LR_RATIO" \
  --weight_decay "$WEIGHT_DECAY" \
  --val_seqs "$VAL_SEQS" \
  --log_every "$LOG_EVERY" \
  --eval_every "$EVAL_EVERY" \
  --eval_batches "$EVAL_BATCHES" \
  --save_every "$SAVE_EVERY" \
  --resume "$RESUME" \
  --seed "$SEED" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
