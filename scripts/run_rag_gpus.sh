#!/usr/bin/env bash
# Run RAG experiments maximizing GPU utilization.
#
# Strategy:
#   1. Run multiple models in parallel, one process per GPU (each with CUDA_VISIBLE_DEVICES=i).
#   2. Each process uses batched inference (--batch_size) to keep its GPU busy.
#
# Usage:
#   cd $REPO_ROOT && bash scripts/run_rag_gpus.sh
#
# Options (edit below or pass env vars):
#   RAG_DATA_DIR   - directory with promptviews.jsonl and itemspecs.jsonl (default: generate pilot)
#   RAG_OUT_BASE   - results base dir (default: results/rag_pilot)
#   RAG_BATCH_SIZE - batch size per model (default: 32 for 7B; reduce if OOM)
#   RAG_MAX_ITEMS  - cap items (default: empty = all)
#   RAG_MODELS     - space-separated ids (default: ten paper models from paper_model_ids.inc.sh)
#
# Requires: PYTHONPATH=src, and at least one GPU.

set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export PYTHONPATH=src
# shellcheck source=paper_model_ids.inc.sh
source "$REPO_ROOT/scripts/paper_model_ids.inc.sh"

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR="${RAG_DATA_DIR:-}"
OUT_BASE="${RAG_OUT_BASE:-results/rag_pilot}"
BATCH_SIZE="${RAG_BATCH_SIZE:-32}"
MAX_ITEMS_ARG=""
if [[ -n "${RAG_MAX_ITEMS:-}" ]]; then
  MAX_ITEMS_ARG="--max_items $RAG_MAX_ITEMS"
fi

# If no dataset path, generate RAG pilot
if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="datasets/anchorbench_rag_pilot"
  if [[ ! -f "$DATA_DIR/promptviews.jsonl" ]]; then
    echo "Generating RAG pilot dataset at $DATA_DIR ..."
    python -m anchorbench_v1.generate --suite rag --size pilot --out_dir "$DATA_DIR" --seed 42
  else
    echo "Using existing RAG dataset at $DATA_DIR"
  fi
fi

PROMPTVIEWS="$DATA_DIR/promptviews.jsonl"
ITEMSPECS="$DATA_DIR/itemspecs.jsonl"
if [[ ! -f "$PROMPTVIEWS" ]] || [[ ! -f "$ITEMSPECS" ]]; then
  echo "ERROR: Missing $PROMPTVIEWS or $ITEMSPECS"
  exit 1
fi

DATA_ARGS="--promptviews $PROMPTVIEWS --itemspecs $ITEMSPECS"
OPTS="--max_tokens 512 --batch_size $BATCH_SIZE --llm_fallback --fallback_model Qwen/Qwen2.5-1.5B-Instruct --fallback_device auto $MAX_ITEMS_ARG"

# Number of GPUs to use (default: all visible)
NGPU="${NGPU:-}"
if [[ -z "$NGPU" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  else
    NGPU=1
  fi
fi
if [[ -z "${RAG_MODELS:-}" ]]; then
  MODELS=("${PAPER_ANCHORBENCH_MODEL_IDS[@]}")
else
  read -r -a MODELS <<< "${RAG_MODELS}"
fi
NMODELS=${#MODELS[@]}
echo "Using $NGPU GPU(s); models: $NMODELS (paper list unless RAG_MODELS set); waves if needed."

failed=0
for ((start = 0; start < NMODELS; start += NGPU)); do
  wave_end=$((start + NGPU))
  [[ $wave_end -gt $NMODELS ]] && wave_end=$NMODELS
  echo ""
  echo "--- Wave: models $((start + 1))–${wave_end} of ${NMODELS} ---"
  pids=()
  slot=0
  for ((idx = start; idx < wave_end; idx++)); do
    model="${MODELS[$idx]}"
    out_name="${model//\//_}"
    out_dir="$OUT_BASE/${out_name}"
    mkdir -p "$out_dir"
    export CUDA_VISIBLE_DEVICES=$slot
    echo "[GPU $slot] $model -> $out_dir"
    python scripts/eval/run_rag.py $DATA_ARGS --model_id "$model" --out_dir "$out_dir" $OPTS &
    pids+=($!)
    slot=$((slot + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
done

[[ "$failed" -eq 0 ]] || exit 1

echo ""
echo "All RAG runs finished. Summaries:"
for model in "${MODELS[@]}"; do
  out_name="${model//\//_}"
  echo "  $OUT_BASE/${out_name}/summary.json"
done
