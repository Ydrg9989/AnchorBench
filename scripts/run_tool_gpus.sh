#!/usr/bin/env bash
# DEPRECATED: prefer `python scripts/run.py eval --suite tool --model ...` or
#   `python scripts/run.py experiment --name paper_main` for the full panel.
# This script remains for backward compatibility with prior shell pipelines.
#
# Run Tool experiments maximizing GPU utilization.
#
# Strategy:
#   1. Run multiple models in parallel, one process per GPU (CUDA_VISIBLE_DEVICES=i).
#   2. Each process uses batched inference (--batch_size) with chat-template tool messages.
#
# Usage:
#   cd $REPO_ROOT && bash scripts/run_tool_gpus.sh
#
# Options (edit below or pass env vars):
#   TOOL_DATA_DIR   - directory with promptviews.jsonl and itemspecs.jsonl
#   TOOL_OUT_BASE   - results base dir (default: results/tool_pilot)
#   TOOL_BATCH_SIZE - batch size per model (default: 32 for 7B; reduce if OOM)
#   TOOL_MAX_ITEMS  - cap items (default: empty = all)
#   TOOL_MODELS     - space-separated ids (default: ten paper models)

set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export PYTHONPATH=src
# shellcheck source=paper_model_ids.inc.sh
source "$REPO_ROOT/scripts/paper_model_ids.inc.sh"

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR="${TOOL_DATA_DIR:-}"
OUT_BASE="${TOOL_OUT_BASE:-results/tool_pilot}"
BATCH_SIZE="${TOOL_BATCH_SIZE:-32}"
MAX_ITEMS_ARG=""
if [[ -n "${TOOL_MAX_ITEMS:-}" ]]; then
  MAX_ITEMS_ARG="--max_items $TOOL_MAX_ITEMS"
fi

# If no dataset path, generate Tool pilot
if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="datasets/anchorbench_tool_pilot"
  if [[ ! -f "$DATA_DIR/promptviews.jsonl" ]]; then
    echo "Generating Tool pilot dataset at $DATA_DIR ..."
    python -m anchorbench_v1.generate --suite tool --size pilot --out_dir "$DATA_DIR" --seed 42
  else
    echo "Using existing Tool dataset at $DATA_DIR"
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
if [[ -z "${TOOL_MODELS:-}" ]]; then
  MODELS=("${PAPER_ANCHORBENCH_MODEL_IDS[@]}")
else
  read -r -a MODELS <<< "${TOOL_MODELS}"
fi
NMODELS=${#MODELS[@]}
echo "Using $NGPU GPU(s); models: $NMODELS (paper list unless TOOL_MODELS set); waves if needed."

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
    export CUDA_VISIBLE_DEVICES=$slot
    echo "[GPU $slot] $model -> $OUT_BASE/${model//\//_}/"
    python scripts/eval/run_tool.py $DATA_ARGS --model_id "$model" --out_dir "$OUT_BASE" $OPTS &
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
echo "All Tool runs finished. Summaries:"
for model in "${MODELS[@]}"; do
  out_name="${model//\//_}"
  echo "  $OUT_BASE/${out_name}/summary.json"
done
