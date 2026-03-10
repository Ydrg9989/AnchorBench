#!/usr/bin/env bash
# Run Tool v2 experiments maximizing GPU utilization.
#
# Strategy:
#   1. Run multiple models in parallel, one process per GPU (CUDA_VISIBLE_DEVICES=i).
#   2. Each process uses batched inference (--batch_size) with chat-template tool messages.
#
# Usage:
#   cd /data/yiderigun/LLM_anchoring && bash scripts/run_tool_v2_gpus.sh
#
# Options (edit below or pass env vars):
#   TOOL_V2_DATA_DIR   - directory with promptviews.jsonl and itemspecs.jsonl
#   TOOL_V2_OUT_BASE   - results base dir (default: results/tool_v2_pilot)
#   TOOL_V2_BATCH_SIZE - batch size per model (default: 32 for 7B; reduce if OOM)
#   TOOL_V2_MAX_ITEMS  - cap items (default: empty = all)

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR="${TOOL_V2_DATA_DIR:-}"
OUT_BASE="${TOOL_V2_OUT_BASE:-results/tool_v2_pilot}"
BATCH_SIZE="${TOOL_V2_BATCH_SIZE:-32}"
MAX_ITEMS_ARG=""
if [[ -n "${TOOL_V2_MAX_ITEMS:-}" ]]; then
  MAX_ITEMS_ARG="--max_items $TOOL_V2_MAX_ITEMS"
fi

# If no dataset path, generate Tool v2 pilot
if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="datasets/anchorbench_v2_tool_pilot"
  if [[ ! -f "$DATA_DIR/promptviews.jsonl" ]]; then
    echo "Generating Tool v2 pilot dataset at $DATA_DIR ..."
    python -m anchorbench_v1.generate --v2 --suites tool_v2 --size pilot --out_dir "$DATA_DIR" --seed 42
  else
    echo "Using existing Tool v2 dataset at $DATA_DIR"
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
echo "Using $NGPU GPU(s)."

# ── Model list ───────────────────────────────────────────────────────
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
)

NJOBS=$(( NGPU < ${#MODELS[@]} ? NGPU : ${#MODELS[@]} ))
echo "Running $NJOBS model(s) in parallel (one per GPU)."

pids=()
for i in $(seq 0 $((NJOBS - 1))); do
  model="${MODELS[$i]}"
  out_name=$(echo "$model" | sed 's|/|_|g')
  out_dir="$OUT_BASE/${out_name}"
  mkdir -p "$out_dir"
  export CUDA_VISIBLE_DEVICES=$i
  echo "[GPU $i] Starting $model -> $out_dir"
  python scripts/eval/run_tool_v2.py $DATA_ARGS --model_id "$model" --out_dir "$OUT_BASE" $OPTS &
  pids+=($!)
done

# Wait for all jobs
for i in $(seq 0 $((NJOBS - 1))); do
  pid=${pids[$i]}
  model="${MODELS[$i]}"
  if wait $pid; then
    echo "[GPU $i] $model finished successfully."
  else
    echo "[GPU $i] $model exited with error."
    exit 1
  fi
done

echo ""
echo "All Tool v2 runs finished. Summaries:"
for i in $(seq 0 $((NJOBS - 1))); do
  model="${MODELS[$i]}"
  out_name=$(echo "$model" | sed 's|/|_|g')
  echo "  $OUT_BASE/${out_name}/summary.json"
done
