#!/usr/bin/env bash
# Run RAG v2 experiments maximizing GPU utilization.
#
# Strategy:
#   1. Run multiple models in parallel, one process per GPU (each with CUDA_VISIBLE_DEVICES=i).
#   2. Each process uses batched inference (--batch_size) to keep its GPU busy.
#
# Usage:
#   cd /data/yiderigun/LLM_anchoring && bash scripts/run_rag_v2_gpus.sh
#
# Options (edit below or pass env vars):
#   RAG_V2_DATA_DIR   - directory with promptviews.jsonl and itemspecs.jsonl (default: generate pilot)
#   RAG_V2_OUT_BASE   - results base dir (default: results/rag_v2_pilot)
#   RAG_V2_BATCH_SIZE - batch size per model (default: 32 for 7B; reduce if OOM)
#   RAG_V2_MAX_ITEMS  - cap items (default: empty = all)
#
# Requires: PYTHONPATH=src, and at least one GPU.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR="${RAG_V2_DATA_DIR:-}"
OUT_BASE="${RAG_V2_OUT_BASE:-results/rag_v2_pilot}"
BATCH_SIZE="${RAG_V2_BATCH_SIZE:-32}"
MAX_ITEMS_ARG=""
if [[ -n "${RAG_V2_MAX_ITEMS:-}" ]]; then
  MAX_ITEMS_ARG="--max_items $RAG_V2_MAX_ITEMS"
fi

# If no dataset path, generate RAG v2 pilot
if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="datasets/anchorbench_v2_rag_pilot"
  if [[ ! -f "$DATA_DIR/promptviews.jsonl" ]]; then
    echo "Generating RAG v2 pilot dataset at $DATA_DIR ..."
    python -m anchorbench_v1.generate --v2 --suites rag --size pilot --out_dir "$DATA_DIR" --seed 42
  else
    echo "Using existing RAG v2 dataset at $DATA_DIR"
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

# ── Model list: one model per GPU (edit to match your GPUs and VRAM) ────
# For 4× H100/A100: 4× 7B–8B models fit (one per GPU). Adjust if OOM.
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
)

# If more GPUs than models, we run only len(MODELS) jobs. If more models than GPUs, run first NGPU models.
NJOBS=$(( NGPU < ${#MODELS[@]} ? NGPU : ${#MODELS[@]} ))
echo "Running $NJOBS model(s) in parallel (one per GPU)."

# Run one model per GPU in background (CUDA_VISIBLE_DEVICES pins to one GPU)
pids=()
for i in $(seq 0 $((NJOBS - 1))); do
  model="${MODELS[$i]}"
  # Sanitize model name for directory
  out_name=$(echo "$model" | sed 's|/|_|g')
  out_dir="$OUT_BASE/${out_name}"
  mkdir -p "$out_dir"
  export CUDA_VISIBLE_DEVICES=$i
  echo "[GPU $i] Starting $model -> $out_dir"
  python scripts/eval/run_rag_v2.py $DATA_ARGS --model_id "$model" --out_dir "$out_dir" $OPTS &
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
echo "All RAG v2 runs finished. Summaries:"
for i in $(seq 0 $((NJOBS - 1))); do
  model="${MODELS[$i]}"
  out_name=$(echo "$model" | sed 's|/|_|g')
  echo "  $OUT_BASE/${out_name}/summary.json"
done
