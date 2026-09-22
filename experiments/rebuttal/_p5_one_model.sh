#!/usr/bin/env bash
# P5 (uncertain-judgment) launcher for a single open-weight model.
# Usage: bash experiments/rebuttal/_p5_one_model.sh <model_id> <gpu> [max_items]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu> [max_items]}"
DEVICES="${2:?usage: $0 <model_id> <gpu> [max_items]}"
MAX_ITEMS="${3:-}"

OUT_DIR="${OUT_DIR:-results/rebuttal/uncertain}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
SLUG="${MODEL_ID//\//_}"
LOG="$LOG_DIR/${SLUG}.log"

echo "[$(date -Iseconds)] P5[uncertain] $MODEL_ID on GPU $DEVICES (max_items=$MAX_ITEMS)" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES="$DEVICES"
EXTRA_ARGS=()
if [ -n "$MAX_ITEMS" ]; then
    EXTRA_ARGS+=( --max_items "$MAX_ITEMS" )
fi

bash scripts/run_with_env.sh \
    python -m anchorbench.runners.rebuttal_uncertain \
        --model_id "$MODEL_ID" \
        --out_dir "$OUT_DIR" \
        --backend vllm \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --batch_size 32 \
        --max_tokens 512 \
        "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] P5[uncertain] done $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
