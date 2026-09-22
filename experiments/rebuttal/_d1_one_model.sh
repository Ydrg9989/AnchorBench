#!/usr/bin/env bash
# Plausibility-intensity probe (D1 / P1) launcher for a single open-weight model.
# Usage:
#   bash experiments/rebuttal/_d1_one_model.sh <model_id> <gpu_index> [suite]
# If suite is omitted, runs External (original D1).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu> [suite]}"
DEVICES="${2:?usage: $0 <model_id> <gpu> [suite]}"
SUITE="${3:-external}"

case "$SUITE" in
    external) DEFAULT_OUT="results/rebuttal/intensity" ;;
    rag)      DEFAULT_OUT="results/rebuttal/intensity_rag" ;;
    history)  DEFAULT_OUT="results/rebuttal/intensity_history" ;;
    *) echo "Unknown suite: $SUITE" >&2; exit 1 ;;
esac

OUT_DIR="${OUT_DIR:-$DEFAULT_OUT}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
SLUG="${MODEL_ID//\//_}"
LOG="$LOG_DIR/${SLUG}.log"

echo "[$(date -Iseconds)] D1[$SUITE] starting $MODEL_ID on GPU $DEVICES" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES="$DEVICES"
bash scripts/run_with_env.sh \
    python -m anchorbench.runners.rebuttal_intensity \
        --model_id "$MODEL_ID" \
        --suite "$SUITE" \
        --out_dir "$OUT_DIR" \
        --backend vllm \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --batch_size 32 \
        --max_tokens 512 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] D1[$SUITE] done $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
