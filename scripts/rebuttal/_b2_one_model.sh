#!/usr/bin/env bash
# Helper invoked by the tmux orchestrator: runs B2 (plausibility-spectrum
# placebo+authority eval) for a single model on one GPU (or via OpenRouter).
#
# Usage:
#   bash scripts/rebuttal/_b2_one_model.sh <model_id> <cuda_or_api>
#     cuda_or_api: an integer (GPU index) for open-weight, or the literal
#     string "api" to use OpenRouter.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu|api>}"
DEVICES="${2:?usage: $0 <model_id> <gpu|api>}"

OUT_DIR="${OUT_DIR:-results/rebuttal/spectrum}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
SLUG="${MODEL_ID//\//_}"
LOG="$LOG_DIR/${SLUG}.log"

echo "[$(date -Iseconds)] B2 starting $MODEL_ID on $DEVICES" | tee -a "$LOG"

if [ "$DEVICES" = "api" ]; then
    bash scripts/run_with_env.sh \
        python -m anchorbench.runners.rebuttal_spectrum \
            --model_id "$MODEL_ID" \
            --backend openrouter \
            --out_dir "$OUT_DIR" \
            --max_concurrent 16 2>&1 | tee -a "$LOG"
else
    export CUDA_VISIBLE_DEVICES="$DEVICES"
    bash scripts/run_with_env.sh \
        python -m anchorbench.runners.rebuttal_spectrum \
            --model_id "$MODEL_ID" \
            --backend vllm \
            --tensor_parallel_size 1 \
            --gpu_memory_utilization 0.85 \
            --out_dir "$OUT_DIR" \
            --batch_size 32 \
            --max_tokens 512 2>&1 | tee -a "$LOG"
fi

echo "[$(date -Iseconds)] B2 done $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
