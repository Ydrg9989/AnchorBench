#!/usr/bin/env bash
# Helper invoked by the tmux orchestrator: runs B1 (CoT extension) for
# a single open-weight model on one GPU. Logs everything to a per-model
# log file under <OUT_DIR>/_logs/.
#
# Usage:
#   bash scripts/rebuttal/_b1_one_model.sh <model_id> <cuda_visible_devices> [extra args...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <cuda_visible_devices>}"
CUDA_DEVICES="${2:?usage: $0 <model_id> <cuda_visible_devices>}"
shift 2

OUT_DIR="${OUT_DIR:-results/rebuttal/cot_extended}"
SUITES=(${SUITES:-external rag history})
STRATEGIES=(${STRATEGIES:-baseline cot})

SLUG="${MODEL_ID//\//_}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLUG}.log"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

echo "[$(date -Iseconds)] B1 starting $MODEL_ID on GPU(s) $CUDA_DEVICES" | tee -a "$LOG"
echo "  suites:     ${SUITES[*]}" | tee -a "$LOG"
echo "  strategies: ${STRATEGIES[*]}" | tee -a "$LOG"
echo "  out_dir:    $OUT_DIR" | tee -a "$LOG"
echo "  log:        $LOG"

bash scripts/run_with_env.sh \
    python -m anchorbench.runners.rebuttal_cot \
        --model_id "$MODEL_ID" \
        --backend vllm \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --suites "${SUITES[@]}" \
        --strategies "${STRATEGIES[@]}" \
        --out_dir "$OUT_DIR" \
        --batch_size 32 \
        --max_tokens 768 "$@" 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] B1 done $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
