#!/usr/bin/env bash
# P3 (Tool realism) launcher for a single open-weight model.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu>}"
DEVICES="${2:?usage: $0 <model_id> <gpu>}"
OUT_DIR="${OUT_DIR:-results/rebuttal/tool_realism}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
SLUG="${MODEL_ID//\//_}"
LOG="$LOG_DIR/${SLUG}.log"

echo "[$(date -Iseconds)] P3[tool_realism] starting $MODEL_ID on GPU $DEVICES" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES="$DEVICES"
bash scripts/run_with_env.sh \
    python -m anchorbench.runners.rebuttal_tool_realism \
        --model_id "$MODEL_ID" \
        --out_dir "$OUT_DIR" \
        --backend vllm \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --batch_size 32 \
        --max_tokens 512 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] P3[tool_realism] done $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
