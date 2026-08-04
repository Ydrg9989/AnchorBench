#!/usr/bin/env bash
# P4 (task-specification ablation) launcher — runs the rebuttal_cot runner
# with the new {rule, judgment} strategies on External-only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu>}"
DEVICES="${2:?usage: $0 <model_id> <gpu>}"
OUT_DIR="${OUT_DIR:-results/rebuttal/task_spec}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
SLUG="${MODEL_ID//\//_}"
LOG="$LOG_DIR/${SLUG}.log"

echo "[$(date -Iseconds)] P4[task_spec] $MODEL_ID on GPU $DEVICES" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES="$DEVICES"
bash scripts/run_with_env.sh \
    python -m anchorbench.runners.rebuttal_cot \
        --model_id "$MODEL_ID" \
        --backend vllm \
        --suites external \
        --strategies baseline rule judgment \
        --out_dir "$OUT_DIR" \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --batch_size 32 \
        --max_tokens 512 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] P4[task_spec] done $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
