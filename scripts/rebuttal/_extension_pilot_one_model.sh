#!/usr/bin/env bash
# REVIEWER-2 extension pilot: run the law + consumer pilot datasets for one
# model on one GPU. Mirrors _c1_one_model.sh but uses the *_other_pilot
# datasets.
#
# Usage: bash scripts/rebuttal/_extension_pilot_one_model.sh <model_id> <gpu>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu>}"
DEVICES="${2:?usage: $0 <model_id> <gpu>}"

EXT_DATASET="${EXT_DATASET:-datasets/anchorbench_external_other_pilot}"
HIST_DATASET="${HIST_DATASET:-datasets/anchorbench_history_other_pilot}"
OUT_DIR="${OUT_DIR:-results/rebuttal/extension_pilot}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
SLUG="${MODEL_ID//\//_}"
LOG="$LOG_DIR/${SLUG}.log"

echo "[$(date -Iseconds)] REVIEWER-2 starting $MODEL_ID on $DEVICES" | tee -a "$LOG"
echo "  ext dataset: $EXT_DATASET" | tee -a "$LOG"
echo "  hist dataset: $HIST_DATASET" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES="$DEVICES"
echo "[$(date -Iseconds)] REVIEWER-2/external $MODEL_ID" | tee -a "$LOG"
bash scripts/run_with_env.sh \
    python -m anchorbench.runners.external \
        --promptviews "$EXT_DATASET/promptviews_core.jsonl" \
        --itemspecs "$EXT_DATASET/itemspecs.jsonl" \
        --model_id "$MODEL_ID" \
        --out_dir "$OUT_DIR/external" \
        --backend vllm \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --batch_size 32 \
        --max_tokens 512 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] REVIEWER-2/history $MODEL_ID" | tee -a "$LOG"
bash scripts/run_with_env.sh \
    python -m anchorbench.runners.history \
        --promptviews "$HIST_DATASET/promptviews.jsonl" \
        --itemspecs "$HIST_DATASET/itemspecs.jsonl" \
        --model_id "$MODEL_ID" \
        --out_dir "$OUT_DIR/history" \
        --baseline_condition control_twostage \
        --backend vllm \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --batch_size 1 \
        --max_tokens 512 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] REVIEWER-2 done $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
