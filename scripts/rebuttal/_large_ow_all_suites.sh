#!/usr/bin/env bash
# Run a 70B-class open-weight model on every suite (External, RAG, History,
# ICL, Tool) using tensor parallelism on a pair of GPUs.
#
# Usage:
#   bash scripts/rebuttal/_large_ow_all_suites.sh <model_id> <devices>
#     devices: comma-separated GPU indices, e.g. "0,1" for TP=2
#
# Output layout (matches the published benchmark layout, so downstream
# analyzers and tables pick it up automatically):
#   results/rebuttal/large_ow/<suite>/<model_slug>/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <devices>}"
DEVICES="${2:?usage: $0 <model_id> <devices>}"

# Tensor-parallel size = number of GPUs in DEVICES
IFS=',' read -ra DEV_ARR <<< "$DEVICES"
TP_SIZE=${#DEV_ARR[@]}

OUT_BASE="${OUT_BASE:-results/rebuttal/large_ow}"
LOG_DIR="$OUT_BASE/_logs"
mkdir -p "$LOG_DIR"
SLUG="${MODEL_ID//\//_}"
LOG="$LOG_DIR/${SLUG}.log"

# vLLM long-context + KV cache settings tuned for 70B BF16 on H100 96GB pair:
#   - max_model_len 8k is enough for every AnchorBench suite (longest = ICL/RAG ~3k input)
#   - gpu_memory_utilization 0.92 squeezes max KV cache room out of 96GB cards
#   - batch_size 16 keeps memory headroom for two-stage History runs
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.92}"
BATCH="${BATCH:-16}"
MAX_TOK="${MAX_TOK:-512}"

echo "[$(date -Iseconds)] LARGE_OW start: $MODEL_ID on GPUs $DEVICES (TP=$TP_SIZE)" | tee -a "$LOG"
echo "  max_model_len=$MAX_LEN  gpu_util=$GPU_UTIL  batch=$BATCH  max_tok=$MAX_TOK" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES="$DEVICES"

# Common vLLM args
COMMON_OW_ARGS=(
    --model_id "$MODEL_ID"
    --backend vllm
    --tensor_parallel_size "$TP_SIZE"
    --gpu_memory_utilization "$GPU_UTIL"
    --max_model_len "$MAX_LEN"
    --batch_size "$BATCH"
    --max_tokens "$MAX_TOK"
)

run_suite() {
    local suite="$1"
    local runner="$2"
    local pv="$3"
    local sp="$4"
    local out_dir="$OUT_BASE/$suite"
    local target="$out_dir/$SLUG/results.jsonl"
    if [ -f "$target" ]; then
        echo "[$(date -Iseconds)] SKIP $suite (already have $target)" | tee -a "$LOG"
        return 0
    fi
    echo "[$(date -Iseconds)] >>> $suite for $MODEL_ID" | tee -a "$LOG"
    local extra=()
    if [ "$suite" = "history" ]; then
        extra+=(--baseline_condition control_twostage)
    fi
    bash scripts/run_with_env.sh \
        python -m "anchorbench.runners.$runner" \
            --promptviews "$pv" \
            --itemspecs "$sp" \
            --out_dir "$out_dir" \
            "${extra[@]}" \
            "${COMMON_OW_ARGS[@]}" 2>&1 | tee -a "$LOG"
}

# External (single-stage)
run_suite external external \
    datasets/anchorbench_external_core/promptviews_core.jsonl \
    datasets/anchorbench_external_core/itemspecs.jsonl

# RAG
run_suite rag rag \
    datasets/anchorbench_rag_core/promptviews_core.jsonl \
    datasets/anchorbench_rag_core/itemspecs.jsonl

# ICL
run_suite icl icl \
    datasets/anchorbench_icl_core/promptviews_core.jsonl \
    datasets/anchorbench_icl_core/itemspecs.jsonl

# Tool (use full promptviews because tool needs the schema metadata)
run_suite tool tool \
    datasets/anchorbench_tool_core/promptviews.jsonl \
    datasets/anchorbench_tool_core/itemspecs.jsonl

# History (two-stage, slow; do last so we can early-stop if needed)
run_suite history history \
    datasets/anchorbench_history_core/promptviews.jsonl \
    datasets/anchorbench_history_core/itemspecs.jsonl

echo "[$(date -Iseconds)] LARGE_OW done: $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
