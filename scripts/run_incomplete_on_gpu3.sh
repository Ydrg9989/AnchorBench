#!/usr/bin/env bash
# Rerun only incomplete (missing or <1800 lines) suite×model combinations on GPU 3.
# Uses same env as full benchmark (core views, GPU_MEM_UTIL=0.70).
#
# Usage:
#   cd $REPO_ROOT && conda activate LLM_anchoring && bash scripts/run_incomplete_on_gpu3.sh
#
# Logs: logs/incomplete_gpu3_<timestamp>.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GPU=3
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
LOGFILE="${LOGDIR}/incomplete_gpu3_${TIMESTAMP}.log"
mkdir -p "$LOGDIR"

export ANCHORBENCH_VIEWS=promptviews_core.jsonl
export ANCHORBENCH_BATCH_SIZE=64
# GPU 3 dedicated – safe to use 0.75 so 32B fits (69 GiB free observed)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
GPU_MEM_UTIL_32B="${GPU_MEM_UTIL_32B:-0.75}"

echo "================================================================"
echo " Rerun incomplete experiments on GPU $GPU"
echo " Log: $LOGFILE"
echo " GPU mem: $GPU_MEM_UTIL (32B: $GPU_MEM_UTIL_32B)"
echo "================================================================"
echo ""

run_model() {
    local model_id="$1"
    local gpu_mem="$2"
    echo "[$(date +%H:%M:%S)] >>> $model_id (GPU mem $gpu_mem)"
    if GPU_MEM_UTIL=$gpu_mem bash scripts/run_model_vllm.sh "$model_id" "$GPU" 512 "$gpu_mem" 4096 1 >> "$LOGFILE" 2>&1; then
        echo "[$(date +%H:%M:%S)] ✓ $model_id done"
    else
        echo "[$(date +%H:%M:%S)] ✗ $model_id failed (see $LOGFILE)"
        return 1
    fi
}

# Models with one or more incomplete suites (run_model_vllm skips complete ones)
# Order: smaller models first so 32B runs when GPU is warm/empty
run_model "google/gemma-3-4b-it" "$GPU_MEM_UTIL" || true
run_model "meta-llama/Llama-3.2-3B-Instruct" "$GPU_MEM_UTIL" || true
run_model "Qwen/Qwen2.5-3B-Instruct" "$GPU_MEM_UTIL" || true
run_model "allenai/OLMo-2-1124-13B-Instruct" "$GPU_MEM_UTIL" || true
run_model "google/gemma-3-1b-it" "$GPU_MEM_UTIL" || true
run_model "allenai/OLMo-2-0325-32B-Instruct" "$GPU_MEM_UTIL_32B" || true

echo ""
echo "================================================================"
echo " Incomplete rerun finished. Log: $LOGFILE"
echo " Recompute unified metrics:"
echo "   bash scripts/run_with_env.sh python scripts/eval/recompute_all_unified.py --results_dir results/full_benchmark"
echo "================================================================"
