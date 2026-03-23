#!/bin/bash
# ============================================================================
# AnchorBench Smoke Test — vLLM Backend (tmux orchestration)
#
# Runs 10% of each suite on 5 open-source models via vLLM.
# Models run sequentially per GPU to avoid OOM; 2 GPUs run in parallel.
#
# Usage:
#   bash scripts/smoke_test_vllm.sh
# ============================================================================

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

MAX_ITEMS=36
MAX_TOKENS=512
SEED=42
GPU_MEM=0.90
MAX_MODEL_LEN=4096

RESULTS_ROOT="results/smoke_test"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$LOG_DIR"

SUITES=(external icl rag tool history)

run_one_suite() {
    local model_id="$1"
    local suite="$2"
    local gpu="$3"

    local model_slug="${model_id//\//_}"
    local data_dir="datasets/anchorbench_${suite}_core"
    local out_dir="$RESULTS_ROOT/${suite}/${model_slug}"
    local log_file="$LOG_DIR/${suite}_${model_slug}_vllm.log"

    mkdir -p "$out_dir"

    local runner="scripts/eval/run_${suite}.py"
    local args=(
        --promptviews "$data_dir/promptviews.jsonl"
        --itemspecs "$data_dir/itemspecs.jsonl"
        --model_id "$model_id"
        --max_items "$MAX_ITEMS"
        --max_tokens "$MAX_TOKENS"
        --seed "$SEED"
        --backend vllm
        --tensor_parallel_size 1
        --gpu_memory_utilization "$GPU_MEM"
        --max_model_len "$MAX_MODEL_LEN"
        --out_dir "$out_dir"
    )

    if [ "$suite" = "history" ]; then
        args+=(--request_final_line)
    fi

    echo "[$(date +%H:%M:%S)] GPU $gpu | $suite | $model_id"
    CUDA_VISIBLE_DEVICES="$gpu" python "$runner" "${args[@]}" > "$log_file" 2>&1 || {
        echo "[$(date +%H:%M:%S)] FAILED: $suite | $model_id (see $log_file)"
        return 1
    }
    echo "[$(date +%H:%M:%S)] DONE:  $suite | $model_id"
}

run_model_all_suites() {
    local model_id="$1"
    local gpu="$2"

    echo ""
    echo "====== GPU $gpu | $model_id ======"
    for suite in "${SUITES[@]}"; do
        run_one_suite "$model_id" "$suite" "$gpu" || true
    done
    echo "====== GPU $gpu | FINISHED: $model_id ======"
}

run_gpu_queue() {
    local gpu="$1"
    shift
    for model_id in "$@"; do
        run_model_all_suites "$model_id" "$gpu"
    done
}

echo "============================================"
echo "  AnchorBench Smoke Test — vLLM Backend"
echo "  Items per suite: $MAX_ITEMS (10%)"
echo "  Max tokens: $MAX_TOKENS"
echo "  GPU mem util: $GPU_MEM"
echo "============================================"

GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"

# Queue assignment: balance by total model size
# GPU A: Gemma-3 1B → Llama 8B → OLMo 13B  (~22B total)
# GPU B: Gemma-3 4B → Qwen 7B               (~11B total, but 2 models only)
run_gpu_queue "$GPU_A" \
    "google/gemma-3-1b-it" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "allenai/OLMo-2-1124-13B-Instruct" &
PID_A=$!

run_gpu_queue "$GPU_B" \
    "google/gemma-3-4b-it" \
    "Qwen/Qwen2.5-7B-Instruct" &
PID_B=$!

wait $PID_A || echo "WARNING: GPU $GPU_A queue exited non-zero"
wait $PID_B || echo "WARNING: GPU $GPU_B queue exited non-zero"

echo ""
echo "============================================"
echo "  All vLLM smoke tests complete!"
echo "  Results: $RESULTS_ROOT/"
echo "  Logs:    $LOG_DIR/"
echo "============================================"
