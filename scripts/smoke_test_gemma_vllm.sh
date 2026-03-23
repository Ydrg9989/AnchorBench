#!/bin/bash
# ============================================================================
# AnchorBench Smoke Test — Gemma 3 models (vLLM + tmux)
#
# Runs 36 items (10%) per suite on Gemma 3 1B and Gemma 3 4B.
# Requires HF access to google/gemma-3-* (huggingface-cli login).
#
# Usage:
#   bash scripts/smoke_test_gemma_vllm.sh
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

echo "============================================"
echo "  Gemma 3 Smoke Test — vLLM Backend"
echo "  Models: gemma-3-1b-it, gemma-3-4b-it"
echo "  Items per suite: $MAX_ITEMS"
echo "============================================"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

# Run both Gemma models in parallel on two GPUs
run_model_all_suites "google/gemma-3-1b-it" "$GPU_A" &
PID_A=$!
run_model_all_suites "google/gemma-3-4b-it" "$GPU_B" &
PID_B=$!

wait $PID_A || echo "WARNING: GPU $GPU_A queue exited non-zero"
wait $PID_B || echo "WARNING: GPU $GPU_B queue exited non-zero"

echo ""
echo "============================================"
echo "  Gemma smoke tests complete!"
echo "  Results: $RESULTS_ROOT/"
echo "  Logs:    $LOG_DIR/"
echo "============================================"
