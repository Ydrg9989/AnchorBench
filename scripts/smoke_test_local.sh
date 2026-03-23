#!/bin/bash
# ============================================================================
# AnchorBench Smoke Test — Local HF Models
#
# Runs 10% of each suite (36 items = 180 prompts/suite) on local HF models.
#
# Phase 1: 4 small models run in parallel (1 GPU each)
# Phase 2: 70B model runs on 2 GPUs
#
# Usage:
#   PYTHONPATH=src bash scripts/smoke_test_local.sh
# ============================================================================

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

MAX_ITEMS=36
MAX_TOKENS=512
SEED=42
BATCH_SIZE=16

RESULTS_ROOT="results/smoke_test"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$LOG_DIR"

run_model_all_suites() {
    local model_id="$1"
    local gpu="$2"

    export CUDA_VISIBLE_DEVICES="$gpu"
    local model_slug="${model_id//\//_}"

    echo "[$(date +%H:%M:%S)] GPU $gpu | Starting $model_id"

    for suite in external icl rag tool history; do
        local out_dir="$RESULTS_ROOT/${suite}/${model_slug}"
        local data_dir="datasets/anchorbench_${suite}_core"
        local log_file="$LOG_DIR/${suite}_${model_slug}.log"

        mkdir -p "$out_dir"

        local runner="scripts/eval/run_${suite}.py"
        local args=(
            --promptviews "$data_dir/promptviews.jsonl"
            --itemspecs "$data_dir/itemspecs.jsonl"
            --model_id "$model_id"
            --max_items "$MAX_ITEMS"
            --max_tokens "$MAX_TOKENS"
            --seed "$SEED"
            --device "cuda:0"
            --llm_fallback
        )

        case "$suite" in
            external|history)
                args+=(--out_dir "$out_dir")
                if [ "$suite" = "history" ]; then
                    args+=(--request_final_line)
                fi
                ;;
            icl|tool)
                args+=(--out_dir "$RESULTS_ROOT/${suite}" --batch_size "$BATCH_SIZE")
                ;;
            rag)
                args+=(--out_dir "$out_dir" --batch_size "$BATCH_SIZE")
                ;;
        esac

        echo "[$(date +%H:%M:%S)] GPU $gpu | $suite | $model_id"
        python "$runner" "${args[@]}" > "$log_file" 2>&1 || {
            echo "[$(date +%H:%M:%S)] GPU $gpu | FAILED: $suite | $model_id (see $log_file)"
        }
    done

    echo "[$(date +%H:%M:%S)] GPU $gpu | Done: $model_id"
    unset CUDA_VISIBLE_DEVICES
}

echo "============================================"
echo "  AnchorBench Smoke Test — Local HF Models"
echo "  Items per suite: $MAX_ITEMS (10% of 360)"
echo "  Max tokens: $MAX_TOKENS"
echo "============================================"

# Phase 1: 4 small models, 1 per GPU
echo ""
echo "--- Phase 1: Small models (parallel, 1 GPU each) ---"

run_model_all_suites "meta-llama/Llama-3.2-1B-Instruct"  0 &
PID0=$!
run_model_all_suites "meta-llama/Llama-3.2-3B-Instruct"  1 &
PID1=$!
run_model_all_suites "Qwen/Qwen2.5-1.5B-Instruct"       2 &
PID2=$!
run_model_all_suites "Qwen/Qwen2.5-3B-Instruct"          3 &
PID3=$!

for pid in $PID0 $PID1 $PID2 $PID3; do
    wait "$pid" || echo "WARNING: process $pid exited non-zero"
done
echo "--- Phase 1 complete ---"

# Phase 2: 70B model, 2 GPUs
echo ""
echo "--- Phase 2: 70B model (multi-GPU) ---"
export CUDA_VISIBLE_DEVICES="0,1"

MODEL70B="meta-llama/Llama-3.1-70B-Instruct"
MODEL70B_SLUG="${MODEL70B//\//_}"

for suite in external icl rag tool history; do
    out_dir="$RESULTS_ROOT/${suite}/${MODEL70B_SLUG}"
    data_dir="datasets/anchorbench_${suite}_core"
    log_file="$LOG_DIR/${suite}_${MODEL70B_SLUG}.log"

    mkdir -p "$out_dir"

    runner="scripts/eval/run_${suite}.py"
    args=(
        --promptviews "$data_dir/promptviews.jsonl"
        --itemspecs "$data_dir/itemspecs.jsonl"
        --model_id "$MODEL70B"
        --max_items "$MAX_ITEMS"
        --max_tokens "$MAX_TOKENS"
        --seed "$SEED"
        --device "auto"
        --llm_fallback
    )

    case "$suite" in
        external|history)
            args+=(--out_dir "$out_dir")
            if [ "$suite" = "history" ]; then
                args+=(--request_final_line)
            fi
            ;;
        icl|tool)
            args+=(--out_dir "$RESULTS_ROOT/${suite}" --batch_size 4)
            ;;
        rag)
            args+=(--out_dir "$out_dir" --batch_size 4)
            ;;
    esac

    echo "[$(date +%H:%M:%S)] Starting $suite | $MODEL70B"
    python "$runner" "${args[@]}" > "$log_file" 2>&1 || {
        echo "[$(date +%H:%M:%S)] FAILED: $suite | $MODEL70B (see $log_file)"
    }
    echo "[$(date +%H:%M:%S)] Done: $suite | $MODEL70B"
done

unset CUDA_VISIBLE_DEVICES
echo "--- Phase 2 complete ---"

echo ""
echo "============================================"
echo "  All local smoke tests complete!"
echo "  Results: $RESULTS_ROOT/"
echo "  Logs:    $LOG_DIR/"
echo "============================================"
