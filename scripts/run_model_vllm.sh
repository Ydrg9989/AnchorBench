#!/usr/bin/env bash
# Run all 5 AnchorBench suites for a single model using vLLM backend.
#
# Each suite loads the vLLM engine, runs inference on all core items,
# writes results, and then shuts down before the next suite starts.
#
# Usage:
#   bash scripts/run_model_vllm.sh <MODEL_ID> <GPU_ID> [MAX_TOKENS] [GPU_MEM_UTIL] [MAX_MODEL_LEN] [TP_SIZE]
#
# Optional env (for core-views run and max GPU util):
#   ANCHORBENCH_VIEWS=promptviews_core.jsonl  use core views only (1800/suite)
#   ANCHORBENCH_BATCH_SIZE=64                 batch size for icl/rag/tool (default 32)
#   ANCHORBENCH_FORCE_RERUN=1                 do not skip suites; rerun even if results exist
#
# Example:
#   bash scripts/run_model_vllm.sh meta-llama/Llama-3.1-8B-Instruct 0
#   ANCHORBENCH_VIEWS=promptviews_core.jsonl bash scripts/run_model_vllm.sh meta-llama/Llama-3.1-8B-Instruct 0 512 0.95
set -euo pipefail

MODEL_ID="$1"
GPU="$2"
MAX_TOKENS="${3:-512}"
GPU_MEM_UTIL="${4:-0.92}"
MAX_MODEL_LEN="${5:-4096}"
TP_SIZE="${6:-1}"

# Core-views only and batch size (optional)
VIEWS_FILE="${ANCHORBENCH_VIEWS:-promptviews.jsonl}"
BATCH_SIZE="${ANCHORBENCH_BATCH_SIZE:-32}"

export CUDA_VISIBLE_DEVICES="$GPU"
MODEL_SLUG="${MODEL_ID//\//_}"
RESULTS_ROOT="results/full_benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================================================"
echo " Model:     $MODEL_ID"
echo " GPU:       $GPU (TP=$TP_SIZE)"
echo " Tokens:    $MAX_TOKENS"
echo " GPU Mem:   $GPU_MEM_UTIL"
echo " Model Len: $MAX_MODEL_LEN"
echo " Output:    $RESULTS_ROOT/*/${MODEL_SLUG}/"
echo " Views:     $VIEWS_FILE"
echo " Started:   $TIMESTAMP"
echo "================================================================"

VLLM_COMMON="--backend vllm --tensor_parallel_size $TP_SIZE \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --max_model_len $MAX_MODEL_LEN \
    --max_tokens $MAX_TOKENS --seed 42"

run_suite() {
    local suite="$1"
    local data_dir="datasets/anchorbench_${suite}_core"
    local runner="scripts/eval/run_${suite}.py"
    local out_dir
    local extra_args=()

    case "$suite" in
        external)
            out_dir="$RESULTS_ROOT/external/${MODEL_SLUG}"
            ;;
        icl)
            out_dir="$RESULTS_ROOT/icl"
            extra_args+=(--batch_size "$BATCH_SIZE")
            ;;
        rag)
            out_dir="$RESULTS_ROOT/rag/${MODEL_SLUG}"
            extra_args+=(--batch_size "$BATCH_SIZE")
            ;;
        tool)
            out_dir="$RESULTS_ROOT/tool"
            extra_args+=(--batch_size "$BATCH_SIZE")
            ;;
        history)
            out_dir="$RESULTS_ROOT/history/${MODEL_SLUG}"
            extra_args+=(--request_final_line)
            ;;
    esac

    mkdir -p "$out_dir"

    local results_file
    case "$suite" in
        icl|tool)
            results_file="$out_dir/${MODEL_SLUG}/results.jsonl"
            ;;
        *)
            results_file="$out_dir/results.jsonl"
            ;;
    esac

    if [ -z "${ANCHORBENCH_FORCE_RERUN:-}" ]; then
        if [ -f "$results_file" ]; then
            local n_lines
            n_lines=$(wc -l < "$results_file")
            if [ "$n_lines" -ge 1800 ]; then
                echo "[$(date +%H:%M:%S)] SKIP $suite — already complete ($n_lines lines)"
                return 0
            fi
        fi
    fi

    echo ""
    echo "[$(date +%H:%M:%S)] ▶ $suite | $MODEL_ID | GPU $GPU"
    echo "  Data:    $data_dir"
    echo "  Output:  $out_dir"

    local start_ts
    start_ts=$(date +%s)

    bash scripts/run_with_env.sh python "$runner" \
        --promptviews "$data_dir/$VIEWS_FILE" \
        --itemspecs "$data_dir/itemspecs.jsonl" \
        --model_id "$MODEL_ID" \
        --out_dir "$out_dir" \
        $VLLM_COMMON \
        "${extra_args[@]}" 2>&1

    local end_ts elapsed_min
    end_ts=$(date +%s)
    elapsed_min=$(( (end_ts - start_ts) / 60 ))

    echo "[$(date +%H:%M:%S)] ✓ $suite DONE (${elapsed_min}m)"
}

TOTAL_START=$(date +%s)
FAILED_SUITES=()

for suite in external icl rag tool history; do
    if ! run_suite "$suite"; then
        echo "[$(date +%H:%M:%S)] ✗ $suite FAILED for $MODEL_ID"
        FAILED_SUITES+=("$suite")
    fi
done

TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo ""
echo "================================================================"
echo " ALL SUITES COMPLETE: $MODEL_ID"
echo " Total time: ${TOTAL_MIN} minutes"
echo " Results:    $RESULTS_ROOT/*/${MODEL_SLUG}/"
if [ ${#FAILED_SUITES[@]} -gt 0 ]; then
    echo " FAILED:     ${FAILED_SUITES[*]}"
fi
echo "================================================================"
