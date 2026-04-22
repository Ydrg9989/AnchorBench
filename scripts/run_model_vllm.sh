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

    out_dir="$RESULTS_ROOT/$suite"
    case "$suite" in
        icl|rag|tool)
            extra_args+=(--batch_size "$BATCH_SIZE")
            ;;
        history)
            extra_args+=(--request_final_line)
            ;;
    esac

    mkdir -p "$out_dir"

    local results_file="$out_dir/${MODEL_SLUG}/results.jsonl"

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

    # Post-run validation
    if [ -f "$results_file" ]; then
        local n_lines
        n_lines=$(wc -l < "$results_file")
        if [ "$n_lines" -lt 1 ]; then
            echo "WARNING: $suite results.jsonl is empty (0 lines)"
        else
            echo "  Validation: $n_lines lines in results.jsonl"
        fi

        local expected_prefix
        case "$suite" in
            external) expected_prefix="EXT-" ;;
            icl)      expected_prefix="ICL-" ;;
            rag)      expected_prefix="RAG-" ;;
            tool)     expected_prefix="TOOL-" ;;
            history)  expected_prefix="HIST-" ;;
        esac

        local first_id
        first_id=$(head -1 "$results_file" | python3 -c "import sys,json; print(json.load(sys.stdin).get('item_id',''))" 2>/dev/null || true)
        if [ -n "$first_id" ] && [ -n "$expected_prefix" ]; then
            case "$first_id" in
                "${expected_prefix}"*)
                    echo "  Validation: first item_id='$first_id' matches expected prefix '$expected_prefix'"
                    ;;
                *)
                    echo "WARNING: first item_id='$first_id' does NOT start with expected prefix '$expected_prefix'"
                    ;;
            esac
        fi
    else
        echo "WARNING: $suite results.jsonl not found after run"
    fi
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
