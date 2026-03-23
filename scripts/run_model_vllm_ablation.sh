#!/usr/bin/env bash
# Run ablation suites (external, icl, rag, history) for a single model via vLLM.
set -euo pipefail

MODEL_ID="$1"
GPU="$2"
MAX_TOKENS="${3:-512}"
GPU_MEM_UTIL="${4:-0.92}"
MAX_MODEL_LEN="${5:-4096}"
TP_SIZE="${6:-1}"

VIEWS_FILE="${ANCHORBENCH_VIEWS:-promptviews_ablation.jsonl}"
BATCH_SIZE="${ANCHORBENCH_BATCH_SIZE:-64}"
RESULTS_ROOT="${ANCHORBENCH_RESULTS_ROOT:-results/ablation_full_benchmark}"

export CUDA_VISIBLE_DEVICES="$GPU"
MODEL_SLUG="${MODEL_ID//\//_}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

VLLM_COMMON="--backend vllm --tensor_parallel_size $TP_SIZE \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --max_model_len $MAX_MODEL_LEN \
    --max_tokens $MAX_TOKENS"

expected_lines() {
    local suite="$1"
    local file="datasets/anchorbench_${suite}_core/${VIEWS_FILE}"
    if [ -f "$file" ]; then
        wc -l < "$file"
    else
        echo 0
    fi
}

run_suite() {
    local suite="$1"
    local data_dir="datasets/anchorbench_${suite}_core"
    local runner="scripts/eval/run_ablation_suite.py"
    local out_dir
    local extra_args=()

    case "$suite" in
        external) out_dir="$RESULTS_ROOT/external/${MODEL_SLUG}" ;;
        icl) out_dir="$RESULTS_ROOT/icl/${MODEL_SLUG}"; extra_args+=(--batch_size "$BATCH_SIZE") ;;
        rag) out_dir="$RESULTS_ROOT/rag/${MODEL_SLUG}"; extra_args+=(--batch_size "$BATCH_SIZE") ;;
        history) out_dir="$RESULTS_ROOT/history/${MODEL_SLUG}"; extra_args+=(--batch_size "$BATCH_SIZE") ;;
        *) echo "Unknown suite: $suite"; return 1 ;;
    esac

    mkdir -p "$out_dir"

    local results_file="$out_dir/results.jsonl"

    local expected
    expected=$(expected_lines "$suite")

    if [ -z "${ANCHORBENCH_FORCE_RERUN:-}" ] && [ -f "$results_file" ] && [ "$expected" -gt 0 ]; then
        local n_lines
        n_lines=$(wc -l < "$results_file")
        if [ "$n_lines" -ge "$expected" ]; then
            echo "[$(date +%H:%M:%S)] SKIP $suite — already complete ($n_lines/$expected lines)"
            return 0
        fi
    fi

    echo "[$(date +%H:%M:%S)] ▶ ablation $suite | $MODEL_ID | GPU $GPU | views=$VIEWS_FILE"
    bash scripts/run_with_env.sh python "$runner" \
        --suite "$suite" \
        --promptviews "$data_dir/$VIEWS_FILE" \
        --itemspecs "$data_dir/itemspecs.jsonl" \
        --model_id "$MODEL_ID" \
        --out_dir "$out_dir" \
        $VLLM_COMMON \
        "${extra_args[@]}"
    echo "[$(date +%H:%M:%S)] ✓ ablation $suite DONE"
}

for suite in external icl rag history; do
    run_suite "$suite"
done

echo "ALL ABLATION SUITES COMPLETE: $MODEL_ID"
