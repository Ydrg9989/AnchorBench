#!/usr/bin/env bash
# =============================================================================
# AnchorBench smoke test experiment — maximize GPU utilization
#
# Uses **core** benchmark + --max_items (same footprint as old smoke sets).
# Runs N models in parallel (N = number of GPUs), each running all 5 suites
# sequentially on its dedicated GPU. Uses vLLM backend.
#
# Usage:
#   cd $REPO_ROOT && bash scripts/run_smoke_experiment.sh
#
# Optional env:
#   SMOKE_RESULTS_DIR  default: results/smoke_experiment
#   SMOKE_MAX_GPUS     max GPUs to use (default: all available)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

RESULTS_ROOT="${SMOKE_RESULTS_DIR:-results/smoke_experiment}"
LOG_DIR="$RESULTS_ROOT/logs"
MAX_TOKENS=512
SEED=42
# Lower GPU mem so multiple processes fit when other jobs use GPUs (override with SMOKE_GPU_MEM=0.9 if GPUs are free)
GPU_MEM="${SMOKE_GPU_MEM:-0.55}"
MAX_MODEL_LEN=4096

mkdir -p "$LOG_DIR"

# Detect number of GPUs
if command -v nvidia-smi &>/dev/null; then
    NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NGPU=0
fi
if [[ -n "${SMOKE_MAX_GPUS:-}" ]]; then
    NGPU=$(( NGPU < SMOKE_MAX_GPUS ? NGPU : SMOKE_MAX_GPUS ))
fi
if [[ "$NGPU" -lt 1 ]]; then
    echo "ERROR: No GPUs detected. Need at least one GPU for vLLM."
    exit 1
fi

echo "============================================"
echo "  AnchorBench Smoke Experiment"
echo "  Datasets: anchorbench_*_core + max_items (ext/icl/rag/tool: 6, history: 2)"
echo "  Backend: vLLM"
echo "  GPUs: $NGPU (one model per GPU, all suites per model)"
echo "  Results: $RESULTS_ROOT"
echo "============================================"

# Models: one per GPU (small models for fast smoke)
MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
)
NJOBS=$(( NGPU < ${#MODELS[@]} ? NGPU : ${#MODELS[@]} ))

run_one_suite() {
    local gpu="$1"
    local model_id="$2"
    local suite="$3"
    local model_slug="${model_id//\//_}"
    local data_dir="datasets/anchorbench_${suite}_core"
    # ICL and Tool write to out_dir/model_slug; others write to out_dir
    if [[ "$suite" == "icl" || "$suite" == "tool" ]]; then
        local out_dir="$RESULTS_ROOT/${suite}"
    else
        local out_dir="$RESULTS_ROOT/${suite}/${model_slug}"
    fi
    local log_file="$LOG_DIR/${suite}_${model_slug}.log"

    [[ -f "$data_dir/promptviews_core.jsonl" ]] || { echo "Skip $suite: no $data_dir/promptviews_core.jsonl"; return 0; }
    mkdir -p "$(dirname "$out_dir")" "$out_dir"

    local runner="scripts/eval/run_${suite}.py"
    local max_items=6
    [[ "$suite" == "history" ]] && max_items=2
    local args=(
        --promptviews "$data_dir/promptviews_core.jsonl"
        --itemspecs "$data_dir/itemspecs.jsonl"
        --max_items "$max_items"
        --model_id "$model_id"
        --max_tokens "$MAX_TOKENS"
        --seed "$SEED"
        --backend vllm
        --tensor_parallel_size 1
        --gpu_memory_utilization "$GPU_MEM"
        --max_model_len "$MAX_MODEL_LEN"
        --out_dir "$out_dir"
    )

    if [[ "$suite" == "history" ]]; then
        args+=(--request_final_line)
    fi
    if [[ "$suite" == "rag" || "$suite" == "icl" || "$suite" == "tool" ]]; then
        args+=(--batch_size 8)
    fi
    if [[ "$suite" == "tool" ]]; then
        args+=(--tool_plaintext)
    fi
    # Optional LLM fallback for parsing (can skip for smoke to save time)
    # args+=(--llm_fallback)

    echo "[$(date +%H:%M:%S)] GPU$gpu $suite $model_id"
    CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_with_env.sh python "$runner" "${args[@]}" >> "$log_file" 2>&1 || {
        echo "[$(date +%H:%M:%S)] FAILED GPU$gpu $suite $model_id (see $log_file)"
        return 1
    }
    echo "[$(date +%H:%M:%S)] DONE GPU$gpu $suite $model_id"
}

run_all_suites_one_model() {
    local gpu="$1"
    local model_id="$2"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo ""
    echo "====== GPU $gpu | $model_id ======"
    for suite in external history icl rag tool; do
        run_one_suite "$gpu" "$model_id" "$suite" || true
    done
    echo "====== GPU $gpu | FINISHED $model_id ======"
}

# Launch one process per GPU (each runs all suites)
pids=()
for i in $(seq 0 $((NJOBS - 1))); do
    model="${MODELS[$i]}"
    run_all_suites_one_model "$i" "$model" &
    pids+=($!)
done

for i in $(seq 0 $((NJOBS - 1))); do
    pid=${pids[$i]}
    model="${MODELS[$i]}"
    if wait "$pid"; then
        echo "[OK] GPU $i $model"
    else
        echo "[FAIL] GPU $i $model"
    fi
done

echo ""
echo "============================================"
echo "  Smoke experiment runs finished."
echo "  Results: $RESULTS_ROOT/"
echo "  Logs:    $LOG_DIR/"
echo "  Next:    bash scripts/validate_smoke_results.sh $RESULTS_ROOT"
echo "============================================"
