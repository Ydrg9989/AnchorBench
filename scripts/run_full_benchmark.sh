#!/usr/bin/env bash
# Full AnchorBench benchmark re-run on 4 H100 GPUs.
#
# === DEPRECATED in favour of the unified CLI ===
# This script is kept for backward compatibility. The recommended way to
# launch the full paper benchmark is now:
#
#   python scripts/run.py experiment --name paper_main
#
# which reads `model_tiers:` and `experiments.paper_main` from
# `configs/benchmark.yaml`. Setting USE_NEW_CLI=1 will dispatch this
# script to the CLI; otherwise the legacy bash logic below is used.
#
# Runs all 10 open-weight models across 5 suites (External, ICL, RAG, Tool, History).
# GPU allocation:
#   - Small models (1B-4B):  1 GPU each, 4 models in parallel
#   - Medium models (7B-8B): 1 GPU each, up to 4 in parallel
#   - Large models (13B):    1 GPU, run serially
#   - Very large (32B OLMo): 4 GPUs (tensor_parallel_size=4)
#
# Usage:
#   bash scripts/run_full_benchmark.sh                    # legacy bash logic
#   USE_NEW_CLI=1 bash scripts/run_full_benchmark.sh      # delegate to CLI
#   bash scripts/run_full_benchmark.sh 2>&1 | tee logs/full_benchmark_$(date +%Y%m%d).log
#
set -euo pipefail

if [ "${USE_NEW_CLI:-0}" = "1" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  echo "[deprecated] dispatching to: python scripts/run.py experiment --name paper_main"
  exec bash "$SCRIPT_DIR/run_with_env.sh" python "$REPO_ROOT/scripts/run.py" \
    experiment --name paper_main "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source scripts/paper_model_ids.inc.sh

OUT_BASE="results/full_benchmark"
BATCH_SIZE=64
MAX_MODEL_LEN=4096
GPU_UTIL=0.90

SUITES=("external" "icl" "rag" "tool" "history")
SUITE_DATASETS=(
  "datasets/anchorbench_external_core"
  "datasets/anchorbench_icl_core"
  "datasets/anchorbench_rag_core"
  "datasets/anchorbench_tool_core"
  "datasets/anchorbench_history_core"
)

run_model_all_suites() {
  local model_id="$1"
  local gpu_id="$2"
  local tp_size="${3:-1}"
  local model_slug="${model_id//\//_}"

  echo "[$(date +%H:%M:%S)] Starting $model_id on GPU $gpu_id (TP=$tp_size)"

  for i in "${!SUITES[@]}"; do
    local suite="${SUITES[$i]}"
    local data_dir="${SUITE_DATASETS[$i]}"
    local runner="scripts/eval/run_${suite}.py"
    local out_dir="${OUT_BASE}/${suite}"

    if [ ! -f "$runner" ]; then
      echo "  WARNING: $runner not found, skipping $suite"
      continue
    fi

    local summary_file="$out_dir/${model_slug}/summary.json"
    if [ -f "$summary_file" ]; then
      echo "  SKIP: $suite/$model_slug (summary.json exists)"
      continue
    fi

    local extra_args=""
    if [ "$suite" = "history" ]; then
      extra_args="--baseline_condition control"
    fi

    echo "  [$(date +%H:%M:%S)] Running $suite..."
    CUDA_VISIBLE_DEVICES="$gpu_id" bash scripts/run_with_env.sh python "$runner" \
      --model_id "$model_id" \
      --promptviews "${data_dir}/promptviews_core.jsonl" \
      --itemspecs "${data_dir}/itemspecs.jsonl" \
      --out_dir "$out_dir" \
      --backend vllm \
      --batch_size "$BATCH_SIZE" \
      --gpu_memory_utilization "$GPU_UTIL" \
      --max_model_len "$MAX_MODEL_LEN" \
      --tensor_parallel_size "$tp_size" \
      $extra_args \
      2>&1 | tail -20

    local results_file="$out_dir/${model_slug}/results.jsonl"
    if [ -f "$results_file" ]; then
      local count
      count=$(wc -l < "$results_file")
      echo "  OK: $suite/$model_slug -> $count records"
    else
      echo "  WARNING: No results.jsonl produced for $suite/$model_slug"
    fi
  done

  echo "[$(date +%H:%M:%S)] Done: $model_id"
}

echo "=========================================="
echo "AnchorBench Full Benchmark Re-run"
echo "Models: ${#PAPER_ANCHORBENCH_MODEL_IDS[@]}"
echo "Suites: ${SUITES[*]}"
echo "Output: $OUT_BASE"
echo "=========================================="
echo ""

mkdir -p "$OUT_BASE"

# ── Tier 1: Small models (1B-4B) — 4 in parallel, 1 GPU each ──
echo "=== Tier 1: Small models (4 parallel, 1 GPU each) ==="
SMALL_MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "google/gemma-3-1b-it"
  "Qwen/Qwen2.5-3B-Instruct"
)
pids=()
for i in "${!SMALL_MODELS[@]}"; do
  run_model_all_suites "${SMALL_MODELS[$i]}" "$i" 1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo ""

# Second batch of small
echo "=== Tier 1b: More small models ==="
SMALL2_MODELS=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "google/gemma-3-4b-it"
)
pids=()
for i in "${!SMALL2_MODELS[@]}"; do
  run_model_all_suites "${SMALL2_MODELS[$i]}" "$i" 1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo ""

# ── Tier 2: Medium models (7B-8B) — 2 parallel, 1 GPU each ──
echo "=== Tier 2: Medium models (2 parallel, 1 GPU each) ==="
MED_MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)
pids=()
for i in "${!MED_MODELS[@]}"; do
  run_model_all_suites "${MED_MODELS[$i]}" "$i" 1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo ""

# ── Tier 3: Large model (13B) — 1 GPU ──
echo "=== Tier 3: Large models (serial, 1 GPU) ==="
run_model_all_suites "allenai/OLMo-2-1124-13B-Instruct" "0" 1
echo ""

# ── Tier 4: Very large (32B) — 4 GPUs, TP=4 ──
echo "=== Tier 4: 32B model (TP=4, all GPUs) ==="
run_model_all_suites "allenai/OLMo-2-0325-32B-Instruct" "0,1,2,3" 4
echo ""

echo "=========================================="
echo "Full benchmark complete!"
echo "Results in: $OUT_BASE/"
echo "=========================================="
