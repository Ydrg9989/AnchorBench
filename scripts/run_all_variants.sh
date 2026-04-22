#!/usr/bin/env bash
# Run all AnchorBench variant/ablation experiments.
#
# === DEPRECATED in favour of the unified CLI ===
# This script is kept for backward compatibility. Each variant is now
# also exposed as a named experiment in `configs/benchmark.yaml`:
#
#   python scripts/run.py experiment --name paper_icl_dist
#   python scripts/run.py experiment --name paper_sampling
#   python scripts/run.py experiment --name paper_mitigation_headroom
#
# Set USE_NEW_CLI=1 to delegate to the CLI (runs paper_icl_dist +
# paper_sampling + paper_mitigation_headroom in sequence).
#
# These produce appendix results and should run AFTER the core benchmark
# (run_full_benchmark.sh) is complete.
#
# Variants:
#   1. ICL-dist (OW):        10 models x ICL-dist dataset
#   2. ICL-dist (API):       4 API models x ICL-dist dataset
#   3. Sampling robustness:  selected models, greedy vs T=0.7
#   4. Mitigation baseline:  ignore-anchor suffix, all 5 suites
#   5. Mitigation headroom:  4 strategies (baseline/ignore/self_check/cot)
#
# Usage:
#   bash scripts/run_all_variants.sh                    # legacy bash logic
#   USE_NEW_CLI=1 bash scripts/run_all_variants.sh      # delegate to CLI
#   bash scripts/run_all_variants.sh 2>&1 | tee logs/variants_$(date +%Y%m%d).log
#
set -euo pipefail

if [ "${USE_NEW_CLI:-0}" = "1" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  for recipe in paper_icl_dist paper_sampling paper_mitigation_headroom; do
    echo "[deprecated] python scripts/run.py experiment --name ${recipe}"
    bash "$SCRIPT_DIR/run_with_env.sh" python "$REPO_ROOT/scripts/run.py" \
      experiment --name "$recipe" || exit $?
  done
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source scripts/paper_model_ids.inc.sh

GPU_UTIL=0.90
MAX_MODEL_LEN=4096
BATCH_SIZE=64

SAMPLING_MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

MITIGATION_MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

echo "=========================================="
echo "AnchorBench Variant Experiments"
echo "=========================================="
echo ""

# ── 1. ICL-dist (OW models) ───────────────────────────────────────
echo "=== Variant 1: ICL-dist (10 OW models) ==="
for model in "${PAPER_ANCHORBENCH_MODEL_IDS[@]}"; do
  model_slug="${model//\//_}"
  summary="results/icl_dist_core/icl/${model_slug}/summary.json"
  if [ -f "$summary" ]; then
    echo "  SKIP: $model_slug (summary.json exists)"
    continue
  fi
  echo "  [$(date +%H:%M:%S)] $model..."
  bash scripts/run_with_env.sh python scripts/eval/run_icl.py \
    --model_id "$model" \
    --promptviews datasets/anchorbench_icl_dist_core/promptviews_core.jsonl \
    --itemspecs datasets/anchorbench_icl_dist_core/itemspecs.jsonl \
    --out_dir results/icl_dist_core/icl \
    --backend vllm --batch_size "$BATCH_SIZE" \
    --gpu_memory_utilization "$GPU_UTIL" --max_model_len "$MAX_MODEL_LEN" \
    2>&1 | tail -5
done
echo ""

# ── 2. ICL-dist (API models) ──────────────────────────────────────
echo "=== Variant 2: ICL-dist (4 API models) ==="
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  for model in openai/gpt-5.4-mini anthropic/claude-haiku-4.5 \
               google/gemini-2.5-flash x-ai/grok-3-mini-beta; do
    echo "  [$(date +%H:%M:%S)] $model..."
    PYTHONPATH=src python scripts/eval/run_icl_dist_api.py \
      --model_id "$model" \
      --out_dir results/icl_dist_api \
      2>&1 | tail -5
  done
else
  echo "  SKIP: OPENROUTER_API_KEY not set"
fi
echo ""

# ── 3. Sampling robustness ────────────────────────────────────────
echo "=== Variant 3: Sampling robustness ==="
for model in "${SAMPLING_MODELS[@]}"; do
  model_slug="${model//\//_}"
  summary="results/decoding_sampling_robustness/${model_slug}/summary.json"
  if [ -f "$summary" ]; then
    echo "  SKIP: $model_slug (summary.json exists)"
    continue
  fi
  echo "  [$(date +%H:%M:%S)] $model..."
  bash scripts/run_with_env.sh python scripts/eval/run_sampling_robustness.py \
    --model_id "$model" \
    --backend vllm --batch_size "$BATCH_SIZE" \
    --out_dir results/decoding_sampling_robustness \
    --gpu_memory_utilization "$GPU_UTIL" --max_model_len "$MAX_MODEL_LEN" \
    2>&1 | tail -5
done
echo ""

# ── 4. Mitigation baseline (ignore-anchor) ────────────────────────
echo "=== Variant 4: Mitigation baseline ==="
for model in "${MITIGATION_MODELS[@]}"; do
  echo "  [$(date +%H:%M:%S)] $model..."
  bash scripts/run_with_env.sh python scripts/eval/run_mitigation_baseline.py \
    --model_id "$model" \
    --backend vllm --batch_size "$BATCH_SIZE" \
    --out_dir results/mitigation_ignore_anchor \
    --gpu_memory_utilization "$GPU_UTIL" --max_model_len "$MAX_MODEL_LEN" \
    2>&1 | tail -5
done
echo ""

# ── 5. Mitigation headroom (4 strategies) ─────────────────────────
echo "=== Variant 5: Mitigation headroom ==="
for model in "${MITIGATION_MODELS[@]}"; do
  echo "  [$(date +%H:%M:%S)] $model..."
  bash scripts/run_with_env.sh python scripts/eval/run_mitigation_headroom.py \
    --model_id "$model" \
    --backend vllm --force \
    --gpu_memory_utilization "$GPU_UTIL" --max_model_len "$MAX_MODEL_LEN" \
    2>&1 | tail -5
done
echo ""

echo "=========================================="
echo "All variant experiments complete!"
echo "=========================================="
