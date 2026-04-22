#!/usr/bin/env bash
# Run AnchorBench API benchmark for all 4 API models across 5 suites.
#
# Requires: OPENROUTER_API_KEY set in environment or .env
#
# Usage:
#   bash scripts/run_api_benchmark_all.sh
#   bash scripts/run_api_benchmark_all.sh 2>&1 | tee logs/api_benchmark_$(date +%Y%m%d).log
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="results/api_benchmark"
MAX_CONCURRENT="${API_MAX_CONCURRENT:-50}"
SUITES="external icl rag tool history"

API_MODELS=(
  "openai/gpt-5.4-mini"
  "anthropic/claude-haiku-4.5"
  "google/gemini-2.5-flash"
  "x-ai/grok-3-mini-beta"
)

# Load API key from .env if not already set
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
  export "$(grep -E '^OPENROUTER_API_KEY=' .env | head -1)"
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set. Export it or add to .env"
  exit 1
fi

echo "=========================================="
echo "AnchorBench API Benchmark"
echo "Models: ${#API_MODELS[@]}"
echo "Suites: $SUITES"
echo "Output: $OUT_DIR"
echo "=========================================="
echo ""

mkdir -p "$OUT_DIR"

for model in "${API_MODELS[@]}"; do
  model_slug="${model//\//_}"
  echo "[$(date +%H:%M:%S)] Running $model..."

  PYTHONPATH=src python scripts/eval/run_api_benchmark.py \
    --model_id "$model" \
    --suites $SUITES \
    --out_dir "$OUT_DIR" \
    --max_concurrent "$MAX_CONCURRENT" \
    2>&1 | tail -30

  echo "[$(date +%H:%M:%S)] Done: $model"
  echo ""
done

echo "=========================================="
echo "Recomputing unified metrics..."
PYTHONPATH=src python scripts/eval/recompute_all_unified.py \
  --results_dir "$OUT_DIR"

echo ""
echo "API benchmark complete!"
echo "Results in: $OUT_DIR/"
echo "=========================================="
