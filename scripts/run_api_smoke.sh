#!/usr/bin/env bash
# Smoke test: run all 4 API models on 10 items per suite.
# Estimated cost: ~$0.39 total, ~2 minutes wall time.
set -euo pipefail
cd "$(dirname "$0")/.."

# Load API key
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export PYTHONPATH=src
OUT_DIR="results/api_smoke"
MAX_ITEMS=10
MAX_CONCURRENT=30
SUITES="external icl rag tool history"

MODELS=(
    "google/gemini-2.5-flash-lite"
    "x-ai/grok-3-mini-beta"
    "openai/gpt-5.4-mini"
    "anthropic/claude-sonnet-4.5"
)

echo "============================================"
echo "AnchorBench API Smoke Test"
echo "  Items per suite: $MAX_ITEMS"
echo "  Suites: $SUITES"
echo "  Models: ${#MODELS[@]}"
echo "  Output: $OUT_DIR"
echo "============================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> Running: $model"
    python scripts/eval/run_api_benchmark.py \
        --model_id "$model" \
        --suites $SUITES \
        --out_dir "$OUT_DIR" \
        --max_items "$MAX_ITEMS" \
        --max_concurrent "$MAX_CONCURRENT" \
        --max_tokens 512 \
        --seed 42
    echo "<<< Done: $model"
done

echo ""
echo "============================================"
echo "All smoke tests complete. Results in $OUT_DIR/"
echo "============================================"
