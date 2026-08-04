#!/usr/bin/env bash
# Smoke test 4 LARGE frontier API models across all 5 suites.
# - Uses --max_items <N> (default 10) to bound cost / wall time.
# - Captures token usage from each summary.json and prints a total + extrapolation.
# - Output layout matches the rest of the pipeline so analyzers pick it up.
#
# Usage:
#   bash scripts/rebuttal/api_smoke_large.sh [n_items]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

N_ITEMS="${1:-10}"
OUT_BASE="${OUT_BASE:-results/rebuttal/large_api_smoke}"
LOG_DIR="$OUT_BASE/_logs"
mkdir -p "$LOG_DIR"

MODELS=(
    "openai/gpt-5.4"
    "anthropic/claude-sonnet-4.6"
    "google/gemini-2.5-pro"
    "x-ai/grok-4.3"
)
SUITES=(external rag history icl tool)

for MODEL_ID in "${MODELS[@]}"; do
    SLUG="${MODEL_ID//\//_}"
    LOG="$LOG_DIR/${SLUG}.log"
    echo "[$(date -Iseconds)] SMOKE start: $MODEL_ID (n_items=$N_ITEMS)" | tee -a "$LOG"
    bash scripts/run_with_env.sh \
        python -m anchorbench.runners.api \
            --model_id "$MODEL_ID" \
            --suites "${SUITES[@]}" \
            --out_dir "$OUT_BASE" \
            --max_items "$N_ITEMS" \
            --max_concurrent 8 \
            --max_tokens 1024 2>&1 | tee -a "$LOG"
    echo "[$(date -Iseconds)] SMOKE done: $MODEL_ID" | tee -a "$LOG"
done

echo "[$(date -Iseconds)] All smoke runs complete; computing cost estimate..."
PYTHONPATH=src bash scripts/run_with_env.sh \
    python -m anchorbench.analysis.api_cost_estimate \
        --in_dir "$OUT_BASE" \
        --n_items_smoke "$N_ITEMS" \
        --n_items_full 360 2>&1 | tee "$OUT_BASE/cost_estimate.txt"
echo "Wrote $OUT_BASE/cost_estimate.txt"
