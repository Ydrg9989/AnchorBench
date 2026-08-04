#!/usr/bin/env bash
# Full-panel run for ONE large frontier API model across all 5 suites.
# Usage: bash scripts/rebuttal/_large_api_full.sh <model_id>
#
# Output: results/rebuttal/large_api/<suite>/<slug>/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id>}"
OUT_BASE="${OUT_BASE:-results/rebuttal/large_api}"
LOG_DIR="$OUT_BASE/_logs"
mkdir -p "$LOG_DIR"
SLUG="${MODEL_ID//\//_}"
LOG="$LOG_DIR/${SLUG}.log"

# Concurrency tuned to model: lower for reasoning models that block on
# rate limits, higher for fast non-reasoning ones.
case "$MODEL_ID" in
    *gpt-5.4)               CONC=16 ;;
    *claude-sonnet-4.6)     CONC=8  ;;   # Anthropic is rate-limited hardest
    *grok-4.3)              CONC=12 ;;
    *)                      CONC=8  ;;
esac

echo "[$(date -Iseconds)] LARGE_API start: $MODEL_ID (max_concurrent=$CONC)" | tee -a "$LOG"

bash scripts/run_with_env.sh \
    python -m anchorbench.runners.api \
        --model_id "$MODEL_ID" \
        --suites external rag history icl tool \
        --out_dir "$OUT_BASE" \
        --max_concurrent "$CONC" \
        --max_tokens 1024 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] LARGE_API done: $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
