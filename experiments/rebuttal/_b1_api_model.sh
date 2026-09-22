#!/usr/bin/env bash
# Run the API model under both baseline and CoT, then reshape outputs into
# results/rebuttal/cot_extended/<suite>/<slug>/<strategy>/ for the analyzer.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${API_MODEL:-openai/gpt-5.4-mini}"
OUT_DIR="${OUT_DIR:-results/rebuttal/cot_extended}"
SUITES=(${SUITES:-external rag history})
SLUG="${MODEL_ID//\//_}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLUG}.log"

COT_SUFFIX=$'\n\nThink step by step. List the relevant evidence, compute your estimate from that evidence only, then provide your final numeric answer on the last line.'

echo "[$(date -Iseconds)] B1 starting API $MODEL_ID" | tee -a "$LOG"
echo "  suites:  ${SUITES[*]}" | tee -a "$LOG"
echo "  out_dir: $OUT_DIR" | tee -a "$LOG"

for STRAT in baseline cot; do
    if [ "$STRAT" = "cot" ]; then SUFFIX="$COT_SUFFIX"; else SUFFIX=""; fi
    STAGING="$OUT_DIR/_api_${STRAT}"
    echo "[$(date -Iseconds)] API run: strategy=$STRAT" | tee -a "$LOG"
    bash scripts/run_with_env.sh \
        python -m anchorbench.runners.api \
            --model_id "$MODEL_ID" \
            --suites "${SUITES[@]}" \
            --out_dir "$STAGING" \
            --prompt_suffix "$SUFFIX" \
            --max_concurrent 16 2>&1 | tee -a "$LOG"
    for SUITE in "${SUITES[@]}"; do
        SRC="$STAGING/$SUITE/$SLUG"
        DST="$OUT_DIR/$SUITE/$SLUG/$STRAT"
        mkdir -p "$DST"
        if [ -d "$SRC" ]; then
            cp -f "$SRC/results.jsonl" "$DST/results.jsonl" 2>/dev/null || true
            cp -f "$SRC/summary.json" "$DST/summary.json" 2>/dev/null || true
        fi
    done
done

echo "[$(date -Iseconds)] B1 done API $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
