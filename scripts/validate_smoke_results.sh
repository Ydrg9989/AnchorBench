#!/usr/bin/env bash
# =============================================================================
# Validate smoke experiment results
#
# 1. Check each results.jsonl exists and has expected record count
# 2. Run unified metrics (recompute_all_unified.py) on the results dir
# 3. Optionally run parse sensitivity on one suite
#
# Usage:
#   bash scripts/validate_smoke_results.sh [results_dir]
#   Default results_dir: results/smoke_sanity (or pass e.g.
#   results/archive/smoke_runs_20260318/smoke_experiment)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

RESULTS_ROOT="${1:-results/smoke_sanity}"

echo "============================================"
echo "  Validating smoke results: $RESULTS_ROOT"
echo "============================================"

# Expected prompt counts per suite (smoke datasets)
declare -A EXPECTED_PROMPTS
EXPECTED_PROMPTS[external]=30
EXPECTED_PROMPTS[history]=10
EXPECTED_PROMPTS[icl]=30
EXPECTED_PROMPTS[rag]=30
EXPECTED_PROMPTS[tool]=30

FAILED=0

for suite in external history icl rag tool; do
    suite_dir="$RESULTS_ROOT/$suite"
    [[ -d "$suite_dir" ]] || continue
    expected="${EXPECTED_PROMPTS[$suite]:-0}"
    echo ""
    echo "--- $suite (expected $expected records per model) ---"
    for model_dir in "$suite_dir"/*/; do
        [[ -d "$model_dir" ]] || continue
        model_slug=$(basename "$model_dir")
        # ICL/Tool write to out_dir/model_slug/results.jsonl (nested)
        results_file="$model_dir/results.jsonl"
        [[ -f "$results_file" ]] || results_file="$model_dir/$model_slug/results.jsonl"
        if [[ ! -f "$results_file" ]]; then
            echo "  MISSING $model_slug: no results.jsonl"
            FAILED=1
            continue
        fi
        count=$(wc -l < "$results_file")
        if [[ $count -ne $expected ]]; then
            echo "  COUNT MISMATCH $model_slug: got $count, expected $expected"
            FAILED=1
        else
            echo "  OK $model_slug: $count records"
        fi
        # Quick schema check: first record has required fields
        if [[ $count -gt 0 ]]; then
            first=$(head -1 "$results_file")
            if ! echo "$first" | python3 -c "
import json, sys
r = json.load(sys.stdin)
required = ['item_id', 'condition', 'answer_int', 'parsed_ok', 'y_star_evidence']
for k in required:
    if k not in r:
        print(f'Missing field: {k}', file=sys.stderr)
        sys.exit(1)
" 2>/dev/null; then
                echo "  SCHEMA $model_slug: first record missing required fields"
                FAILED=1
            fi
        fi
    done
done

if [[ $FAILED -eq 1 ]]; then
    echo ""
    echo "Some checks failed."
fi

# Run unified metrics (recompute) on this results dir
echo ""
echo "--- Running unified metrics (recompute_all_unified) ---"
if [[ -f scripts/eval/recompute_all_unified.py ]]; then
    # recompute_all_unified discovers results from <results_dir>/<suite>/<model>/results.jsonl
    # Our structure is results/smoke_experiment/<suite>/<model_slug>/results.jsonl
    python scripts/eval/recompute_all_unified.py --results_dir "$RESULTS_ROOT" --epsilon 3.0 2>&1 | tail -80
else
    echo "  recompute_all_unified.py not found, skipping"
fi

echo ""
echo "============================================"
echo "  Validation complete. Results: $RESULTS_ROOT"
echo "  Unified summary: $RESULTS_ROOT/unified_all_suites.json"
echo "============================================"

exit $FAILED
