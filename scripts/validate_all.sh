#!/usr/bin/env bash
# validate_all.sh — Canonical script to validate all AnchorBench suites.
#
# Usage:
#   bash scripts/validate_all.sh [SIZE]
#
# Arguments:
#   SIZE  "smoke" | "pilot" | "core"  (default: pilot)
#
# Checks:
#   - schema completeness
#   - paired-condition completeness
#   - duplicate detection
#   - domain/difficulty balance
#   - gold-answer validity
#   - answer-format instruction presence
#   - manifest consistency
#
# After validation passes, freeze with:
#   bash scripts/freeze_dataset.sh [SIZE]

set -euo pipefail

# shellcheck source=scripts/_env.sh
source "$(dirname "$0")/_env.sh"   # puts src/ on PYTHONPATH

SIZE="${1:-pilot}"
BASE="datasets"
if [[ "$SIZE" == "smoke" ]]; then
    if [[ -d "${BASE}/archive/generated_smoke/anchorbench_external_smoke" ]]; then
        BASE="${BASE}/archive/generated_smoke"
    elif [[ -d "${BASE}/archive/smoke_benchmark_20260318/anchorbench_external_smoke" ]]; then
        BASE="${BASE}/archive/smoke_benchmark_20260318"
    fi
fi

DIRS=()
for suite in external history icl icl_dist rag tool; do
    dir="${BASE}/anchorbench_${suite}_${SIZE}"
    if [ -d "${dir}" ]; then
        DIRS+=("${dir}")
    else
        echo "WARNING: ${dir} not found, skipping."
    fi
done

if [ ${#DIRS[@]} -eq 0 ]; then
    echo "ERROR: No dataset directories found for size=${SIZE}."
    exit 1
fi

echo "============================================================"
echo "  AnchorBench — Validate all suites (size=${SIZE})"
echo "============================================================"

python -m anchorbench.data.validate --data_dir "${DIRS[@]}"
