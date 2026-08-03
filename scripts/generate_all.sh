#!/usr/bin/env bash
# generate_all.sh -- thin wrapper that calls `anchorbench generate` for
# every paper suite at the given size.
#
# Usage:
#   bash scripts/generate_all.sh [SIZE] [SEED]
#
# Args:
#   SIZE  "smoke" | "pilot" | "core"  (default: core)
#   SEED  master random seed          (default: 42)

set -euo pipefail

SIZE="${1:-core}"
SEED="${2:-42}"

echo "============================================================"
echo "  AnchorBench -- Generate all suites (size=${SIZE} seed=${SEED})"
echo "============================================================"

for suite in external history icl icl_dist rag tool; do
    echo ""
    echo "--- Generating ${suite} ---"
    anchorbench generate "data=${suite}" "+size=${SIZE}" "seed=${SEED}"
done

echo ""
echo "Done. Validate with: bash scripts/validate_all.sh ${SIZE}"
