#!/usr/bin/env bash
# generate_all.sh — Canonical script to generate all AnchorBench suites.
#
# Usage:
#   bash scripts/generate_all.sh [SIZE] [SEED]
#
# Arguments:
#   SIZE  "smoke" | "pilot" | "core"  (default: core)
#   SEED  master random seed          (default: 42)
#
# Target counts (core, seed=42):
#   External   6 domains × 2 diffs × 3 offsets × 10 = 360 items, 1800 views
#   ICL        6 domains × 2 diffs × 3 offsets × 10 = 360 items, 1800 views
#   ICL-dist   6 domains × 2 diffs × 3 offsets × 10 = 360 items, 1800 views
#   RAG        6 domains × 2 diffs × 3 offsets × 10 = 360 items, 1800 views
#   Tool       6 domains × 2 diffs × 3 offsets × 10 = 360 items, 1800 views
#   History    6 domains × 2 diffs × 1          × 30 = 360 items, 1800 views
#   Total                                            2160 items, 10800 views
#
# After generating, run:
#   bash scripts/validate_all.sh [SIZE]

set -euo pipefail

SIZE="${1:-core}"
SEED="${2:-42}"
BASE="datasets"
# Tiny smoke sets go under archive/ so the repo root only keeps *_core.
OUT_BASE="${BASE}"
if [[ "$SIZE" == "smoke" ]]; then
    OUT_BASE="${BASE}/archive/generated_smoke"
    mkdir -p "${OUT_BASE}"
    echo "(smoke → ${OUT_BASE}/anchorbench_*_smoke — not at repo root)"
fi

# History has no offset dimension (domain × difficulty only), so it needs
# a larger n_per_cell to match the 360-item target of offset-stratified suites.
HISTORY_N_PER_CELL=30
if [[ "$SIZE" == "smoke" ]]; then
    HISTORY_N_PER_CELL=1
elif [[ "$SIZE" == "pilot" ]]; then
    HISTORY_N_PER_CELL=15
fi

echo "============================================================"
echo "  AnchorBench — Generate all suites"
echo "  Size: ${SIZE}    Seed: ${SEED}"
echo "============================================================"

for suite in external icl icl_dist rag tool; do
    out_dir="${OUT_BASE}/anchorbench_${suite}_${SIZE}"
    echo ""
    echo "--- Generating ${suite} → ${out_dir} ---"
    PYTHONPATH=src python -m anchorbench_v1.generate \
        --suite "${suite}" --size "${SIZE}" --seed "${SEED}" \
        --out_dir "${out_dir}"
done

# History: use --n_per_cell override
out_dir="${OUT_BASE}/anchorbench_history_${SIZE}"
echo ""
echo "--- Generating history → ${out_dir} (n_per_cell=${HISTORY_N_PER_CELL}) ---"
PYTHONPATH=src python -m anchorbench_v1.generate \
    --suite history --size "${SIZE}" --seed "${SEED}" \
    --out_dir "${out_dir}" --n_per_cell "${HISTORY_N_PER_CELL}"

echo ""
echo "============================================================"
echo "  All suites generated.  Next: bash scripts/validate_all.sh ${SIZE}"
echo "============================================================"
