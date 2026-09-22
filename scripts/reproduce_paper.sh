#!/usr/bin/env bash
# reproduce_paper.sh -- end-to-end reproduction of every numeric claim,
# figure, and LaTeX table in the COLM 2026 paper.
#
# Pipeline:
#   1. Validate that core datasets are present (regenerates if missing).
#   2. Run the frozen `paper_main` experiment recipe (14 models x 5 suites).
#   3. Recompute the unified metrics aggregator.
#   4. Regenerate every paper figure and LaTeX table.
#   5. Run the data-driven verifier; fail loudly if any claim drifts.
#
# Usage:
#   bash scripts/reproduce_paper.sh             # full run
#   DRY_RUN=1 bash scripts/reproduce_paper.sh   # enumerate cells without launching
#
# Requirements:
#   * vLLM-capable GPU(s) for the open-weight tier
#   * OPENROUTER_API_KEY exported for the API tier
#   * Approx. 24h wall time on 4xA100 + ~$300 OpenRouter spend at full size

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/_env.sh
source "$SCRIPT_DIR/_env.sh"   # sets ANCHORBENCH and PYTHONPATH

DRY_RUN_FLAG=""
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_FLAG="+dry_run=true"
    echo "[reproduce] DRY_RUN=1 -- will only enumerate planned cells."
fi

echo "============================================================"
echo "  AnchorBench -- reproducing COLM 2026 paper artifacts"
echo "============================================================"

echo ""
echo "[1/5] Verifying core datasets ..."
missing=0
for suite in external history icl rag tool; do
    if [[ ! -d "datasets/anchorbench_${suite}_core" ]]; then
        echo "  MISSING datasets/anchorbench_${suite}_core"
        missing=$((missing+1))
    fi
done
if [[ "${missing}" -gt 0 ]]; then
    echo "  -> regenerating missing suites at size=core ..."
    bash scripts/generate_all.sh core 42
fi

echo ""
echo "[2/5] Running paper_main experiment recipe ..."
"${ANCHORBENCH[@]}" experiment +experiment=paper_main ${DRY_RUN_FLAG}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo ""
    echo "[3/5][skip] DRY_RUN=1: skipping aggregation"
    echo "[4/5][skip] DRY_RUN=1: skipping artifact generation"
    echo "[5/5][skip] DRY_RUN=1: skipping verifier"
    exit 0
fi

echo ""
echo "[3/5] Recomputing unified_all_suites.json ..."
python -m anchorbench.analysis.unified --results_dir results/full_benchmark
python -m anchorbench.analysis.unified --results_dir results/api_benchmark

echo ""
echo "[4/5] Regenerating paper figures and LaTeX tables ..."
"${ANCHORBENCH[@]}" tables --paper

echo ""
echo "[5/5] Verifying every numeric claim ..."
"${ANCHORBENCH[@]}" verify

echo ""
echo "============================================================"
echo "  Done. Figures: outputs/figures/  Tables: outputs/tables/"
echo "============================================================"
