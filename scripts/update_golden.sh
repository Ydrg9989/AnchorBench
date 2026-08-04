#!/usr/bin/env bash
# Refresh the golden manifests in tests/golden/.
#
# These pin what the generators produce *right now*, so a refactor that is
# meant to be behaviour-preserving can be proven behaviour-preserving. They
# are deliberately NOT a check against the paper -- that is verify.py's job,
# and the two must not be conflated. A divergence from the paper belongs in
# docs/RECONCILIATION.md, never in a widened tolerance here.
#
# Run this only when a change is *intended* to alter generator output, and
# say why in the commit message. Requires the full results/ tree.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOLDEN="$ROOT/tests/golden"
cd "$ROOT"

mkdir -p "$GOLDEN"

hash_into() {  # hash_into <manifest> <find-args...>
    local out="$1"; shift
    find "$@" -type f 2>/dev/null | LC_ALL=C sort | xargs -d'\n' -r sha256sum > "$out"
    echo "  $(wc -l < "$out") files -> ${out#$ROOT/}"
}

echo "Regenerating paper tables and figures ..."
PYTHONPATH=src python3 -m anchorbench.paper.tables_main    >/dev/null
PYTHONPATH=src python3 -m anchorbench.paper.tables_appendix >/dev/null
PYTHONPATH=src python3 -m anchorbench.paper.fig4_dose_response >/dev/null
PYTHONPATH=src python3 -m anchorbench.paper.fig5_acc_vs_disc   >/dev/null

echo "Hashing ..."
# Regenerable from results/: the main + appendix table set.
hash_into "$GOLDEN/tables.sha256" outputs/tables -name '*.tex'

# Pin-only: the 13 tables the camera-ready \input-s, plus the orphaned
# generator outputs alongside them. Their inputs are the bulk rebuttal
# results, which are not committed, so these are pinned rather than
# regenerated. Every one is committed, so this catches accidental edits.
hash_into "$GOLDEN/rebuttal_tables.sha256" results/rebuttal -name '*.tex'

# Figures: PNG only. Matplotlib writes /CreationDate and /ID into PDFs, so
# PDF hashes differ on every run even when the plot is identical -- verified.
hash_into "$GOLDEN/figures.sha256" COLM/figures -name '*.png'

# The two unified summaries every table is derived from. Belt and braces:
# catches an accidental regeneration of the inputs themselves.
hash_into "$GOLDEN/unified.sha256" results -name 'unified_all_suites.json'

echo
echo "Done. Review the diff before committing:"
echo "  git diff --stat tests/golden/"
