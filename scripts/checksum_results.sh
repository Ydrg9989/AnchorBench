#!/usr/bin/env bash
# Checksum the raw model generations under results/.
#
# results/ is ~631 MB, almost all of it per-record generations, so it is
# gitignored and published as downloadable tarballs instead. This manifest is what
# git carries: it lets anyone verify a downloaded tarball holds the same
# bytes the paper's numbers were computed from.
#
# Usage:
#   bash scripts/checksum_results.sh              # write results/CHECKSUMS.sha256
#   bash scripts/checksum_results.sh --verify     # check the tree against it
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${ROOT}/results/CHECKSUMS.sha256"

cd "$ROOT"

if [ "${1:-}" = "--verify" ]; then
    echo "Verifying results/ against ${MANIFEST#$ROOT/}"
    sha256sum -c "$MANIFEST"
    exit
fi

# Cover everything that goes into the published tarballs. Run logs are excluded:
# they embed absolute paths from the machine that produced them, so they are
# neither reproducible nor publishable.
echo "Hashing results/ ..."
find results -type f \
    -not -name '*.log' \
    -not -path '*/_logs/*' \
    -not -name 'CHECKSUMS.sha256' \
    | LC_ALL=C sort \
    | xargs -d'\n' sha256sum > "$MANIFEST"

echo "Wrote ${MANIFEST#$ROOT/}"
echo "  files:  $(wc -l < "$MANIFEST")"
echo "  covers: $(cut -c67- "$MANIFEST" | xargs -d'\n' du -shc 2>/dev/null | tail -1 | cut -f1)"
