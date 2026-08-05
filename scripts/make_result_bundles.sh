#!/usr/bin/env bash
# Build the results/ tarballs for publication.
#
# results/ is ~631 MB of raw model generations, too large for git. It is
# published as downloadable tarballs instead, one per experiment.
#
# NOTE: Google Drive gives no DOI and no permanence guarantee. If a citable
# archive is wanted later, these same tarballs upload to Zenodo unchanged.
#
# results/CHECKSUMS.sha256 (committed) covers every file in every bundle, so
# a downloader can verify the bytes are the ones the paper's numbers came
# from.
#
# Usage:
#   bash scripts/make_result_bundles.sh [OUT_DIR]
#
# Default OUT_DIR is dist/bundles/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist/bundles}"
cd "$ROOT"

# One bundle per experiment. scratch/ is an empty leftover directory and
# smoke_sanity/ is a 300 KB pipeline smoke test; neither backs a paper number.
BUNDLES=(
    full_benchmark
    api_benchmark
    rebuttal
    revision
    decoding_sampling_robustness
    mitigation_ignore_anchor
    icl_dist_core
    icl_dist_api
)

mkdir -p "$OUT"

for name in "${BUNDLES[@]}"; do
    [ -d "results/$name" ] || { echo "skip: results/$name not present"; continue; }
    archive="$OUT/anchorbench-results-${name}.tar.gz"
    echo "packing results/$name ..."
    # Exclude run logs: they embed absolute paths from the machine that
    # produced them, and CHECKSUMS.sha256 does not cover them either.
    tar --exclude='_logs' --exclude='*.log' \
        -czf "$archive" "results/$name"
    printf "  %-52s %s\n" "$(basename "$archive")" "$(du -h "$archive" | cut -f1)"
done

cp results/CHECKSUMS.sha256 "$OUT/"

( cd "$OUT" && sha256sum ./*.tar.gz > BUNDLE_CHECKSUMS.sha256 )

echo
echo "Bundles in $OUT"
echo "Next: upload these to the results folder, then check the link in"
echo "      README.md and datasets/VERSIONS.md still resolves."
