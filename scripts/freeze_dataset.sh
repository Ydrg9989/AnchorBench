#!/usr/bin/env bash
# freeze_dataset.sh — Archive a validated dataset for reproducible release.
#
# Usage:
#   bash scripts/freeze_dataset.sh [SIZE]
#
# Creates:
#   datasets/anchorbench_frozen_{SIZE}.tar.gz
#
# Prerequisites:
#   1. Generate:  bash scripts/generate_all.sh SIZE SEED
#   2. Validate:  bash scripts/validate_all.sh SIZE
#   3. Freeze:    bash scripts/freeze_dataset.sh SIZE

set -euo pipefail

SIZE="${1:-pilot}"
BASE="datasets"
ARCHIVE="${BASE}/anchorbench_frozen_${SIZE}.tar.gz"

DIRS=()
for suite in external history icl rag tool; do
    dir="${BASE}/anchorbench_${suite}_${SIZE}"
    if [ -d "${dir}" ]; then
        DIRS+=("${dir}")
    else
        echo "ERROR: ${dir} not found. Run generate_all.sh first."
        exit 1
    fi
done

echo "============================================================"
echo "  AnchorBench — Freeze dataset (size=${SIZE})"
echo "============================================================"

echo "Computing SHA-256 checksums..."
for dir in "${DIRS[@]}"; do
    for f in "${dir}"/*.jsonl "${dir}"/manifest.json; do
        if [ -f "$f" ]; then
            sha256sum "$f"
        fi
    done
done | tee "${BASE}/anchorbench_${SIZE}_checksums.sha256"

echo ""
echo "Creating archive: ${ARCHIVE}"
tar -czf "${ARCHIVE}" "${DIRS[@]}" "${BASE}/anchorbench_${SIZE}_checksums.sha256"

echo ""
echo "Archive size: $(du -h "${ARCHIVE}" | cut -f1)"
echo "Checksums:    ${BASE}/anchorbench_${SIZE}_checksums.sha256"
echo ""
echo "Done. The frozen archive is ready for submission."
