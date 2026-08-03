#!/usr/bin/env bash
# smoke_test.sh -- validate the full AnchorBench pipeline without a GPU.
#
# Steps:
#   1. Generate smoke-size datasets for every suite via `anchorbench generate`.
#   2. Validate the generated datasets (`scripts/validate_all.sh smoke`).
#   3. Run the test suite (pytest).
#
# Usage:
#   bash scripts/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== AnchorBench smoke test ==="

echo "[1/3] Generating smoke datasets ..."
bash scripts/generate_all.sh smoke 42

echo "[2/3] Validating smoke datasets ..."
bash scripts/validate_all.sh smoke

echo "[3/3] Running pytest ..."
python -m pytest tests/ -v --tb=short

echo "=== Smoke test complete -- all checks passed ==="
