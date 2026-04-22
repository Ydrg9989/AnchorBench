#!/usr/bin/env bash
# AnchorBench smoke test — validates the full pipeline without GPU.
#
# Runs:
#   1. Dataset generation (smoke size, seed 42)
#   2. Dataset validation
#   3. Unit + integration tests (pytest)
#
# Usage:
#   bash scripts/smoke_test.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== AnchorBench Smoke Test ==="
echo ""

# ── 1. Generate smoke datasets ─────────────────────────────────────
echo "[1/3] Generating smoke datasets (6 items × 5 suites)..."
bash scripts/generate_all.sh smoke 42
echo "  ✓ Dataset generation complete"
echo ""

# ── 2. Validate generated datasets ────────────────────────────────
echo "[2/3] Validating smoke datasets..."
bash scripts/validate_all.sh smoke
echo "  ✓ Dataset validation passed"
echo ""

# ── 3. Run pytest ──────────────────────────────────────────────────
echo "[3/3] Running unit and integration tests..."
python -m pytest tests/ -v --tb=short
echo "  ✓ All tests passed"
echo ""

echo "=== Smoke test complete — all checks passed ==="
