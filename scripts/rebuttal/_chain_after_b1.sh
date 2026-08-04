#!/usr/bin/env bash
# Wait for B1 to complete for a given model, then chain B2 → B3 → C1.
# B3 (weighted-mean) is skipped for the API model (the api runner reads
# from the standard SUITE_DATASETS["external"] path).
#
# Usage: bash scripts/rebuttal/_chain_after_b1.sh <model_id> <gpu|api>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu|api>}"
DEVICES="${2:?usage: $0 <model_id> <gpu|api>}"
SLUG="${MODEL_ID//\//_}"

B1_DONE="results/rebuttal/cot_extended/_logs/${SLUG}.done"
echo "[$(date -Iseconds)] chain: waiting for B1 to finish ($B1_DONE)..."
while [ ! -f "$B1_DONE" ]; do
    sleep 30
done
echo "[$(date -Iseconds)] chain: B1 done for $MODEL_ID"

# --- B2: plausibility spectrum -----------------------------------------
echo "[$(date -Iseconds)] chain: launching B2 for $MODEL_ID"
bash scripts/rebuttal/_b2_one_model.sh "$MODEL_ID" "$DEVICES" || \
    echo "[$(date -Iseconds)] WARN: B2 failed for $MODEL_ID"

# --- B3: weighted-mean (open-weight only) ------------------------------
if [ "$DEVICES" != "api" ]; then
    echo "[$(date -Iseconds)] chain: launching B3 for $MODEL_ID"
    bash scripts/rebuttal/_b3_one_model.sh "$MODEL_ID" "$DEVICES" || \
        echo "[$(date -Iseconds)] WARN: B3 failed for $MODEL_ID"
else
    echo "[$(date -Iseconds)] chain: skipping B3 for API model"
fi

# --- C1: medical pilot --------------------------------------------------
echo "[$(date -Iseconds)] chain: launching C1 for $MODEL_ID"
bash scripts/rebuttal/_c1_one_model.sh "$MODEL_ID" "$DEVICES" || \
    echo "[$(date -Iseconds)] WARN: C1 failed for $MODEL_ID"

echo "[$(date -Iseconds)] chain: ALL DONE for $MODEL_ID"
touch "results/rebuttal/_chain_${SLUG}.done"
