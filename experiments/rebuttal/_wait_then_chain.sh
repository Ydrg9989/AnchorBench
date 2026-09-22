#!/usr/bin/env bash
# Wait for the P1 history run to finish for a given model, THEN start the
# P2->P5 chain on the same GPU.
#
# Usage: bash experiments/rebuttal/_wait_then_chain.sh <model_id> <gpu_index>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu>}"
DEVICES="${2:?usage: $0 <model_id> <gpu>}"
SLUG="${MODEL_ID//\//_}"
DONE_FILE="results/rebuttal/intensity_history/_logs/${SLUG}.done"

echo "[$(date -Iseconds)] WAIT for $DONE_FILE"
while [ ! -f "$DONE_FILE" ]; do
    sleep 30
done
echo "[$(date -Iseconds)] WAIT cleared; chaining P2-P5 for $MODEL_ID on GPU $DEVICES"
bash experiments/rebuttal/_chain_p2_to_p5.sh "$MODEL_ID" "$DEVICES"
