#!/usr/bin/env bash
# Run P2 -> P3 -> P4 -> P5 sequentially on ONE GPU for ONE model.
# Each step writes its own logs and .done marker; the chain just orders them.
#
# Usage: bash scripts/rebuttal/_chain_p2_to_p5.sh <model_id> <gpu_index>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu>}"
DEVICES="${2:?usage: $0 <model_id> <gpu>}"
SLUG="${MODEL_ID//\//_}"

CHAIN_LOG_DIR="results/rebuttal/_chain_logs"
mkdir -p "$CHAIN_LOG_DIR"
CHAIN_LOG="$CHAIN_LOG_DIR/${SLUG}.log"

echo "[$(date -Iseconds)] CHAIN start $MODEL_ID on GPU $DEVICES" | tee -a "$CHAIN_LOG"

steps=("p2:scripts/rebuttal/_p2_one_model.sh"
       "p3:scripts/rebuttal/_p3_one_model.sh"
       "p4:scripts/rebuttal/_p4_one_model.sh"
       "p5:scripts/rebuttal/_p5_one_model.sh")

for step in "${steps[@]}"; do
    NAME="${step%%:*}"
    SCRIPT="${step##*:}"
    echo "[$(date -Iseconds)] CHAIN[$NAME] $MODEL_ID" | tee -a "$CHAIN_LOG"
    bash "$SCRIPT" "$MODEL_ID" "$DEVICES"
    echo "[$(date -Iseconds)] CHAIN[$NAME] $MODEL_ID OK" | tee -a "$CHAIN_LOG"
done

echo "[$(date -Iseconds)] CHAIN done $MODEL_ID" | tee -a "$CHAIN_LOG"
touch "$CHAIN_LOG_DIR/${SLUG}.done"
