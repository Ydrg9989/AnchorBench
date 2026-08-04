#!/usr/bin/env bash
# Wait for all 5-model chains (B1 -> B2 -> B3 -> C1) to finish, then run
# every post-hoc analyzer and the deliverables aggregator.
#
# Designed to live in its own tmux window: it polls for ``.done`` flags
# from `_chain_after_b1.sh` and only kicks off analysis when ALL chains
# have completed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODELS=(
    "Qwen_Qwen2.5-7B-Instruct"
    "meta-llama_Llama-3.1-8B-Instruct"
    "google_gemma-3-4b-it"
    "allenai_OLMo-2-1124-13B-Instruct"
    "openai_gpt-5.4-mini"
)

CHAIN_DONE_DIR="results/rebuttal"
LOG="results/rebuttal/_wrap_up.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date -Iseconds)] wrap_up: waiting for chains to finish..." | tee -a "$LOG"
while :; do
    DONE=0
    PENDING=()
    for SLUG in "${MODELS[@]}"; do
        if [ -f "$CHAIN_DONE_DIR/_chain_${SLUG}.done" ]; then
            DONE=$((DONE+1))
        else
            PENDING+=("$SLUG")
        fi
    done
    echo "[$(date -Iseconds)] wrap_up: chain progress $DONE/${#MODELS[@]} done" | tee -a "$LOG"
    if [ "$DONE" -eq "${#MODELS[@]}" ]; then break; fi
    [ "${#PENDING[@]}" -gt 0 ] && echo "  pending: ${PENDING[*]}" | tee -a "$LOG"
    sleep 120
done

echo "[$(date -Iseconds)] wrap_up: all chains done; running analyzers" | tee -a "$LOG"

set +e
PYTHONPATH=src bash scripts/run_with_env.sh \
    python -m anchorbench.analysis.cot_reasoning_extended 2>&1 | tee -a "$LOG"
PYTHONPATH=src bash scripts/run_with_env.sh \
    python -m anchorbench.analysis.spectrum 2>&1 | tee -a "$LOG"
PYTHONPATH=src bash scripts/run_with_env.sh \
    python -m anchorbench.analysis.weighted_mean 2>&1 | tee -a "$LOG"
PYTHONPATH=src bash scripts/run_with_env.sh \
    python -m anchorbench.analysis.medical_pilot 2>&1 | tee -a "$LOG"
PYTHONPATH=src bash scripts/run_with_env.sh \
    python -m anchorbench.analysis.rebuttal_deliverables 2>&1 | tee -a "$LOG"
set -e

echo "[$(date -Iseconds)] wrap_up: ALL DONE" | tee -a "$LOG"
touch "$CHAIN_DONE_DIR/_wrap_up.done"
