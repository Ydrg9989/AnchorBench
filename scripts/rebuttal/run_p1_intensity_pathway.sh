#!/usr/bin/env bash
# P1 — Cross-pathway intensity probe.
# Extends D1 (External, 3 OW) to {External, RAG, History} on a 4-OW panel.
#
# GPU layout (assumes 4 H100s available):
#   GPU 0: OLMo-13B    (external + rag + history; OLMo had no prior D1)
#   GPU 1: Qwen-7B     (rag + history; external already in results/rebuttal/intensity)
#   GPU 2: Llama-8B    (rag + history; external already done)
#   GPU 3: Gemma-4B    (rag + history; external already done)
#
# Launches each model on its own GPU in the background and waits for all
# to finish. Logs go to results/rebuttal/intensity{,_rag,_history}/_logs/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

run_model_suites() {
    local model_id="$1"; shift
    local gpu="$1"; shift
    local suites=("$@")
    for suite in "${suites[@]}"; do
        echo "[$(date -Iseconds)] launching $model_id on GPU $gpu (suite=$suite)"
        bash scripts/rebuttal/_d1_one_model.sh "$model_id" "$gpu" "$suite"
    done
}

# Background launches: one process per GPU, each looping over its suites.
( run_model_suites "allenai/OLMo-2-1124-13B-Instruct"   0 external rag history ) &
PID_OLMO=$!
( run_model_suites "Qwen/Qwen2.5-7B-Instruct"           1 rag history ) &
PID_QWEN=$!
( run_model_suites "meta-llama/Llama-3.1-8B-Instruct"   2 rag history ) &
PID_LLAMA=$!
( run_model_suites "google/gemma-3-4b-it"               3 rag history ) &
PID_GEMMA=$!

echo "Spawned PIDs: olmo=$PID_OLMO qwen=$PID_QWEN llama=$PID_LLAMA gemma=$PID_GEMMA"
wait $PID_OLMO $PID_QWEN $PID_LLAMA $PID_GEMMA

echo "[$(date -Iseconds)] P1 all intensity runs complete."
