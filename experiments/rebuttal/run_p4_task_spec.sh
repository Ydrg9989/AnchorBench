#!/usr/bin/env bash
# P4 — Task-specification ablation on a 4-OW panel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

( bash experiments/rebuttal/_p4_one_model.sh "allenai/OLMo-2-1124-13B-Instruct" 0 ) &
P0=$!
( bash experiments/rebuttal/_p4_one_model.sh "Qwen/Qwen2.5-7B-Instruct"         1 ) &
P1=$!
( bash experiments/rebuttal/_p4_one_model.sh "meta-llama/Llama-3.1-8B-Instruct" 2 ) &
P2=$!
( bash experiments/rebuttal/_p4_one_model.sh "google/gemma-3-4b-it"             3 ) &
P3=$!

wait $P0 $P1 $P2 $P3
echo "[$(date -Iseconds)] P4 all task-spec runs complete."
