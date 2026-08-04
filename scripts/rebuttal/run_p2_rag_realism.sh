#!/usr/bin/env bash
# P2 — RAG realism ablation on a 4-OW panel (4 GPUs, parallel).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

( bash scripts/rebuttal/_p2_one_model.sh "allenai/OLMo-2-1124-13B-Instruct" 0 ) &
P0=$!
( bash scripts/rebuttal/_p2_one_model.sh "Qwen/Qwen2.5-7B-Instruct"         1 ) &
P1=$!
( bash scripts/rebuttal/_p2_one_model.sh "meta-llama/Llama-3.1-8B-Instruct" 2 ) &
P2=$!
( bash scripts/rebuttal/_p2_one_model.sh "google/gemma-3-4b-it"             3 ) &
P3=$!

wait $P0 $P1 $P2 $P3
echo "[$(date -Iseconds)] P2 all RAG realism runs complete."
