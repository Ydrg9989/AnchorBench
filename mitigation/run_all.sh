#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true
export PYTHONUNBUFFERED=1

CFG="mitigation/config.yaml"
SUITES="external,rag,tool"

echo "====== Starting mitigation experiments ======"
echo "Time: $(date)"

# B1 — "Ignore the anchor" instruction
echo ">>> B1 — Ignore instruction"
python mitigation/run_mitigation.py --config $CFG --mitigation B1 \
    --run_name mit_b1 --suites $SUITES 2>&1

# B2 — Awareness reminder
echo ">>> B2 — Awareness reminder"
python mitigation/run_mitigation.py --config $CFG --mitigation B2 \
    --run_name mit_b2 --suites $SUITES 2>&1

# B3 — Chain-of-thought
echo ">>> B3 — Chain-of-thought"
python mitigation/run_mitigation.py --config $CFG --mitigation B3 \
    --run_name mit_b3 --suites $SUITES 2>&1

# EFR — Evidence-First Restructuring
echo ">>> EFR — Evidence-First Restructuring"
python mitigation/run_mitigation.py --config $CFG --mitigation EFR \
    --run_name mit_efr --suites $SUITES 2>&1

# CxDP — Context-Distilled Prompting
echo ">>> CxDP — Context-Distilled Prompting"
python mitigation/run_mitigation.py --config $CFG --mitigation CxDP \
    --run_name mit_cxdp --suites $SUITES 2>&1

echo "====== All mitigation experiments complete ======"
echo "Time: $(date)"

# Post-hoc evaluation
echo ">>> Computing post-hoc mitigations and full evaluation"
python mitigation/eval_mitigations.py \
    --dataset poc_dataset/poc_v0.2.jsonl \
    --baseline runner_outputs/poc_v0_openrouter.jsonl runner_outputs/api_v02.jsonl \
    --mitigation runner_outputs/mit_b1.jsonl \
    --posthoc B4 CAC PLD \
    --run_name eval_all_b1 2>&1

python mitigation/eval_mitigations.py \
    --dataset poc_dataset/poc_v0.2.jsonl \
    --baseline runner_outputs/poc_v0_openrouter.jsonl runner_outputs/api_v02.jsonl \
    --mitigation runner_outputs/mit_b2.jsonl \
    --posthoc B4 CAC PLD \
    --run_name eval_all_b2 2>&1

python mitigation/eval_mitigations.py \
    --dataset poc_dataset/poc_v0.2.jsonl \
    --baseline runner_outputs/poc_v0_openrouter.jsonl runner_outputs/api_v02.jsonl \
    --mitigation runner_outputs/mit_b3.jsonl \
    --posthoc B4 CAC PLD \
    --run_name eval_all_b3 2>&1

python mitigation/eval_mitigations.py \
    --dataset poc_dataset/poc_v0.2.jsonl \
    --baseline runner_outputs/poc_v0_openrouter.jsonl runner_outputs/api_v02.jsonl \
    --mitigation runner_outputs/mit_efr.jsonl \
    --posthoc B4 CAC PLD \
    --run_name eval_all_efr 2>&1

python mitigation/eval_mitigations.py \
    --dataset poc_dataset/poc_v0.2.jsonl \
    --baseline runner_outputs/poc_v0_openrouter.jsonl runner_outputs/api_v02.jsonl \
    --mitigation runner_outputs/mit_cxdp.jsonl \
    --posthoc B4 CAC PLD \
    --run_name eval_all_cxdp 2>&1

echo ">>> All evaluations complete"
echo "Time: $(date)"
