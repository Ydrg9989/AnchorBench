#!/usr/bin/env bash
# Run External inference for Llama-3.1-8B and Qwen2.5-7B (sequential).
#
# Runs two models back-to-back on a single GPU, writing results and raw-text
# exports to results/external_pilot_*/. Each model uses LLM fallback
# parsing for improved parse rates on CoT outputs.
#
# Usage:
#   cd $REPO_ROOT && bash scripts/run_external_8B_7B.sh
#
# Environment:
#   Requires PYTHONPATH=src (set automatically), one CUDA GPU, and the
#   External pilot dataset at datasets/anchorbench_external_pilot/.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src

DATA="--promptviews datasets/anchorbench_external_pilot/promptviews.jsonl --itemspecs datasets/anchorbench_external_pilot/itemspecs.jsonl"
OPTS="--max_tokens 512 --llm_fallback --fallback_model Qwen/Qwen2.5-1.5B-Instruct --fallback_device auto"

echo "=== 1/2 Llama-3.1-8B-Instruct ==="
python scripts/eval/run_external.py $DATA --model_id meta-llama/Llama-3.1-8B-Instruct --out_dir results/external_pilot_Llama31_8B_512 $OPTS
python scripts/export_raw_texts.py results/external_pilot_Llama31_8B_512/results.jsonl || true

echo "=== 2/2 Qwen2.5-7B-Instruct ==="
python scripts/eval/run_external.py $DATA --model_id Qwen/Qwen2.5-7B-Instruct --out_dir results/external_pilot_Qwen25_7B_512 $OPTS
python scripts/export_raw_texts.py results/external_pilot_Qwen25_7B_512/results.jsonl || true

echo "Done. Summaries: results/external_pilot_Llama31_8B_512/summary.json and results/external_pilot_Qwen25_7B_512/summary.json"
echo "Raw text exports: .../raw_texts_export.txt in each run dir"
