#!/usr/bin/env bash
# Run Llama-3.1-8B and Qwen2.5-7B on External v2 pilot (sequential = one GPU used at a time).
# Run this script in your own terminal (not Cursor background) so the process has GPU access.
# Usage: cd /data/yiderigun/LLM_anchoring && bash scripts/run_external_v2_8B_7B.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src

DATA="--promptviews datasets/anchorbench_v2_external_pilot/promptviews.jsonl --itemspecs datasets/anchorbench_v2_external_pilot/itemspecs.jsonl"
OPTS="--max_tokens 512 --llm_fallback --fallback_model Qwen/Qwen2.5-1.5B-Instruct --fallback_device auto"

echo "=== 1/2 Llama-3.1-8B-Instruct ==="
python scripts/eval/run_external_v2.py $DATA --model_id meta-llama/Llama-3.1-8B-Instruct --out_dir results/external_v2_pilot_Llama31_8B_512 $OPTS
python scripts/export_raw_texts.py results/external_v2_pilot_Llama31_8B_512/results.jsonl || true

echo "=== 2/2 Qwen2.5-7B-Instruct ==="
python scripts/eval/run_external_v2.py $DATA --model_id Qwen/Qwen2.5-7B-Instruct --out_dir results/external_v2_pilot_Qwen25_7B_512 $OPTS
python scripts/export_raw_texts.py results/external_v2_pilot_Qwen25_7B_512/results.jsonl || true

echo "Done. Summaries: results/external_v2_pilot_Llama31_8B_512/summary.json and results/external_v2_pilot_Qwen25_7B_512/summary.json"
echo "Raw text exports: .../raw_texts_export.txt in each run dir"
