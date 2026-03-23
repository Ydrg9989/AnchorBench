#!/usr/bin/env bash
# Run the 4 models that are missing tool-suite results, one per GPU (max utilization).
# Skips external/icl/rag/history when already complete (run_model_vllm.sh).
#
# Usage:
#   cd $REPO_ROOT && conda activate LLM_anchoring && bash scripts/run_tool_missing_4gpu.sh
#
# Logs: logs/tool_missing_4gpu_<timestamp>/gpu{0..3}_*.log

set -euo pipefail
cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/tool_missing_4gpu_${TS}"
mkdir -p "$LOGDIR"

export ANCHORBENCH_VIEWS=promptviews_core.jsonl
export ANCHORBENCH_BATCH_SIZE=96
MAX_TOKENS=512
MAX_MODEL_LEN=4096

echo "================================================================"
echo " Tool suite — 4 models in parallel (GPUs 0–3)"
echo " Logs: $LOGDIR/"
echo " Started: $TS"
echo "================================================================"

# GPU 0–3: small → large; higher mem util on smaller models
bash scripts/run_model_vllm.sh google/gemma-3-4b-it 0 $MAX_TOKENS 0.93 $MAX_MODEL_LEN 1 \
  >"$LOGDIR/gpu0_gemma4b.log" 2>&1 &
PID0=$!

bash scripts/run_model_vllm.sh google/gemma-3-1b-it 1 $MAX_TOKENS 0.93 $MAX_MODEL_LEN 1 \
  >"$LOGDIR/gpu1_gemma1b.log" 2>&1 &
PID1=$!

bash scripts/run_model_vllm.sh allenai/OLMo-2-1124-13B-Instruct 2 $MAX_TOKENS 0.92 $MAX_MODEL_LEN 1 \
  >"$LOGDIR/gpu2_olmo13b.log" 2>&1 &
PID2=$!

bash scripts/run_model_vllm.sh allenai/OLMo-2-0325-32B-Instruct 3 $MAX_TOKENS 0.88 $MAX_MODEL_LEN 1 \
  >"$LOGDIR/gpu3_olmo32b.log" 2>&1 &
PID3=$!

echo "PIDs: gemma4b=$PID0 gemma1b=$PID1 olmo13b=$PID2 olmo32b=$PID3"
echo "Waiting for all jobs..."
wait $PID0 || echo "GPU0 exit: $?"
wait $PID1 || echo "GPU1 exit: $?"
wait $PID2 || echo "GPU2 exit: $?"
wait $PID3 || echo "GPU3 exit: $?"

echo ""
echo "================================================================"
echo " Done. Check tool results:"
echo "   wc -l results/full_benchmark/tool/*/results.jsonl"
echo " Then: bash scripts/run_with_env.sh python scripts/eval/recompute_all_unified.py --results_dir results/full_benchmark"
echo "================================================================"
