#!/usr/bin/env bash
# Run Tool suite only for Gemma 1B/4B + OLMo 13B/32B (plaintext tool prompts).
# One model per GPU. After fix in scripts/eval/run_tool.py for template issues.
#
#   conda activate LLM_anchoring && bash scripts/run_tool_gemma_olmo_4gpu.sh

set -euo pipefail
cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/tool_gemma_olmo_${TS}"
mkdir -p "$LOGDIR"

DATA_VIEWS="datasets/anchorbench_tool_core/promptviews_core.jsonl"
DATA_SPECS="datasets/anchorbench_tool_core/itemspecs.jsonl"
OUT="results/full_benchmark/tool"
OPTS="--backend vllm --max_tokens 512 --batch_size 48 --tensor_parallel_size 1 --max_model_len 4096"

run_one() {
  local model="$1" gpu="$2" mem="$3" log="$4"
  echo "GPU $gpu: $model (mem $mem) -> $log"
  CUDA_VISIBLE_DEVICES=$gpu bash scripts/run_with_env.sh python scripts/eval/run_tool.py \
    --promptviews "$DATA_VIEWS" --itemspecs "$DATA_SPECS" \
    --model_id "$model" --out_dir "$OUT" \
    $OPTS --gpu_memory_utilization "$mem" \
    >"$log" 2>&1
}

run_one "google/gemma-3-4b-it" 0 0.92 "$LOGDIR/gpu0_gemma4b.log" &
run_one "google/gemma-3-1b-it" 1 0.92 "$LOGDIR/gpu1_gemma1b.log" &
run_one "allenai/OLMo-2-1124-13B-Instruct" 2 0.90 "$LOGDIR/gpu2_olmo13b.log" &
run_one "allenai/OLMo-2-0325-32B-Instruct" 3 0.85 "$LOGDIR/gpu3_olmo32b.log" &
wait

echo "Done. Logs: $LOGDIR"
wc -l "$OUT"/google_gemma-3-4b-it/results.jsonl \
      "$OUT"/google_gemma-3-1b-it/results.jsonl \
      "$OUT"/allenai_OLMo-2-1124-13B-Instruct/results.jsonl \
      "$OUT"/allenai_OLMo-2-0325-32B-Instruct/results.jsonl 2>/dev/null || true
