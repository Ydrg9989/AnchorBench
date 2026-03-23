#!/usr/bin/env bash
# Run ICL-style experiments (default: icl_dist core) with one model per GPU.
#
# Default model list: all ten instruction-tuned models from the paper
# (COLM/sections/appendix/setup.tex, Table model-details), including Gemma~3 and OLMo~2.
# If you have fewer GPUs than models, jobs run in waves (full GPU use each wave).
#
# Usage:
#   cd "$REPO_ROOT" && bash scripts/run_icl_gpus.sh
#
# Environment overrides:
#   ICL_DATA_DIR     - dataset dir (default: datasets/anchorbench_icl_dist_core)
#   ICL_OUT_BASE     - results root (default: results/icl_dist_core)
#   ICL_BATCH_SIZE   - batch size for HF/vLLM batched path (default: 32)
#   ICL_BACKEND      - hf | vllm (default: hf)
#   ICL_MODELS       - space-separated model ids (optional; else built-in list)
#   NGPU             - max parallel jobs (default: all visible GPUs)
#   ICL_MAX_ITEMS    - cap items for debugging
#   ICL_TP           - vLLM tensor_parallel_size per process (default: 1)
#   ICL_GPU_MEM      - vLLM --gpu_memory_utilization (default: 0.9)

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
# shellcheck source=paper_model_ids.inc.sh
source "$REPO_ROOT/scripts/paper_model_ids.inc.sh"

DATA_DIR="${ICL_DATA_DIR:-datasets/anchorbench_icl_dist_core}"
OUT_BASE="${ICL_OUT_BASE:-results/icl_dist_core}"
BATCH_SIZE="${ICL_BATCH_SIZE:-32}"
BACKEND="${ICL_BACKEND:-hf}"
TP_SIZE="${ICL_TP:-1}"
GPU_MEM="${ICL_GPU_MEM:-0.9}"

MAX_ITEMS_ARG=()
if [[ -n "${ICL_MAX_ITEMS:-}" ]]; then
  MAX_ITEMS_ARG=(--max_items "$ICL_MAX_ITEMS")
fi

if [[ -z "${ICL_MODELS:-}" ]]; then
  MODELS=("${PAPER_ANCHORBENCH_MODEL_IDS[@]}")
else
  read -r -a MODELS <<< "${ICL_MODELS}"
fi

if [[ ! -f "$DATA_DIR/promptviews.jsonl" ]] || [[ ! -f "$DATA_DIR/itemspecs.jsonl" ]]; then
  echo "ERROR: Missing itemspecs/promptviews under $DATA_DIR"
  echo "Generate with: PYTHONPATH=src python -m anchorbench_v1.generate --suite icl_dist --size core"
  exit 1
fi

NGPU="${NGPU:-}"
if [[ -z "$NGPU" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  else
    NGPU=1
  fi
fi

NMODELS=${#MODELS[@]}
echo "ICL suite data: $DATA_DIR"
echo "Backend: $BACKEND | Models: $NMODELS (paper list) | GPUs: $NGPU (waves if models > GPUs)"

EXTRA_BACKEND=()
if [[ "$BACKEND" == "vllm" ]]; then
  EXTRA_BACKEND=(
    --backend vllm
    --tensor_parallel_size "$TP_SIZE"
    --gpu_memory_utilization "$GPU_MEM"
  )
else
  EXTRA_BACKEND=(--backend hf)
fi

failed=0
for ((start = 0; start < NMODELS; start += NGPU)); do
  wave_end=$((start + NGPU))
  [[ $wave_end -gt $NMODELS ]] && wave_end=$NMODELS
  echo ""
  echo "--- Wave: models $((start + 1))–${wave_end} of ${NMODELS} ---"
  pids=()
  slot=0
  for ((idx = start; idx < wave_end; idx++)); do
    model="${MODELS[$idx]}"
    out_name="${model//\//_}"
    export CUDA_VISIBLE_DEVICES=$slot
    echo "[GPU $slot] $model -> $OUT_BASE/$out_name"
    bash scripts/run_with_env.sh python scripts/eval/run_icl.py \
      --promptviews "$DATA_DIR/promptviews.jsonl" \
      --itemspecs "$DATA_DIR/itemspecs.jsonl" \
      --model_id "$model" \
      --out_dir "$OUT_BASE" \
      --max_tokens 512 \
      --batch_size "$BATCH_SIZE" \
      --llm_fallback \
      --fallback_model Qwen/Qwen2.5-1.5B-Instruct \
      --fallback_device "cuda:0" \
      "${EXTRA_BACKEND[@]}" \
      "${MAX_ITEMS_ARG[@]}" &
    pids+=($!)
    slot=$((slot + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo ""
echo "Done. Summaries:"
for model in "${MODELS[@]}"; do
  out_name="${model//\//_}"
  echo "  $OUT_BASE/$out_name/summary.json"
done
