#!/usr/bin/env bash
# DEPRECATED: prefer `python scripts/run.py eval --suite history --model ...` or
#   `python scripts/run.py experiment --name paper_main` for the full panel.
# This script remains for backward compatibility with prior shell pipelines.
#
# Run History suite (default: two-stage baseline) with one model per GPU.
#
# Default models: same ten open-weight IDs as the paper (scripts/paper_model_ids.inc.sh).
#
# Usage:
#   cd "$REPO_ROOT" && bash scripts/run_history_gpus.sh
#
# Environment overrides:
#   HIST_DATA_DIR          - dataset dir (default: datasets/anchorbench_history_core)
#   HIST_OUT_BASE          - results root (default: results/history_core_twostage_baseline)
#   HIST_BACKEND           - hf | vllm (default: hf)
#   HIST_MODELS            - space-separated model ids (optional)
#   HIST_BASELINE          - control_twostage | control (default: control_twostage)
#   NGPU                   - max parallel jobs
#   HIST_MAX_ITEMS         - cap items for debugging
#   HIST_TP, HIST_GPU_MEM  - vLLM options (defaults: 1, 0.9)

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
# shellcheck source=paper_model_ids.inc.sh
source "$REPO_ROOT/scripts/paper_model_ids.inc.sh"

DATA_DIR="${HIST_DATA_DIR:-datasets/anchorbench_history_core}"
OUT_BASE="${HIST_OUT_BASE:-results/history_core_twostage_baseline}"
BACKEND="${HIST_BACKEND:-hf}"
BASELINE="${HIST_BASELINE:-control_twostage}"
TP_SIZE="${HIST_TP:-1}"
GPU_MEM="${HIST_GPU_MEM:-0.9}"

MAX_ITEMS_ARG=()
if [[ -n "${HIST_MAX_ITEMS:-}" ]]; then
  MAX_ITEMS_ARG=(--max_items "$HIST_MAX_ITEMS")
fi

if [[ -z "${HIST_MODELS:-}" ]]; then
  MODELS=("${PAPER_ANCHORBENCH_MODEL_IDS[@]}")
else
  read -r -a MODELS <<< "${HIST_MODELS}"
fi

if [[ ! -f "$DATA_DIR/promptviews.jsonl" ]] || [[ ! -f "$DATA_DIR/itemspecs.jsonl" ]]; then
  echo "ERROR: Missing itemspecs/promptviews under $DATA_DIR"
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
echo "History data: $DATA_DIR | baseline: $BASELINE"
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
    export CUDA_VISIBLE_DEVICES=$slot
    echo "[GPU $slot] $model -> $OUT_BASE/${model//\//_}/"
    bash scripts/run_with_env.sh python scripts/eval/run_history.py \
      --promptviews "$DATA_DIR/promptviews.jsonl" \
      --itemspecs "$DATA_DIR/itemspecs.jsonl" \
      --model_id "$model" \
      --out_dir "$OUT_BASE" \
      --baseline_condition "$BASELINE" \
      --max_tokens 512 \
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
[[ "$failed" -eq 0 ]] || exit 1

echo ""
echo "Done. Summaries:"
for model in "${MODELS[@]}"; do
  slug="${model//\//_}"
  echo "  $OUT_BASE/$slug/summary.json"
done
