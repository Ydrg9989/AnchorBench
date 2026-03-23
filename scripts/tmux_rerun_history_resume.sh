#!/usr/bin/env bash
# Start in-flight History jobs under tmux with --resume (keeps partial results.jsonl).
#
#   bash scripts/tmux_rerun_history_resume.sh
#
# Creates session "anchorbench" if missing, then adds three windows (or starts fresh).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
SESSION="${TMUX_SESSION:-anchorbench}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate LLM_anchoring

HIST_PV="$REPO/datasets/anchorbench_history_core/promptviews.jsonl"
HIST_IS="$REPO/datasets/anchorbench_history_core/itemspecs.jsonl"
HIST_OUT="$REPO/results/history_core_twostage_baseline"

run_one() {
  local name="$1"
  local cmd="$2"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" -n "$name" "bash -lc $(printf '%q' "$cmd; exec bash")"
  else
    tmux new-session -d -s "$SESSION" -n "$name" "bash -lc $(printf '%q' "$cmd; exec bash")"
  fi
}

G="cd $REPO && export CUDA_VISIBLE_DEVICES=3 && bash scripts/run_with_env.sh python scripts/eval/run_history.py \
  --promptviews $HIST_PV --itemspecs $HIST_IS \
  --model_id google/gemma-3-4b-it \
  --out_dir $HIST_OUT --baseline_condition control_twostage \
  --max_tokens 512 --backend hf --llm_fallback \
  --fallback_model Qwen/Qwen2.5-1.5B-Instruct --fallback_device cuda:0 \
  --resume"

O13="cd $REPO && export CUDA_VISIBLE_DEVICES=0 && bash scripts/run_with_env.sh python scripts/eval/run_history.py \
  --promptviews $HIST_PV --itemspecs $HIST_IS \
  --model_id allenai/OLMo-2-1124-13B-Instruct \
  --out_dir $HIST_OUT --baseline_condition control_twostage \
  --max_tokens 512 --backend vllm --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.95 --max_model_len 4096 \
  --resume"

O32="cd $REPO && export CUDA_VISIBLE_DEVICES=1,2 && bash scripts/run_with_env.sh python scripts/eval/run_history.py \
  --promptviews $HIST_PV --itemspecs $HIST_IS \
  --model_id allenai/OLMo-2-0325-32B-Instruct \
  --out_dir $HIST_OUT --baseline_condition control_twostage \
  --max_tokens 512 --backend vllm --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.95 --max_model_len 4096 \
  --resume"

run_one "gemma4b-hist" "$G"
run_one "olmo13b-hist" "$O13"
run_one "olmo32b-hist" "$O32"

echo "Started 3 History jobs with --resume in tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
tmux list-windows -t "$SESSION"
