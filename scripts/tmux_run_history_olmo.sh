#!/usr/bin/env bash
# Launch OLMo History experiments inside tmux windows.
#
# Usage:
#   bash scripts/tmux_run_history_olmo.sh          # default: both 13B + 32B
#   OLMO_ONLY=13b bash scripts/tmux_run_history_olmo.sh
#   OLMO_ONLY=32b bash scripts/tmux_run_history_olmo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

SESSION="${TMUX_SESSION:-anchorbench}"
ONLY="${OLMO_ONLY:-both}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate LLM_anchoring

tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION"

run_in_tmux() {
  local win_name="$1"; shift
  local cmd="$*"
  if tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx "$win_name"; then
    tmux send-keys -t "$SESSION:$win_name" C-c
    sleep 1
    tmux send-keys -t "$SESSION:$win_name" "$cmd" Enter
  else
    tmux new-window -t "$SESSION" -n "$win_name" "bash -c '$cmd; exec bash'"
  fi
}

HIST_PV="$REPO/datasets/anchorbench_history_core/promptviews.jsonl"
HIST_IS="$REPO/datasets/anchorbench_history_core/itemspecs.jsonl"
HIST_OUT="$REPO/results/history_core_twostage_baseline"

if [[ "$ONLY" == "both" || "$ONLY" == "13b" ]]; then
  run_in_tmux "olmo13b-hist" \
    "export CUDA_VISIBLE_DEVICES=0 && bash $REPO/scripts/run_with_env.sh python $REPO/scripts/eval/run_history.py \
      --promptviews $HIST_PV --itemspecs $HIST_IS \
      --model_id allenai/OLMo-2-1124-13B-Instruct \
      --out_dir $HIST_OUT --baseline_condition control_twostage \
      --max_tokens 512 --backend vllm --tensor_parallel_size 1 \
      --gpu_memory_utilization 0.95 --max_model_len 4096 --resume"
  echo "Started OLMo-13B History in tmux window 'olmo13b-hist'"
fi

if [[ "$ONLY" == "both" || "$ONLY" == "32b" ]]; then
  run_in_tmux "olmo32b-hist" \
    "export CUDA_VISIBLE_DEVICES=1,2 && bash $REPO/scripts/run_with_env.sh python $REPO/scripts/eval/run_history.py \
      --promptviews $HIST_PV --itemspecs $HIST_IS \
      --model_id allenai/OLMo-2-0325-32B-Instruct \
      --out_dir $HIST_OUT --baseline_condition control_twostage \
      --max_tokens 512 --backend vllm --tensor_parallel_size 2 \
      --gpu_memory_utilization 0.95 --max_model_len 4096 --resume"
  echo "Started OLMo-32B History in tmux window 'olmo32b-hist'"
fi

echo ""
echo "Monitor: tmux attach -t $SESSION"
tmux list-windows -t "$SESSION"
