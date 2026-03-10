#!/usr/bin/env bash
# Tmux-based experiment orchestrator for 4x H100 (~93.6GB each) + OpenRouter API
# Usage: bash scripts/run_all.sh [--smoke|--full]
# Monitor: tmux attach -t anchoring
set -euo pipefail

SESSION="anchoring"
MODE=${1:---full}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/$TIMESTAMP"
mkdir -p "$LOGDIR"
ln -sfn "$TIMESTAMP" logs/latest

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

COMMON="--promptviews datasets/anchorbench_v1/promptviews.jsonl --itemspecs datasets/anchorbench_v1/itemspecs.jsonl --seed 42"

if [[ "$MODE" == "--smoke" ]]; then
    MAX_MI=10
    MAX_MIT=10
    MECH_FLAGS=""
    MIT_STRATEGIES="B0,COUNTERFACTUAL_ENSEMBLE"
    FALLBACK_FLAGS=""
    echo "=== SMOKE TEST MODE (10 items, minimal mitigations) ==="
else
    MAX_MI=200
    MAX_MIT=600
    MECH_FLAGS="--do_patching --do_heads --top_k_layers 8"
    MIT_STRATEGIES="ALL"
    FALLBACK_FLAGS="--llm_fallback --fallback_device cpu"
    echo "=== FULL RUN MODE (200 mech / 600 mit items, all strategies) ==="
fi

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with control window
tmux new-session -d -s "$SESSION" -n control
tmux send-keys -t "$SESSION":control \
    "echo '=== Anchoring Experiments ===' && echo 'Started: $TIMESTAMP' && echo 'Mode: $MODE' && echo 'Logs: $LOGDIR/' && echo '' && echo 'Windows: mech_gpu0, mech_gpu1, mit_gpu2, api_all' && echo 'Switch: Ctrl-b w (list) | Ctrl-b n/p (next/prev)'" C-m

# ── GPU 0: Llama-1B mech -> Qwen3-4B mech -> Qwen3-4B mitigation ──
tmux new-window -t "$SESSION" -n mech_gpu0
tmux send-keys -t "$SESSION":mech_gpu0 \
    "cd $(pwd) && set -a && source .env 2>/dev/null; set +a && \
echo '>>> [GPU 0] Llama-1B mechinterp' && \
CUDA_VISIBLE_DEVICES=0 python -m src.mechinterp.cli run \
    --model_id meta-llama/Llama-3.2-1B-Instruct --device cuda:0 \
    --max_items $MAX_MI $MECH_FLAGS $COMMON \
    --out_dir results/mechinterp/ 2>&1 | tee $LOGDIR/mech_1b.log && \
echo '>>> [GPU 0] Qwen3-4B mechinterp' && \
CUDA_VISIBLE_DEVICES=0 python -m src.mechinterp.cli run \
    --model_id Qwen/Qwen3-4B-Instruct-2507-FP8 --device cuda:0 \
    --max_items $MAX_MI $MECH_FLAGS $COMMON \
    --out_dir results/mechinterp/ 2>&1 | tee $LOGDIR/mech_4b.log && \
echo '>>> [GPU 0] Qwen3-4B mitigation' && \
CUDA_VISIBLE_DEVICES=0 python -m src.mitigation_eval.cli run \
    --model_id Qwen/Qwen3-4B-Instruct-2507-FP8 --device cuda:0 \
    --backend hf --mitigation $MIT_STRATEGIES --batch_size 32 \
    --max_items $MAX_MIT $FALLBACK_FLAGS $COMMON \
    --out_dir results/mitigation/ 2>&1 | tee $LOGDIR/mit_4b_hf.log && \
echo '>>> [GPU 0] ALL DONE'" C-m

# ── GPU 1: Llama-3B mech -> Qwen3-30B-A3B mitigation ──
tmux new-window -t "$SESSION" -n mech_gpu1
tmux send-keys -t "$SESSION":mech_gpu1 \
    "cd $(pwd) && set -a && source .env 2>/dev/null; set +a && \
echo '>>> [GPU 1] Llama-3B mechinterp' && \
CUDA_VISIBLE_DEVICES=1 python -m src.mechinterp.cli run \
    --model_id meta-llama/Llama-3.2-3B-Instruct --device cuda:0 \
    --max_items $MAX_MI $MECH_FLAGS $COMMON \
    --out_dir results/mechinterp/ 2>&1 | tee $LOGDIR/mech_3b.log && \
echo '>>> [GPU 1] Qwen3-30B-A3B mitigation' && \
CUDA_VISIBLE_DEVICES=1 python -m src.mitigation_eval.cli run \
    --model_id Qwen/Qwen3-30B-A3B-Instruct-2507 --device cuda:0 \
    --backend hf --mitigation $MIT_STRATEGIES --batch_size 16 \
    --max_items $MAX_MIT $FALLBACK_FLAGS $COMMON \
    --out_dir results/mitigation/ 2>&1 | tee $LOGDIR/mit_30b_hf.log && \
echo '>>> [GPU 1] ALL DONE'" C-m

# ── GPU 2: OLMo-3.1-32B mitigation ──
tmux new-window -t "$SESSION" -n mit_gpu2
tmux send-keys -t "$SESSION":mit_gpu2 \
    "cd $(pwd) && set -a && source .env 2>/dev/null; set +a && \
echo '>>> [GPU 2] OLMo-3.1-32B mitigation' && \
CUDA_VISIBLE_DEVICES=2 python -m src.mitigation_eval.cli run \
    --model_id allenai/Olmo-3.1-32B-Instruct --device cuda:0 \
    --backend hf --mitigation $MIT_STRATEGIES --batch_size 16 \
    --max_items $MAX_MIT $FALLBACK_FLAGS $COMMON \
    --out_dir results/mitigation/ 2>&1 | tee $LOGDIR/mit_olmo32b_hf.log && \
echo '>>> [GPU 2] ALL DONE'" C-m

# ── API models (no GPU) ──
tmux new-window -t "$SESSION" -n api_all
tmux send-keys -t "$SESSION":api_all \
    "cd $(pwd) && set -a && source .env 2>/dev/null; set +a && \
echo '>>> [API] GPT-5.2' && \
python -m src.mitigation_eval.cli run \
    --backend openrouter --max_concurrent 20 \
    --model_id openai/gpt-5.2 --mitigation $MIT_STRATEGIES \
    --max_items $MAX_MIT $FALLBACK_FLAGS $COMMON \
    --out_dir results/mitigation/ 2>&1 | tee $LOGDIR/mit_gpt52.log && \
echo '>>> [API] Claude Sonnet 4.6' && \
python -m src.mitigation_eval.cli run \
    --backend openrouter --max_concurrent 20 \
    --model_id anthropic/claude-sonnet-4.6 --mitigation $MIT_STRATEGIES \
    --max_items $MAX_MIT $FALLBACK_FLAGS $COMMON \
    --out_dir results/mitigation/ 2>&1 | tee $LOGDIR/mit_claude46.log && \
echo '>>> [API] Gemini 2.5 Pro' && \
python -m src.mitigation_eval.cli run \
    --backend openrouter --max_concurrent 20 \
    --model_id google/gemini-2.5-pro --mitigation $MIT_STRATEGIES \
    --max_items $MAX_MIT $FALLBACK_FLAGS $COMMON \
    --out_dir results/mitigation/ 2>&1 | tee $LOGDIR/mit_gemini25.log && \
echo '>>> [API] ALL DONE'" C-m

# Back to control window
tmux select-window -t "$SESSION":control

echo "=========================================="
echo "Tmux session '$SESSION' created with all jobs."
echo "Attach:  tmux attach -t $SESSION"
echo "Windows: Ctrl-b w  (list)  |  Ctrl-b n/p  (next/prev)"
echo "Logs:    $LOGDIR/"
echo "=========================================="
