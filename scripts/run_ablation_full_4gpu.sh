#!/usr/bin/env bash
# AnchorBench ablation benchmark: same 10 open models × 4 suites (external, icl, rag, history)
# on 4 GPUs via tmux. Uses promptviews_ablation.jsonl from *_core datasets.
#
# Usage:
#   bash scripts/run_ablation_full_4gpu.sh           # resume (skip complete suites)
#   bash scripts/run_ablation_full_4gpu.sh fresh     # archive old results, rerun all
#
# Monitor:  tmux attach -t ablation_benchmark
#
# Output:   results/ablation_full_benchmark/
# Archive:  results/archive/ablation_full_benchmark_<timestamp>/
set -euo pipefail

SESSION="ablation_benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/ablation_${TIMESTAMP}"
mkdir -p "$LOGDIR"
ln -sfn "ablation_${TIMESTAMP}" logs/latest_ablation

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

FORCE_RERUN=
case "${1:-}" in
    force|rerun|fresh|archive|new|--force|--rerun|--fresh) FORCE_RERUN=1 ;;
esac
if [ -n "${ANCHORBENCH_FORCE_RERUN:-}" ]; then FORCE_RERUN=1; fi
if [ -n "$FORCE_RERUN" ]; then
    ARCHIVE_ROOT="results/archive"
    ARCHIVE_DIR="${ARCHIVE_ROOT}/ablation_full_benchmark_${TIMESTAMP}"
    if [ -d results/ablation_full_benchmark ] && [ -n "$(ls -A results/ablation_full_benchmark 2>/dev/null)" ]; then
        mkdir -p "$ARCHIVE_ROOT"
        echo "Archiving: results/ablation_full_benchmark → $ARCHIVE_DIR"
        mv results/ablation_full_benchmark "$ARCHIVE_DIR"
        mkdir -p results/ablation_full_benchmark
        echo "$TIMESTAMP — archived before ablation re-run" >> "${ARCHIVE_ROOT}/ablation_benchmark_manifest.txt"
        echo "  $ARCHIVE_DIR" >> "${ARCHIVE_ROOT}/ablation_benchmark_manifest.txt"
    else
        mkdir -p results/ablation_full_benchmark
    fi
    export ANCHORBENCH_FORCE_RERUN=1
    FORCE_RERUN_EXPORT="ANCHORBENCH_FORCE_RERUN=1 "
else
    FORCE_RERUN_EXPORT=""
fi

export ANCHORBENCH_VIEWS=promptviews_ablation.jsonl
export ANCHORBENCH_BATCH_SIZE=64
export ANCHORBENCH_RESULTS_ROOT=results/ablation_full_benchmark
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
GPU_MEM_UTIL_32B="${GPU_MEM_UTIL_32B:-0.88}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "================================================================"
echo " AnchorBench Ablation Benchmark"
echo " Started: $TIMESTAMP"
echo " Logs:    $LOGDIR/"
echo " Views:   promptviews_ablation.jsonl"
echo " Suites:  external, icl, rag, history (no tool ablation in pipeline)"
echo " Output:  results/ablation_full_benchmark/"
if [ -n "$FORCE_RERUN" ]; then echo " Mode:    FRESH (archived prior ablation dir if any)"; fi
echo " GPUs:    4 (mem util $GPU_MEM_UTIL, 32B: $GPU_MEM_UTIL_32B)"
echo "================================================================"

if [ -z "$FORCE_RERUN" ] && [ -d results/ablation_full_benchmark/external ] && [ -n "$(ls -A results/ablation_full_benchmark/external 2>/dev/null)" ]; then
    echo ""
    echo "  NOTE: Ablation results already exist — completed suites SKIP (no GPU)."
    echo "  Full re-run from scratch:"
    echo "    bash scripts/run_ablation_full_4gpu.sh fresh"
    echo ""
fi

tmux new-session -d -s "$SESSION" -n control
tmux send-keys -t "$SESSION":control "echo 'Ablation benchmark | $TIMESTAMP'; echo 'Logs: $LOGDIR'; echo 'tmux attach -t $SESSION'" C-m

tmux new-window -t "$SESSION" -n gpu0
tmux send-keys -t "$SESSION":gpu0 "export ${FORCE_RERUN_EXPORT}ANCHORBENCH_VIEWS=promptviews_ablation.jsonl ANCHORBENCH_BATCH_SIZE=64 ANCHORBENCH_RESULTS_ROOT=results/ablation_full_benchmark GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_TOKENS=$MAX_TOKENS MAX_MODEL_LEN=$MAX_MODEL_LEN PROJECT_ROOT=$PROJECT_ROOT && cd \$PROJECT_ROOT && \
echo 'GPU0: Llama-1B → Qwen-1.5B → Gemma-1B' && \
bash scripts/run_model_vllm_ablation.sh meta-llama/Llama-3.2-1B-Instruct 0 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu0_llama_1b.log && \
bash scripts/run_model_vllm_ablation.sh Qwen/Qwen2.5-1.5B-Instruct 0 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu0_qwen_1.5b.log && \
bash scripts/run_model_vllm_ablation.sh google/gemma-3-1b-it 0 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu0_gemma_1b.log && \
echo 'GPU0 DONE'" C-m

tmux new-window -t "$SESSION" -n gpu1
tmux send-keys -t "$SESSION":gpu1 "export ${FORCE_RERUN_EXPORT}ANCHORBENCH_VIEWS=promptviews_ablation.jsonl ANCHORBENCH_BATCH_SIZE=64 ANCHORBENCH_RESULTS_ROOT=results/ablation_full_benchmark GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_TOKENS=$MAX_TOKENS MAX_MODEL_LEN=$MAX_MODEL_LEN PROJECT_ROOT=$PROJECT_ROOT && cd \$PROJECT_ROOT && \
echo 'GPU1: Llama-3B → Qwen-3B → Gemma-4B' && \
bash scripts/run_model_vllm_ablation.sh meta-llama/Llama-3.2-3B-Instruct 1 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu1_llama_3b.log && \
bash scripts/run_model_vllm_ablation.sh Qwen/Qwen2.5-3B-Instruct 1 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu1_qwen_3b.log && \
bash scripts/run_model_vllm_ablation.sh google/gemma-3-4b-it 1 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu1_gemma_4b.log && \
echo 'GPU1 DONE'" C-m

tmux new-window -t "$SESSION" -n gpu2
tmux send-keys -t "$SESSION":gpu2 "export ${FORCE_RERUN_EXPORT}ANCHORBENCH_VIEWS=promptviews_ablation.jsonl ANCHORBENCH_BATCH_SIZE=64 ANCHORBENCH_RESULTS_ROOT=results/ablation_full_benchmark GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_TOKENS=$MAX_TOKENS MAX_MODEL_LEN=$MAX_MODEL_LEN PROJECT_ROOT=$PROJECT_ROOT && cd \$PROJECT_ROOT && \
echo 'GPU2: Llama-8B → Qwen-7B → OLMo-13B' && \
bash scripts/run_model_vllm_ablation.sh meta-llama/Llama-3.1-8B-Instruct 2 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu2_llama_8b.log && \
bash scripts/run_model_vllm_ablation.sh Qwen/Qwen2.5-7B-Instruct 2 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu2_qwen_7b.log && \
bash scripts/run_model_vllm_ablation.sh allenai/OLMo-2-1124-13B-Instruct 2 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu2_olmo_13b.log && \
echo 'GPU2 DONE'" C-m

tmux new-window -t "$SESSION" -n gpu3
tmux send-keys -t "$SESSION":gpu3 "export ${FORCE_RERUN_EXPORT}ANCHORBENCH_VIEWS=promptviews_ablation.jsonl ANCHORBENCH_BATCH_SIZE=64 ANCHORBENCH_RESULTS_ROOT=results/ablation_full_benchmark GPU_MEM_UTIL=$GPU_MEM_UTIL_32B MAX_TOKENS=$MAX_TOKENS MAX_MODEL_LEN=$MAX_MODEL_LEN PROJECT_ROOT=$PROJECT_ROOT && cd \$PROJECT_ROOT && \
echo 'GPU3: OLMo-32B' && \
bash scripts/run_model_vllm_ablation.sh allenai/OLMo-2-0325-32B-Instruct 3 $MAX_TOKENS $GPU_MEM_UTIL_32B $MAX_MODEL_LEN 1 2>&1 | tee $LOGDIR/gpu3_olmo_32b.log && \
echo 'GPU3 DONE'" C-m

echo ""
echo "tmux session: $SESSION"
echo "Attach:  tmux attach -t $SESSION"
echo "Logs:    $LOGDIR/"
echo "Results: results/ablation_full_benchmark/"
