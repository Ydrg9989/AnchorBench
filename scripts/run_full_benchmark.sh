#!/usr/bin/env bash
# Full AnchorBench benchmark: 10 open models × 5 suites on 4× H100 NVL GPUs.
#
# Uses core views only (promptviews_core.jsonl, 1800 prompts/suite). Launches
# a tmux session with one window per GPU. Each GPU runs its assigned models
# sequentially (all 5 suites per model before moving to the next). Maximizes
# GPU utilization (0.95 memory, batch_size 64 where applicable).
#
# Usage:
#   bash scripts/run_full_benchmark.sh              # resume: skip suites already ≥1800 lines
#   bash scripts/run_full_benchmark.sh fresh        # archive results/full_benchmark → archive/, rerun all
#   bash scripts/run_full_benchmark.sh force        # same as fresh (alias)
#
# Monitor:
#   tmux attach -t benchmark
#   tmux select-window -t benchmark:gpu0   # switch to GPU 0
#
# Resume after interruption:
#   Just re-run — completed suites (≥1800 lines) are automatically skipped.
#
# GPU scheduling (4× H100 NVL, 95.8GB each):
#
#   GPU 0: Llama-3.2-1B → Qwen2.5-1.5B → Gemma-3-1B → recompute_metrics
#          (~1.5GB + ~3GB + ~2GB; fast models, ~1.5h total)
#
#   GPU 1: Llama-3.2-3B → Qwen2.5-3B → Gemma-3-4B
#          (~6GB + ~6GB + ~8GB; small-medium models, ~3h total)
#
#   GPU 2: Llama-3.1-8B → Qwen2.5-7B → OLMo-2-1124-13B
#          (~16GB + ~14GB + ~26GB; medium models, ~4h total)
#
#   GPU 3: OLMo-2-0325-32B
#          (~64GB; single large model, ~3h total)
set -euo pipefail

SESSION="benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/benchmark_${TIMESTAMP}"
mkdir -p "$LOGDIR"
ln -sfn "benchmark_${TIMESTAMP}" logs/latest_benchmark

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env if present (for any API keys)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Fresh run: archive previous full_benchmark, then run every suite from scratch
FORCE_RERUN=
case "${1:-}" in
    force|rerun|fresh|archive|new|--force|--rerun|--fresh) FORCE_RERUN=1 ;;
esac
if [ -n "${ANCHORBENCH_FORCE_RERUN:-}" ]; then FORCE_RERUN=1; fi
if [ -n "$FORCE_RERUN" ]; then
    ARCHIVE_ROOT="results/archive"
    ARCHIVE_DIR="${ARCHIVE_ROOT}/full_benchmark_${TIMESTAMP}"
    if [ -d results/full_benchmark ] && [ -n "$(ls -A results/full_benchmark 2>/dev/null)" ]; then
        mkdir -p "$ARCHIVE_ROOT"
        echo "Archiving: results/full_benchmark → $ARCHIVE_DIR"
        mv results/full_benchmark "$ARCHIVE_DIR"
        mkdir -p results/full_benchmark
        echo "$TIMESTAMP — archived before full re-run (run_full_benchmark.sh)" >> "${ARCHIVE_ROOT}/full_benchmark_manifest.txt"
        echo "  $ARCHIVE_DIR" >> "${ARCHIVE_ROOT}/full_benchmark_manifest.txt"
    else
        mkdir -p results/full_benchmark
    fi
    export ANCHORBENCH_FORCE_RERUN=1
    FORCE_RERUN_EXPORT="ANCHORBENCH_FORCE_RERUN=1 "
else
    FORCE_RERUN_EXPORT=""
fi

# Core views only (1800 prompts/suite).
# Dedicated 4×H100: default high mem util. Lower if you share GPUs or OOM.
export ANCHORBENCH_VIEWS=promptviews_core.jsonl
export ANCHORBENCH_BATCH_SIZE=64
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
GPU_MEM_UTIL_32B="${GPU_MEM_UTIL_32B:-0.88}"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "================================================================"
echo " AnchorBench Full Benchmark (core views)"
echo " Started: $TIMESTAMP"
echo " Logs:    $LOGDIR/"
echo " Views:   promptviews_core.jsonl (1800/suite)"
if [ -n "$FORCE_RERUN" ]; then echo " Mode:    FRESH RUN (archived prior results, all suites)"; fi
echo " Models:  10 open models × 5 suites"
echo " GPUs:    4× H100 NVL (mem util $GPU_MEM_UTIL)"
echo "================================================================"
if [ -z "$FORCE_RERUN" ] && [ -f results/full_benchmark/external/meta-llama_Llama-3.2-1B-Instruct/results.jsonl ]; then
    n=$(wc -l < results/full_benchmark/external/meta-llama_Llama-3.2-1B-Instruct/results.jsonl 2>/dev/null || echo 0)
    if [ "${n:-0}" -ge 1800 ] 2>/dev/null; then
        echo ""
        echo "  NOTE: Existing results look COMPLETE — suites will SKIP (no GPU load)."
        echo "  To re-run inference from scratch:"
        echo "    bash scripts/run_full_benchmark.sh fresh"
        echo ""
    fi
fi
echo ""
echo "Launching tmux session '$SESSION'..."
echo "  Monitor: tmux attach -t $SESSION"
echo ""

# ── Create tmux session with control window ──
tmux new-session -d -s "$SESSION" -n control
tmux send-keys -t "$SESSION":control "cat <<'BANNER'
╔══════════════════════════════════════════════════════════════╗
║  AnchorBench Full Benchmark                                 ║
║  Started: $TIMESTAMP                                        ║
║  Logs:    $LOGDIR/                                          ║
║                                                             ║
║  Windows:                                                   ║
║    gpu0 — Llama-1B, Qwen-1.5B, Gemma-1B                    ║
║    gpu1 — Llama-3B, Qwen-3B, Gemma-4B                      ║
║    gpu2 — Llama-8B, Qwen-7B, OLMo-13B                      ║
║    gpu3 — OLMo-32B                                          ║
║                                                             ║
║  Navigate: Ctrl-b w (list) | Ctrl-b n/p (next/prev)        ║
║  Detach:   Ctrl-b d                                         ║
╚══════════════════════════════════════════════════════════════╝
BANNER" C-m

# Common vLLM settings (max GPU util)
MAX_TOKENS=512
MAX_MODEL_LEN=4096

# ── GPU 0: Small models (1B-1.5B) ──
tmux new-window -t "$SESSION" -n gpu0
tmux send-keys -t "$SESSION":gpu0 "export ${FORCE_RERUN_EXPORT}ANCHORBENCH_VIEWS=promptviews_core.jsonl ANCHORBENCH_BATCH_SIZE=64 GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_TOKENS=512 MAX_MODEL_LEN=4096 LOGDIR=$LOGDIR PROJECT_ROOT=$PROJECT_ROOT && cd \$PROJECT_ROOT && \\
echo '━━━ GPU 0: Small models ━━━' && \\
echo '' && \\
echo '>>> [1/3] Llama-3.2-1B-Instruct' && \\
bash scripts/run_model_vllm.sh meta-llama/Llama-3.2-1B-Instruct 0 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 \\
    2>&1 | tee $LOGDIR/gpu0_llama_1b.log && \\
echo '' && \\
echo '>>> [2/3] Qwen2.5-1.5B-Instruct' && \\
bash scripts/run_model_vllm.sh Qwen/Qwen2.5-1.5B-Instruct 0 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 \\
    2>&1 | tee $LOGDIR/gpu0_qwen_1.5b.log && \\
echo '' && \\
echo '>>> [3/3] Gemma-3-1B-it' && \\
bash scripts/run_model_vllm.sh google/gemma-3-1b-it 0 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 \\
    2>&1 | tee $LOGDIR/gpu0_gemma_1b.log && \\
echo '' && \\
echo '━━━ GPU 0: Recomputing unified metrics ━━━' && \\
bash scripts/run_with_env.sh python scripts/eval/recompute_all_unified.py \\
    --results_dir results/full_benchmark 2>&1 | tee $LOGDIR/recompute_metrics.log && \\
echo '' && \\
echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' && \\
echo '  GPU 0: ALL DONE' && \\
echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'" C-m

# ── GPU 1: Small-medium models (3B-4B) ──
tmux new-window -t "$SESSION" -n gpu1
tmux send-keys -t "$SESSION":gpu1 "export ${FORCE_RERUN_EXPORT}ANCHORBENCH_VIEWS=promptviews_core.jsonl ANCHORBENCH_BATCH_SIZE=64 GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_TOKENS=512 MAX_MODEL_LEN=4096 LOGDIR=$LOGDIR PROJECT_ROOT=$PROJECT_ROOT && cd \$PROJECT_ROOT && \\
echo '━━━ GPU 1: Small-medium models ━━━' && \\
echo '' && \\
echo '>>> [1/3] Llama-3.2-3B-Instruct' && \\
bash scripts/run_model_vllm.sh meta-llama/Llama-3.2-3B-Instruct 1 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 \\
    2>&1 | tee $LOGDIR/gpu1_llama_3b.log && \\
echo '' && \\
echo '>>> [2/3] Qwen2.5-3B-Instruct' && \\
bash scripts/run_model_vllm.sh Qwen/Qwen2.5-3B-Instruct 1 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 \\
    2>&1 | tee $LOGDIR/gpu1_qwen_3b.log && \\
echo '' && \\
echo '>>> [3/3] Gemma-3-4B-it' && \\
bash scripts/run_model_vllm.sh google/gemma-3-4b-it 1 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 \\
    2>&1 | tee $LOGDIR/gpu1_gemma_4b.log && \\
echo '' && \\
echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' && \\
echo '  GPU 1: ALL DONE' && \\
echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'" C-m

# ── GPU 2: Medium models (7B-13B) ──
tmux new-window -t "$SESSION" -n gpu2
tmux send-keys -t "$SESSION":gpu2 "export ${FORCE_RERUN_EXPORT}ANCHORBENCH_VIEWS=promptviews_core.jsonl ANCHORBENCH_BATCH_SIZE=64 GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_TOKENS=512 MAX_MODEL_LEN=4096 LOGDIR=$LOGDIR PROJECT_ROOT=$PROJECT_ROOT && cd \$PROJECT_ROOT && \\
echo '━━━ GPU 2: Medium models ━━━' && \\
echo '' && \\
echo '>>> [1/3] Llama-3.1-8B-Instruct' && \\
bash scripts/run_model_vllm.sh meta-llama/Llama-3.1-8B-Instruct 2 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 \\
    2>&1 | tee $LOGDIR/gpu2_llama_8b.log && \\
echo '' && \\
echo '>>> [2/3] Qwen2.5-7B-Instruct' && \\
bash scripts/run_model_vllm.sh Qwen/Qwen2.5-7B-Instruct 2 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 \\
    2>&1 | tee $LOGDIR/gpu2_qwen_7b.log && \\
echo '' && \\
echo '>>> [3/3] OLMo-2-1124-13B-Instruct' && \\
bash scripts/run_model_vllm.sh allenai/OLMo-2-1124-13B-Instruct 2 $MAX_TOKENS $GPU_MEM_UTIL $MAX_MODEL_LEN 1 \\
    2>&1 | tee $LOGDIR/gpu2_olmo_13b.log && \\
echo '' && \\
echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' && \\
echo '  GPU 2: ALL DONE' && \\
echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'" C-m

# ── GPU 3: Large model (32B) ──
tmux new-window -t "$SESSION" -n gpu3
tmux send-keys -t "$SESSION":gpu3 "export ${FORCE_RERUN_EXPORT}ANCHORBENCH_VIEWS=promptviews_core.jsonl ANCHORBENCH_BATCH_SIZE=64 GPU_MEM_UTIL=$GPU_MEM_UTIL_32B MAX_TOKENS=512 MAX_MODEL_LEN=4096 LOGDIR=$LOGDIR PROJECT_ROOT=$PROJECT_ROOT && cd \$PROJECT_ROOT && \\
echo '━━━ GPU 3: Large model (mem util $GPU_MEM_UTIL_32B) ━━━' && \\
echo '' && \\
echo '>>> [1/1] OLMo-2-0325-32B-Instruct' && \\
bash scripts/run_model_vllm.sh allenai/OLMo-2-0325-32B-Instruct 3 \$MAX_TOKENS \$GPU_MEM_UTIL 4096 1 \\
    2>&1 | tee $LOGDIR/gpu3_olmo_32b.log && \\
echo '' && \\
echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' && \\
echo '  GPU 3: ALL DONE' && \\
echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'" C-m

echo ""
echo "Tmux session '$SESSION' launched with 5 windows:"
echo "  control  — status overview"
echo "  gpu0     — Llama-1B, Qwen-1.5B, Gemma-1B  (+ metrics recompute)"
echo "  gpu1     — Llama-3B, Qwen-3B, Gemma-4B"
echo "  gpu2     — Llama-8B, Qwen-7B, OLMo-13B"
echo "  gpu3     — OLMo-32B"
echo ""
echo "Attach with:  tmux attach -t $SESSION"
echo "Logs at:      $LOGDIR/"
echo ""
