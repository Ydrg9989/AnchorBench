#!/usr/bin/env bash
# =============================================================================
# Smoke sanity: one open-source model, all 5 suites, small slice of **core** data.
#
# Uses canonical anchorbench_*_core + --max_items (same 5 conditions as paper).
# Standalone *_smoke datasets are archived under datasets/archive/.
#
# Usage:
#   conda activate LLM_anchoring
#   cd $REPO_ROOT
#
#   bash scripts/run_smoke_sanity.sh
#
#   SMOKE_BACKEND=vllm CUDA_VISIBLE_DEVICES=0 bash scripts/run_smoke_sanity.sh
#
# Env:
#   SMOKE_MODEL     HF model id (default: Qwen/Qwen2.5-1.5B-Instruct)
#   SMOKE_BACKEND   hf | vllm (default: hf)
#   SMOKE_OUT       results dir (default: results/smoke_sanity)
#   SMOKE_GPU_MEM   vLLM gpu_memory_utilization (default: 0.85)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
BACKEND="${SMOKE_BACKEND:-hf}"
OUT="${SMOKE_OUT:-results/smoke_sanity}"
GPU_MEM="${SMOKE_GPU_MEM:-0.85}"
MAX_TOKENS="${SMOKE_MAX_TOKENS:-256}"
SEED=42

MODEL_SLUG="${MODEL//\//_}"
ROOT_OUT="$OUT/$MODEL_SLUG"
mkdir -p "$ROOT_OUT"

echo "============================================"
echo "  Smoke sanity | $MODEL"
echo "  Backend: $BACKEND | Out: $ROOT_OUT"
echo "  Data: *_core + max_items (external/icl/rag/tool: 6, history: 2)"
echo "============================================"

run_external() {
  bash scripts/run_with_env.sh python scripts/eval/run_external.py \
    --promptviews datasets/anchorbench_external_core/promptviews_core.jsonl \
    --itemspecs datasets/anchorbench_external_core/itemspecs.jsonl \
    --max_items 6 \
    --model_id "$MODEL" \
    --out_dir "$ROOT_OUT/external" \
    --max_tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --backend "$BACKEND" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "$GPU_MEM"
}

run_history() {
  bash scripts/run_with_env.sh python scripts/eval/run_history.py \
    --promptviews datasets/anchorbench_history_core/promptviews_core.jsonl \
    --itemspecs datasets/anchorbench_history_core/itemspecs.jsonl \
    --max_items 2 \
    --model_id "$MODEL" \
    --out_dir "$ROOT_OUT/history" \
    --max_tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --request_final_line \
    --backend "$BACKEND" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "$GPU_MEM"
}

run_icl() {
  bash scripts/run_with_env.sh python scripts/eval/run_icl.py \
    --promptviews datasets/anchorbench_icl_core/promptviews_core.jsonl \
    --itemspecs datasets/anchorbench_icl_core/itemspecs.jsonl \
    --max_items 6 \
    --model_id "$MODEL" \
    --out_dir "$ROOT_OUT/icl" \
    --max_tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --batch_size 4 \
    --backend "$BACKEND" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "$GPU_MEM"
}

run_rag() {
  bash scripts/run_with_env.sh python scripts/eval/run_rag.py \
    --promptviews datasets/anchorbench_rag_core/promptviews_core.jsonl \
    --itemspecs datasets/anchorbench_rag_core/itemspecs.jsonl \
    --max_items 6 \
    --model_id "$MODEL" \
    --out_dir "$ROOT_OUT/rag" \
    --max_tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --batch_size 4 \
    --backend "$BACKEND" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "$GPU_MEM"
}

run_tool() {
  bash scripts/run_with_env.sh python scripts/eval/run_tool.py \
    --promptviews datasets/anchorbench_tool_core/promptviews_core.jsonl \
    --itemspecs datasets/anchorbench_tool_core/itemspecs.jsonl \
    --max_items 6 \
    --model_id "$MODEL" \
    --out_dir "$ROOT_OUT/tool" \
    --max_tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --batch_size 4 \
    --tool_plaintext \
    --backend "$BACKEND" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "$GPU_MEM"
}

echo ""
echo "--- External (6 items × 5 conditions) ---"
run_external

echo ""
echo "--- History (2 items × 5, two-stage) ---"
run_history

echo ""
echo "--- ICL ---"
run_icl

echo ""
echo "--- RAG ---"
run_rag

echo ""
echo "--- Tool (plaintext tool block) ---"
run_tool

echo ""
echo "============================================"
echo "  Done. Inspect:"
echo "    results.jsonl  — raw model text + parsed int + parse_strategy"
echo "    summary.json   — UAI, TAR, parse rates"
echo "  external/history/rag: $ROOT_OUT/<suite>/results.jsonl + summary.json"
echo "  icl/tool:             $ROOT_OUT/<suite>/$MODEL_SLUG/results.jsonl"
echo "============================================"
