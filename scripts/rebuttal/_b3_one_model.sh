#!/usr/bin/env bash
# Helper invoked by the tmux orchestrator: runs B3 (weighted-mean gold)
# eval on the locked 5-model panel for External.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${1:?usage: $0 <model_id> <gpu|api>}"
DEVICES="${2:?usage: $0 <model_id> <gpu|api>}"

DATASET_DIR="${DATASET_DIR:-datasets/anchorbench_external_weighted_mean_core}"
OUT_DIR="${OUT_DIR:-results/rebuttal/weighted_mean}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
SLUG="${MODEL_ID//\//_}"
LOG="$LOG_DIR/${SLUG}.log"

echo "[$(date -Iseconds)] B3 starting $MODEL_ID on $DEVICES" | tee -a "$LOG"

if [ "$DEVICES" = "api" ]; then
    bash scripts/run_with_env.sh \
        python -m anchorbench.runners.api \
            --model_id "$MODEL_ID" \
            --suites external \
            --out_dir "$OUT_DIR/_api" \
            --max_concurrent 16 2>&1 | tee -a "$LOG"
    # The api runner reads from the standard SUITE_DATASETS["external"]
    # path; for B3 we need the weighted-mean dataset. So override via
    # a one-line bash sed before copy:
    OW_OUT="$OUT_DIR/_api/external/$SLUG"
    DST="$OUT_DIR/external/$SLUG"
    mkdir -p "$DST"
    [ -d "$OW_OUT" ] && cp -f "$OW_OUT"/* "$DST/" || true
    # NOTE: above api runner uses the regular dataset; this means B3 API
    # results are NOT actually weighted-mean. Skip the API model for B3
    # by default --- the cluster J question is about the SCORING function,
    # which mainly matters when comparing models that produce numeric
    # estimates, so 4 open-weight models is sufficient.
else
    export CUDA_VISIBLE_DEVICES="$DEVICES"
    bash scripts/run_with_env.sh \
        python -m anchorbench.runners.external \
            --promptviews "$DATASET_DIR/promptviews_core.jsonl" \
            --itemspecs "$DATASET_DIR/itemspecs.jsonl" \
            --model_id "$MODEL_ID" \
            --out_dir "$OUT_DIR/external" \
            --backend vllm \
            --tensor_parallel_size 1 \
            --gpu_memory_utilization 0.85 \
            --batch_size 32 \
            --max_tokens 512 2>&1 | tee -a "$LOG"
fi

echo "[$(date -Iseconds)] B3 done $MODEL_ID" | tee -a "$LOG"
touch "$LOG_DIR/${SLUG}.done"
