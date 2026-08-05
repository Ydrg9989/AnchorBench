#!/usr/bin/env bash
# Stage 3: re-run the two experiments whose inputs were never preserved.
#
#   tab:history_matched   10 open-weight models, History two-stage
#   tab:tool_plaintext     5 models, Tool suite forced to plaintext
#
# Both are recorded as D5 in docs/RECONCILIATION.md. Their numbers will NOT
# match the published ones exactly -- vLLM is not bitwise reproducible at
# temperature 0 (see results/rebuttal/cot_replication/README.md) -- so they
# are published as a versioned addendum beside the frozen PDF values, never
# substituted for them.
#
# Runs on a chosen pair of GPUs, saturating both. `anchorbench experiment`
# executes its cells sequentially, so driving both cards means running two
# processes with disjoint model lists rather than relying on the tier
# gpu_ids, which only pin CUDA_VISIBLE_DEVICES per cell.
#
# Usage:
#   bash scripts/run_stage3_reruns.sh                 # GPUs 2,3, both experiments
#   PHASES=history bash scripts/run_stage3_reruns.sh  # history only
#   PHASES=tool    bash scripts/run_stage3_reruns.sh  # tool only
#   GPU_A=0 GPU_B=1 bash scripts/run_stage3_reruns.sh
#   WAIT=1 bash scripts/run_stage3_reruns.sh          # wait for them to free up
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PHASES="${PHASES:-all}"      # all | history | tool
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-85}"
POLL_SECONDS="${POLL_SECONDS:-300}"

LOG_DIR="$ROOT/logs/stage3"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MAIN_LOG"; }

free_gib() {  # free_gib <index>
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" \
        | awk '{print int($1/1024)}'
}

wait_for_pair() {
    local stable=0
    log "waiting for GPUs $GPU_A and $GPU_B to have >= ${MIN_FREE_GB} GiB free"
    while :; do
        local a b
        a=$(free_gib "$GPU_A"); b=$(free_gib "$GPU_B")
        if [ "$a" -ge "$MIN_FREE_GB" ] && [ "$b" -ge "$MIN_FREE_GB" ]; then
            stable=$((stable + 1))
            log "  free (${stable}/3): GPU$GPU_A=${a}GiB GPU$GPU_B=${b}GiB"
            [ "$stable" -ge 3 ] && return 0
        else
            [ "$stable" -gt 0 ] && log "  busy again, resetting"
            stable=0
            log "  busy: GPU$GPU_A=${a}GiB GPU$GPU_B=${b}GiB"
        fi
        sleep "$POLL_SECONDS"
    done
}

# run_stream <gpus> <logtag> <recipe> <model,list>
run_stream() {
    local gpus="$1" tag="$2" recipe="$3" models="$4"
    CUDA_VISIBLE_DEVICES="$gpus" bash scripts/run_with_env.sh \
        python -m anchorbench.cli.experiment "+experiment=$recipe" \
        "++models=[$models]" >>"$LOG_DIR/$tag.log" 2>&1
    local rc=$?
    log "  stream $tag finished (rc=$rc)"
    return $rc
}

log "Stage 3 starting; pid $$; GPUs $GPU_A and $GPU_B"
log "GPU$GPU_A free $(free_gib "$GPU_A") GiB, GPU$GPU_B free $(free_gib "$GPU_B") GiB"

[ "${WAIT:-0}" = "1" ] && wait_for_pair

rc_total=0

if [ "$PHASES" = "all" ] || [ "$PHASES" = "history" ]; then
# --- Phase 1: OLMo-32B needs both cards (TP=2), so it runs alone -----------
log "=== phase 1: history / OLMo-32B on GPUs $GPU_A,$GPU_B (TP=2) ==="
run_stream "$GPU_A,$GPU_B" "history_olmo32b" paper_history_matched "olmo_32b" || rc_total=1

# --- Phase 2: the other 9 history models, two streams, one per card -------
# Split to balance total parameters (~19.5B vs ~22B) rather than model count.
log "=== phase 2: history / 9 models, two parallel streams ==="
run_stream "$GPU_A" "history_a" paper_history_matched \
    "olmo_13b,gemma_4b,qwen_1_5b,llama_1b" &
PID_A=$!
run_stream "$GPU_B" "history_b" paper_history_matched \
    "llama_8b,qwen_7b,llama_3b,qwen_3b,gemma_1b" &
PID_B=$!
wait $PID_A || rc_total=1
wait $PID_B || rc_total=1
fi

if [ "$PHASES" = "all" ] || [ "$PHASES" = "tool" ]; then
# --- Phase 3: tool plaintext, five models, two streams --------------------
log "=== phase 3: tool_plaintext / 5 models, two parallel streams ==="
run_stream "$GPU_A" "tool_a" paper_tool_plaintext "qwen_7b,llama_3b,qwen_1_5b" &
PID_A=$!
run_stream "$GPU_B" "tool_b" paper_tool_plaintext "llama_8b,qwen_3b" &
PID_B=$!
wait $PID_A || rc_total=1
wait $PID_B || rc_total=1
fi

log "regenerating the two appendix tables from the new runs"
bash scripts/run_with_env.sh python -m anchorbench.paper.tables_appendix \
    --history_matched "$ROOT/results/history_matched" \
    --tool_plaintext  "$ROOT/results/tool_plaintext" \
    >>"$LOG_DIR/tables.log" 2>&1

log "checksumming the new results"
bash scripts/checksum_results.sh >>"$MAIN_LOG" 2>&1

log "DONE (rc=$rc_total)"
log "Review outputs/tables/tab_history_matched.tex and tab_tool_plaintext.tex,"
log "then write the addendum. These are an ADDENDUM -- do not paste them over"
log "the published numbers."
exit $rc_total
