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
# Designed to run unattended in tmux. It waits for the GPUs to be free before
# starting, so it can be launched while someone else's job is still running.
#
# Usage:
#   bash scripts/run_stage3_reruns.sh              # wait for GPUs, then run
#   NO_WAIT=1 bash scripts/run_stage3_reruns.sh    # start immediately
#   MIN_FREE_GB=80 bash scripts/run_stage3_reruns.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs/stage3"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run.log"

# OLMo-32B needs tensor-parallel across all four cards, so every GPU has to be
# genuinely free -- not just the one a small model would land on.
MIN_FREE_GB="${MIN_FREE_GB:-85}"
STABLE_CHECKS="${STABLE_CHECKS:-3}"     # consecutive passes before starting
POLL_SECONDS="${POLL_SECONDS:-300}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MAIN_LOG"; }

gpus_free() {
    local min_free_mib=$((MIN_FREE_GB * 1024))
    local free
    while read -r free; do
        [ "$free" -lt "$min_free_mib" ] && return 1
    done < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    return 0
}

wait_for_gpus() {
    local stable=0
    log "waiting for all GPUs to have >= ${MIN_FREE_GB} GiB free"
    log "(needs ${STABLE_CHECKS} consecutive passes ${POLL_SECONDS}s apart, so a"
    log " brief dip in someone else's job does not trigger a start)"
    while :; do
        if gpus_free; then
            stable=$((stable + 1))
            log "  GPUs free (${stable}/${STABLE_CHECKS})"
            [ "$stable" -ge "$STABLE_CHECKS" ] && return 0
        else
            [ "$stable" -gt 0 ] && log "  GPUs busy again, resetting"
            stable=0
            log "  busy: $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' ')"
        fi
        sleep "$POLL_SECONDS"
    done
}

run_experiment() {                        # run_experiment <recipe>
    local name="$1"
    log "=== START $name ==="
    bash scripts/run_with_env.sh python -m anchorbench.cli.experiment \
        "+experiment=$name" 2>&1 | tee -a "$LOG_DIR/$name.log"
    local rc=${PIPESTATUS[0]}
    log "=== END $name (rc=$rc) ==="
    return $rc
}

log "Stage 3 re-runs starting; pid $$"
log "host $(hostname), repo $ROOT"

if [ "${NO_WAIT:-0}" != "1" ]; then
    wait_for_gpus
    log "GPUs are free; starting"
else
    log "NO_WAIT=1, skipping the GPU wait"
fi

rc_total=0
# History first: OLMo-32B needs all four cards, so run it while the machine is
# known-quiet rather than after an hour of Tool work.
run_experiment paper_history_matched || rc_total=1
run_experiment paper_tool_plaintext  || rc_total=1

log "regenerating the two appendix tables from the new runs"
bash scripts/run_with_env.sh python -m anchorbench.paper.tables_appendix \
    --history_matched "$ROOT/results/history_matched" \
    --tool_plaintext  "$ROOT/results/tool_plaintext" \
    2>&1 | tee -a "$LOG_DIR/tables.log"

log "checksumming the new results"
bash scripts/checksum_results.sh 2>&1 | tee -a "$MAIN_LOG"

log "DONE (rc=$rc_total)"
log "Next: review outputs/tables/tab_history_matched.tex and"
log "      outputs/tables/tab_tool_plaintext.tex, then write the addendum."
log "Do NOT paste these over the published numbers -- they are an addendum."
exit $rc_total
