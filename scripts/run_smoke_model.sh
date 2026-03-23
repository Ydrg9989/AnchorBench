#!/bin/bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

MODEL_ID="$1"
GPU="$2"
MAX_ITEMS="${3:-36}"
MAX_TOKENS="${4:-512}"

export CUDA_VISIBLE_DEVICES="$GPU"
MODEL_SLUG="${MODEL_ID//\//_}"
RESULTS_ROOT="results/smoke_test"

echo "[$(date +%H:%M:%S)] Starting $MODEL_ID on GPU $GPU"

for suite in external icl rag tool history; do
    data_dir="datasets/anchorbench_${suite}_core"
    runner="scripts/eval/run_${suite}.py"

    case "$suite" in
        external|history)
            out_dir="$RESULTS_ROOT/${suite}/${MODEL_SLUG}"
            extra=()
            [ "$suite" = "history" ] && extra+=(--request_final_line)
            ;;
        icl|tool)
            out_dir="$RESULTS_ROOT/${suite}"
            extra=(--batch_size 16)
            ;;
        rag)
            out_dir="$RESULTS_ROOT/${suite}/${MODEL_SLUG}"
            extra=(--batch_size 16)
            ;;
    esac

    mkdir -p "$out_dir"

    echo "[$(date +%H:%M:%S)] GPU $GPU | $suite | $MODEL_ID"
    python "$runner" \
        --promptviews "$data_dir/promptviews.jsonl" \
        --itemspecs "$data_dir/itemspecs.jsonl" \
        --model_id "$MODEL_ID" \
        --out_dir "$out_dir" \
        --max_items "$MAX_ITEMS" \
        --max_tokens "$MAX_TOKENS" \
        --seed 42 \
        --device cuda:0 \
        "${extra[@]}" 2>&1 | tail -20

    echo "[$(date +%H:%M:%S)] GPU $GPU | $suite DONE"
    echo "---"
done

echo "[$(date +%H:%M:%S)] ALL DONE: $MODEL_ID"
