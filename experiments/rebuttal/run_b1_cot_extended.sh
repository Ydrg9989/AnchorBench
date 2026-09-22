#!/usr/bin/env bash
# Tier B1 — CoT (reasoning-allowed) extension on the locked 5-model panel.
#
# Does anchoring persist when models are allowed to
# reason before answering?
#
# Runs four open-weight models locally (via vLLM) and one API model
# (GPT-5.4-mini via OpenRouter) on three suites (External, RAG, History)
# under both the baseline and the CoT prompt suffix.
#
# Expected wall-time on 4xH200: ~4-6h total when run serially; can shard
# models across GPUs by launching multiple instances with different
# tensor_parallel_size / CUDA_VISIBLE_DEVICES.
#
# Outputs end up in results/rebuttal/cot_extended/<suite>/<slug>/<strategy>/.
# After this script completes, run:
#   PYTHONPATH=src python -m anchorbench.analysis.cot_reasoning \
#       --input results/rebuttal/cot_extended/rebuttal_cot_comparison.json \
#       --out_dir results/rebuttal/cot_reasoning_extended

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-results/rebuttal/cot_extended}"
SUITES=("external" "rag" "history")
STRATEGIES=("baseline" "cot")

OW_MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    "allenai/OLMo-2-1124-13B-Instruct"
)

API_MODEL="${API_MODEL:-openai/gpt-5.4-mini}"

COT_SUFFIX=$'\n\nThink step by step. List the relevant evidence, compute your estimate from that evidence only, then provide your final numeric answer on the last line.'

echo "=== Tier B1: CoT extension ==="
echo "  out_dir:    $OUT_DIR"
echo "  suites:     ${SUITES[*]}"
echo "  strategies: ${STRATEGIES[*]}"
echo "  ow models:  ${OW_MODELS[*]}"
echo "  api model:  $API_MODEL"
echo

# -- Open-weight models (vLLM) -----------------------------------------------
for MODEL in "${OW_MODELS[@]}"; do
    echo "==== [B1/OW] $MODEL ===="
    bash scripts/run_with_env.sh \
        python -m anchorbench.runners.rebuttal_cot \
            --model_id "$MODEL" \
            --backend vllm \
            --suites "${SUITES[@]}" \
            --strategies "${STRATEGIES[@]}" \
            --out_dir "$OUT_DIR" \
            --batch_size 32 \
            --max_tokens 768
done

# -- API model (OpenRouter) --------------------------------------------------
# The API runner writes to <out_dir>/<suite>/<slug>/results.jsonl directly
# (no strategy subdir), so we invoke it twice: once for baseline (no suffix),
# once for cot (with suffix), each into a strategy-subdir.
for STRAT in "${STRATEGIES[@]}"; do
    if [ "$STRAT" = "cot" ]; then
        SUFFIX="$COT_SUFFIX"
    else
        SUFFIX=""
    fi
    SLUG="${API_MODEL//\//_}"
    STRAT_OUT="$OUT_DIR/_api_${STRAT}"
    echo "==== [B1/API] $API_MODEL (strategy=$STRAT) ===="
    bash scripts/run_with_env.sh \
        python -m anchorbench.runners.api \
            --model_id "$API_MODEL" \
            --suites "${SUITES[@]}" \
            --out_dir "$STRAT_OUT" \
            --prompt_suffix "$SUFFIX" \
            --max_concurrent 16
    # Reshape into <suite>/<slug>/<strategy>/ for the analyzer
    for SUITE in "${SUITES[@]}"; do
        SRC="$STRAT_OUT/$SUITE/$SLUG"
        DST="$OUT_DIR/$SUITE/$SLUG/$STRAT"
        mkdir -p "$DST"
        if [ -d "$SRC" ]; then
            cp -n "$SRC/results.jsonl" "$DST/results.jsonl" || true
            cp -n "$SRC/summary.json" "$DST/summary.json" || true
        fi
    done
done

# -- Aggregate comparison ----------------------------------------------------
echo "==== [B1] Aggregating comparison ===="
bash scripts/run_with_env.sh \
    python -c "
import json
from pathlib import Path
from anchorbench.eval.constants import MODEL_SHORT
from anchorbench.eval.io import load_records
from anchorbench.eval.metrics import compute_unified_metrics

out_dir = Path('$OUT_DIR')
metric_keys = ['uai_irr','uai_plaus','disc_delta','mae_control','acc10_control',
               'tar_irr','tar_plaus','parse_rate']
rows = []
for suite_dir in sorted(out_dir.iterdir()):
    if not suite_dir.is_dir() or suite_dir.name.startswith('_'):
        continue
    suite = suite_dir.name
    for model_dir in sorted(suite_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for strat_dir in sorted(model_dir.iterdir()):
            if not strat_dir.is_dir():
                continue
            rj = strat_dir / 'results.jsonl'
            if not rj.exists():
                continue
            recs = load_records(str(rj))
            baseline = 'control_twostage' if suite == 'history' else 'control'
            m = compute_unified_metrics(recs, baseline_condition=baseline)
            r = {'suite':suite, 'model': MODEL_SHORT.get(model_dir.name, model_dir.name),
                 'model_slug': model_dir.name, 'strategy': strat_dir.name}
            for k in metric_keys:
                r[k] = m.get(k)
            rows.append(r)
(out_dir/'rebuttal_cot_comparison.json').write_text(json.dumps(rows, indent=2))
print(f'Wrote {out_dir}/rebuttal_cot_comparison.json with {len(rows)} rows')
"

# -- Run analyzer ------------------------------------------------------------
echo "==== [B1] Running CoT analyzer ===="
bash scripts/run_with_env.sh \
    python -m anchorbench.analysis.cot_reasoning \
        --input "$OUT_DIR/rebuttal_cot_comparison.json" \
        --out_dir results/rebuttal/cot_reasoning_extended

echo
echo "Tier B1 done. See results/rebuttal/cot_reasoning_extended/interpretation.md"
