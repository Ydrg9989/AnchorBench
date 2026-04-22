# Reproducibility Guide

## Requirements

- Python 3.10+
- CUDA-capable GPU (for model evaluation only; generation is CPU-only)
- Dependencies: `pip install -r requirements.txt`

## Reproducing the Benchmark Dataset

### Quick Start

```bash
# 1. Generate all 5 suites (pilot size, seed 42)
bash scripts/generate_all.sh pilot 42

# 2. Validate all generated datasets
bash scripts/validate_all.sh pilot

# 3. Freeze dataset for release
bash scripts/freeze_dataset.sh pilot
```

### Step-by-Step

#### Generate a single suite

```bash
PYTHONPATH=src python -m anchorbench_v1.generate \
    --suites external --size pilot --seed 42 \
    --out_dir datasets/anchorbench_external_pilot/
```

Available `--suites`: `external`, `history`, `icl`, `rag`, `tool`

Available `--size`: `smoke` (6 items), `pilot` (180 items), `core` (360 items)

#### Validate

```bash
PYTHONPATH=src python -m anchorbench_v1.validate \
    --data_dir datasets/anchorbench_external_pilot/
```

### Deterministic Regeneration

All generation is deterministic given the same `--seed`. The only non-deterministic field in the output is `manifest.json:timestamp`. To verify:

```bash
# Compare two runs with the same seed
diff <(grep -v '"timestamp"' datasets/run1/manifest.json) \
     <(grep -v '"timestamp"' datasets/run2/manifest.json)

diff datasets/run1/itemspecs.jsonl datasets/run2/itemspecs.jsonl
diff datasets/run1/promptviews.jsonl datasets/run2/promptviews.jsonl
```

## Reproducing Paper Results

### Run a single model on one suite

```bash
PYTHONPATH=src python scripts/eval/run_icl.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --promptviews datasets/anchorbench_icl_pilot/promptviews.jsonl \
    --itemspecs datasets/anchorbench_icl_pilot/itemspecs.jsonl \
    --out_dir results/icl_pilot \
    --batch_size 32 --max_new_tokens 512
```

### Run all models on all suites

```bash
bash scripts/run_full_benchmark.sh

# Or per-suite:
bash scripts/run_icl_gpus.sh
bash scripts/run_tool_gpus.sh
bash scripts/run_rag_gpus.sh
bash scripts/run_history_gpus.sh
```

### Recompute unified metrics

```bash
PYTHONPATH=src python scripts/eval/recompute_all_unified.py --results_dir results/full_benchmark
```

## Decoding Protocol

All evaluations use:
- `temperature = 0.0` (greedy decoding)
- `max_tokens = 512`
- No system prompt by default
- Answer format: integer 0–100 on the last line

## Parsing

Responses are parsed with a multi-stage regex parser (`parse_answer_int` in `src/mitigation_eval/runner.py`):
1. Look for a standalone integer on the last non-empty line
2. Fall back to the first integer in the response
3. Records with no parseable integer are marked `parsed_ok=False`

## Key Outputs

```
results/{suite}_pilot/{model_slug}/
    results.jsonl      — per-prompt raw outputs and parsed answers
    summary.json       — aggregated UAI, TAR, Disc_delta, parse rate
```

## Checksums

After freezing, checksums are stored in `datasets/anchorbench_pilot_checksums.sha256`. Verify with:

```bash
sha256sum -c datasets/anchorbench_pilot_checksums.sha256
```

## Reproducing Paper Figures and Tables

All COLM 2026 figures and tables can be regenerated from the per-record
`results.jsonl` files (no inference). Run the umbrella script:

```bash
PYTHONPATH=src python COLM/scripts/generate_paper_figures.py
```

This calls every per-artifact script and writes outputs to
`COLM/figures/` and `outputs/tables/`. Pass `--figures`, `--tables`, or
`--extensions` to regenerate only one group; pass `--skip_extensions`
to skip the slower gold-shift / sampling / mitigation aggregation.

After regeneration, verify against the published numbers:

```bash
PYTHONPATH=src python scripts/verify_paper_tables.py            # all claims
PYTHONPATH=src python scripts/verify_paper_tables.py --quick    # main only
PYTHONPATH=src python scripts/verify_paper_tables.py --strict   # tighter tols
```

### Per-artifact mapping

Every figure or table in the paper is produced by exactly one script.
Open-weight (OW) results live in `results/full_benchmark/`; API results
live in `results/api_benchmark/`; revision experiments live in
`results/revision/`.

| Paper artifact | LaTeX label | Generator | Output path |
|---|---|---|---|
| Figure 4 (dose-response) | `fig:dose-response` | `scripts/eval/paper/fig4_dose_response.py` | `COLM/figures/fig4_dose_response.{pdf,png}` |
| Figure 5 (acc vs. disc) | `fig:acc-vs-disc` | `scripts/eval/paper/fig5_acc_vs_disc.py` | `COLM/figures/fig5_acc_vs_disc.{pdf,png}` |
| Table 1 (main results) | `tab:main_results` | `scripts/eval/paper/tables_main.py` | `outputs/tables/tab_main_results.tex` |
| Table 2 (UAI by pathway) | `tab:uai-pathway` | `scripts/eval/paper/tables_main.py` | `outputs/tables/tab_uai_pathway.tex` |
| Table 4 (External, full) | `tab:app-external` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_app_external.tex` |
| Table 5 (History, full) | `tab:app-history` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_app_history.tex` |
| Table 6 (ICL, full) | `tab:app-icl` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_app_icl.tex` |
| Table 7 (RAG, full) | `tab:app-rag` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_app_rag.tex` |
| Table 8 (Tool, full) | `tab:app-tool` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_app_tool.tex` |
| Table 9 (UAI summary) | `tab:uai_summary` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_uai_summary.tex` |
| Table 10 (stats inference) | `tab:stats_inference` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_stats_inference.tex` |
| Table 11 (UAI distribution) | `tab:uai_distribution` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_uai_distribution.tex` |
| Table 12 (Δ-MAE) | `tab:anchored_mae` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_anchored_mae.tex` |
| Table 13 (history matched) | `tab:history_matched` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_history_matched.tex` |
| Table 14 (ICL-dist) | `tab:icl_dist` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_icl_dist.tex` |
| Table 15 (Tool plaintext) | `tab:tool_plaintext` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_tool_plaintext.tex` |
| Table 16 (difficulty) | `tab:difficulty` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_difficulty.tex` |
| Boundary subtable | `tab:boundary` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_boundary.tex` |
| Model panel | `tab:model-details` | `scripts/eval/paper/tables_appendix.py` | `outputs/tables/tab_model_details.tex` |
| Gold-shift decomposition | `tab:gold_shift_main` | `scripts/eval/gold_shift_decomposition.py` | `results/revision/gold_shift_decomposition/*.tex` |
| Sampling robustness | `tab:sampling_robustness` | `scripts/eval/sampling_robustness_figures.py` | `results/revision/sampling_robustness/*.tex` |
| Mitigation headroom | `tab:mitigation_headroom` | `scripts/eval/aggregate_mitigation_headroom.py` | `results/revision/mitigation_headroom/*.tex` |

### Recomputing aggregated metrics

If raw `results/.../results.jsonl` files have changed (e.g. after a
re-run), recompute the unified summaries first:

```bash
PYTHONPATH=src python scripts/eval/recompute_all_unified.py \
    --results_dir results/full_benchmark
PYTHONPATH=src python scripts/eval/recompute_all_unified.py \
    --results_dir results/api_benchmark
```

Then re-run the umbrella generator and verifier above.
