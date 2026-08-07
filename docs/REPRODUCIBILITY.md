# Reproducibility Guide

This guide reproduces the AnchorBench dataset and every figure/table in
the COLM 2026 paper using the consolidated `anchorbench` CLI introduced
in v2.0.

## Requirements

- Python 3.10+
- A CUDA-capable GPU (only required for open-weight inference; dataset
  generation and analysis run on CPU)
- `pip install -e ".[all]"`
- For API models: `OPENROUTER_API_KEY` exported in the environment

## One-shot reproduction

```bash
bash scripts/reproduce_paper.sh
```

This regenerates datasets if missing, runs the frozen `paper_main`
recipe (14 models x 5 suites), recomputes unified summaries, rebuilds
every figure/table, and finally calls `anchorbench verify` to confirm
the numbers match the published values. Pass `DRY_RUN=1` to see the
plan without executing.

## Step-by-step

### 1. Generate the dataset

```bash
anchorbench generate data=external
anchorbench generate data=history
anchorbench generate data=icl
anchorbench generate data=icl_dist
anchorbench generate data=rag
anchorbench generate data=tool
```

`anchorbench generate` is a thin wrapper around
`anchorbench.data.generate.generate_suite_dataset`. Each call writes
`itemspecs.jsonl`, `promptviews.jsonl`, and `manifest.json` under
`datasets/anchorbench_<suite>_<size>/`. Generation is deterministic
given the same `seed`.

Validate determinism by re-running with the same seed and diffing the
JSONL files; only `manifest.json:timestamp` should differ.

### 2. Validate the dataset

```bash
bash scripts/validate_all.sh core
```

Internally this calls `python -m anchorbench.data.validate` for each
suite. Validation enforces all schema, condition, and offset-grid
invariants.

### 3. Run inference

A single (model, suite) cell:

```bash
anchorbench eval data=external model=qwen_7b
```

A full named recipe:

```bash
anchorbench experiment +experiment=paper_main          # 14 models x 5 suites
anchorbench experiment +experiment=paper_icl_dist      # Table 14
anchorbench experiment +experiment=paper_history_matched
anchorbench experiment +experiment=paper_sampling
anchorbench experiment +experiment=paper_mitigation_headroom
anchorbench experiment +experiment=paper_gold_shift
```

Override anything from the command line (Hydra syntax):

```bash
anchorbench eval data=icl model=llama_8b model.batch_size=64 +decoding.n_samples=5
```

Use `+dry_run=true` to enumerate the cells without dispatching them to a
backend.

### 4. Recompute unified summaries

```bash
python -m anchorbench.analysis.unified --results_dir results/full_benchmark
python -m anchorbench.analysis.unified --results_dir results/api_benchmark
```

Both commands rebuild `unified_all_suites.json` from the per-record
`results.jsonl` files. Output is byte-identical given the same inputs.

### 5. Regenerate figures and tables

```bash
anchorbench tables --paper          # all figures + tables
anchorbench tables --figures        # only figures
anchorbench tables --tables         # only main + appendix tables
anchorbench tables --extensions     # gold-shift / sampling / mitigation
```

### 6. Verify against published numbers

```bash
anchorbench verify             # all claims
anchorbench verify --quick     # main-table claims only
anchorbench verify --strict    # tighter tolerances
```

`anchorbench verify` reads typed `Claim(suite, metric, value, tolerance)`
entries from `src/anchorbench/paper/verify.py`. CI fails when any claim
drifts, so adding a metric or changing a number is a deliberate update
rather than an accidental drift.

## Decoding protocol

All paper experiments use the `greedy` decoding profile
(`conf/decoding/greedy.yaml`):

- `temperature = 0.0`
- `max_tokens = 512`
- no system prompt
- answer format: integer 0 to 100 on the last line

The sampling robustness experiment uses `sample_t07` (5 samples,
`temperature = 0.7`).

## Per-artifact mapping

Numbers are as printed in the camera-ready. Open-weight results live under
`results/full_benchmark/`, API results under `results/api_benchmark/`, the
appendix extensions under `results/rebuttal/` and `results/revision/`.

### Main paper

| # | Label | Generator | Output |
|---|---|---|---|
| Figure 3 | `fig:dose-response` | `anchorbench.paper.fig4_dose_response` | `outputs/figures/fig4_dose_response.{pdf,png}` |
| Table 1 | `tab:main_results` | `anchorbench.paper.tables_main` | `outputs/tables/tab_main_results_revised.tex` |
| Table 2 | `tab:uncertain-main` | `anchorbench.analysis.uncertain` | hand-condensed from `results/rebuttal/uncertain/` |

Figures 1 and 2 (`fig:overview`, `fig:main-item-format`) are hand-drawn: the
teaser is authored artwork with no generator, and the item-format figure is a
`tcolorbox` written inline in `sections/benchmark.tex`.

Table 1 is generated but hand-styled in the paper: the numbers match
`tab_main_results_revised.tex` cell for cell, while the bolding, the model
name macros and the caption are authored.

### Appendix, generated into `outputs/tables/`

| # | Label | Output |
|---|---|---|
| Table 3 | `tab:model-details` | `tab_model_details.tex` |
| Figure 8 | `fig:acc-vs-disc` | `outputs/figures/fig5_acc_vs_disc.{pdf,png}` |
| Tables 4-8 | `tab:app-{suite}` | `tab_app_{suite}.tex` |
| Table 9 | `tab:uai_summary` | `tab_uai_summary.tex` |
| Table 10 | `tab:stats_inference` | `tab_stats_inference.tex` |
| Table 11 | `tab:uai_distribution` | `tab_uai_distribution.tex` |
| Table 12 | `tab:anchored_mae` | `tab_anchored_mae.tex` |
| Table 13 | `tab:history_matched` | `tab_history_matched.tex` |
| Table 14 | `tab:icl_dist` | `tab_icl_dist.tex` |
| Table 15 | `tab:tool_plaintext` | `tab_tool_plaintext.tex` |
| Table 17 | `tab:difficulty` | `tab_difficulty.tex` |
| — | `tab:boundary` | `tab_boundary.tex` |

All from `anchorbench.paper.tables_appendix`. Tables 10, 11, 12 and 17 are
`\input`-ed by the paper from `COLM_camera_ready/tables/*_body.tex`, synced by
`scripts/sync_paper_tables.py`; the rest are pasted inline.

### Appendix, generated by `analysis/`

| # | Label | Generator |
|---|---|---|
| Table 16 | `tab:gold_shift_main` | `anchorbench.analysis.gold_shift` |
| Table 18 | `tab:sampling_robustness` | `anchorbench.analysis.sampling` |
| Table 19 | `tab:mitigation_headroom` | `anchorbench.analysis.mitigation` |
| Tables 20-23 and others | the 13 `\input`-ed tables | see [APPENDIX_TABLES.md](APPENDIX_TABLES.md) |

Regenerate the last group with `anchorbench tables --appendix`; it needs
`results/rebuttal/`, which is published as tarballs on Google Drive rather than committed.

### Not reproducible from the release

`tab:history_matched` and `tab:tool_plaintext` were produced by runs that were
never preserved. Both were re-run independently; the results and a
published-vs-re-run comparison are in
D5 in [RECONCILIATION.md](RECONCILIATION.md). The published values stand.

## Checksums

After freezing, dataset checksums are stored in
`datasets/anchorbench_core_checksums.sha256`:

```bash
sha256sum -c datasets/anchorbench_core_checksums.sha256
```
