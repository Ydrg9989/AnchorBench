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

Open-weight (OW) results live under `results/full_benchmark/`; API
results live under `results/api_benchmark/`; rebuttal extensions live
under `results/revision/`.

| Paper artifact | LaTeX label | Generator | Output |
|---|---|---|---|
| Figure 4 (dose-response) | `fig:dose-response` | `anchorbench.paper.fig4_dose_response` | `COLM/figures/fig4_dose_response.{pdf,png}` |
| Figure 5 (acc vs. disc) | `fig:acc-vs-disc` | `anchorbench.paper.fig5_acc_vs_disc` | `COLM/figures/fig5_acc_vs_disc.{pdf,png}` |
| Table 1 (main results) | `tab:main_results` | `anchorbench.paper.tables_main` | `outputs/tables/tab_main_results.tex` |
| Table 2 (UAI by pathway) | `tab:uai-pathway` | `anchorbench.paper.tables_main` | `outputs/tables/tab_uai_pathway.tex` |
| Tables 4-8 (per-suite full) | `tab:app-{suite}` | `anchorbench.paper.tables_appendix` | `outputs/tables/tab_app_{suite}.tex` |
| Tables 9-12 (UAI / stats / distribution / MAE) | various | `anchorbench.paper.tables_appendix` | `outputs/tables/tab_*.tex` |
| Table 13 (history matched) | `tab:history_matched` | `anchorbench.paper.tables_appendix` | `outputs/tables/tab_history_matched.tex` |
| Table 14 (ICL-dist) | `tab:icl_dist` | `anchorbench.paper.tables_appendix` | `outputs/tables/tab_icl_dist.tex` |
| Table 15 (Tool plaintext) | `tab:tool_plaintext` | `anchorbench.paper.tables_appendix` | `outputs/tables/tab_tool_plaintext.tex` |
| Table 16 (difficulty) | `tab:difficulty` | `anchorbench.paper.tables_appendix` | `outputs/tables/tab_difficulty.tex` |
| Boundary subtable | `tab:boundary` | `anchorbench.paper.tables_appendix` | `outputs/tables/tab_boundary.tex` |
| Model panel | `tab:model-details` | `anchorbench.paper.tables_appendix` | `outputs/tables/tab_model_details.tex` |
| Gold-shift decomposition | `tab:gold_shift_main` | `anchorbench.analysis.gold_shift` | `results/revision/gold_shift_decomposition/*.tex` |
| Sampling robustness | `tab:sampling_robustness` | `anchorbench.analysis.sampling` | `results/revision/sampling_robustness/*.tex` |
| Mitigation headroom | `tab:mitigation_headroom` | `anchorbench.analysis.mitigation` | `results/revision/mitigation_headroom/*.tex` |

## Checksums

After freezing, dataset checksums are stored in
`datasets/anchorbench_pilot_checksums.sha256`:

```bash
sha256sum -c datasets/anchorbench_pilot_checksums.sha256
```
