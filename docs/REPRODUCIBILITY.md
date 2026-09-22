# Reproducibility guide

How to regenerate the dataset, re-run the benchmark, rebuild every figure and
table of the COLM 2026 paper, and check the numbers without running anything.
The package layout is in [ARCHITECTURE.md](ARCHITECTURE.md); the data objects
are in [DATA.md](DATA.md); known paper-vs-code divergences are recorded in
[RECONCILIATION.md](RECONCILIATION.md).

## Requirements

- Python 3.10 or newer; `pip install -e ".[all]"` installs the package with
  vLLM, the async API client and the test tools. `pip install -e ".[dev]"` is
  enough for generation, validation, tables and tests.
- A CUDA GPU for the open-weight models. Generation, analysis, tables and
  tests run on CPU.
- An OpenRouter key for the four API models. Put it in
  `~/.config/anchorbench/env` as `OPENROUTER_API_KEY=...`; `scripts/run_with_env.sh`
  sources that file. Keys never go inside the working tree
  (`tests/test_no_secrets.py` fails if one appears).

### Conda environment and the vLLM wrapper

The paper runs used the conda environment in `environment.yml`:

```bash
conda env create -f environment.yml      # creates LLM_anchoring
conda activate LLM_anchoring
```

Launch GPU work through the wrapper, which activates that environment, puts
`$CONDA_PREFIX/lib` first on `LD_LIBRARY_PATH` so only one cuDNN is visible,
sets `VLLM_USE_V1=0`, loads the Hugging Face token from its cache and sources
the OpenRouter key file:

```bash
bash scripts/run_with_env.sh anchorbench eval data=external model=llama_8b
ANCHORBENCH_CONDA_ENV=my_env bash scripts/run_with_env.sh ...   # other env name
```

If vLLM fails with `Found 2 libcudnn.so.x`, two cuDNN builds are visible; the
wrapper's `LD_LIBRARY_PATH` ordering and the V0 engine setting are the fix.

## One-shot reproduction

```bash
DRY_RUN=1 bash scripts/reproduce_paper.sh    # list the 70 cells, launch nothing
bash scripts/reproduce_paper.sh              # roughly 24 h on 4x A100 plus ~$300 of API spend
```

The script regenerates any missing core dataset, runs the `paper_main` recipe
(14 models x 5 suites), recomputes both unified summaries, regenerates the
figures and main tables, and finishes with `anchorbench verify`, which exits
non-zero if any numeric claim drifts.

## Step by step

### 1. Generate the datasets

```bash
anchorbench generate data=external              # one suite; +size=smoke|pilot|core (default core), seed=42 from conf/config.yaml
bash scripts/generate_all.sh core 42            # external history icl icl_dist rag tool
```

Output goes to `datasets/anchorbench_<suite>_<size>/`. Generation is
deterministic for a given seed; `tests/test_dataset_regeneration.py` proves
that the committed datasets regenerate byte-for-byte from the current code.

### 2. Validate

```bash
bash scripts/validate_all.sh core
python -m anchorbench.data.validate --data_dir datasets/anchorbench_external_core
```

### 3. Run inference

One cell:

```bash
anchorbench eval data=external model=qwen_7b
anchorbench eval data=tool model=qwen_7b +tool_plaintext=true
```

A named recipe (add `+dry_run=true` to print the cells without launching):

| Recipe | What it runs | Backs |
|---|---|---|
| `paper_main` | 14 models x 5 suites, greedy; API History cells use the runner's two-stage control (ledger D8, D9) | Table 1, Figures 3 and 8, most appendix tables |
| `paper_icl_dist` | ICL distribution-matching variant | Table 14 |
| `paper_history_matched` | History with the two-stage control | Table 13 (re-run, see D5) |
| `paper_tool_plaintext` | Tool suite forced to plaintext, 5 models | Table 15 (re-run, see D5) |
| `paper_sampling` | temperature 0.7, top-p 0.9, 3 seeds | Table 18 |
| `paper_mitigation_headroom` | prompt-based mitigation strategies | Table 19 |
| `paper_gold_shift` | inputs for the error decomposition | Table 16 |

```bash
anchorbench experiment +experiment=paper_main
```

Any config key can be overridden on the command line, for example
`batch_size=64` or `decoding.max_tokens=256`. Every cell runs in a subprocess
through the same evaluator loop; open-weight models load through vLLM, API
models through the OpenRouter backend with bounded concurrency. Every runner
accepts `--backend openrouter`, so any suite can be pointed at a hosted model
directly. History on the API defaults to a real three-turn chat for Stage 2;
`--history_chat_format flat` reproduces the single-message rendering the
published API cells used (D9).

### 4. Recompute the unified summaries

```bash
python -m anchorbench.analysis.unified --results_dir results/full_benchmark
python -m anchorbench.analysis.unified --results_dir results/api_benchmark
```

Each writes `unified_all_suites.json` next to the per-model `results.jsonl`
files. Output is byte-identical for identical inputs.

### 5. Regenerate figures and tables

```bash
anchorbench tables --paper              # figures + tables + gold-shift/sampling/mitigation + verify
anchorbench tables --figures            # Figure 3 and Figure 8 only
anchorbench tables --tables-only        # tables_main + tables_appendix only
anchorbench tables --skip-extensions    # figures + tables, skip the slow analysis modules
anchorbench tables --appendix           # the 13 \input-ed appendix tables (needs results/rebuttal/)
anchorbench tables --dry-run            # print the commands
```

Figures land in `outputs/figures/`, tables in `outputs/tables/`. Both
directories are gitignored; their contents are pinned by
`tests/golden/*.sha256` instead.

### 6. Verify the numbers

```bash
anchorbench verify              # every claim
anchorbench verify --quick      # main-table claims only
anchorbench verify --strict     # tighter tolerances; this is what CI runs
```

`paper/verify.py` holds every number the paper quotes as a
`Claim(suite, metric, value, tolerance)` and re-reads it from the committed
summaries. It runs on a clean clone.

## Decoding protocol

All paper experiments use `conf/decoding/greedy.yaml`: temperature 0,
`max_tokens` 512, no system prompt, and the instruction to return a single
integer 0 to 100 on the last line. The sampling-robustness check uses
`conf/decoding/sample_t07.yaml`: temperature 0.7, three seeds; the top-p 0.9 it
lists was never applied by the runner (D10 in the ledger).
vLLM is not bitwise reproducible at temperature 0; re-running a cell can move
a metric by a few hundredths (`results/rebuttal/cot_replication/README.md`).

## Results: what is committed and what is downloaded

The raw generations are about 630 MB and are not in git. They are published
as one tarball per experiment on
[Google Drive](https://drive.google.com/drive/folders/1Befi102mkvXQomB1zwCPS_m0_4OlKH2M?usp=sharing),
built by `scripts/make_result_bundles.sh`:

```
anchorbench-results-full_benchmark.tar.gz      10 open-weight models x 5 suites
anchorbench-results-api_benchmark.tar.gz       4 API models x 5 suites
anchorbench-results-rebuttal.tar.gz            every appendix experiment
anchorbench-results-revision.tar.gz            gold shift, sampling, mitigation
anchorbench-results-decoding_sampling_robustness.tar.gz
anchorbench-results-mitigation_ignore_anchor.tar.gz
anchorbench-results-icl_dist_core.tar.gz, anchorbench-results-icl_dist_api.tar.gz
```

Extract them into `results/` and check them against the committed manifest:

```bash
bash scripts/checksum_results.sh --verify      # sha256sum -c results/CHECKSUMS.sha256
```

What git carries is the small derived layer that the verifier and the golden
tests need on a clean clone: the two `unified_all_suites.json` files, the
aggregate CSVs and `summary.json` files under `results/revision/`, and every
`.tex`, `.csv`, `.json` and `.md` under `results/rebuttal/`. Per-record
`.jsonl` files and run logs are excluded by `.gitignore`.

## Which code produces which paper artifact

Numbers are as printed in the camera-ready. Open-weight results live under
`results/full_benchmark/`, API results under `results/api_benchmark/`, the
appendix extensions under `results/rebuttal/` and `results/revision/`.

### Main paper

| # | Label | Generator | Output |
|---|---|---|---|
| Figure 3 | `fig:dose-response` | `anchorbench.paper.fig4_dose_response` | `outputs/figures/fig4_dose_response.{pdf,png}` |
| Table 1 | `tab:main_results` | `anchorbench.paper.tables_main` | `outputs/tables/tab_main_results_revised.tex` |
| Table 2 | `tab:uncertain-main` | `anchorbench.analysis.uncertain` | hand-condensed from `results/rebuttal/uncertain/` |

Figures 1 and 2 are hand-drawn. Table 1 is generated but hand-styled: the
numbers match the generated file cell for cell, while the bolding, the model
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
`\input`-ed by the paper from copies of these bodies; the rest are pasted
inline. The LaTeX source ships with the arXiv submission rather than this
repository, and the scripts that synced and compared against it went with it.

### Appendix, generated by `analysis/`

| # | Label | Generator |
|---|---|---|
| Table 16 | `tab:gold_shift_main` | `anchorbench.analysis.gold_shift` |
| Table 18 | `tab:sampling_robustness` | `anchorbench.analysis.sampling` |
| Table 19 | `tab:mitigation_headroom` | `anchorbench.analysis.mitigation` |

### Appendix tables `\input`-ed by the paper

Thirteen tables are read by the paper straight from generated `.tex` files.
`anchorbench tables --appendix` runs the modules below in order; it needs the
`rebuttal` tarball extracted into `results/rebuttal/`. The `.tex` outputs are
committed, so the tables are in git even when the generations are not, and
`tests/test_golden_artifacts.py` pins their hashes.

| # | Paper table | Generator module | Reads | Launcher in `experiments/rebuttal/` |
|---|---|---|---|---|
| 1 | `tab:implied_weight` | `analysis.bayesian_bound` | `full_benchmark`, `api_benchmark` | — |
| 2 | `tab:excess_uai` | `analysis.bayesian_bound` | same run | — |
| 3 | `tab:plausibility_spectrum` | `analysis.spectrum` | `rebuttal/spectrum/` | `_b2_one_model.sh` |
| 4 | `tab:uncertain_k` | `analysis.uncertain` | `rebuttal/uncertain/` | `_p5_one_model.sh` |
| 5 | `tab:intensity_pathway` | `analysis.intensity_pathway` | `rebuttal/intensity{,_rag,_history}/` | `_d1_one_model.sh` |
| 6 | `tab:extension_pilot` | `analysis.extension_pilot` | `rebuttal/extension_pilot/`, `rebuttal/medical/` | `_extension_pilot_one_model.sh`, `_c1_one_model.sh` |
| 7 | `tab:weighted_mean` | `analysis.weighted_mean` | `rebuttal/weighted_mean/` | `_b3_one_model.sh` |
| 8 | `tab:cot_extended` | `analysis.cot_reasoning_extended` | `rebuttal/cot_extended/` | `_b1_*.sh`, `_chain_after_b1.sh` |
| 9 | `tab:task_spec` | `analysis.task_spec` | `rebuttal/task_spec/` | `_p4_one_model.sh` |
| 10 | `tab:rag_realism` | `analysis.rag_realism` | `rebuttal/rag_realism/` | `run_p2_rag_realism.sh` |
| 11 | `tab:tool_realism` | `analysis.tool_realism` | `rebuttal/tool_realism/` | `run_p3_tool_realism.sh` |
| 12 | `tab:large_panel_results` | `analysis.large_panel` | `rebuttal/large_{api,ow}/` | `_large_api_full.sh`, `_large_ow_all_suites.sh` |
| 13 | `tab:case_studies` | `analysis.case_studies` | `full_benchmark/external/` | — |

`analysis.cohens_d` also runs under `--appendix`; `tab:cohens_d` in the paper
is a hand-condensed summary of its `cohens_d_table.tex`.

Naming traps: `cot_reasoning_extended.py` writes `cot_vs_baseline.tex` while
the paper file is `cot_vs_baseline_extended.tex` (same content), and five of
the thirteen were hand-edited from `\begin{table}` to `\begin{table*}` to span
both columns, so a regenerated file differs from the paper copy in that one
token.

Four analysis modules are superseded and produce nothing the camera-ready
uses. They stay because committed launchers call them:

| Module | Superseded by | Called from |
|---|---|---|
| `analysis/intensity.py` | `intensity_pathway.py` | `_wrap_up_p2p5.sh` |
| `analysis/api_cost_estimate.py` | — (a cost projection) | `api_smoke_large.sh` |
| `analysis/cot_reasoning.py` | `cot_reasoning_extended.py` | `run_b1_cot_extended.sh` |
| `analysis/medical_pilot.py` | `extension_pilot.py` | `_wrap_up.sh` |

### Not reproducible from the release

`tab:history_matched` and `tab:tool_plaintext` were produced by runs whose
inputs were not preserved. Both were re-run independently with
`experiments/run_stage3_reruns.sh`; the published values stand, and the
published-vs-re-run comparison is D5 in [RECONCILIATION.md](RECONCILIATION.md).

## Checks that run without a GPU

```bash
pytest                                    # regeneration, parsing, metrics, routing, verifier smoke
ruff check src tests scripts datasets
anchorbench verify --strict
sha256sum -c datasets/anchorbench_core_checksums.sha256
```

`tests/golden/*.sha256` pin what the table and figure generators emit; the
tests that need the bulk results skip when the tarballs are absent. Refresh
the manifests with `scripts/update_golden.sh` only when a change to generator
output is intended, and say why in the commit message.
