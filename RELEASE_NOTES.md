# Release Notes — AnchorBench v2.0.0

Release-quality refactor of the AnchorBench repository for COLM 2026.
Three previously separate Python packages (`anchorbench_v1`,
`anchorbench_eval`, `mitigation_eval`) and a sprawling `scripts/eval/`
tree are consolidated into a single `anchorbench/` package with a
Hydra-driven `anchorbench` console script.

**No experiment logic, metrics, prompts, or dataset bytes have
changed.** The unified summaries (`results/.../unified_all_suites.json`)
recompute byte-identically from the same per-record `results.jsonl`
files, and `anchorbench verify` reproduces every paper claim from the
checked-in summaries.

## Highlights

- **One package, one CLI.** `pip install -e ".[all]"` exposes a single
  `anchorbench` console script with `eval`, `experiment`, `generate`,
  `tables`, `add-model`, and `verify` subcommands.
- **Hydra config tree** under `conf/` replaces the flat
  `configs/benchmark.yaml`: one YAML per data suite, model, tier,
  decoding profile, and named experiment, with full CLI override
  support.
- **End-to-end reproduction** in one command: `bash
  scripts/reproduce_paper.sh` regenerates datasets if needed, runs the
  frozen `paper_main` recipe, recomputes unified summaries, rebuilds
  every figure/table, and verifies the numbers.
- **Release-quality docs**: rewritten `README.md`, new
  `docs/ARCHITECTURE.md` with a data-flow diagram, and retargeted
  `docs/REPRODUCIBILITY.md` and `docs/EXTENDING.md`.

## New layout

```
src/anchorbench/
  data/        # was anchorbench_v1
  eval/        # was anchorbench_eval
  inference/   # was mitigation_eval
  runners/     # was scripts/eval/run_*.py
  analysis/    # was scripts/eval/{recompute_all_unified,gold_shift_*,sampling_*,aggregate_*}.py
  paper/       # was scripts/eval/paper/ + scripts/verify_paper_tables.py
  cli/         # new: Hydra-driven entry points
conf/          # new: Hydra config tree (data, model, tier, decoding, experiment)
scripts/
  reproduce_paper.sh
  generate_all.sh
  validate_all.sh
  smoke_test.sh
  run_with_env.sh
```

## Added

- **`anchorbench` console script** (`pyproject.toml`) with subcommands:
  `eval`, `experiment`, `generate`, `tables`, `add-model`, `verify`.
- **`conf/` Hydra tree**: 6 data suites, 14 model configs, 6 tier
  configs, 2 decoding profiles, 7 named experiment recipes.
- **`scripts/reproduce_paper.sh`**: end-to-end paper reproduction with
  optional `DRY_RUN=1`.
- **`docs/ARCHITECTURE.md`**: package map and mermaid data-flow
  diagram.
- **`tests/test_imports.py`**: smoke test asserting every public
  submodule imports and the deprecation shims redirect correctly.
- **`tests/test_paper_verify.py`**: smoke test asserting
  `anchorbench.paper.verify --quick` runs end-to-end on the bundled
  unified summaries.
- **Hydra + OmegaConf** added to required dependencies.

## Changed

- **Package layout consolidated** to a single `src/anchorbench/`. The
  three legacy package names (`anchorbench_v1`, `anchorbench_eval`,
  `mitigation_eval`) ship as deprecation shims that re-export the new
  submodules and emit a `DeprecationWarning`. They will be removed in
  v2.1.
- **Version** bumped to `2.0.0`.
- **Static constants** (`SUITE_DATASETS`, `MODEL_SHORT`,
  `API_MODEL_IDS`, etc.) hardcoded in
  `src/anchorbench/eval/constants.py`; the runtime no longer depends on
  `configs/benchmark.yaml`.
- **`README.md`** rewritten as a release-quality landing page.
- **`docs/REPRODUCIBILITY.md`** and **`docs/EXTENDING.md`** retargeted
  to the new CLI and package paths.
- **Tests** updated to import from the new package paths; pytest now
  reports 200 passing.

## Removed

- **`configs/benchmark.yaml`** (replaced by `conf/`).
- **`scripts/eval/run_*.py`** (promoted into `anchorbench.runners`).
- **`scripts/eval/paper/`** (promoted into `anchorbench.paper`).
- **`scripts/eval/{recompute_all_unified,gold_shift_decomposition,sampling_robustness_figures,aggregate_mitigation_headroom}.py`**
  (promoted into `anchorbench.analysis`).
- **One-off recovery scripts**: `parse_failure_report.py`,
  `repair_history_api.py`, `reparse_with_fallback.py`,
  `unified_metrics.py`, `export_latex_tables.py`,
  `package_extension_results.py`, `verify_paper_tables_v2.py`,
  `verify_results.py`, `run.py`.
- **Per-suite shell launchers**: `scripts/run_full_benchmark.sh`,
  `scripts/run_all_variants.sh`, `scripts/run_api_benchmark_all.sh`,
  `scripts/run_model_vllm.sh`, `scripts/run_{external,history,icl,rag,tool}_gpus.sh`,
  `scripts/paper_model_ids.inc.sh` (replaced by `conf/tier/*.yaml` and
  `anchorbench experiment`).
- **`tool_read` suite** (deprecated; the agentic `tool` suite is the
  paper-canonical version).
- **64 MB `archive_pre_rerun_20260328.tar.gz`** (legacy archive; not
  needed for reproduction).

## Migration

Old import sites continue to work for one release thanks to the
deprecation shims:

```python
# old
from anchorbench_v1.suites import SUITE_RENDERERS
from anchorbench_eval.metrics import compute_unified_metrics

# new
from anchorbench.data.suites import SUITE_RENDERERS
from anchorbench.eval.metrics import compute_unified_metrics
```

Old shell entry points map onto the new CLI as follows:

| Old                                      | New                                       |
|------------------------------------------|-------------------------------------------|
| `bash scripts/generate_all.sh`           | `anchorbench generate data=<suite>`        |
| `bash scripts/run_full_benchmark.sh`     | `anchorbench experiment +experiment=paper_main` |
| `python scripts/eval/run_external.py`    | `anchorbench eval data=external model=<m>` |
| `python scripts/eval/recompute_all_unified.py` | `python -m anchorbench.analysis.unified` |
| `python scripts/verify_paper_tables.py`  | `anchorbench verify`                       |

## Validation gate

Before tagging the release, all four checks pass on `release/v2.0`:

1. `pytest tests/` — **200 passed**.
2. `python -m anchorbench.paper.verify --quick` — runs end-to-end on
   the bundled unified summaries.
3. `anchorbench experiment +experiment=paper_main +dry_run=true`
   enumerates the same (model, suite, gpu) cells the legacy bash did.
4. `python -m anchorbench.analysis.unified --results_dir results/full_benchmark`
   produces a **byte-identical** `unified_all_suites.json` to the
   pre-refactor copy. Same for `results/api_benchmark`.

## Unchanged (intentionally)

- Experiment logic: suite renderers, ItemSpec generation, prompt
  construction.
- Metrics: UAI, TAR, Disc_Δ, MAE, Acc10, bootstrap CIs, Wilcoxon, BH
  correction.
- Parsing cascade: XML / regex / LLM fallback.
- Dataset files (`datasets/`).
- Per-record `results/.../results.jsonl` outputs.
- COLM 2026 paper source (`COLM/`).
