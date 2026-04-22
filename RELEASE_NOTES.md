# Release Notes — AnchorBench v2.1.0

Public-release cleanup of the AnchorBench repository. All experiment logic,
metrics, parsing, benchmark generation, and dataset files are unchanged.

## Removed

- **`legacy/`** — 116 archived files (superseded prototypes, exploratory
  notebooks, old analysis scripts). Not needed for reproduction.
- **`analysis/`** — 4 ad-hoc analysis scripts (`analyze_ablation_results.py`,
  `anchorbench_bubble_plot.py`, `anchorbench_paper_stats.py`,
  `generate_all_figures.py`). Functionality is covered by
  `COLM/scripts/generate_paper_figures.py` and `scripts/eval/export_latex_tables.py`.
- **`.vscode/`** — Editor-specific settings and extension recommendations.
- **Root zip archives** — `paper.zip`, `colm_backup.zip`.
- **COLM build artifacts** — Stale `.synctex(busy)`, duplicate `COLM/paper/`
  directory, orphaned figure PNGs replaced by PDFs.
- **11 internal planning docs** — `docs/CLEANUP_PLAN.md`,
  `docs/CORE_VS_APPENDIX.md`, `docs/EVAL_REFACTOR_PLAN.md`,
  `docs/camera_ready_gap_report.md`, `docs/freeze_decision_memo.md`,
  `docs/full_benchmark_generation_plan.md`,
  `docs/pre_run_validation_checklist.md`,
  `docs/promptview_fields_for_release.md`, `docs/repo_alignment_plan.md`,
  `docs/validation_necessity.md`, `COLM/sections/README.md`.
- **2 orphaned scripts** — `src/anchorbench_v1/llm_generate.py`,
  `src/anchorbench_v1/model_registry.py` (unused after eval refactor).
- **Smoke/debug scripts** — Ad-hoc smoke tests and one-off GPU launch scripts
  replaced by `scripts/smoke_test.sh`.

## Added

- **`configs/benchmark.yaml`** — Centralized configuration for suite dataset
  paths, model display names, API model IDs, and evaluation defaults. All eval
  scripts load from this file via `constants.py`.
- **`src/anchorbench_eval/constants.py`** — Single-source module for
  `SUITE_DATASETS`, `MODEL_SHORT`, `API_MODEL_IDS`, and `DEFAULTS`. Falls back
  to hardcoded values when the YAML is absent (e.g. library import outside the
  repo tree).
- **`.env.example`** — Template for required environment variables
  (`OPENROUTER_API_KEY`, `HF_TOKEN`).
- **`LICENSE`** — Apache License 2.0.
- **`scripts/smoke_test.sh`** — End-to-end pipeline smoke test (generate →
  validate → pytest) that runs without GPU.
- **`tests/conftest.py`** — Shared pytest fixtures; eliminates `sys.path` hacks
  from individual test files.
- **`pyproject.toml`** — Added `dependencies`, optional dependency groups
  (`vllm`, `api`, `dev`, `all`), console entry points
  (`anchorbench-generate`, `anchorbench-validate`), and `[tool.pytest]` config.

## Changed

- **Version** aligned to **2.1.0** across `pyproject.toml`.
- **`.gitignore`** updated to cover build artifacts, results, and editor files.
- **`SUITE_DATASETS` / `MODEL_SHORT`** — Previously duplicated across
  `runner_utils.py` and individual scripts; now imported from
  `constants.py` everywhere.
- **Dead code removed** from `evaluator.py`, `backends.py`, `metrics.py`,
  `domains.py` (unused imports, commented-out blocks, stale TODOs).
- **Test `sys.path` hacks** replaced with `conftest.py` + `pyproject.toml`
  `pythonpath` config.

## Bug Fixes

- **`run_external.py`** — `batch_size` now forwarded correctly to the backend.
- **`run_icl_gpus.sh`** — Default dataset path corrected to
  `datasets/anchorbench_icl_core`.
- **`run_api_benchmark.py`** — History baseline condition no longer skipped when
  running alongside other suites.
- **`run_mitigation_headroom.py`** — `--force` flag respected (was silently
  ignored).
- **`run_model_vllm.sh`** — Post-run validation step added to catch silent
  failures.

## Unchanged

The following are intentionally untouched to preserve reproducibility:

- Experiment logic (suite renderers, ItemSpec generation)
- Metrics computation (UAI, TAR, Disc_Δ, MAE, Acc10, bootstrap CIs)
- Parsing cascade (XML → regex → LLM fallback)
- Benchmark generation pipeline
- Dataset files (`datasets/`)
