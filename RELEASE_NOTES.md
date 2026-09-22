# Release Notes — AnchorBench

## v2.1.0 (2026-09) — repository cleanup after the camera-ready

No experiment logic, metric, prompt or dataset byte changed. Every committed
table and figure regenerates identically (`tests/golden/`), and
`anchorbench verify --strict` reports zero mismatches.

### Fixed

- **`anchorbench eval` sent Gemma to the API runner.** The routing rule
  prefix-matched `hf_id` against `google/`, so `eval data=external
  model=gemma_4b` launched `runners.api` and would have produced an empty
  summary. The August fix had only reached `anchorbench experiment`. Both
  commands now build their cell command through `anchorbench.cli.cells`,
  which routes by the model's declared `backend`; `tests/test_model_routing.py`
  covers the rule and both commands.
- **Dry runs created result directories.** `anchorbench experiment
  +dry_run=true` left empty `results/full_benchmark/<suite>/` behind, which
  is the marker the golden tests use to decide the bulk results are present,
  so they stopped skipping and failed on a clean clone after any dry run.
- **33 ruff errors** (all import order) in the rebuttal runners, the two
  figure scripts and `scripts/sync_paper_tables.py`.
- **API cells emitted flags the API runner rejects.** Both CLI copies of
  the cell builder passed `--suite` and `--variant`; `runners.api` takes
  `--suites` and has no variant flag, so `eval data=icl_dist
  model=gemini_2_5_flash` failed at argument parsing. The shared builder
  passes the variant as the suite name, and a test checks that every flag
  a built command emits is declared by the runner it targets.
- `scripts/reproduce_paper.sh`, `generate_all.sh` and `validate_all.sh`
  fall back to `python -m anchorbench.cli.main` with `src/` on
  `PYTHONPATH` when the console script is not installed (`scripts/_env.sh`).

### Changed

- `runners/external.py`, `icl.py` and `rag.py` were three copies of one
  loop; the body now lives once in `runners/_single_stage.py`. The
  pass-through `runners/base.py` is removed; import the helpers from
  `anchorbench.eval.runner_utils`.
- `anchorbench eval` accepts `+tool_plaintext=true`, as `experiment` did.
- The 25 appendix-experiment launchers moved from `scripts/rebuttal/` to
  `experiments/rebuttal/`, with `experiments/run_stage3_reruns.sh` beside
  them and a README mapping each launcher to the table it backs. `scripts/`
  now holds only user-facing entry points.
- `docs/` is four guides instead of eleven files: `ARCHITECTURE.md` (package
  map, evaluation internals, extending), `REPRODUCIBILITY.md` (every command,
  results policy, the full artifact-to-module map), `DATA.md` (pipeline,
  record schemas, History design, HF release) and the unchanged
  `RECONCILIATION.md` ledger.
- `datasets/README.md` rewritten; it described scripts and modules that no
  longer exist.
- Hugging Face dataset id is `Yiderigun/AnchorBench` (renamed from
  `Yiderigun/LLM_anchoring`; the old id redirects).
- `pyproject.toml` lists the authors; version 2.1.0.
- `conf/decoding/sample_t07.yaml` says three seeds; its unconsumed
  `n_samples: 5` contradicted the sampling runner's default and the paper.
- `docs/RECONCILIATION.md` gains **D8 (OPEN)**: the four API-tier History
  cells were scored against the two-stage control while the open-weight
  cells used the single-stage control. `paper_main` reproduces the
  committed numbers unchanged; the ledger asks for a decision on the
  Table 1 caveat.

### Added

- GitHub Actions CI: ruff, the full test suite and `anchorbench verify
  --strict` on every push and pull request.
- `CITATION.cff` and `environment.yml`.

### Removed

- `scripts/data_gen/gen_syn_anchors_local.py` (the pre-benchmark SynAnchors
  v0 generator), `scripts/export_raw_texts.py` (debug helper) and
  `scripts/validate_smoke_results.sh` (called a script deleted in v2.0).
- `docs/benchmark_spec_v1_frozen.md`, `docs/EVALUATION_PIPELINE.md`,
  `docs/ENVIRONMENT.md`, `docs/EXTENDING.md`, `docs/APPENDIX_TABLES.md`,
  `docs/DATASET_GENERATION_PIPELINE.md`, `docs/data_objects_reference.md`,
  `docs/history_suite.md` (merged into the four guides above).

## v2.0.0 (2026-08) — release refactor for COLM 2026

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

### Highlights

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

### New layout

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

### Added

- **`anchorbench` console script** (`pyproject.toml`) with subcommands:
  `eval`, `experiment`, `generate`, `tables`, `add-model`, `verify`.
- **`conf/` Hydra tree**: 6 data suites, 14 model configs, 6 tier
  configs, 2 decoding profiles, 7 named experiment recipes.
- **`scripts/reproduce_paper.sh`**: end-to-end paper reproduction with
  optional `DRY_RUN=1`.
- **`docs/ARCHITECTURE.md`**: package map and mermaid data-flow
  diagram.
- **`tests/test_imports.py`**: smoke test asserting every public
  submodule imports.
- **`tests/test_paper_verify.py`**: smoke test asserting
  `anchorbench.paper.verify --quick` runs end-to-end on the bundled
  unified summaries.
- **Hydra + OmegaConf** added to required dependencies.

### Changed

- **Package layout consolidated** to a single `src/anchorbench/`. The
  three pre-2.0 package names (`anchorbench_v1`, `anchorbench_eval`,
  `mitigation_eval`) are **removed**, not shimmed: v2.0.0 is the first
  public release, so there are no downstream importers to keep working,
  and shipping them would have claimed three generic top-level import
  names on PyPI for no benefit. See the import migration map below.
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

### Removed

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

### Migration

Old import sites do **not** keep working -- the pre-2.0 packages were
removed, not shimmed (see Changed, above). Rewrite them as follows:

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

### Validation gate

Before tagging the release, all four checks pass on `release/v2.0`:

1. `pytest tests/` — **200 passed**.
2. `python -m anchorbench.paper.verify --quick` — runs end-to-end on
   the bundled unified summaries.
3. `anchorbench experiment +experiment=paper_main +dry_run=true`
   enumerates the same (model, suite, gpu) cells the legacy bash did.
4. `python -m anchorbench.analysis.unified --results_dir results/full_benchmark`
   produces a **byte-identical** `unified_all_suites.json` to the
   pre-refactor copy. Same for `results/api_benchmark`.

### Unchanged (intentionally)

- Experiment logic: suite renderers, ItemSpec generation, prompt
  construction.
- Metrics: UAI, TAR, Disc_Δ, MAE, Acc10, bootstrap CIs, Wilcoxon, BH
  correction.
- Parsing cascade: XML / regex / LLM fallback.
- Dataset files (`datasets/`).
- Per-record `results/.../results.jsonl` outputs.
- COLM 2026 paper source (`COLM/`).

### Import migration (pre-2.0 → 2.0)

| Old import | New import |
|---|---|
| `anchorbench_v1.*` | `anchorbench.data.*` |
| `anchorbench_v1.suites` | `anchorbench.data.suites` |
| `anchorbench_eval.*` | `anchorbench.eval.*` |
| `anchorbench_eval.metrics` | `anchorbench.eval.metrics` |
| `anchorbench_eval.runner_utils` | `anchorbench.eval.runner_utils` |
| `mitigation_eval.*` | `anchorbench.inference.*` |

```diff
-from anchorbench_v1.suites import SUITE_RENDERERS
-from anchorbench_eval.metrics import compute_unified_metrics
+from anchorbench.data.suites import SUITE_RENDERERS
+from anchorbench.eval.metrics import compute_unified_metrics
```
