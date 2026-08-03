# Extending AnchorBench

This guide describes the five most common rebuttal-time extensions:
adding a model, a suite, a condition, a metric, or a named experiment.
Each section is self-contained and lists the minimum set of files to
touch in the v2.0 layout.

The benchmark is driven by a Hydra config tree under `conf/` and a
single `anchorbench` console script. Anything you can do via a YAML
file you can also do via a command-line override.

---

## Add a Model

For both open-weight (vLLM-served) and OpenRouter API models.

1. Drop a YAML file in `conf/model/<short_name>.yaml` (or use the
   helper):

   ```bash
   anchorbench add-model my-org/my-cool-model-7b
   ```

   The helper appends `conf/model/my_cool_model_7b.yaml` and prints a
   smoke-test command. Edit the file by hand to set
   `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`,
   and the figure/table label (`short`).

2. (Open-weight only) add the new model to a tier in
   `conf/tier/<tier>.yaml` so `experiment=paper_main` picks it up:

   ```yaml
   models:
     - qwen_7b
     - my_cool_model_7b
   ```

3. Smoke-test on one suite:

   ```bash
   anchorbench eval data=external model=my_cool_model_7b model.batch_size=8
   ```

4. Run all five suites:

   ```bash
   for suite in external history icl rag tool; do
     anchorbench eval data=$suite model=my_cool_model_7b
   done
   ```

5. Refresh paper artifacts:

   ```bash
   python -m anchorbench.analysis.unified --results_dir results/full_benchmark
   anchorbench tables --paper
   ```

The model appears in every figure/table automatically because the
generators key on the unified summary files plus `conf/tier/*.yaml`
ordering.

---

## Add a Suite

A suite is a (renderer, item generator, runner, config) quadruple.

1. **Renderer**: add `src/anchorbench/data/suites/<suite>.py`
   exporting `render_<suite>(item: ItemSpec, **kwargs) -> list[PromptView]`.
   Register it in `src/anchorbench/data/suites/__init__.py` under
   `SUITE_RENDERERS`.

2. **Item generator** (if the existing helpers in
   `src/anchorbench/data/itemspec_gen.py` are not enough): add a
   `generate_<suite>_itemspecs` function and wire it into
   `_ITEMSPEC_GENERATORS` in `src/anchorbench/data/generate.py`.

3. **Runner**: copy `src/anchorbench/runners/external.py` to
   `src/anchorbench/runners/<suite>.py` and add any suite-specific
   flags. The runner should re-use
   `anchorbench.runners.base.add_common_args` and
   `anchorbench.runners.base.make_backend`.

4. **Hydra config**: add `conf/data/<suite>.yaml`:

   ```yaml
   suite: <suite>
   dataset_path: datasets/anchorbench_<suite>_core
   conditions: [control, irrelevant, plausible]
   ```

5. **Generate, validate, run**:

   ```bash
   anchorbench generate data=<suite>
   bash scripts/validate_all.sh core
   anchorbench eval data=<suite> model=qwen_7b
   ```

6. **Paper artifacts**: extend
   `src/anchorbench/paper/tables_appendix.py` (and optionally
   `tables_main.py`) so the new suite gets a per-suite table.

---

## Add a Condition

Conditions are the columns of every per-item record (control,
irrelevant, plausible, plus optional ablations such as `placebo_*`,
`authority_*`, `neutral_*`).

1. Extend `CONDITIONS` (paper-canonical) or `EXTENDED_CONDITIONS`
   (ablation-only) in `src/anchorbench/eval/metrics.py`.

2. Update each renderer in `src/anchorbench/data/suites/` that should
   emit the new condition. Each renderer owns its own prompt
   construction; only renderers where the new condition makes sense
   need to be touched.

3. If the condition feeds into a metric (e.g. a new TAR variant), add
   a helper in `metrics.py` and surface it in
   `compute_unified_metrics` or `compute_extended_metrics`.

4. Re-run inference for the affected (model, suite) cells. Existing
   conditions remain untouched in the JSONL files; only the new column
   is added.

---

## Add a Metric

1. Define the metric in `src/anchorbench/eval/metrics.py` (a function
   that takes per-item records and returns a scalar plus optional
   confidence interval). Re-use `bootstrap_ci`, `paired_wilcoxon`, and
   `bh_correction` for statistical helpers.

2. Add the metric to the unified summary by extending
   `compute_unified_metrics` (paper-canonical) or
   `compute_extended_metrics` (ablation-only).

3. Re-emit unified files:

   ```bash
   python -m anchorbench.analysis.unified --results_dir results/full_benchmark
   python -m anchorbench.analysis.unified --results_dir results/api_benchmark
   ```

4. Add a column to the relevant generator under
   `src/anchorbench/paper/tables_*.py`. If the metric belongs in the
   main table, also extend `tab_main_results.tex` and the corresponding
   helper in `_common.py`.

5. If the metric maps to a paper claim, register it in
   `src/anchorbench/paper/verify.py` so future drift is caught by
   `anchorbench verify`.

---

## Add a Named Experiment

Useful when a rebuttal asks for a self-contained run that is more than
a single (model, suite) cell.

1. Create `conf/experiment/rebuttal_long_anchor.yaml`:

   ```yaml
   defaults:
     - override /data: external
     - override /decoding: greedy

   description: Long anchor sentences ablation
   suites: [external]
   tiers:  [medium]
   out_dir: results/rebuttal/long_anchor
   ```

2. Run it:

   ```bash
   anchorbench experiment +experiment=rebuttal_long_anchor
   ```

3. (Optional) Pin its expected outputs under
   `conf/experiment/paper_run.yaml` so the reproducibility verifier
   checks for it.

---

## Where to Look

| Concern                | File |
|------------------------|------|
| Model + tier config    | `conf/model/*.yaml`, `conf/tier/*.yaml` |
| Decoding profile       | `conf/decoding/*.yaml` |
| Named experiment       | `conf/experiment/*.yaml` |
| CLI dispatch           | `src/anchorbench/cli/main.py` |
| Per-suite runners      | `src/anchorbench/runners/*.py` |
| Renderers              | `src/anchorbench/data/suites/` |
| Item generation        | `src/anchorbench/data/generate.py` |
| Metrics + statistics   | `src/anchorbench/eval/metrics.py` |
| Unified summary        | `src/anchorbench/analysis/unified.py` |
| Paper figures + tables | `src/anchorbench/paper/` |
| Numeric verification   | `src/anchorbench/paper/verify.py` |

For end-to-end reproduction commands, see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). For the package layout and
data flow, see [ARCHITECTURE.md](ARCHITECTURE.md).
