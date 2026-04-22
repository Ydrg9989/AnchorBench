# Extending AnchorBench

This guide describes the four most common rebuttal-time extensions:
adding a model, a suite, a condition, or a metric. Each section is
self-contained and lists the minimum set of files to touch.

The benchmark is driven by `configs/benchmark.yaml` and a small set of
Python entry points; the new unified CLI lives at `scripts/run.py`.

---

## Add a Model

For both open-weight (vLLM-served) and OpenRouter API models.

1. Append the HF Hub or OpenRouter id to `configs/benchmark.yaml`:

   ```yaml
   ow_models:                  # or: api_models
     - my-org/my-cool-model-7b
   model_short_names:
     my-org_my-cool-model-7b: MyModel-7B   # used in figure/table labels
   ```

2. Decide which GPU tier it belongs to (open-weight only) and add it
   under `model_tiers:`. Existing tiers expose
   `tensor_parallel`, `gpu_ids`, and `models` — re-use a tier or define
   a new one (e.g. `medium_extra`).

3. Smoke-test:

   ```bash
   # Or use:
   #   python scripts/run.py add-model my-org/my-cool-model-7b
   # which appends to YAML and prints this command.
   python scripts/run.py eval --model my-org/my-cool-model-7b \
       --suite external --gpu_ids 0 --batch_size 8
   ```

4. Run all five suites for the new model:

   ```bash
   for suite in external history icl rag tool; do
     python scripts/run.py eval --model my-org/my-cool-model-7b \
         --suite "$suite" --gpu_ids 0
   done
   ```

5. Recompute the unified summaries and regenerate paper artifacts:

   ```bash
   PYTHONPATH=src python scripts/eval/recompute_all_unified.py \
       --results_dir results/full_benchmark
   python scripts/run.py tables --paper
   ```

The model will appear in every paper figure/table automatically because
generators key on the unified summary files plus the YAML `model_tiers`
ordering.

---

## Add a Suite

A suite is a (renderer, item generator, runner) trio.

1. **Renderer**: add `src/anchorbench_v1/suites/<suite>.py` exporting
   `render_<suite>(item: ItemSpec, **kwargs) -> list[PromptView]`.
   Register it in
   [`src/anchorbench_v1/suites/__init__.py`](../src/anchorbench_v1/suites/__init__.py)
   under `SUITE_RENDERERS`.

2. **Item generator** (if needed): if the existing ItemSpec generator
   in `src/anchorbench_v1/itemspec.py` is not enough, add a generator
   under `src/anchorbench_v1/itemspec_<suite>.py` and wire it into
   `anchorbench_v1.generate.main`.

3. **Per-suite runner**: copy `scripts/eval/run_external.py` to
   `scripts/eval/run_<suite>.py` and add any suite-specific flags. The
   runner should accept the standard CLI from
   [`runner_utils.add_common_args`](../src/anchorbench_eval/runner_utils.py).
   Then register the runner in `SUITE_RUNNERS` inside
   [`scripts/run.py`](../scripts/run.py).

4. **Dataset config**: add the dataset directory to
   `configs/benchmark.yaml`:

   ```yaml
   suite_datasets:
     <suite>: datasets/anchorbench_<suite>_core
   ```

5. **Generate, validate, run**:

   ```bash
   bash scripts/generate_all.sh core 42        # rebuilds all suites
   bash scripts/validate_all.sh core
   python scripts/run.py eval --model <m> --suite <suite> --gpu_ids 0
   ```

6. **Paper artifacts**: extend `tables_appendix.py` (and optionally
   `tables_main.py`) so the new suite gets its own per-suite table.

---

## Add a Condition

Conditions are the columns of every per-item record (control,
irrelevant, plausible, plus optional ablations).

1. Extend `CONDITIONS` (paper-canonical) or `EXTENDED_CONDITIONS`
   (ablation-only) in
   [`src/anchorbench_eval/metrics.py`](../src/anchorbench_eval/metrics.py).
   The benchmark already supports `placebo_*`, `authority_*`,
   `neutral_*` for non-mainline experiments.

2. Update each renderer that should emit the new condition. Each
   renderer owns its prompt construction logic; only renderers that
   make sense for the new condition need to be touched.

3. If the condition feeds into a metric (e.g. a new TAR variant), add
   a helper in `metrics.py` and surface it in `compute_unified_metrics`
   or `compute_extended_metrics`.

4. Re-run inference for the affected (model, suite) cells. Existing
   conditions remain untouched in the JSONL files; only the new column
   is added.

---

## Add a Metric

1. Define the metric in
   [`src/anchorbench_eval/metrics.py`](../src/anchorbench_eval/metrics.py)
   (a function that takes per-item records and returns a scalar plus
   optional CI). Re-use `bootstrap_ci`, `paired_wilcoxon`, and
   `bh_correction` for statistical helpers.

2. Add the metric to the unified summary by extending
   `compute_unified_metrics` (paper-canonical) or
   `compute_extended_metrics` (ablation-only).

3. Re-emit unified files:

   ```bash
   PYTHONPATH=src python scripts/eval/recompute_all_unified.py \
       --results_dir results/full_benchmark
   PYTHONPATH=src python scripts/eval/recompute_all_unified.py \
       --results_dir results/api_benchmark
   ```

4. Add a column to the relevant generator under
   `scripts/eval/paper/tables_*.py`. If the metric belongs in the main
   table, also extend `tab_main_results.tex` and the corresponding
   helper in `_common.py`.

5. If the metric maps to a paper claim, register it in
   `scripts/verify_paper_tables.py` so future drift is caught.

---

## Add a Named Experiment

Useful when a rebuttal asks for a self-contained run that is more than
a single (model, suite) cell.

1. Add an entry under `experiments:` in `configs/benchmark.yaml`:

   ```yaml
   experiments:
     rebuttal_long_anchor:
       description: Long anchor sentences ablation
       suites: [external]
       model_tiers: [medium]
       decoding: {temperature: 0.0, max_tokens: 512}
       out_dir: results/rebuttal/long_anchor
   ```

2. Run it:

   ```bash
   python scripts/run.py experiment --name rebuttal_long_anchor
   ```

3. (Optional) Pin it inside `paper_run.expected_outputs` so the
   reproducibility verifier checks for it.

---

## Where to Look

| Concern                | File |
|------------------------|------|
| Model + tier config    | `configs/benchmark.yaml` |
| CLI dispatch           | `scripts/run.py` |
| Per-suite runners      | `scripts/eval/run_*.py` |
| Renderers              | `src/anchorbench_v1/suites/` |
| Item generation        | `src/anchorbench_v1/generate.py` |
| Metrics + statistics   | `src/anchorbench_eval/metrics.py` |
| Paper figures + tables | `scripts/eval/paper/` |
| Numeric verification   | `scripts/verify_paper_tables.py` |

For end-to-end reproduction commands, see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
