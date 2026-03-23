# AnchorBench Statistical Analysis Summary

## Source files used
- `results/full_benchmark/unified_all_suites.json` (machine-readable model-suite summaries)
- `results/full_benchmark/**/results.jsonl` (item-level records)

## Feasible analyses performed
- Bootstrap CIs for `Acc10_c`, `UAI_irr`, `UAI_pls`, `DiscDelta`, `Parse` at model-suite level from item-level vectors.
- Paired nonparametric tests (Wilcoxon signed-rank) for requested comparisons where pairing is valid.
- BH multiple-comparison correction across reported raw p-values.

## Bootstrap method
- Nonparametric bootstrap over analysis units with replacement, `n_boot=2000`, `seed=42`, two-sided 95% percentile CI.
- Item-level vectors are suite/metric-specific; parse failures are preserved and explicitly reflected in parse-rate vectors.

## Parse and missing-data handling
- `Parse` uses all records (parsed vs not parsed).
- `Acc10_c` uses control items with parsed control answers.
- `UAI`/`DiscDelta` use items with required parsed control+anchor answers and denominator filter `|a - y_ctrl| >= 3.0`.

## Limitations / blockers
- Tool-Read vs Tool-Agentic is blocked in `results/full_benchmark`: those suites are not present as source-of-truth runs.
- History vs ICL item-level pairing is not valid because suites are different task interfaces; model-level paired tests were used conservatively.
- For some broad claims (e.g., interface dependence omnibus), we report bootstrap effect CIs without forcing potentially invalid omnibus p-values.

## Dataset scope
- Model-suite runs: 50 (10 models x 5 suites)
- Parse-rate range across runs: [0.0056, 1.0000]

## Output artifacts
- `outputs/stats/bootstrap_model_suite_metrics.csv` and `.json`
- `outputs/tables/main_claims_statistical_table.csv` and `.tex`
- `outputs/tables/appendix_bootstrap_metrics_table.csv` and `.tex`
- `outputs/tables/appendix_paired_comparisons_table.csv` and `.tex`
- `outputs/stats/STATISTICAL_ANALYSIS_SUMMARY.md`