# AnchorBench Main-Paper Result Package

## Source of truth
- `results/full_benchmark/unified_all_suites.json` (machine-readable unified metrics)
- `results/full_benchmark/**/results.jsonl` (raw outputs for parse summaries)

## Main figures
- `outputs/figures/disc_delta_heatmap.(png|pdf)` supports claim (1): interface dependence.
- `outputs/figures/acc10_vs_disc_delta_scatter.(png|pdf)` supports claim (3): control accuracy vs robustness dissociation.
- `outputs/figures/uai_irr_vs_uai_plaus_scatter.(png|pdf)` supports claim (2): plausible > irrelevant trends.

## Main tables
- `outputs/tables/main_condensed_benchmark_summary.(csv|tex)` compact COLM-style table (`Acc10_c`, `DiscΔ`, `Parse`).
- `outputs/tables/optional_headline_effects_model_suite.(csv|tex)` optional effects with CI / BH-adjusted p-value at model-suite granularity.
- `outputs/tables/tab_benchmark_disc_delta_wide.tex` — model × suite **DiscΔ** (booktabs; missing → `---`).
- `outputs/tables/tab_benchmark_parse_rate_wide.tex` — model × suite **parse rate** (\%).
- `outputs/tables/tab_benchmark_condensed_pct.tex` — long-form Acc$_{10}^c$ / DiscΔ / Parse as percentages.
- Regenerate wide/percent tables: `python scripts/eval/export_latex_tables.py`.

## Appendix
- `outputs/tables/appendix_full_metric_table.(csv|tex)` full metric matrix.
- `outputs/tables/appendix_uai_decomposition.(csv|tex)` low/high UAI decomposition.
- `outputs/tables/appendix_parse_failure_summary.(csv|tex)` parse/protocol failure counts.

## Notes
- Tool suite has full metric coverage in current source-of-truth files, but parse/protocol failures remain non-negligible for some model-suite runs; this is explicitly surfaced via `Parse` and appendix parse-failure summary.
- No fabricated metrics were used.