# Ablation Study Results Summary

## Data sources
- `results/ablation_full_benchmark/*/*/results.jsonl` (ablation runs)
- `results/full_benchmark/*/*/results.jsonl` (core controls for baseline)

## Coverage
- Suites analyzed: external, icl, rag, history
- Models per suite: 10 (all complete)
- Tool ablations are not included because no ablation promptviews exist for tool suites.

## Key observations (descriptive)
- External: authority/plausible-style ablations show stronger UAI than placebo on average (0.409 vs 0.063).
- ICL neutral-header ablation mean UAI: 0.008.
- History control_twostage mean shift vs core control: 6.649 (non-trivial; two-stage format can shift responses).
- RAG order effect (first vs last): 0.099 vs 0.092 mean UAI.
- RAG disclaimer removal (nodiscl) mean UAI: 0.026; irrelevant order-based baseline mean UAI: 0.041.

## Generated artifacts
- `outputs/stats/ablation_joined_item_level.csv`
- `outputs/tables/ablation_coverage_and_parse.(csv|tex)`
- `outputs/tables/ablation_condition_summary.(csv|tex)`
- `outputs/tables/ablation_model_summary.(csv|tex)`
- `outputs/figures/ablation_parse_rate_heatmap.png`
- `outputs/figures/ablation_parse_rate_heatmap.pdf`
- `outputs/figures/ablation_condition_uai_bar.png`
- `outputs/figures/ablation_condition_uai_bar.pdf`
- `outputs/figures/ablation_history_control_twostage_shift_hist.png`
- `outputs/figures/ablation_history_control_twostage_shift_hist.pdf`

## Notes
- Analysis is descriptive and anchored to core-control baselines from the same model/suite/item.
- No missing values were imputed.
- Parse failures are preserved in parse-rate reporting.