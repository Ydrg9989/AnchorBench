# AnchorBench Supplementary Bubble Plot Summary

## Source file
- `results/full_benchmark/unified_all_suites.json`

## Metadata availability
- Chatbot Arena score: not found in repository; used fallback `mean_acc10_c` for x-axis.
- Parameter count: parsed from model names/slugs (B-scale values).
- Model family/developer: inferred from model slug (`Qwen`, `Llama`, `Gemma`, `OLMo`).

## Aggregation formulas (one row per model)
- `mean_abs_uai = average_over_suites( mean(|UAI_irr|, |UAI_pls|) )`
- `mean_disc = average_over_suites(DiscΔ)`
- `mean_abs_disc = average_over_suites(|DiscΔ|)`
- `mean_acc10_c = average_over_suites(Acc10_c)`

## Artifacts
- `outputs/figures/bubble_capability_vs_mean_abs_uai_all.png`
- `outputs/figures/bubble_capability_vs_mean_abs_uai_all.pdf`
- `outputs/figures/bubble_capability_vs_mean_disc_all.png`
- `outputs/figures/bubble_capability_vs_mean_disc_all.pdf`
- `outputs/figures/bubble_capability_vs_mean_abs_uai_excl_tool.png`
- `outputs/figures/bubble_capability_vs_mean_abs_uai_excl_tool.pdf`
- `outputs/figures/bubble_capability_vs_mean_disc_excl_tool.png`
- `outputs/figures/bubble_capability_vs_mean_disc_excl_tool.pdf`
- `outputs/stats/bubble_model_aggregate_all_suites.csv`
- `outputs/stats/bubble_model_aggregate_excl_tool.csv`
- `outputs/tables/bubble_model_aggregate_all_suites.tex`
- `outputs/tables/bubble_model_aggregate_excl_tool.tex`

## Interpretation caution
- This is a global supplementary summary, not a primary causal/inferential figure.
- Tool-inclusive and Tool-excluded versions are both provided to avoid overclaiming under protocol-sensitive settings.