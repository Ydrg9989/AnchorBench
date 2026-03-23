# AnchorBench Claim-Facing Figures Summary

## Source files used
- `results/full_benchmark/unified_all_suites.json` (model × suite machine-readable metrics)

## Figures and claim mapping
- `disc_delta_heatmap`: supports claim (1), interface dependence of anchoring strength.
- `acc10_c_heatmap`: supports claim (3), capability differences independent from anchoring.
- `acc10_disc_two_panel_heatmap`: directly contrasts capability vs anchoring across interfaces.
- `uai_irr_vs_uai_plaus_dumbbell_by_suite`: supports claim (2), plausible vs irrelevant within interfaces.
- `appendix_small_multiples_acc10_vs_disc_by_suite` (optional): per-suite association diagnostics; avoids overclaiming a single global trend.

## Limitations / caveats
- No missing values were fabricated; any absent model-suite cells are shown as `NA` (gray).
- Tool suite metric cells missing: 0. Tool parse-rate range across models: [0.9694, 1.0000].
- Tool should be interpreted with protocol-compliance caveats; parse/protocol issues may contribute to observed behavior.

## Generated artifacts
- `outputs/figures/disc_delta_heatmap.png`
- `outputs/figures/disc_delta_heatmap.pdf`
- `outputs/figures/acc10_c_heatmap.png`
- `outputs/figures/acc10_c_heatmap.pdf`
- `outputs/figures/acc10_disc_two_panel_heatmap.png`
- `outputs/figures/acc10_disc_two_panel_heatmap.pdf`
- `outputs/figures/uai_irr_vs_uai_plaus_dumbbell_by_suite.png`
- `outputs/figures/uai_irr_vs_uai_plaus_dumbbell_by_suite.pdf`
- `outputs/figures/appendix_small_multiples_acc10_vs_disc_by_suite.png`
- `outputs/figures/appendix_small_multiples_acc10_vs_disc_by_suite.pdf`
- `outputs/figures/ANCHORBENCH_CLAIM_FIGURES_SUMMARY.md`