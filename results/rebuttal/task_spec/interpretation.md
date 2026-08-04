# P4 — Task-specification ablation

Addresses reviewer **REVIEWER-2** (weak-model low control accuracy may reflect task confusion rather than anchoring) and reinforces answers to **REVIEWER-1** (gold = arithmetic mean) and **REVIEWER-1** (task realism).

Two new prompt suffixes on External, alongside the published baseline prompt:

- `+rule`: "Your estimate should be the unweighted arithmetic mean of the visible ratings, rounded to the nearest integer."
- `+judgment`: "Your estimate should be a weighted average of the available evidence, weighting each piece according to which sources you find more or less credible."

## Mean across panel

| Metric | Baseline | +Rule | +Judgment |
|---|---:|---:|---:|
| UAI_plausible | 0.33 | 0.06 | 0.26 |
| UAI_irrelevant | 0.09 | 0.01 | 0.12 |

## Interpretation

- **+Rule vs Baseline**: a meaningful drop in UAI_pls under the explicit-rule prompt indicates the published anchoring effect is partly attributable to task underspecification rather than to a deep numerical-anchoring tendency. Importantly, *some* residual UAI_pls under +Rule is the bias-attributable portion that cannot be explained by confusion alone.
- **+Judgment vs Baseline**: this serves as an upper-bound check; the explicit invitation to weight sources should raise UAI_pls toward (but not above) the Bayesian rational ceiling computed in Table 4 (A1).
- **UAI_irr** should be near-zero across all three conditions for capable models — irrelevant anchors should remain ignored regardless of task framing.
