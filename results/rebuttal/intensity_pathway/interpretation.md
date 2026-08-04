# P1 — Cross-pathway plausibility-intensity

Addresses reviewer **REVIEWER-2** (matched intensity across pathways) and reinforces **REVIEWER-1** (graded relevance).

Same 3-point source-credibility scale (mild / standard / strong) applied to three pathways:

- External: preamble injected before the question.
- RAG: preamble forms the body of the anchored retrieved doc.
- History: preamble injected at Stage 2 alongside the complete evidence (compounds with the model-generated self-anchor).

## Per-pathway dose-response (mean UAI)

| Pathway | Mild | Standard | Strong | Δ(strong-mild) |
|---|---:|---:|---:|---:|
| External | 0.25 | 0.32 | 0.40 | 0.15 |
| RAG | 0.05 | 0.08 | 0.18 | 0.14 |
| History | 0.47 | 0.31 | 0.90 | 0.43 |

## Interpretation

A monotone Mild < Standard < Strong curve indicates the model weights source credibility (consistent with rational Bayesian updating). A flat-but-high curve indicates classical number-anchoring (the numeric value drives the shift independent of source). Cross-pathway: if the *shape* of the intensity curve is preserved across pathways, the underlying credibility-sensitivity mechanism is pathway-invariant.
