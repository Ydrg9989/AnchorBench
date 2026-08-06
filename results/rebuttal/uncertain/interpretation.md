# Uncertain-judgment

Tests whether anchoring requires genuine judgment under uncertainty; the published task is arithmetic mean of visible numbers). We render External items with only k of 5 ratings visible, but score against the original full-5 mean. Now there is true epistemic uncertainty, and we can compare measured anchoring against the rational Bayesian ceiling.

## Rational Bayesian baseline (anchor weighted as 1 extra rating)

- k=1: rational UAI_pls ceiling = +0.500
- k=2: rational UAI_pls ceiling = +0.333
- k=3: rational UAI_pls ceiling = +0.250

## Per-model curves (mean UAI_pls)

| Model | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| Gemma-4B | 0.21 | 0.20 | 0.25 |
| Llama-8B | 0.68 | 0.36 | 0.50 |
| OLMo-13B | 0.48 | 0.34 | 0.29 |
| Qwen-7B | 0.53 | 0.34 | 0.36 |

## Interpretation

- **Measured UAI_pls > rational ceiling at the same k**: the model exceeds what a Bayesian updater treating the anchor as a single extra rating would predict — clear classical anchoring bias on top of legitimate updating.
- **Measured UAI_pls ≈ rational ceiling**: behaviour is indistinguishable from rational Bayesian updating; the published anchoring effect on standard External (k=5 visible) would then be defensible as evidence integration rather than bias.
- **Decreasing UAI_pls with k**: the model treats the anchor as one piece of evidence among many — exactly the rational behaviour. A flat curve indicates the anchor has fixed salience regardless of how much evidence is visible.
- **UAI_irr near zero at every k**: irrelevant anchors should stay ignored regardless of uncertainty.
