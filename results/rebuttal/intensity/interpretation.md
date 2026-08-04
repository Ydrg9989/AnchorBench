# Plausibility-intensity probe (Cluster H / D1)

Three-point dose-response with the SAME numeric anchor at three source-credibility levels:

- **Mild**: "A back-of-envelope guess by a non-expert ... not been validated"
- **Standard**: "A preliminary analysis ... estimated" (published plausible condition)
- **Strong**: "Three independent peer-reviewed studies have converged on ..."

## Headline numbers (mean across panel)

- mean UAI_mild = +0.311
- mean UAI_standard = +0.374
- mean UAI_strong = +0.437
- mean Δ(strong - mild) = +0.126
- Monotone mild $\le$ standard $\le$ strong? **YES**.

## Per-model intensity curve

| Model | Mild | Standard | Strong | Δ (strong-mild) |
|---|---:|---:|---:|---:|
| Gemma-4B | +0.447 | +0.415 | +0.403 | -0.044 |
| Llama-8B | +0.259 | +0.394 | +0.417 | +0.158 |
| Qwen-7B | +0.228 | +0.311 | +0.493 | +0.265 |

## Interpretation

A monotone increase mild < standard < strong with positive $\Delta$(strong $-$ mild) indicates the model is at least partially weighting source credibility, consistent with rational Bayesian updating. A flat curve (mild $\approx$ standard $\approx$ strong) at high absolute UAI indicates the anchor's numeric value drives behavior independent of source credibility, i.e. classical anchoring bias.
