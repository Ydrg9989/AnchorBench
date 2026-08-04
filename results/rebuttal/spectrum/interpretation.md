# Plausibility spectrum (Cluster H response)

Four-point spectrum on External: placebo / irrelevant / plausible / authority.

## Headline numbers (mean across panel)

- mean UAI_placebo = +0.093
- mean UAI_irrelevant = +0.054
- mean UAI_plausible = +0.270
- mean UAI_authority = +0.488

Monotone increase placebo < irrelevant < plausible < authority? **NO**.
A monotone increase would be consistent with the model rationally weighting source credibility; non-monotonicity (especially placebo close to irrelevant) is strong evidence of anchoring on the numeric value itself rather than its source.

## Per-model spectrum

| Model | Placebo | Irrelevant | Plausible | Authority |
|---|---:|---:|---:|---:|
| GPT-5.4-mini | -0.01 | +0.01 | +0.13 | +0.40 |
| Gemma-4B | +0.38 | +0.20 | +0.43 | +0.58 |
| Llama-8B | +0.08 | +0.07 | +0.36 | +0.42 |
| OLMo-13B | +0.02 | -0.01 | +0.16 | +0.49 |
| Qwen-7B | -0.00 | +0.01 | +0.27 | +0.56 |

## Bias-above-noise check

- 2/5 models have UAI_placebo > 0.05, i.e., shift toward a numeric value that is *literally just the document age in days*.
