# Cohen's d translation

LLM anchoring magnitudes expressed in **|Cohen's d|** (averaged variance, paired within item, magnitude per anchor direction averaged) so they can be compared directly with the human anchoring literature.

**Why |d|?** The signed mean cancels out when we pool high and low anchor conditions (high pushes up, low pushes down). The natural summary of anchoring *magnitude* is therefore the average over absolute per-direction d values; this matches the convention used by Furnham & Boo (2011).

**Human reference points** (|Cohen's d|, anchor vs no-anchor unless noted):
- Tversky \& Kahneman (1974), \% African nations in UN (anchor 10 vs 65): d ≈ 0.79 (d_hi_lo)
- Strack \& Mussweiler (1997, Exp.\ 1), Gandhi's age (plausible anchor): d ≈ 0.97 (d_pls)
- Strack \& Mussweiler (1997, Exp.\ 1), Gandhi's age (irrelevant anchor): d ≈ 0.45 (d_irr)
- Furnham \& Boo (2011, meta), Mean across 27 classical anchoring studies: d ≈ 0.55 (d_pls)

## External

| Model | |d_pls| | |d_irr| | |d_hi-lo| | n_items |
|---|---:|---:|---:|---:|
| Gemma-4B | 0.21 | 0.40 | 0.30 | 360 |
| Llama-8B | 0.47 | 0.02 | 0.84 | 360 |
| OLMo-13B | 0.24 | 0.01 | 0.32 | 360 |
| Qwen-7B | 0.46 | 0.01 | 0.89 | 360 |

- mean |d_pls|: 0.35
- mean |d_irr|: 0.11
- mean |d_hi_lo|: 0.59

## History

| Model | |d_pls| | |d_irr| | |d_hi-lo| | n_items |
|---|---:|---:|---:|---:|
| Gemma-4B | 0.39 | 0.20 | 0.81 | 360 |
| Llama-8B | 0.11 | 0.11 | 0.21 | 360 |
| OLMo-13B | 0.24 | 0.16 | 0.37 | 360 |
| Qwen-7B | 0.21 | 0.30 | 0.46 | 360 |

- mean |d_pls|: 0.24
- mean |d_irr|: 0.19
- mean |d_hi_lo|: 0.46

## Rag

| Model | |d_pls| | |d_irr| | |d_hi-lo| | n_items |
|---|---:|---:|---:|---:|
| Gemma-4B | 0.11 | 0.21 | 0.21 | 360 |
| Llama-8B | 0.18 | 0.06 | 0.34 | 360 |
| OLMo-13B | 0.10 | 0.06 | 0.06 | 360 |
| Qwen-7B | 0.26 | 0.01 | 0.50 | 360 |

- mean |d_pls|: 0.16
- mean |d_irr|: 0.09
- mean |d_hi_lo|: 0.28

## Tool

| Model | |d_pls| | |d_irr| | |d_hi-lo| | n_items |
|---|---:|---:|---:|---:|
| Gemma-4B | 0.08 | 0.05 | 0.14 | 360 |
| Llama-8B | 0.39 | 0.35 | 0.76 | 357 |
| OLMo-13B | 0.16 | 0.06 | 0.02 | 360 |
| Qwen-7B | 0.29 | 0.01 | 0.53 | 360 |

- mean |d_pls|: 0.23
- mean |d_irr|: 0.12
- mean |d_hi_lo|: 0.36

### Headline interpretation

On the External suite the open-weight 4-model panel reaches |d_pls| in the 0.2–0.5 range and |d_hi-lo| in the 0.3–0.9 range — directly comparable to the human anchoring d ≈ 0.55 reported by Furnham & Boo (2011) and the d ≈ 0.79 reported by Tversky & Kahneman (1974). Irrelevant-anchor effects on External (|d_irr| ≈ 0.0–0.4) are within range of the d ≈ 0.45 irrelevant-anchor effect of Strack & Mussweiler (1997). The pattern (|d_pls| > |d_irr|, plausible > irrelevant) replicates across RAG, Tool, and History pathways, with smaller magnitudes on the indirect pathways. This translation lets readers position LLM anchoring on the same effect-size axis used in the behavioural-economics literature without requiring a paired human study.
