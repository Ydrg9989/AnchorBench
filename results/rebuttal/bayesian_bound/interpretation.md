# Bayesian-Bound Analysis
## Setup
- n_evidence = 5 (visible ratings on easy items)
- Reference w = 1.0 (anchor worth one evidence item)
- Rational UAI ceiling = w/(n+w) = 0.167
## Interpretation rules
- **Irrelevant anchors:** the rational weight is 0 (they carry no
  information about the answer). Any positive UAI on irrelevant
  anchors is unambiguously bias above rational updating.
- **Plausible anchors:** UAI_plaus > w/(n+w) implies the model
  weights the anchor more than the reference assumption. The
  implied weight w_imp = UAI/(1-UAI) * n is the smallest weight
  consistent with rationality.
- **w_imp > n** means the model treats a single anonymous anchor
  as more credible than ALL visible evidence combined --- a
  pattern that is hard to defend on rational grounds.
## Per-suite summary
### External
- mean UAI_pls = 0.216, mean UAI_irr = 0.047 (14 models)
- median implied w (plausible) = 1.03 (reference w = 1); 7/14 cells exceed reference; 0/14 cells imply w > n = 5 (anchor weighted more than all visible evidence combined)
- median implied w using the 95% CI LOWER bound for UAI_pls (conservative) = 0.75
- 5/14 cells have UAI_pls 95% CI entirely above the rational ceiling 0.167 (i.e., the anchor effect is significantly above rational at p<0.05)
- top 3 implied weights:
  - Gemma-4B: UAI_pls=0.428 => w_imp=3.74
  - OLMo-32B: UAI_pls=0.395 => w_imp=3.27
  - Llama-8B: UAI_pls=0.360 => w_imp=2.82

### History
- mean UAI_pls = 0.337, mean UAI_irr = 0.054 (14 models)
- median implied w (plausible) = 1.91 (reference w = 1); 8/14 cells exceed reference; 4/14 cells imply w > n = 5 (anchor weighted more than all visible evidence combined)
- median implied w using the 95% CI LOWER bound for UAI_pls (conservative) = 1.06
- 7/14 cells have UAI_pls 95% CI entirely above the rational ceiling 0.167 (i.e., the anchor effect is significantly above rational at p<0.05)
- top 3 implied weights:
  - Qwen-1.5B: UAI_pls=0.985 => w_imp=319.68
  - OLMo-32B: UAI_pls=0.752 => w_imp=15.17
  - Llama-3B: UAI_pls=0.727 => w_imp=13.32

### Rag
- mean UAI_pls = 0.129, mean UAI_irr = 0.025 (14 models)
- median implied w (plausible) = 0.49 (reference w = 1); 5/14 cells exceed reference; 0/14 cells imply w > n = 5 (anchor weighted more than all visible evidence combined)
- median implied w using the 95% CI LOWER bound for UAI_pls (conservative) = 0.26
- 2/14 cells have UAI_pls 95% CI entirely above the rational ceiling 0.167 (i.e., the anchor effect is significantly above rational at p<0.05)
- top 3 implied weights:
  - OLMo-32B: UAI_pls=0.447 => w_imp=4.04
  - Qwen-1.5B: UAI_pls=0.227 => w_imp=1.46
  - Llama-3B: UAI_pls=0.222 => w_imp=1.43

### Tool
- mean UAI_pls = 0.176, mean UAI_irr = 0.070 (14 models)
- median implied w (plausible) = 0.40 (reference w = 1); 6/13 cells exceed reference; 1/13 cells imply w > n = 5 (anchor weighted more than all visible evidence combined)
- median implied w using the 95% CI LOWER bound for UAI_pls (conservative) = 0.25
- 2/14 cells have UAI_pls 95% CI entirely above the rational ceiling 0.167 (i.e., the anchor effect is significantly above rational at p<0.05)
- top 3 implied weights:
  - Qwen-1.5B: UAI_pls=0.825 => w_imp=23.51
  - Llama-8B: UAI_pls=0.478 => w_imp=4.58
  - Gemma-4B: UAI_pls=0.215 => w_imp=1.37

### Icl
- mean UAI_pls = 0.026, mean UAI_irr = 0.025 (14 models)
- median implied w (plausible) = 0.00 (reference w = 1); 1/14 cells exceed reference; 0/14 cells imply w > n = 5 (anchor weighted more than all visible evidence combined)
- median implied w using the 95% CI LOWER bound for UAI_pls (conservative) = 0.00
- 0/14 cells have UAI_pls 95% CI entirely above the rational ceiling 0.167 (i.e., the anchor effect is significantly above rational at p<0.05)
- top 3 implied weights:
  - Llama-1B: UAI_pls=0.217 => w_imp=1.38
  - Llama-8B: UAI_pls=0.149 => w_imp=0.88
  - Llama-3B: UAI_pls=0.135 => w_imp=0.78

## Headline numbers
Across 55 non-ICL (model, suite) cells:
- **16/55 cells (29%) have the 95% CI of UAI_pls entirely above the rational ceiling (0.167, w=1).** This is the primary headline.
- median implied w (point estimate) = 0.94 (reference = 1; mean is skewed by tail outliers).
- median implied w using 95% CI lower bound = 0.52 (the most conservative estimate).
- 5/55 cells imply w > n = 5, i.e., the anchor is treated as more credible than all visible evidence combined --- a regime that is hard to defend on rational grounds.
- For irrelevant anchors, the rational weight is 0, so ANY positive UAI_irr is bias above rational updating (see Table 1 / Table 2 of the paper).
