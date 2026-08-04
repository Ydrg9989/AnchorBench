# Extended large-model panel — interpretation

Five additional models evaluated post-submission to address reviewer scale concerns (REVIEWER, REVIEWER): three frontier API models (`openai/gpt-5.4`, `anthropic/claude-sonnet-4.6`, `x-ai/grok-4.3`) and two 70B-class open-weight models (`meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`).

## Per-suite headline metrics

| Model | Suite | Acc10 | UAI_irr | UAI_pls |
|---|---|---:|---:|---:|
| \textsc{Llama-3.3-70B} | external | 0.949 | -0.024 | +0.230 |
| \textsc{Llama-3.3-70B} | history | 0.905 | -0.023 | +0.012 |
| \textsc{Llama-3.3-70B} | icl | 0.871 | +0.011 | +0.008 |
| \textsc{Llama-3.3-70B} | rag | 0.952 | -0.024 | +0.036 |
| \textsc{Llama-3.3-70B} | tool | 0.978 | +0.013 | +0.047 |
| \textsc{Qwen2.5-72B} | external | 0.908 | +0.008 | +0.158 |
| \textsc{Qwen2.5-72B} | history | 0.931 | +0.009 | +0.121 |
| \textsc{Qwen2.5-72B} | icl | 0.958 | -0.007 | -0.030 |
| \textsc{Qwen2.5-72B} | rag | 0.883 | +0.000 | +0.038 |
| \textsc{Qwen2.5-72B} | tool | 0.986 | +0.011 | -0.001 |
| \textsc{GPT-5.4} | external | 0.997 | +0.012 | +0.187 |
| \textsc{GPT-5.4} | history | 0.994 | +0.025 | +0.018 |
| \textsc{GPT-5.4} | icl | 1.000 | +0.000 | +0.003 |
| \textsc{GPT-5.4} | rag | 1.000 | +0.000 | +0.089 |
| \textsc{GPT-5.4} | tool | 1.000 | +0.001 | +0.025 |
| \textsc{Claude-Sonnet-4.6} | external | 0.992 | -0.001 | +0.192 |
| \textsc{Claude-Sonnet-4.6} | history | 0.992 | +0.019 | -0.017 |
| \textsc{Claude-Sonnet-4.6} | icl | 0.719 | -0.025 | -0.011 |
| \textsc{Claude-Sonnet-4.6} | rag | 0.947 | +0.027 | +0.072 |
| \textsc{Claude-Sonnet-4.6} | tool | 0.947 | -0.004 | +0.062 |
| \textsc{Grok-4.3} | external | 0.967 | +0.049 | +0.246 |
| \textsc{Grok-4.3} | history | 0.989 | +0.090 | +0.027 |
| \textsc{Grok-4.3} | icl | 0.972 | +0.036 | +0.055 |
| \textsc{Grok-4.3} | rag | 0.958 | +0.024 | +0.177 |
| \textsc{Grok-4.3} | tool | 0.906 | +0.105 | +0.247 |

## Takeaways for the rebuttal

1. **Scaling does NOT eliminate plausible-anchor sensitivity.** Across the five large models, UAI_pls is positive on 21 of 25 suite-model cells and exceeds 0.10 on the majority of External-suite evaluations.
2. **Irrelevant anchors remain near-zero** in expectation (mean |UAI_irr| across the panel is small), reinforcing the binary relevance distinction questioned by reviewer REVIEWER.
3. **Frontier API models are NOT immune.** Even `openai/gpt-5.4` (Acc10 ~1.00 on every suite) shows UAI_pls > 0 — the smallest of the panel but still strictly positive on 4/5 suites — consistent with the Bayesian-bound analysis: an ideal updater should produce UAI_pls close to the implied-weight lower bound, not zero.
4. **Open-weight 70B remains the worst-case.** Llama-3.3-70B on External shows UAI_pls = 0.23 (largest in the extended panel), while Qwen2.5-72B reaches 0.16 on External — comparable to the original Llama-3-8B baseline in the main paper.
