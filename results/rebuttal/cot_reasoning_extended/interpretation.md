# Reasoning-allowed CoT extension (Cluster F / B1)

Extended from 2 to 5 models and from 2 suites to 3 suites (External, RAG, History).

## External

- mean uai_irr: baseline=+0.051, CoT=+0.024, Δ=-0.027
- mean uai_plaus: baseline=+0.263, CoT=+0.214, Δ=-0.048
- mean disc_delta: baseline=+0.212, CoT=+0.190, Δ=-0.022

| Model | UAI_pls base | UAI_pls CoT | Δ UAI_pls | Disc base | Disc CoT | Δ Disc |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.4-mini | 0.13 | 0.14 | +0.01 | 0.12 | 0.14 | +0.02 |
| Gemma-4B | 0.37 | 0.23 | -0.14 | 0.28 | 0.22 | -0.06 |
| Llama-8B | 0.39 | 0.37 | -0.03 | 0.31 | 0.31 | -0.00 |
| OLMo-13B | 0.15 | 0.18 | +0.03 | 0.08 | 0.13 | +0.04 |
| Qwen-7B | 0.27 | 0.16 | -0.12 | 0.26 | 0.15 | -0.11 |

## Rag

- mean uai_irr: baseline=+0.028, CoT=+0.034, Δ=+0.007
- mean uai_plaus: baseline=+0.095, CoT=+0.096, Δ=+0.001
- mean disc_delta: baseline=+0.068, CoT=+0.062, Δ=-0.006

| Model | UAI_pls base | UAI_pls CoT | Δ UAI_pls | Disc base | Disc CoT | Δ Disc |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.4-mini | 0.09 | 0.08 | -0.01 | 0.09 | 0.08 | -0.00 |
| Gemma-4B | -0.05 | 0.25 | +0.31 | -0.03 | 0.19 | +0.22 |
| Llama-8B | 0.20 | 0.12 | -0.08 | 0.12 | 0.06 | -0.06 |
| OLMo-13B | 0.05 | -0.01 | -0.06 | -0.03 | -0.05 | -0.02 |
| Qwen-7B | 0.19 | 0.03 | -0.16 | 0.19 | 0.03 | -0.16 |

## History

- mean uai_irr: baseline=+0.026, CoT=+0.132, Δ=+0.106
- mean uai_plaus: baseline=+0.361, CoT=+0.238, Δ=-0.123
- mean disc_delta: baseline=+0.335, CoT=+0.106, Δ=-0.229

| Model | UAI_pls base | UAI_pls CoT | Δ UAI_pls | Disc base | Disc CoT | Δ Disc |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.4-mini | 0.02 | -0.03 | -0.05 | 0.03 | -0.03 | -0.06 |
| Gemma-4B | 0.57 | 0.64 | +0.07 | 0.32 | 0.33 | +0.00 |
| Llama-8B | 0.29 | 0.28 | -0.01 | 0.21 | 0.01 | -0.20 |
| OLMo-13B | 0.51 | 0.17 | -0.34 | 0.60 | 0.08 | -0.52 |
| Qwen-7B | 0.42 | 0.13 | -0.29 | 0.52 | 0.15 | -0.37 |

## Summary

- CoT reduces UAI_pls in 11/15 (suite, model) cells
- mean ΔUAI_pls (CoT - baseline) = -0.057
- UAI_pls > 0 under CoT in 13/15 cells (CoT mitigates but does NOT eliminate anchoring).
