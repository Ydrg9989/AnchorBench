# AnchorBench Full Benchmark Analysis

## 1. Dataset Summary

| Suite | Items | Prompts/Suite | Domains | Conditions | Difficulties | Version |
|-------|-------|---------------|---------|------------|-------------|---------|
| External | 360 | 1,800 | 6 | 5 core + 4 ablation | easy/hard | 2.1.0 |
| History | 360 | 1,800 | 6 | 5 core + 1 ablation | easy/hard | 2.1.0 |
| ICL | 360 | 1,800 | 6 | 5 core + 2 ablation | easy/hard | 2.1.0 |
| RAG | 360 | 1,800 | 6 | 5 core + 12 ablation | easy/hard | 2.1.0 |
| Tool | 360 | 1,800 | 6 | 5 core | easy/hard | 2.1.0 |
| **Total** | **1,800** | **9,000** | 6 | 5 core per item | 2 | 2.1.0 |

### Domains (6)
- Legal / Policy compliance
- Market / Demographics adoption
- Operations / Time estimation
- Pricing / Willingness-to-pay
- Resource consumption / Efficiency
- Transportation / Logistics reliability

### Core Conditions (5 per item)
- `control` (no anchor)
- `irrelevant_low` (anchor = θ - 30, arbitrary metadata)
- `irrelevant_high` (anchor = θ + 30, arbitrary metadata)
- `plausible_low` (anchor = θ - 30, weakly credible source)
- `plausible_high` (anchor = θ + 30, weakly credible source)

---

## 2. Models Evaluated (10)

| Model | Family | Parameters | Backend |
|-------|--------|-----------|---------|
| Qwen2.5-1.5B-Instruct | Qwen | 1.5B | HF |
| Qwen2.5-3B-Instruct | Qwen | 3B | HF |
| Qwen2.5-7B-Instruct | Qwen | 7B | HF |
| Llama-3.2-1B-Instruct | Llama | 1B | HF |
| Llama-3.2-3B-Instruct | Llama | 3B | HF |
| Llama-3.1-8B-Instruct | Llama | 8B | HF |
| Gemma-3-1b-it | Gemma | 1B | HF |
| Gemma-3-4b-it | Gemma | 4B | HF |
| OLMo-2-1124-13B-Instruct | OLMo | 13B | HF |
| OLMo-2-0325-32B-Instruct | OLMo | 32B | HF |

---

## 3. Experiment Completion Status

| Suite | Models Completed | Status |
|-------|-----------------|--------|
| External | 10/10 | ✅ Complete |
| History | 10/10 | ✅ Complete |
| ICL | 10/10 | ✅ Complete |
| RAG | 10/10 | ✅ Complete |
| Tool | 10/10 | ✅ Complete (Llama-1B re-run with plaintext) |
| **Total** | **50/50** | **✅ All complete** |

---

## 4. Full Results Table

### 4.1 External Suite

| Model | MAE_c | Acc10 | UAI_irr | UAI_pls | TAR_irr | TAR_pls | Disc_Δ | Parse |
|-------|-------|-------|---------|---------|---------|---------|--------|-------|
| Gemma-1B | 15.55 | 36.3% | 0.01 | 0.16 | 0.29 | 0.47 | 0.15 | 99.4% |
| Qwen-1.5B | 18.05 | 33.6% | 0.02 | 0.22 | 0.22 | 0.49 | 0.20 | 100% |
| Llama-1B | 15.74 | 52.9% | 0.23 | 0.24 | 0.37 | 0.51 | 0.01 | 92.3% |
| Llama-3B | 10.43 | 67.6% | 0.08 | 0.28 | 0.37 | 0.62 | 0.21 | 99.2% |
| Qwen-3B | 9.36 | 65.1% | 0.01 | 0.11 | 0.22 | 0.49 | 0.10 | 99.1% |
| Gemma-4B | 15.18 | 60.5% | 0.18 | 0.41 | 0.55 | 0.67 | 0.22 | 98.4% |
| Qwen-7B | 7.71 | 73.9% | 0.01 | 0.29 | 0.16 | 0.67 | 0.28 | 100% |
| Llama-8B | 5.05 | 85.4% | 0.04 | 0.32 | 0.21 | 0.70 | 0.28 | 98.8% |
| OLMo-13B | **3.75** | **88.3%** | 0.03 | 0.10 | 0.26 | 0.48 | 0.07 | 99.4% |
| OLMo-32B | 13.07 | 51.9% | 0.05 | 0.40 | 0.13 | 0.55 | **0.35** | 100% |

### 4.2 History Suite

| Model | MAE_c | Acc10 | UAI_irr | UAI_pls | TAR_irr | TAR_pls | Disc_Δ | Parse |
|-------|-------|-------|---------|---------|---------|---------|--------|-------|
| Gemma-1B | 14.71 | 40.9% | -0.06 | 0.15 | 0.46 | 0.45 | 0.21 | 99.8% |
| Qwen-1.5B | 5.49 | 83.6% | 0.04 | **0.93** | 0.38 | 0.81 | **0.89** | 99.9% |
| Llama-1B | 15.71 | 51.7% | 0.52 | 0.53 | 0.65 | 0.63 | 0.01 | 93.5% |
| Llama-3B | 9.46 | 69.0% | 0.12 | 0.34 | 0.49 | 0.62 | 0.22 | 98.1% |
| Qwen-3B | 3.65 | 88.8% | 0.02 | 0.21 | 0.28 | 0.47 | 0.18 | 99.2% |
| Gemma-4B | 12.03 | 67.1% | 0.20 | 0.43 | 0.55 | 0.64 | 0.24 | 98.9% |
| Qwen-7B | 3.77 | **91.9%** | -0.01 | 0.17 | 0.32 | 0.48 | 0.18 | 100% |
| Llama-8B | 7.02 | 80.7% | 0.21 | 0.29 | 0.33 | 0.44 | 0.08 | 98.5% |
| OLMo-13B | 5.40 | 86.9% | -0.13 | 0.16 | 0.29 | 0.41 | 0.29 | 98.8% |
| OLMo-32B | **4.13** | 87.8% | -0.04 | 0.07 | 0.33 | 0.33 | 0.12 | 100% |

### 4.3 ICL Suite

| Model | MAE_c | Acc10 | UAI_irr | UAI_pls | TAR_irr | TAR_pls | Disc_Δ | Parse |
|-------|-------|-------|---------|---------|---------|---------|--------|-------|
| Gemma-1B | 13.38 | 46.4% | -0.00 | -0.03 | 0.16 | 0.24 | -0.02 | 100% |
| Qwen-1.5B | 10.25 | 59.2% | 0.01 | 0.00 | 0.23 | 0.24 | -0.01 | 100% |
| Llama-1B | 15.56 | 54.2% | 0.22 | 0.15 | 0.37 | 0.41 | -0.08 | 93.8% |
| Llama-3B | 7.49 | 78.2% | 0.08 | 0.09 | 0.37 | 0.46 | 0.00 | 99.2% |
| Qwen-3B | 7.16 | 77.5% | -0.04 | -0.04 | 0.15 | 0.20 | 0.00 | 100% |
| Gemma-4B | 13.52 | 38.9% | 0.00 | -0.07 | 0.31 | 0.33 | -0.08 | 100% |
| Qwen-7B | **5.50** | **87.5%** | -0.03 | -0.01 | 0.17 | 0.23 | 0.01 | 100% |
| Llama-8B | 7.26 | 77.0% | 0.02 | 0.18 | 0.40 | 0.52 | **0.17** | 97.1% |
| OLMo-13B | 10.23 | 56.9% | 0.01 | -0.01 | 0.17 | 0.29 | -0.03 | 99.9% |
| OLMo-32B | 8.14 | 69.7% | -0.02 | -0.01 | 0.14 | 0.18 | 0.01 | 100% |

### 4.4 RAG Suite

| Model | MAE_c | Acc10 | UAI_irr | UAI_pls | TAR_irr | TAR_pls | Disc_Δ | Parse |
|-------|-------|-------|---------|---------|---------|---------|--------|-------|
| Gemma-1B | 14.98 | 37.2% | -0.00 | 0.04 | 0.12 | 0.23 | 0.04 | 100% |
| Qwen-1.5B | 22.57 | 20.9% | 0.04 | 0.25 | 0.19 | 0.49 | 0.20 | 99.9% |
| Llama-1B | 14.08 | 58.6% | 0.04 | 0.14 | 0.26 | 0.36 | 0.10 | 95.1% |
| Llama-3B | 10.74 | 70.8% | 0.16 | 0.12 | 0.31 | 0.39 | -0.04 | 99.4% |
| Qwen-3B | 8.22 | 73.8% | -0.02 | 0.07 | 0.15 | 0.39 | 0.10 | 99.8% |
| Gemma-4B | 13.57 | 45.6% | 0.00 | -0.04 | 0.22 | 0.31 | -0.05 | 99.9% |
| Qwen-7B | 9.83 | 63.9% | -0.00 | 0.20 | 0.11 | 0.48 | 0.20 | 100% |
| Llama-8B | 4.85 | 84.9% | -0.04 | 0.13 | 0.21 | 0.36 | 0.17 | 99.3% |
| OLMo-13B | **4.58** | **87.7%** | 0.00 | 0.02 | 0.27 | 0.34 | 0.02 | 99.5% |
| OLMo-32B | 10.92 | 58.6% | -0.00 | **0.45** | 0.06 | 0.57 | **0.45** | 100% |

### 4.5 Tool Suite

| Model | MAE_c | Acc10 | UAI_irr | UAI_pls | TAR_irr | TAR_pls | Disc_Δ | Parse |
|-------|-------|-------|---------|---------|---------|---------|--------|-------|
| Gemma-1B | 20.50 | 25.8% | -0.04 | -0.03 | 0.23 | 0.31 | 0.02 | 99.9% |
| Qwen-1.5B | 26.15 | 45.9% | 0.24 | **0.82** | 0.28 | 0.72 | **0.58** | 97.5% |
| Llama-1B | 16.57 | 55.6% | 0.14 | 0.12 | 0.45 | 0.49 | -0.03 | 93.6% |
| Llama-3B | 3.07 | 92.0% | 0.01 | 0.25 | 0.19 | 0.37 | 0.23 | 79.7% |
| Qwen-3B | 4.45 | 89.7% | 0.09 | 0.17 | 0.24 | 0.38 | 0.08 | 97.7% |
| Gemma-4B | 15.20 | 60.8% | 0.24 | 0.27 | 0.52 | 0.56 | 0.03 | 96.9% |
| Qwen-7B | **0.67** | **98.9%** | 0.00 | 0.17 | 0.08 | 0.51 | 0.17 | 100% |
| Llama-8B | 12.13 | 73.3% | 0.31 | 0.47 | 0.28 | 0.65 | 0.16 | 95.5% |
| OLMo-13B | 8.73 | 70.2% | 0.15 | 0.13 | 0.36 | 0.45 | -0.02 | 99.7% |
| OLMo-32B | 7.88 | 72.5% | -0.00 | 0.04 | 0.10 | 0.16 | 0.05 | 100% |

---

## 5. Suite-Level Summary Statistics

| Suite | Mean UAI_pls | Mean Disc_Δ | Mean Acc10 | Mean Parse |
|-------|-------------|-------------|-----------|-----------|
| External | 0.25 | 0.19 | 61.5% | 98.7% |
| History | 0.33 | 0.24 | 74.8% | 98.7% |
| ICL | 0.02 | -0.00 | 64.5% | 99.0% |
| RAG | 0.14 | 0.12 | 60.2% | 99.3% |
| Tool | 0.24 | 0.13 | 68.5% | 96.0% |

---

## 6. Key Findings

### Finding 1: Anchoring is Interface-Dependent
- Disc_Δ > 0 in 41/50 model-suite combinations (82%)
- ICL shows near-zero effects; External and History show strongest effects
- Model rankings change across suites (no model is consistently best/worst)

### Finding 2: ICL Priming is Qualitatively Different
- Mean UAI_pls = 0.02 across all models on ICL
- Mean Disc_Δ = -0.00 on ICL
- Metadata numbers in few-shot demos don't function as effective anchors

### Finding 3: Accuracy and Discrimination are Orthogonal
- OLMo-13B: highest accuracy on External (88.3%) but low discrimination (0.07)
- Qwen-1.5B: moderate History accuracy (83.6%) but extreme discrimination (0.89)
- Within-family scale does not monotonically improve robustness

### Finding 4: Notable Outliers
- Qwen-1.5B shows UAI_pls = 0.93 on History and 0.82 on Tool
- OLMo-32B shows UAI_pls = 0.45 on RAG (highest RAG discrimination)
- Llama-1B shows indiscriminate anchoring on History (UAI_irr ≈ UAI_pls ≈ 0.52)

---

## 7. Generated Figures

| Figure | Description | Path |
|--------|-------------|------|
| Fig 1 | Disc_Δ heatmap (model × suite) | `outputs/figures/fig1_disc_delta_heatmap.{png,pdf}` |
| Fig 2 | UAI dumbbell chart (irr vs pls, faceted by suite) | `outputs/figures/fig2_uai_dumbbell.{png,pdf}` |
| Fig 3 | Acc10 vs Disc_Δ bubble plot | `outputs/figures/fig3_acc_vs_disc_bubble.{png,pdf}` |
| Fig 4 | UAI_pls grouped bar chart with CIs | `outputs/figures/fig4_uai_plaus_bar.{png,pdf}` |
| Fig 5 | Parse rate heatmap | `outputs/figures/fig5_parse_rate_heatmap.{png,pdf}` |
| Fig 6 | Acc10 heatmap | `outputs/figures/fig6_acc10_heatmap.{png,pdf}` |
| Fig 7 | Two-panel: Acc10 + Disc_Δ heatmaps | `outputs/figures/fig7_acc_disc_two_panel.{png,pdf}` |
| Fig 8 | Suite comparison bar charts | `outputs/figures/fig8_suite_comparison.{png,pdf}` |

---

## 8. Statistical Notes

- Bootstrap CIs (n_boot=2000, seed=42) computed for all metrics
- Significance markers: * p < 0.05, ** p < 0.01, *** p < 0.001 (vs zero)
- BH correction applied for multiple comparisons
- ε = 3.0 filtering for UAI denominator
