# Core vs. Appendix Structure

This document specifies what goes into the main paper body versus the appendix.

## Main Paper (Core Results)

### Benchmark design (§3 Setup)
- 6 suites × 5 core conditions = the main comparison
- Core conditions: control, irrelevant_low/high, plausible_low/high
- 6 domains × 2 difficulties
- Gold answer construction
- Tool-Read vs Tool-Agentic as separate suites (distinct channels)

### Metrics (§3.2)
- UAI_irr, UAI_plaus, TAR_irr, TAR_plaus, Disc_Δ, MAE_control, Acc10_control
- Bootstrap 95% CIs for all primary metrics
- Paired Wilcoxon signed-rank test + BH correction for UAI ≠ 0 and plaus ≠ irr
- Parse rate with per-condition denominators

### Main tables (§4 Results)
- **Table 1**: Cross-suite comparison — all models × all suites, showing UAI_irr, UAI_plaus, Disc_Δ with CIs and significance markers
- **Table 2**: History suite — including ACR, RR, control_twostage comparison
- **Table 3**: Tool-Read vs Tool-Agentic — side-by-side comparison

### Key figures
- **Figure 1**: UAI by suite (bar chart with CIs)
- **Figure 2**: Disc_Δ across model scale (trend plot)

### Inline findings
- Relevance discrimination: plausible > irrelevant (universally positive Disc_Δ in External)
- Suite ranking: which channels show strongest anchoring
- Scale effects: larger models show less/more susceptibility

---

## Appendix (Robustness & Ablation Results)

### A. Extended Controls

#### A.1 Placebo / Authority controls (External suite)
- UAI_placebo, UAI_authority compared to UAI_irrelevant, UAI_plausible
- Key question: Does placebo ≈ irrelevant (salience control)? Does authority > plausible (credibility gradient)?

#### A.2 History two-stage control
- control_twostage vs single-stage control
- Shows whether two-stage format itself (not the anchor) changes estimates
- Compute UAI with control_twostage as baseline → should be similar to UAI with control

#### A.3 ICL neutral-header ablation
- neutral_low/high: same numbers as irrelevant, but semantically neutral header
- Key question: Is the ICL effect purely numeric priming or does header semantics matter?

### B. RAG Ablations

#### B.1 Document order
- *_order_first (anchor in position 1) vs *_order_last (anchor in position 3) vs baseline (position 2)
- Tests primacy/recency effects in retrieval augmentation

#### B.2 Disclaimer removal
- irrelevant_*_nodiscl: removes "This identifier is assigned sequentially and is unrelated..."
- Tests whether explicit disclaimers reduce irrelevant anchor effects

#### B.3 Authority framing
- plausible_*_authority: upgrades source to "expert panel commissioned by regulatory body"
- Tests credibility gradient within the RAG channel

### C. Parse-Policy Sensitivity

- Table showing UAI/TAR/Disc_Δ under 4 parsing policies: strict_regex, regex_final, full_cascade, last_only
- Demonstrates results are robust to parsing choices
- Report max |Δ UAI| across policies

### D. Anchor-Distance Analysis

- Per-offset breakdown (offset = {15, 25, 40})
- Does UAI increase/decrease with anchor distance?
- Monotonicity check: is anchoring proportional to distance?

### E. Difficulty Analysis

- By-difficulty breakdown (easy vs hard)
- Key question: Are harder items (conflicting/missing evidence) more susceptible to anchoring?

### F. Full Per-Model Tables

- Complete metrics for all models × all suites (including ablation conditions)
- All CIs and p-values

### G. Dataset Construction Details

- ItemSpec generation procedure
- Evidence generation with difficulty-dependent noise
- Anchor offset stratification
- Domain-specific templates

---

## File Mapping

| Content | Paper location | Data source |
|---------|---------------|-------------|
| Core 5-condition results | §4 main tables | `promptviews_core.jsonl` |
| Placebo/authority (External) | Appendix A.1 | `promptviews_ablation.jsonl` (external) |
| Two-stage control (History) | Appendix A.2 | `promptviews_ablation.jsonl` (history) |
| Neutral-header (ICL) | Appendix A.3 | `promptviews_ablation.jsonl` (icl) |
| Order/disclaimer/authority (RAG) | Appendix B | `promptviews_ablation.jsonl` (rag) |
| Parse sensitivity | Appendix C | `parse_sensitivity.py` output |
| Per-offset | Appendix D | `compute_by_offset()` in `metrics.py` |
| Per-difficulty | Appendix E | `compute_by_difficulty()` in `metrics.py` |
| Full tables | Appendix F | `unified_all_suites.json` |

## Metrics Available

| Metric | Core | Appendix |
|--------|------|----------|
| UAI_irr, UAI_plaus | ✓ (with 95% CI) | ✓ (all ablation conditions) |
| TAR_irr, TAR_plaus | ✓ | ✓ |
| Disc_Δ | ✓ (with CI + p-value) | ✓ |
| MAE_control, Acc10_control | ✓ (with CI) | ✓ |
| Parse rate | ✓ (aggregate) | ✓ (per-condition denominators) |
| Bootstrap CIs | ✓ (all primary metrics) | ✓ |
| Wilcoxon + BH | ✓ (plaus≠irr, UAI≠0) | ✓ (all comparisons) |
| ACR, RR (History only) | ✓ | ✓ |
| UAI_placebo, UAI_authority | — | ✓ |
| UAI_neutral (ICL) | — | ✓ |
| Per-offset breakdown | — | ✓ |
| Per-difficulty breakdown | — | ✓ |
| Parse-policy sensitivity | — | ✓ |
