# Freeze Decision Memo

**Date:** 2026-03-10  
**Purpose:** Practical summary of what must happen before and after the benchmark freeze.

---

## 1. Final Recommended Primary Benchmark Scope

- **5 suites:** External, History, ICL, RAG, Tool
- **5 conditions per item:** control, irrelevant_low, irrelevant_high, plausible_low, plausible_high
- **6 domains**, balanced
- **2 difficulty levels** (easy, hard), balanced
- **3 anchor offsets** (15, 25, 40) for External/ICL/RAG/Tool; fixed 25 for History
- **Pilot scale:** 780 items, 3,900 prompt views (after External fix)
- **4 models:** Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct
- **Greedy decoding** (temperature=0, max_tokens=512)
- **3 primary metrics:** UAI, TAR, Disc_delta
- **Parser:** regex primary + LLM fallback

---

## 2. Optional Extensions (do NOT block submission)

| Extension | Status | Notes |
|---|---|---|
| Informative relevance level | Not implemented | Future validation only |
| Stress/robustness subset | Removed (`stress.py` was unused by pipeline) | Can reintroduce if needed |
| Ecological tool-calling | Implemented (`run_tool_ecological.py`) | ~50 items, appendix only |
| Mitigation strategies (B1–B3, SELFHELP, LATE_BINDING) | Implemented (`src/mitigation_eval/`) | Uses 3-condition layout; appendix |
| Mechanistic interpretability | Implemented (`src/mechinterp/`) | Appendix |
| Core/full scale (10 items/cell) | Not generated | Future extension |
| Sampling-based analysis (temperature > 0) | Not run | Future extension |
| Epsilon sensitivity sweep | Trivial to compute | Appendix |
| Per-domain / per-difficulty breakdowns | Trivial to compute | Appendix |

---

## 3. Unresolved Risks

### High Priority

| Risk | Impact | Mitigation |
|---|---|---|
| External pilot uses only 3 domains | Paper claims 6 domains; reviewer will notice | Regenerate External pilot with 6 domains before final runs [MUST] |
| Metric inconsistency across scripts | Local `compute_metrics()` in run_external/rag/history uses different UAI formula | Unify all scripts to call `compute_unified_metrics()` [MUST] |
| No bootstrap CIs in unified_metrics | Paper should report CIs for credibility | Add bootstrap to `unified_metrics.py` [SHOULD] |

### Medium Priority

| Risk | Impact | Mitigation |
|---|---|---|
| History control is single-stage vs. two-stage anchored | Potential confound; reviewer may object | Already documented in paper Limitations section; conservative choice |
| Tool suite parse rate for Llama (79–86%) | Lower than other suites | Report parse rates; discuss in paper |
| Fallback parser device defaults differ across scripts | Minor reproducibility issue | Standardize to `--fallback_device cuda:0` [SHOULD] |

### Low Priority

| Risk | Impact | Mitigation |
|---|---|---|
| Legacy code still referenced by mechinterp/mitigation | Confusing for reviewers reading code | Document in README that mechinterp/mitigation use 3-condition layout |
| `anchor_preambles["comparative"]` defined but never used | Dead code | Remove from domains.py [SHOULD] |

---

## 4. Must Fix Before Full Runs

These are blocking issues. Do not start final model runs until these are resolved.

1. **Regenerate External pilot with 6 domains.** Change `generate_external_itemspecs` to default to `DOMAIN_IDS` (all 6) instead of `DOMAIN_IDS[:3]`. Regenerate with `--size pilot --seed 42`. This will produce 180 items (matching ICL/RAG/Tool) instead of 90. [MUST]

2. **Unify metrics across all eval scripts.** Replace local `compute_metrics()` in `run_external.py`, `run_rag.py`, and `run_history.py` with calls to `compute_unified_metrics()` from `unified_metrics.py`. Ensure all scripts use `epsilon = 3` and `denom = a - y_control`. [MUST]

3. **Update paper to match actual counts.** After External regeneration, update `paper/sections/dataset.tex` and `paper/sections/results.tex` to reflect 180 External items and 6 domains. [MUST]

4. **Validate all five pilot datasets.** Run the full validation suite (`validate_all` from `validators.py`) against all five pilot datasets. Fix any errors. [MUST]

---

## 5. Can Wait Until After Full Experiments

These are improvements that should not delay experiment runs.

- Add bootstrap CIs to `unified_metrics.py` [SHOULD — before final paper tables]
- Standardize fallback device defaults across scripts [SHOULD]
- Remove unused `anchor_preambles["comparative"]` from `domains.py` [SHOULD]
- Create `DATASET_CARD.md` and `REPRODUCIBILITY.md` docs [MUST — before submission, not before runs]
- Anonymize git history for artifact release [MUST — before submission]
- Epsilon sensitivity analysis [OPTIONAL — appendix]
- Per-domain and per-difficulty breakdowns [OPTIONAL — appendix]
- Clean up legacy condition naming in mechinterp/mitigation code [SHOULD — before submission]

---

*End of memo.*
