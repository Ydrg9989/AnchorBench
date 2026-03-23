# Repository Alignment Plan

**Date:** 2026-03-10  
**Purpose:** Minimum set of changes needed to align the codebase with the frozen benchmark spec before full model runs.

---

## Priority 1: MUST Fix Before Runs

### 1.1 External Pilot Domain Bug

**Problem:** `generate_external_itemspecs()` in `src/anchorbench_v1/itemspec_gen.py` defaults to `domains or DOMAIN_IDS[:3]` (3 domains), while all other generators default to `domains or DOMAIN_IDS` (6 domains). The External pilot manifest shows only 3 domains; the paper claims 6.

**File:** `src/anchorbench_v1/itemspec_gen.py`

**Fix:** Change the default in `generate_external_itemspecs()`:
```python
# Before
domains = domains or DOMAIN_IDS[:3]
# After
domains = domains or DOMAIN_IDS
```

**Also fix:** `src/anchorbench_v1/generate.py` line ~246 where `effective_domains` may be overridden for External pilot. Ensure pilot uses all 6 domains.

**Impact:** External pilot must be regenerated. New count: 6 domains × 2 difficulties × 3 offsets × 5 items = 180 items, 900 promptviews.

**Severity:** MUST — paper-code conflict visible to reviewers.

---

### 1.2 Unify Metrics Across All Eval Scripts

**Problem:** Three scripts compute metrics locally with different formulas:

| Script | Denominator | Exclusion | Function |
|---|---|---|---|
| `run_external.py` | `anchor_val - y_star` | `\|denom\| < 1` | local `compute_metrics()` |
| `run_rag.py` | `anchor_val - y_star` | `\|denom\| < 1` | local `compute_metrics()` |
| `run_history.py` | `anchor_val - y_star` | `\|denom\| < 1` | local `compute_metrics()` |
| `run_icl.py` | `a - y_control` | `\|denom\| < 3` | `compute_unified_metrics()` |
| `run_tool.py` | `a - y_control` | `\|denom\| < 3` | `compute_unified_metrics()` |

The frozen spec mandates `denom = a - y_control` with `epsilon = 3` for all suites.

**Files:**
- `scripts/eval/run_external.py` — replace local `compute_metrics()` with import of `compute_unified_metrics`
- `scripts/eval/run_rag.py` — same
- `scripts/eval/run_history.py` — same (History needs special handling: `a = stage1_answer`)

**Fix approach:** Each script should:
1. Write `results.jsonl` as now.
2. Call `compute_unified_metrics(records, epsilon=3.0)` from `unified_metrics.py` for the summary.
3. Remove the local `compute_metrics()` function.

**Severity:** MUST — metric inconsistency would undermine cross-suite comparison.

---

### 1.3 Regenerate External Pilot Dataset

**After** fixing 1.1, regenerate:

```bash
PYTHONPATH=src python -m anchorbench_v1.generate \
  --suites external --size pilot --seed 42 \
  --out_dir datasets/anchorbench_external_pilot
```

Then run validators:

```bash
PYTHONPATH=src python -m anchorbench_v1.validate \
  --suite external \
  --itemspecs datasets/anchorbench_external_pilot/itemspecs.jsonl \
  --promptviews datasets/anchorbench_external_pilot/promptviews.jsonl
```

**Severity:** MUST — blocked by 1.1.

---

### 1.4 Update Paper Counts

**Files:**
- `paper/sections/dataset.tex` — update External pilot counts (90 → 180 items, 450 → 900 promptviews, 3 → 6 domains)
- `paper/sections/results.tex` — update External pilot description
- `paper/sections/appendix.tex` — update if External counts are mentioned

**Severity:** MUST — must match regenerated dataset.

---

## Priority 2: SHOULD Fix Before Submission

### 2.1 Add Bootstrap CIs to Unified Metrics

**File:** `scripts/eval/unified_metrics.py`

**Fix:** Add a `bootstrap_ci()` function:
- Input: per-item UAI values
- Output: (mean, ci_low, ci_high) with 2,000 resamples, seed=42
- Call after `compute_unified_metrics()` for each reported aggregate

**Severity:** SHOULD — strengthens reviewer trust in point estimates.

---

### 2.2 Standardize Fallback Device Defaults

**Problem:** Different scripts default to different `--fallback_device` values:

| Script | Default |
|---|---|
| `run_icl.py` | `cuda:0` |
| `run_tool.py` | `cuda:0` |
| `run_external.py` | `auto` |
| `run_rag.py` | `cpu` |
| `run_history.py` | (varies) |

**Fix:** Standardize all to `cuda:0` (or whatever device the main model is not using).

**Severity:** SHOULD — minor reproducibility issue.

---

### 2.3 Remove Dead Code

| Item | File | Action |
|---|---|---|
| `anchor_preambles["comparative"]` | `src/anchorbench_v1/domains.py` | Remove from all 6 domain definitions. Never used by any suite. (May already be removed.) |
| `SUITE_PREFIX["tool"]`, `SUITE_PREFIX["icl"]` | `src/anchorbench_v1/itemspec_gen.py` | Verify these are used; remove if not. |

**Severity:** SHOULD — cleaner artifact.

---

### 2.4 Standardize Output Directory Layout

**Problem:** Scripts differ in output structure:

| Script | Layout |
|---|---|
| ICL, Tool, RAG | `out_dir / model_slug / results.jsonl` |
| External, History | `out_dir / results.jsonl` (no model subdir) |

**Fix:** All scripts should use `out_dir / model_slug / results.jsonl`. This may require changes to `run_external.py` and `run_history.py` and corresponding shell scripts.

**Severity:** SHOULD — inconsistency confuses `recompute_all_unified.py`.

---

### 2.5 Standardize sys.path Manipulation

**Problem:** Scripts add different parent directories to `sys.path`:

| Script | sys.path additions |
|---|---|
| `run_icl.py` | `parents[1] / "src"` AND `parents[2] / "src"` |
| `run_external.py` | `parents[1] / "src"` only |

**Fix:** All scripts should use a single consistent pattern. Preferred: rely on `PYTHONPATH=src` set by shell scripts, remove `sys.path` manipulation from Python scripts.

**Severity:** SHOULD — fragile import resolution.

---

### 2.6 Create Missing Documentation

| Document | Location | Content |
|---|---|---|
| `DATASET_CARD.md` | `docs/DATASET_CARD.md` | Motivation, composition, collection, intended use, limitations, ethical considerations, license |
| `REPRODUCIBILITY.md` | `docs/REPRODUCIBILITY.md` | Exact commands to regenerate datasets, run all models, reproduce tables/figures, verify checksums |

**Severity:** SHOULD — expected by COLM reviewers.

---

## Priority 3: OPTIONAL Improvements

### 3.1 Anonymization Audit

Before creating the submission artifact:

- [ ] Grep for author names, emails, affiliations in all files
- [ ] Grep for server hostnames, internal paths, usernames
- [ ] Grep for API keys, tokens, `.env` references
- [ ] Squash or rewrite git history
- [ ] Verify W&B project names are anonymized (if W&B logging is referenced)
- [ ] Remove `.cursor/`, `.codex/`, editor-specific files from artifact

**Severity:** MUST before submission, but does not block runs.

---

### 3.2 Pin Exact Dependency Versions

**File:** `requirements.txt`

Current has minimum versions (`torch>=2.1`). For the frozen artifact, pin exact versions used for the paper experiments.

**Severity:** SHOULD for reproducibility.

---

### 3.3 Condition Names in Mechinterp/Mitigation

**Problem:** `src/mitigation_eval/` and `src/mechinterp/` use 3-condition layout (plausible-only). The frozen primary benchmark uses 5 conditions.

**Decision:** Keep as-is. Mechinterp and mitigation results in the paper are clearly marked as appendix analyses. Renaming would require reworking these modules with no scientific benefit.

**Action:** Add a note in README or docs that mechinterp/mitigation use 3-condition layout.

**Severity:** OPTIONAL.

---

## Summary: Changes by File

| File | Change | Priority |
|---|---|---|
| `src/anchorbench_v1/itemspec_gen.py` | External default domains → all 6 | MUST |
| `src/anchorbench_v1/generate.py` | Ensure External pilot uses all 6 domains | MUST |
| `scripts/eval/run_external.py` | Replace local metrics with `compute_unified_metrics` | MUST |
| `scripts/eval/run_rag.py` | Replace local metrics with `compute_unified_metrics` | MUST |
| `scripts/eval/run_history.py` | Replace local metrics with `compute_unified_metrics` | MUST |
| `datasets/anchorbench_external_pilot/` | Regenerate | MUST |
| `paper/sections/dataset.tex` | Update External counts | MUST |
| `paper/sections/results.tex` | Update External description | MUST |
| `scripts/eval/unified_metrics.py` | Add bootstrap CIs | SHOULD |
| `scripts/eval/run_*.py` | Standardize fallback device defaults | SHOULD |
| `scripts/eval/run_external.py` | Output to `out_dir/model_slug/` | SHOULD |
| `scripts/eval/run_history.py` | Output to `out_dir/model_slug/` | SHOULD |
| `src/anchorbench_v1/domains.py` | Remove `comparative` preamble | SHOULD |
| `docs/DATASET_CARD.md` | Create | SHOULD |
| `docs/REPRODUCIBILITY.md` | Create | SHOULD |
| `requirements.txt` | Pin exact versions | SHOULD |
| All files | Anonymization audit | MUST (pre-submission) |

---

*End of alignment plan.*
