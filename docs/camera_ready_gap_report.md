# Camera-Ready Gap Report

Status of the AnchorBench codebase for anonymous COLM 2026 submission.

## Fixes Applied

### Critical (Trust-Breaking)

| Issue | Status | Details |
|-------|--------|---------|
| External suite defaults to 3 domains | **FIXED** | Changed `itemspec_gen.py` and `generate.py` to use all 6 domains by default |
| Docstrings say "90 items" for pilot | **FIXED** | Updated to "180 items (6 domains)" in generate.py |
| ICL demo fallback uses non-deterministic `hash()` | **FIXED** | Replaced with `hashlib.sha256` for deterministic behavior |
| Unused `comparative` preamble in domains | **FIXED** | Removed from all 6 domains in domains.py |
| Internal paths in shell scripts | **FIXED** | Replaced hardcoded internal paths with `$REPO_ROOT` in 4 scripts |
| Missing History pairing validator | **FIXED** | Added `validate_history_pairing` to validators.py |
| Missing duplicate detection in validators | **FIXED** | Added `validate_no_duplicate_ids` |
| Missing balance validation | **FIXED** | Added `validate_domain_difficulty_balance` |
| Missing manifest consistency check | **FIXED** | Added `validate_manifest_consistency`; `validate_all` now accepts optional manifest |
| Validator CLI single-dir only | **FIXED** | `validate.py` now accepts `--data_dir` with multiple directories |

### Documentation

| Document | Status | Purpose |
|----------|--------|---------|
| `docs/data_generation_pipeline_readme.md` | **CREATED** | Pipeline overview with code references |
| `docs/data_objects_reference.md` | **CREATED** | ItemSpec/PromptView field reference |
| `docs/full_benchmark_generation_plan.md` | **CREATED** | Complete generation plan with counts |
| `docs/REPRODUCIBILITY.md` | **CREATED** | Step-by-step reproduction guide |
| `docs/REPO_STRUCTURE.md` | **CREATED** | Repository layout explanation |
| `docs/camera_ready_gap_report.md` | **CREATED** | This file |
| `README.md` | **UPDATED** | Camera-ready version |

### Workflow Scripts

| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/generate_all.sh` | **CREATED** | Generate all 5 suites |
| `scripts/validate_all.sh` | **CREATED** | Validate all 5 suites |
| `scripts/freeze_dataset.sh` | **CREATED** | Archive + checksum for submission |

## Remaining Blockers Before Full Dataset Generation

| Item | Priority | Description |
|------|----------|-------------|
| Regenerate External pilot | **MUST** | Old External datasets used 3 domains; must regenerate with 6 |
| Validate all pilot datasets | **MUST** | Run `validate_all.sh pilot` after regeneration |
| Verify deterministic regeneration | **SHOULD** | Generate twice, diff outputs (except timestamps) |

## Remaining Blockers Before Submission

| Item | Priority | Description |
|------|----------|-------------|
| Metric unification in eval scripts | **SHOULD** | `run_external.py` and `run_rag.py` use inline NAI with denom `(a - y_star)` and epsilon=1 instead of unified UAI with `(a - y_control)` and epsilon=3. Add a unified metrics recomputation step after each eval run. |
| Bootstrap CIs | **SHOULD** | `unified_metrics.py` does not compute bootstrap confidence intervals. Can be added as a thin wrapper without changing the core computation. |
| Paper count updates | **MUST** | After External regeneration, update paper tables to reflect 180 External items and 6 domains. |
| Pittsburgh zip code in paper | **SHOULD** | `paper/colm2026_conference.tex` line 43 has Pittsburgh, PA 15213. Verify COLM anonymization requirements. |
| `.env` exclusion | **MUST** | Ensure `.env` is excluded from the submission archive (already in `.gitignore`). |
| Final anonymization scan | **SHOULD** | Run `grep -rn "/home/\|/Users/" .` before final tarball. |

## Metric Consistency Summary

| Script | Metric | Denominator | Epsilon | Status |
|--------|--------|-------------|---------|--------|
| `unified_metrics.py` | UAI | `a - y_control` | 3.0 | **Canonical** |
| `run_icl.py` | UAI | `a - y_control` | 3.0 | OK (uses unified) |
| `run_tool.py` | UAI | `a - y_control` | 3.0 | OK (uses unified) |
| `run_external.py` | NAI | `a - y_star` | 1.0 | **Needs update** |
| `run_rag.py` | NAI | `a - y_star` | 1.0 | **Needs update** |
| `run_history.py` | ACR/RR | custom | 1.0 | **Needs review** |
| `recompute_all_unified.py` | UAI | `a - y_control` | 3.0 | OK (uses unified) |

**Recommended action:** After each eval run, `recompute_all_unified.py` already recomputes all metrics with the canonical formula. The inline metrics in individual eval scripts are used for quick feedback during runs but are superseded by the unified recomputation. This is acceptable for submission as long as the paper reports unified metrics.

## Anonymization Status

| Check | Status |
|-------|--------|
| No hardcoded API keys | PASS |
| No author names in code | PASS |
| No internal paths in code | PASS (fixed) |
| `.env` in .gitignore | PASS |
| Paper author block | Uses COLM template placeholders |
| GitHub URLs | None in code |
| HuggingFace org paths | None identifying |
