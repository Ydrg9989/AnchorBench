---
name: History Suite Redesign
overview: Redesign the History suite from experimenter-injected prior-estimate approximation into a genuine two-stage self-generated anchoring benchmark, aligned with AnchorBench v2 (same domains, same evidence-first task, same y_star_evidence; only anchor source changes to conversation history).
todos:
  - id: schema
    content: "Add `history: Optional[Dict]` field to ItemSpec in schema.py"
    status: pending
  - id: itemspec-gen
    content: Add generate_history_v2_itemspecs() with subset-search and same-domain warmup logic
    status: pending
  - id: renderer
    content: "Rewrite history.py: add render_history_v2() with explicit prompt protocol wording"
    status: pending
  - id: runner
    content: Add generate_chat(messages) multi-turn method to HFRunner in runner.py
    status: pending
  - id: eval-script
    content: Create run_history_v2.py with two-stage loop and ACR/RR + irrelevant-shift metrics
    status: pending
  - id: validators
    content: Add History v2 validators (same-domain warmup, subset pressure, stage1/stage2 storage)
    status: pending
  - id: generate-cli
    content: Wire History v2 into suites/__init__.py and generate.py CLI
    status: pending
  - id: docs
    content: Create docs/history_suite_v2.md with old vs new and usage
    status: pending
isProject: false
---

# History Suite Redesign (v2): Genuine Self-Generated Anchoring

## Overview

The History suite instantiates **Paradigm P2: self-generated anchoring** (Epley & Gilovich). The current implementation only approximates it by injecting a prior assistant estimate into history. This plan revises the suite into a genuine two-stage protocol where the anchor comes from the model’s own earlier estimate, with the same benchmark philosophy as External v2.

## Goal

- Replace fake injected history with a **two-stage protocol**: Stage 1 produces the model’s prior estimate; Stage 2 presents full evidence (or a new case) with that prior in conversation history.
- Preserve **benchmark alignment**: same evidence-first task, same `y_star_evidence` as evaluation target, same relevance axis, same 5-condition core; only the **source** of the anchor changes (conversation history vs external prompt text).
- Keep the old History v1 renderer for backward compatibility where useful.

## Invariants Shared with External v2

The History suite **must** use the same backbone as the external anchoring dataset:

- **Same six domains** (no separate History-only domain set):
  1. pricing_wtp (Pricing / Willingness-to-Pay)
  2. operations_time (Operations / Time Estimation)
  3. transportation_logistics (Transportation / Logistics Reliability)
  4. resource_consumption (Resource Consumption / Efficiency)
  5. market_demographics (Market / Demographics Adoption)
  6. legal_policy (Legal / Policy Compliance)

- **Same domain config**: scenario templates, evidence labels, question templates, and metric naming come from the same `DOMAINS` / `DomainConfig` used for External v2.
- **Same gold logic**: `y_star_evidence = round(mean(visible_evidence))` for the **target item**; evaluation is always vs the target item’s `y_star_evidence`.
- Cross-suite comparison (History vs External) should differ **only in anchor source**, not in domain composition or task structure.

## Conditions

| Condition        | Anchor source                          | Stage 1                          | Stage 2                                  |
|------------------|----------------------------------------|-----------------------------------|------------------------------------------|
| control          | None                                   | —                                 | Full evidence, direct question           |
| plausible_low    | Model’s own estimate (same case)       | Partial evidence (low-pressure subset) | Same case, full evidence, revised estimate |
| plausible_high   | Model’s own estimate (same case)       | Partial evidence (high-pressure subset) | Same case, full evidence, revised estimate |
| irrelevant_low   | Model’s own estimate (different case)  | Warmup case (same domain)         | New case, same domain, full evidence     |
| irrelevant_high  | Model’s own estimate (different case)  | Warmup case (same domain)         | New case, same domain, full evidence     |

## Data Curation Logic

- **Base item**: Same `ItemSpec` structure as elsewhere: `item_id`, `suite`, `domain`, `template_family`, `evidence_structured`, `y_star_evidence`, `anchors`, etc. Generated from the same domain backbone as External v2.
- **Three families**: (1) control, (2) plausible same-case self-generated, (3) irrelevant different-case self-generated.
- **Plausible**: Stage 1 = partial-evidence view; Stage 2 = full evidence for the **same** case. Stage 1 answer becomes the self-generated anchor.
- **Irrelevant**: Stage 1 = warmup case (same domain, different case identity); Stage 2 = target case; warmup answer stays in history; Stage 2 wording states it is a **new case** and the previous answer was for a different case.

## Same-Domain Warmup Rule (Irrelevant Conditions)

- **Warmup and target must share `domain_id`.** Do not use a different domain for warmup.
- **Warmup and target must be different cases**: different `item_id` / case identity, different evidence and `y_star_evidence`.
- **Rationale**: Avoids domain-shift confounds; keeps wording, evidence scale, and numeric context comparable; isolates the history effect (prior number from same domain but normatively irrelevant to the target case).

Warmup cases are generated in the same domain with theta in [15, 30] for irrelevant_low and [70, 85] for irrelevant_high, with their own evidence and `y_star_evidence`. The target item’s `y_star_evidence` remains the sole evaluation target for Stage 2.

## Stage 1 Subset Search Rule (Plausible Conditions)

Do **not** hard-code “2 lowest” or “2 highest” evidence ratings.

- **Enumerate** all 2-of-5 evidence subsets (by index).
- **Compute** each subset’s mean.
- **Choose**:
  - **Low-pressure subset**: subset whose mean is closest to `y_star_evidence - delta` (e.g. delta = 10 or 15 for easy items).
  - **High-pressure subset**: subset whose mean is closest to `y_star_evidence + delta`.
- If no subset hits the target exactly, take the nearest and **record the realized gap** (e.g. in `spec.history`) for analysis.

**Why this is better than lowest/highest**: (1) More controlled low/high pressure relative to the evidence-supported answer. (2) More comparable across items with different evidence spreads. (3) Less brittle when evidence has ties or skewed distributions.

## Prompt Protocol

- **Plausible Stage 2**: Must explicitly state that this is the **same case** and that **full evidence is now available**; then ask for the revised estimate (0–100).
- **Irrelevant Stage 2**: Must explicitly state that this is a **new case** from the **same domain**, and that the **previous answer was for a different case** (not for this assessment). Then present target scenario + evidence + question.

The relevance distinction (same-case revision vs new-case with prior in history) must be clear in the wording so the benchmark does not conflate the two.

## Metrics

### Plausible (same-case self-generated)

**Primary:**

- **Adjustment Completion Ratio (ACR)**  
  `ACR = (stage2 - stage1) / (y_star_evidence - stage1)`  
  Interpretation: ACR = 1 ⇒ full adjustment toward evidence-supported target; 0 < ACR < 1 ⇒ under-adjustment.

- **Residual Reduction (RR)**  
  `RR = 1 - |stage2 - y_star_evidence| / |stage1 - y_star_evidence|`  
  Interpretation: How much of the initial error (distance from `y_star_evidence`) was corrected in Stage 2.

**Secondary (optional):** Stage 1 MAE vs `y_star_evidence` for plausible conditions.

### Irrelevant (different-case history)

**Primary:**

- **History-induced shift vs control**  
  `Shift_irr = stage2_irr - control`  
  Report mean shift for irrelevant_low and irrelevant_high; high–low contrast; and “toward-anchor” rate (e.g. proportion of Stage 2 answers that move toward the Stage 1 value relative to control).

**Optional secondary:** An aggregated history NAI using dataset-level stage1 low/high separation (e.g. median stage1 in “low” vs “high” conditions) rather than a per-item denominator.

**Why not use the original per-item NAI for irrelevant:** For irrelevant, Stage 1 is a **different case**. The denominator `(stage1_answer - y_star_evidence)` mixes the warmup case’s answer with the target’s gold; it is not meaningfully tied to the target item, and the ratio can be unstable or misleading. Shift and toward-anchor rate are more interpretable.

### Standard accuracy (all conditions)

- Stage 2 MAE vs `y_star_evidence`
- Stage 2 accuracy within ±10
- Parse rate; optional Stage 1 MAE for plausible

## Control / Confound Note

Control is **single-stage**; non-control conditions are **two-stage**. A possible confound is that any difference could partly reflect **history structure** (having a prior turn) rather than anchoring per se.

- **Document** this as a limitation.
- **Optional validation-only condition**: `neutral_history` — same two-turn structure as anchored conditions but Stage 1 is a neutral task (e.g. unrelated question) so the prior answer is not a numeric anchor for the target. Compare Stage 2 in neutral_history vs control to check for pure structure effects. This is **not** part of the core 5-condition benchmark but is documented for internal validation.

## Validators

Validators must check:

- **Domain consistency**: For irrelevant conditions, warmup and target share the same `domain_id`; warmup and target case IDs (e.g. `item_id`) differ.
- **Plausible subsets**: Stage 1 subsets belong to the same target item; low vs high subset pressure actually differs (e.g. subset means on opposite sides of `y_star_evidence`, or recorded gaps verify direction).
- **Gold**: `y_star_evidence` is always the **target** item’s evaluation gold for Stage 2.
- **Control**: Control condition has no prior answer; single-stage only.
- **Non-control**: Non-control conditions store both `stage1_answer` and `stage2_answer` (or equivalent in result records).

## Implementation Plan

1. **Schema** ([src/anchorbench_v1/schema.py](src/anchorbench_v1/schema.py)): Add `history: Optional[Dict[str, Any]] = None` to `ItemSpec`; add `"history"` to `to_dict` null-drop list. No change to `PromptView`; two-stage data stays in `prompt_components`.

2. **ItemSpec generation** ([src/anchorbench_v1/itemspec_gen.py](src/anchorbench_v1/itemspec_gen.py)): Add `generate_history_v2_itemspecs()` using the **same six domains** and same theta/evidence generation backbone as External v2. For each item: run subset search for plausible low/high; generate same-domain warmup cases (different case ID) for irrelevant_low and irrelevant_high; store in `spec.history` (subset indices, subset means/gaps, warmup case data). Use same `y_star_evidence` logic as External v2 for the target item.

3. **Renderer** ([src/anchorbench_v1/suites/history.py](src/anchorbench_v1/suites/history.py)): Add `render_history_v2(spec)` returning 5 `PromptView`s. Control: same stem as External v2 control. Plausible low/high: `stage1_user_message` = scenario + chosen partial evidence + initial-estimate instruction; `stage2_user_message` = explicit “same case, full evidence now available” + full evidence + revised-estimate instruction. Irrelevant low/high: `stage1_user_message` = warmup scenario + warmup evidence + question; `stage2_user_message` = explicit “new case, same domain, previous answer was for a different case” + target scenario + target evidence + question. Set `anchor_relevance` per condition. Keep `render_history()` (v1) for backward compatibility.

4. **Runner** ([src/mitigation_eval/runner.py](src/mitigation_eval/runner.py)): Add `generate_chat(messages, max_tokens, temperature)` to `HFRunner` for multi-turn generation. Leave `generate()` unchanged.

5. **Eval script** (`scripts/eval/run_history_v2.py`): Load History v2 promptviews and itemspecs; for control run single-stage; for non-control run Stage 1 then Stage 2 as multi-turn (user → assistant → user). Record `stage1_answer`, `stage1_raw_text`, `stage2_answer`, `stage2_raw_text`. Compute ACR and RR for plausible; shift, mean shift low/high, high–low contrast, toward-anchor rate for irrelevant; MAE and accuracy ±10 for Stage 2 vs `y_star_evidence`.

6. **Validators** ([src/anchorbench_v1/validators.py](src/anchorbench_v1/validators.py)): Add `validate_history_v2_itemspec` (history dict, same-domain warmup, different case IDs, subset indices and pressure) and `validate_history_v2_promptview` (two-stage components, anchor_relevance, same-stem where applicable). Wire into `validate_all()`.

7. **Suite registry and CLI**: Register `"history_v2": render_history_v2` in [src/anchorbench_v1/suites/__init__.py](src/anchorbench_v1/suites/__init__.py). In [src/anchorbench_v1/generate.py](src/anchorbench_v1/generate.py), add `generate_history_v2_dataset()` and support `--v2 --suites history` (or equivalent) so History v2 uses the same six domains and generation flow.

8. **Documentation** (`docs/history_suite_v2.md`): Describe old History (approximation), new two-stage design, domain consistency with External v2, subset and warmup rules, prompt protocol, metrics (ACR, RR, irrelevant shift), validators, how to generate and run eval, output fields, and limitations (including control confound and optional `neutral_history`).

## Remaining Limitations

- Stage 1 direction (low/high) is controlled by subset choice; the exact Stage 1 value is model-dependent.
- Multi-turn context and length; possible context-window effects for long conversations.
- Determinism: even with temperature=0, two-stage runs may vary across environments.
- Control vs two-stage structure confound; optional `neutral_history` is for validation only, not core benchmark.

## Changes from Previous Plan

- **Domain consistency**: History suite uses the **exact same six domains** as External v2; no separate History-only domain set. Irrelevant warmups are **same-domain, different-case** (not different domain) to avoid domain-shift confounds and keep cross-suite comparison focused on anchor source.
- **Plausible Stage 1**: Replaced fixed “2 lowest / 2 highest” with **subset search**: enumerate 2-of-5 subsets, choose low subset as mean closest to `y_star_evidence - delta`, high as mean closest to `y_star_evidence + delta` (delta e.g. 10–15); record realized gap. Rationale: more controlled pressure, more comparable across items, less brittle.
- **Metrics**: Replaced single per-item NAI with (1) **Plausible**: ACR and RR as primary; (2) **Irrelevant**: shift vs control, mean shift low/high, high–low contrast, toward-anchor rate; (3) standard Stage 2 MAE and accuracy ±10. Documented why per-item NAI is not conceptually stable for irrelevant (Stage 1 from different case; denominator not tied to target).
- **Control confound**: Added explicit note that control is single-stage and non-control is two-stage; documented optional **neutral_history** validation condition (not in core 5 conditions).
- **Prompt protocol**: Plausible Stage 2 must state “same case” and “full evidence now available”; irrelevant Stage 2 must state “new case,” “same domain,” and “previous answer was for a different case.”
- **Validators**: Require warmup and target same domain and different case IDs; plausible subsets from same target with distinct low/high pressure; `y_star_evidence` always target gold; control has no prior; non-control records store stage1 and stage2 answers.
