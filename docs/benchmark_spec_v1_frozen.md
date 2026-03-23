# AnchorBench v1.0 — Frozen Benchmark Specification

**Status:** FROZEN  
**Date:** 2026-03-10  
**Scope:** Primary benchmark for COLM 2026 submission  
**Rule:** No changes to benchmark semantics after this freeze unless a true bug is discovered.

---

## A. Goal and Non-Goals

### Goal

Measure how much the numeric estimates produced by instruction-tuned LLMs shift toward salient numeric anchors delivered through five distinct interface channels, and whether models discriminate between transparently irrelevant and plausibly informative anchors.

### Non-Goals

- **Not** a general accuracy benchmark. Task accuracy is a diagnostic, not the primary outcome.
- **Not** a jailbreak or safety benchmark.
- **Not** an evaluation of free-form reasoning quality.
- **Not** a test of whether anchoring is always harmful (plausible anchors *should* influence).
- **Not** a multi-language benchmark (English only).
- **Not** a test of anchoring on non-numeric tasks.

---

## B. Primary Benchmark Claim

> When LLMs are asked to produce numeric estimates, their outputs systematically shift toward salient reference numbers embedded in context — but the magnitude and direction of this shift depend on (i) the interface channel through which the anchor is delivered and (ii) whether the anchor is transparently irrelevant or plausibly informative. No single model is uniformly best or worst across all channels, and model rankings on both susceptibility and discrimination reverse across suites.

This claim is narrow: it asserts *channel-dependence* and *relevance-sensitivity*, not a universal anchoring law.

---

## C. Suite Definitions

### C1. External

| Property | Value |
|---|---|
| **Anchor source** | Explicit numeric sentence inserted before the question |
| **Interface channel** | Prompt-embedded textual cue |
| **Human analogue** | Classic external anchoring (Tversky & Kahneman, 1974) |
| **Manipulated** | Presence and value of anchor sentence; relevance framing |
| **Held fixed** | Scenario text, evidence values, question, gold answer, answer format, decoding |
| **Known confounds** | Anchor sentence adds tokens; control is shorter. Minimal: single sentence. |

### C2. History

| Property | Value |
|---|---|
| **Anchor source** | Model's own Stage 1 estimate, retained in conversation history |
| **Interface channel** | Multi-turn conversation history |
| **Human analogue** | Self-generated anchoring (Epley & Gilovich, 2001) |
| **Manipulated** | Stage 1 setup: partial evidence (plausible) vs. warmup case (irrelevant) vs. absent (control) |
| **Held fixed** | Target scenario, target evidence, target question, gold answer, answer format, decoding |
| **Known confounds** | Control is single-stage; anchored conditions are two-stage. Structural asymmetry is inherent to the paradigm and documented as a limitation. |

### C3. ICL (In-Context Learning)

| Property | Value |
|---|---|
| **Anchor source** | Metadata in few-shot demonstration headers |
| **Interface channel** | Few-shot ICL examples |
| **Human analogue** | Numeric priming (Wilson et al., 1996) |
| **Manipulated** | Demo header metadata containing anchor-laden text |
| **Held fixed** | Demo evidence, demo answers, target scenario, target evidence, target question, gold answer, answer format, decoding |
| **Known confounds** | Anchor is in metadata, not demo answers; tests priming rather than ICL distribution-matching. This is intentional and documented. |

### C4. RAG (Retrieval-Augmented Generation)

| Property | Value |
|---|---|
| **Anchor source** | Anchor-bearing document in a 3-document mini-corpus |
| **Interface channel** | Retrieved document context |
| **Human analogue** | Ecological anchoring via authoritative sources (Chapman & Johnson, 1999) |
| **Manipulated** | Content of the anchor-slot document (control placeholder vs. irrelevant metadata vs. plausible survey reference) |
| **Held fixed** | Core evidence document, filler document, document order, scenario, evidence values, question, gold answer, answer format, decoding |
| **Known confounds** | Fixed retrieval order (core → anchor → filler); no real retrieval ranking. Acceptable for controlled measurement. |

### C5. Tool

| Property | Value |
|---|---|
| **Anchor source** | Field in simulated tool-call response JSON |
| **Interface channel** | Tool-call output |
| **Human analogue** | Ecological anchoring via tool authority (Chapman & Johnson, 1999) |
| **Manipulated** | Output of `check_external_reference` tool: absent (control) vs. `request_id` (irrelevant) vs. `reference_value` (plausible) |
| **Held fixed** | `get_evidence_summary` output, scenario, evidence, question, system prompt, answer format, decoding |
| **Known confounds** | Primary benchmark uses simulated (injected) tool traces, not real multi-turn tool calling. Ecological variant is extension-only. |

---

## D. Condition Matrix

### D1. Primary Benchmark Conditions (FROZEN)

All five suites use exactly these five conditions per item:

| Condition | Relevance | Anchor Direction | Description |
|---|---|---|---|
| `control` | none | — | No anchor present |
| `irrelevant_low` | irrelevant | low | Transparently arbitrary low anchor |
| `irrelevant_high` | irrelevant | high | Transparently arbitrary high anchor |
| `plausible_low` | plausible | low | Weakly credible low anchor |
| `plausible_high` | plausible | high | Weakly credible high anchor |

**Decision:** The `informative` relevance level is **NOT** part of the frozen primary benchmark. It is an OPTIONAL extension for future validation studies. [MUST]

**Decision:** Legacy 3-condition names (`low_anchor`, `high_anchor`) are deprecated. All primary benchmark data and code must use the 5-condition names above. [MUST]

### D2. Conditions per Suite

| Suite | control | irrelevant_low | irrelevant_high | plausible_low | plausible_high |
|---|---|---|---|---|---|
| External | No anchor sentence | Case-number preamble, low value | Case-number preamble, high value | Survey-reference preamble, low value | Survey-reference preamble, high value |
| History | Single-stage, full evidence | Two-stage, warmup case (different scenario) producing low estimate | Same, high estimate | Two-stage, partial evidence (same case) producing low estimate | Same, high estimate |
| ICL | Neutral demo headers | Demo headers with case/batch IDs containing low anchor | Same, high | Demo headers with "prior estimate" containing low anchor | Same, high |
| RAG | Anchor-slot doc says "no data found" | Anchor-slot doc contains batch reference number (low) | Same, high | Anchor-slot doc contains survey reference value (low) | Same, high |
| Tool | `check_external_reference` returns `status: "no_data"` | Returns `request_id` field with low value | Same, high | Returns `reference_value` field with low value | Same, high |

---

## E. Relevance-Level Definitions

### E1. None / Control (FROZEN)

**Operational definition:** The prompt contains no numeric anchor related to the estimation task. The model receives only the scenario, evidence, question, and answer-format instruction (plus any structural elements required by the channel, such as tool schemas or demo structure, but with anchor fields absent or explicitly marked as unavailable).

**Auditable criterion:** No integer in [0, 100] appears in the anchor-bearing component of the prompt that could plausibly be interpreted as a reference point for the estimation task.

### E2. Irrelevant (FROZEN)

**Operational definition:** A numeric value appears in the prompt in a position and framing that is transparently unrelated to the estimation task. A competent human reader would recognize it as an administrative identifier, batch number, or other arbitrary label.

**Auditable criterion:** The anchor number appears inside framing text that explicitly marks it as non-substantive (e.g., "assessment case #X", "document batch reference #X", tool field `request_id`).

### E3. Plausible (FROZEN)

**Operational definition:** A numeric value appears in the prompt in a position and framing that a reader could reasonably interpret as weakly informative about the estimation task, but that is not authoritative or precise enough to serve as the answer. Examples: "a recent survey suggested approximately X", "prior estimate for similar cases: X", tool field `reference_value` with `confidence: "low"`.

**Auditable criterion:** The anchor number appears inside framing text that attributes it to a credible but imprecise source. The framing does not assert the number as the correct answer.

### E4. Informative (EXTENSION ONLY — not in frozen primary benchmark)

**Operational definition:** A numeric value appears with framing that makes it a legitimate, high-confidence data point for the estimation task.

**Status:** Reserved for future validation. Not generated, evaluated, or reported in the primary benchmark. [MUST]

---

## F. Benchmark Invariants / Identification Assumptions

### F1. Universal Invariants (all suites) [MUST]

For every item, across all five conditions:

1. **Same question.** The estimation question is textually identical.
2. **Same evidence.** The numeric evidence values presented to the model are identical.
3. **Same gold answer.** `y_star_evidence = round(mean(visible_evidence))` is identical.
4. **Same answer format.** `"Return only a single integer 0–100 on the last line."` is identical.
5. **Same decoding.** Temperature, top-p, max tokens, and seed are identical.
6. **Only the anchor-bearing component changes.** The sole manipulation is the content that carries (or omits) the anchor value.

### F2. Suite-Specific Invariants

**External:**
- Same scenario text, same evidence block. Only the anchor sentence (or its absence) changes.

**History:**
- Same target scenario, same target evidence, same target question in Stage 2. Stage 1 setup varies by design (this is the manipulation). The anchor is the model's own Stage 1 output.

**ICL:**
- Same demo evidence values, same demo answer values, same target scenario, same target evidence. Only demo header metadata changes.

**RAG:**
- Same core document, same filler document, same document order. Only the anchor-slot document content changes.

**Tool:**
- Same `get_evidence_summary` output, same scenario, same evidence, same system prompt. Only the `check_external_reference` output changes.

### F3. Causal Interpretation

The benchmark supports the claim: "Differences in model output across conditions for the same item are attributable to the anchor manipulation, holding all other inputs constant." This is a within-item paired design. Cross-item comparisons are secondary.

---

## G. Anchor Construction Policy (FROZEN)

### G1. Latent Signal

- `theta ~ Uniform[30, 70]` (integer) [MUST]
- Answer space: integers in [0, 100] [MUST]

### G2. Anchor Offsets

**External, ICL, RAG, Tool suites:**
- Stratified offsets: `delta in {15, 25, 40}` [MUST]
- `anchor_low = max(0, theta - delta)`
- `anchor_high = min(100, theta + delta)`
- Items are generated for each offset value; offset is a stratification variable, not a within-item manipulation.

**History suite:**
- Fixed offset: `delta = 25` [MUST — preserve current design]
- History uses fixed offset because the actual anchor is the model's Stage 1 answer, not the experimenter-set value. The offset controls the *expected direction* of Stage 1 output via subset/warmup selection, not the exact anchor value.

**Rationale for History exception:** The self-generated paradigm inherently produces model-dependent anchors. Using stratified offsets would create a false precision about a quantity the experimenter does not control. Preserving fixed offset is the conservative choice.

### G3. Symmetric Distance

- Not strictly required. Clamping to [0, 100] means `anchor_low` and `anchor_high` may be asymmetric when theta is near boundaries.
- This is acceptable and documented. [MUST preserve]

### G4. Anchor Wording

- Templated per domain and per relevance level. Templates are defined in `domains.py` (`anchor_preambles` dict). [MUST]
- Wording is fixed per (domain, relevance) pair. No random variation in anchor phrasing within a condition. [MUST]

### G5. Irrelevant vs. Plausible Anchor Framing

| Relevance | External | ICL | RAG | Tool |
|---|---|---|---|---|
| Irrelevant | "assessment case #{anchor}" | "Case #{anchor}, Batch {anchor}" in header | "document batch reference #{anchor}" | `request_id: {anchor}` |
| Plausible | "a recent industry report suggested ... around {anchor}" | "Prior estimate for similar cases: {anchor}" in header | "A recent survey ... index of approximately {anchor}" | `reference_value: {anchor}`, `confidence: "low"` |

---

## H. Domain and Difficulty Balance Policy (FROZEN)

### H1. Domain List [MUST]

| ID | Display Name |
|---|---|
| `pricing_wtp` | Pricing / Willingness-to-Pay |
| `operations_time` | Operations / Time Estimation |
| `transportation_logistics` | Transportation / Logistics Reliability |
| `resource_consumption` | Resource Consumption / Efficiency |
| `market_demographics` | Market / Demographics Adoption |
| `legal_policy` | Legal / Policy Compliance |

Six domains. No additions or removals. [MUST]

### H2. Difficulty Levels [MUST]

| Level | Evidence Config |
|---|---|
| `easy` | 5 concordant ratings, sigma = 8 |
| `hard` | 3 visible ratings (2 missing, displayed as "[data not available]"), sigma = 15, 1 conflicting value |

Two levels. No additions. [MUST]

### H3. Balance Requirements

- **All suites MUST be balanced by domain.** Equal item count per domain within each suite. [MUST]
- **All suites MUST be balanced by difficulty.** Equal item count per difficulty within each suite. [MUST]
- **External, ICL, RAG, Tool:** Also balanced by anchor offset (equal items per delta in {15, 25, 40}). [MUST]
- **History:** No offset stratification (fixed delta = 25). [MUST]

### H4. Domain Count per Suite

- **All five suites MUST use all six domains.** [MUST]
- The current External pilot uses only 3 domains (`DOMAIN_IDS[:3]`). This is a **KNOWN BUG** that must be fixed before final generation. See Section I and the alignment plan.

### H5. Minimum Counts

- Minimum 5 items per cell (suite × domain × difficulty × offset) for pilot. [SHOULD]
- Target 10 items per cell for core/full benchmark. [SHOULD]

---

## I. Dataset Counts and Split Policy (FROZEN)

### I1. Pilot Counts (current, for paper submission)

| Suite | Domains | Difficulties | Offsets | Items/cell | Total Items | Conditions | Total PromptViews |
|---|---|---|---|---|---|---|---|
| External | 6 | 2 | 3 | 5 | 180 | 5 | 900 |
| History | 6 | 2 | 1 | 5 | 60 | 5 | 300 |
| ICL | 6 | 2 | 3 | 5 | 180 | 5 | 900 |
| RAG | 6 | 2 | 3 | 5 | 180 | 5 | 900 |
| Tool | 6 | 2 | 3 | 5 | 180 | 5 | 900 |
| **Total** | | | | | **780** | | **3,900** |

**CRITICAL FIX:** External pilot must be regenerated with 6 domains (currently uses 3, producing only 90 items). Target: 180 items matching the other offset-stratified suites. [MUST]

**History:** 60 items is correct (no offset stratification: 6 domains × 2 difficulties × 5 items = 60). [MUST preserve]

### I2. Core Counts (target for full benchmark, OPTIONAL)

| Suite | Items/cell | Total Items | Total PromptViews |
|---|---|---|---|
| External | 10 | 360 | 1,800 |
| History | 10 | 120 | 600 |
| ICL | 10 | 360 | 1,800 |
| RAG | 10 | 360 | 1,800 |
| Tool | 10 | 360 | 1,800 |
| **Total** | | **1,560** | **7,800** |

### I3. Split Policy

- **Pilot:** Used for paper submission. Frozen after generation. [MUST]
- **Smoke:** Development/testing only. Not part of the benchmark artifact. [MUST]
- **Core/Full:** Optional extension for future work. Not required for submission. [OPTIONAL]
- **No train/test split.** This is a benchmark, not a dataset for model training. All items are evaluation items.

### I4. Robustness Subset

- Stress-test items (position, format, magnitude, RAG order controls) are an **OPTIONAL extension**, not part of the primary benchmark. [MUST separate]
- If included, they must be clearly labeled as `split: "robustness"` in the manifest.

---

## J. Prompt and Decoding Protocol (FROZEN)

### J1. System Prompt [MUST]

| Suite | System Prompt |
|---|---|
| External | None (user message only) |
| History | None (user/assistant turns only) |
| ICL | None (user message only) |
| RAG | None (user message only) |
| Tool | "You are an expert assessment analyst. You have access to tools that can help you analyze data and look up external references. Use the tool outputs to inform your assessment, but apply your own judgment when interpreting the results." |

**Decision:** Only the Tool suite uses a system prompt. All others are user-message-only. [MUST]

### J2. Answer Format [MUST]

All suites append: `"Return only a single integer 0–100 on the last line."` [MUST]

### J3. Decoding Parameters [MUST]

| Parameter | Value |
|---|---|
| Temperature | 0.0 (greedy) |
| top_p | 1.0 (no nucleus sampling) |
| max_new_tokens | 512 |
| do_sample | False |
| Seed | 42 |
| Repeated samples | 1 (greedy is deterministic) |

**Decision:** Primary benchmark uses greedy decoding only. Sampling-based analyses (multiple seeds, temperature > 0) are OPTIONAL extensions. [MUST]

### J4. CoT Policy [MUST]

- CoT is **not suppressed.** The answer format instruction asks for "a single integer on the last line" but does not forbid preceding reasoning.
- Parsing must handle CoT output (extract last valid integer). See Section K.

### J5. Tool Suite Protocol [MUST]

- **Primary benchmark:** Simulated tool traces. Tool call and response JSON are injected into the prompt as pre-computed text. The model does not execute real tool calls. [MUST]
- **Ecological variant:** Real multi-turn tool calling where the model issues tool calls and receives executed responses. This is an **OPTIONAL extension** (~50 items, plausible_high + control only). [OPTIONAL]

---

## K. Parsing and Validity Rules (FROZEN)

### K1. Primary Parser [MUST]

`parse_answer_int(raw_text, prompt_text)` from `src/mitigation_eval/runner.py`:

1. If entire text is a single integer in [0, 100] → accept.
2. If all integers in [0, 100] found in text agree → accept.
3. Regex `_COT_NUM_PAT = r"(?:is|=|:\s*)\s*(\d{1,3})\s*(?:[.\s\n]|$)"` → last match in [0, 100].
4. Last line containing exactly one valid integer in [0, 100] → accept.
5. Exclude integers that appear verbatim in prompt text (echo filtering) → take last remaining.
6. Otherwise → parse failure.

### K2. Fallback Parser [SHOULD]

`LLMFallbackExtractor` using a small model (default: `Qwen/Qwen2.5-1.5B-Instruct`):
- Prompt: "Extract the single final numeric answer (integer 0-100)..."
- `max_new_tokens=8`, `do_sample=False`
- Only accepts integers 0–100.

### K3. CoT Parser

`parse_cot_answer(raw_text)` from `src/mitigation_eval/strategies.py`:
- Last line with exactly one valid integer in [0, 100] → accept.
- Fallback: last valid integer in full text.
- Used only when mitigation strategy B3 (CoT) is active. Primary benchmark does not use B3.

### K4. Validity Constraints [MUST]

- Valid output: a single integer in [0, 100].
- Outputs outside this range or non-numeric outputs are marked `parsed_ok = False`.
- Parse strategy is recorded per item: `"regex"`, `"llm_fallback"`, or `"failed"`.

### K5. Parse-Rate Accounting [MUST]

- Parse rate is reported per (model, suite, condition).
- Items with `parsed_ok = False` are **excluded** from anchoring metrics (UAI, TAR) but included in parse-rate reporting.
- If parse rate < 80% for any (model, suite) combination, results for that combination should be flagged and discussed. [SHOULD]

### K6. Output Schema [MUST]

Each record in `results.jsonl`:

```json
{
  "item_id": "str",
  "suite": "str",
  "domain": "str",
  "difficulty": "str",
  "condition": "str",
  "anchor_relevance": "str",
  "anchor_value": "int | null",
  "y_star_evidence": "int",
  "theta": "int",
  "answer_int": "int | null",
  "parsed_ok": "bool",
  "parse_strategy": "str",
  "raw_text": "str"
}
```

History suite adds: `"stage1_answer"`, `"stage1_raw_text"`.

---

## L. Metrics and Statistics Plan (FROZEN)

### L1. Primary Metrics [MUST]

**UAI (Unified Anchor Influence):**

```
UAI_{i,r} = (y_anchor - y_control) / (a_{i,r} - y_control)
```

- `y_anchor`: model answer for the anchored condition
- `y_control`: model answer for the control condition (same item)
- `a_{i,r}`: anchor value for item i, relevance r
  - External, ICL, RAG, Tool: experimenter-set anchor (`theta ± delta`)
  - History: model's Stage 1 answer
- **Exclusion:** Items where `|a_{i,r} - y_control| < epsilon` are excluded. `epsilon = 3` [MUST]
- Reported separately as: `UAI_irr` (mean over irrelevant conditions), `UAI_plaus` (mean over plausible conditions)

**TAR (Toward-Anchor Rate):**

```
TAR_{i,r} = 1  if  (y_anchor - y_control) * (a_{i,r} - y_control) > 0
             0  otherwise
```

- Fraction of items where the shift is toward the anchor.
- Reported separately for irrelevant and plausible conditions.

**Disc_delta (Discrimination Delta):**

```
Disc_delta = UAI_plaus - UAI_irr
```

- Positive values indicate the model shifts more toward plausible anchors than irrelevant ones.

### L2. Secondary Metrics [SHOULD]

**MAE_ctrl:** Mean absolute error on control condition: `mean(|y_control - y_star_evidence|)`.

**Acc10_ctrl:** Fraction of control items where `|y_control - y_star_evidence| <= 10`.

**Parse rate:** Per (model, suite, condition).

**History-specific (appendix only):**
- ACR (Anchoring Correction Ratio): `(stage2 - stage1) / (y_star - stage1)` for plausible conditions
- RR (Revision Ratio): `1 - |stage2 - y_star| / |stage1 - y_star|` for plausible conditions

### L3. Bootstrap CI Plan [SHOULD]

- 2,000 bootstrap resamples over item IDs.
- 95% percentile confidence intervals for UAI_irr, UAI_plaus, and Disc_delta.
- Seed: 42.

**Current gap:** `unified_metrics.py` does not implement bootstrap. [SHOULD fix before final paper tables]

### L4. Paired Significance Tests [OPTIONAL]

- Wilcoxon signed-rank test on paired (UAI_plaus_i - UAI_irr_i) per model×suite.
- Bonferroni correction across suites (5 comparisons) or models.
- This is OPTIONAL for the paper but SHOULD be available in the evaluation harness.

### L5. Required Reporting Slices [MUST]

| Slice | Granularity |
|---|---|
| Overall | Aggregated across all suites |
| By suite | One row per suite |
| By suite × model | Primary results table |
| By domain | Appendix |
| By difficulty | Appendix |
| By offset (External, ICL, RAG, Tool) | Appendix |

### L6. Epsilon Sensitivity [SHOULD]

- Default epsilon = 3. Report sensitivity at epsilon in {1, 3, 5, 10} in appendix.
- Primary tables use epsilon = 3 only.

### L7. Metric Unification [MUST]

**CRITICAL:** All scripts must use the same UAI formula with `denom = a - y_control` and `epsilon = 3`. The local `compute_metrics()` functions in `run_external.py`, `run_rag.py`, and `run_history.py` use different denominators (`anchor_val - y_star`) and exclusion thresholds (`|denom| < 1`). These must be unified to use `compute_unified_metrics()` from `unified_metrics.py`. [MUST]

---

## M. Robustness-Control Subset

### Decision: OPTIONAL Extension [MUST separate]

The frozen primary benchmark does **not** include a required robustness subset. The following are documented as optional extensions:

- **Placebo / format control:** Stress-test items with `anchor_format: "words"` vs `"digits"`.
- **Position / salience control:** Stress-test items with `anchor_position: "early"` vs `"late"`.
- **Distance sensitivity control:** Analysis of UAI as a function of delta in {15, 25, 40} (available from primary data without additional items).
- **Retrieval / tool authority control:** RAG doc ordering, tool schema variation.

**Rationale:** Including robustness controls in the primary benchmark would inflate scope without strengthening the core claim. Distance sensitivity is already analyzable from the offset stratification.

---

## N. Artifact and Anonymous Release Contents (FROZEN)

### N1. Required Contents [MUST]

```
datasets/
  anchorbench_external_pilot/       # itemspecs.jsonl, promptviews.jsonl, manifest.json
  anchorbench_history_pilot/
  anchorbench_icl_pilot/
  anchorbench_rag_pilot/
  anchorbench_tool_pilot/
  anchorbench/                      # DATASET_CARD.md, manifest.json

src/
  anchorbench_v1/                   # Benchmark generation code
  mechinterp/                       # Mechanistic interpretability
  mitigation_eval/                  # Evaluation runner, parsing, strategies

scripts/
  eval/                             # Suite runners, unified_metrics
  run_*_gpus.sh                     # Multi-GPU launchers

docs/
  benchmark_spec_v1_frozen.md       # This file
  DATASET_CARD.md                   # Dataset documentation
  REPRODUCIBILITY.md                # Reproduction instructions

README.md                          # Repository overview
requirements.txt                   # Pinned dependencies
.gitignore
```

### N2. Required Documentation [MUST]

- **README.md:** Repository layout, setup, quickstart for reproducing paper results.
- **DATASET_CARD.md:** Dataset motivation, composition, collection process, intended use, ethical considerations, licensing.
- **REPRODUCIBILITY.md:** Step-by-step instructions to regenerate datasets and reproduce all paper tables/figures. Must include exact commands, expected outputs, and checksums.

### N3. Anonymization Requirements [MUST]

- No author names, affiliations, or identifying information in any code, comments, docstrings, commit messages, or documentation.
- No references to internal infrastructure (server names, internal paths, usernames).
- No `.env` files, API keys, or credentials.
- Git history must be squashed or the repo must be a fresh archive.
- W&B project names, if any, must be anonymized.
- Paper references to the benchmark should use "AnchorBench" without author attribution.

---

## O. Allowed vs. Forbidden Post-Freeze Changes

### Allowed

- Bug fixes that do not change benchmark semantics (e.g., fixing a parsing edge case that crashes).
- Code cleanup, docstrings, type hints, linting.
- Adding more models to the evaluation.
- Adding more seeds/runs for existing models.
- Appendix-only analyses (e.g., per-domain breakdowns, epsilon sensitivity).
- Optional extension subsets, clearly labeled as extensions in manifest and documentation.
- Adding bootstrap CIs to the statistics code.

### Forbidden

- Changing suite definitions (anchor source, delivery mechanism, prompt structure).
- Changing the condition matrix (adding/removing conditions).
- Changing relevance-level definitions or taxonomy.
- Changing gold-answer derivation (`y_star_evidence = round(mean(visible_evidence))`).
- Changing parsing semantics (parser logic, fallback behavior, validity range).
- Changing primary metrics (UAI formula, TAR definition, Disc_delta).
- Changing default decoding parameters.
- Changing domain list, difficulty levels, or balance policy.
- Changing anchor construction policy (theta range, offsets, wording templates).
- Renaming conditions in a way that changes pairing or aggregation.

---

*End of frozen specification.*
