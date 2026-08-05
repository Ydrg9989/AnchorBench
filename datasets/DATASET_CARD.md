---
license: cc-by-4.0
language:
- en
pretty_name: AnchorBench
task_categories:
- text-generation
tags:
- anchoring
- anchoring-bias
- cognitive-bias
- llm-evaluation
- benchmark
size_categories:
- 10K<n<100K
configs:
- config_name: external
  data_files: data/external.jsonl
- config_name: history
  data_files: data/history.jsonl
- config_name: icl
  data_files: data/icl.jsonl
- config_name: rag
  data_files: data/rag.jsonl
- config_name: tool
  data_files: data/tool.jsonl
- config_name: external_uncertain
  data_files: data/external_uncertain.jsonl
---

# AnchorBench

**A multi-paradigm benchmark for anchoring bias in large language models.**

## Dataset Description

AnchorBench measures how much LLM numeric estimates shift toward salient reference numbers delivered through five pathways (prompt text, conversation history, in-context demonstrations, retrieved documents, tool outputs). Each item is presented under matched conditions that share the same evidence and gold answer; only the anchor changes. The benchmark distinguishes **irrelevant** anchors (transparently arbitrary) from **plausible** anchors (weakly credible), so you can measure both raw susceptibility and relevance discrimination.

- **Homepage:** https://github.com/Yiderigun/LLM_anchoring
- **Paper:** AnchorBench: A Multi-Pathway Benchmark for the Anchoring Effect in LLMs (COLM 2026)
- **Language:** English
- **Task:** Numeric estimation (0–100) under anchoring manipulations
- **License:** CC BY 4.0 (items are synthetic and author-generated)
- **Splits:** none. AnchorBench is eval-only; every row is `split: test`.

## Two-Layer Structure

The benchmark is organized into **core** and **ablation** layers:

### Core Layer (main results)

The core layer contains the base 5-condition comparison used for all primary results. Every suite shares these conditions:

| Condition | Anchor | Relevance |
|-----------|--------|-----------|
| `control` | None | — |
| `irrelevant_low` | Low value | Irrelevant (e.g. "case #X") |
| `irrelevant_high` | High value | Irrelevant |
| `plausible_low` | Low value | Plausible (e.g. "survey suggested X") |
| `plausible_high` | High value | Plausible |

**File:** `promptviews_core.jsonl` in each suite directory.

### Ablation Layer (robustness checks)

Suite-specific extended conditions for appendix analyses:

| Suite | Ablation Conditions | Purpose |
|-------|-------------------|---------|
| **External** | `placebo_low/high`, `authority_low/high` | Salience vs. content controls |
| **History** | `control_twostage` | Matched two-stage neutral control |
| **ICL** | `neutral_low/high` | Pure numeric priming (no semantic header) |
| **RAG** | `*_order_first/last`, `*_nodiscl`, `*_authority` | Position, disclaimer, framing ablations |
| **Tool** | (none) | Core-only |

**File:** `promptviews_ablation.jsonl` in each suite directory (where applicable).

## Suites (6)

| Suite | Anchor pathway | Items | Core views | Ablation views | All views |
|---|---|---:|---:|---:|---:|
| **external_core** | Sentence in the prompt | 360 | 1,800 | 1,440 | 3,240 |
| **history_core** | The model's own prior turn | 360 | 1,800 | 360 | 2,160 |
| **icl_core** | Demonstration metadata | 360 | 1,800 | 720 | 2,520 |
| **rag_core** | Retrieved document | 360 | 1,800 | 4,320 | 6,120 |
| **tool_core** | Tool-call response | 360 | 1,800 | 0 | 1,800 |
| **external_uncertain** | Prompt, with only k of 5 ratings visible | 360 | 5,400 | 0 | 5,400 |

**Total core:** 1,800 items, 9,000 core prompt views (5 suites).
**With ablations:** 15,840 prompt views.
**Released on the Hub:** 14,400 rows -- the 9,000 core views plus the 5,400
uncertain views. `external_uncertain` backs Table 2 in the main paper.

## Data Schema

Each file is **JSONL**: one JSON object per line.

**Path layout:** `{suite}_core/promptviews_core.jsonl` (core) or `{suite}_core/promptviews_ablation.jsonl` (ablations).

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | string | Unique item id (same for all conditions of one item). |
| `condition` | string | Condition name (see tables above). |
| `prompt_text` | string | Full prompt to send to the model. |
| `anchor_value` | int \| null | Numeric anchor in the prompt. `null` for controls. |
| `anchor_relevance` | string | `"none"`, `"irrelevant"`, `"plausible"`, `"placebo"`, `"authority"`, or `"neutral"`. |

Ground-truth answer `y_star` is in `itemspecs.jsonl` (matched by `item_id`).

## How to Use

### Load the data

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download(
    repo_id="Yiderigun/LLM_anchoring",
    filename="external_core/promptviews_core.jsonl",
    repo_type="dataset",
)
with open(path) as f:
    for line in f:
        row = json.loads(line)
        # row keys: item_id, condition, prompt_text, anchor_value, ...
        break
```

### Minimal evaluation

1. **Control accuracy:** For `condition == "control"`, compare model output to `y_star` from itemspecs.
2. **UAI (Unified Anchor Influence):**
   `UAI = (model_answer - y_control) / (anchor_value - y_control)`
   (exclude items where `|anchor_value - y_control| < 3`).
3. **TAR (Toward-Anchor Rate):** Fraction of anchor-condition answers that shift toward `anchor_value` relative to control.
4. **Discrimination:** `UAI_plausible - UAI_irrelevant` (positive = model discriminates by relevance).

### Extended evaluation

Use the provided metrics module for bootstrap CIs, paired Wilcoxon tests with BH correction, parse-policy sensitivity, and by-offset/by-difficulty breakdowns:

```python
from anchorbench_eval.metrics import compute_extended_metrics
metrics = compute_extended_metrics(records)
# metrics includes: uai_irr_ci, uai_plaus_ci, disc_delta_ci,
# p_plaus_vs_irr, p_irr_vs_zero, parse_by_condition, etc.
```

## Domains and Design

- **6 domains:** Pricing/WTP, Operations/Time, Transportation/Logistics, Resource Consumption, Market/Demographics, Legal/Policy.
- **2 difficulties:** Easy (concordant evidence), Hard (missing/conflicting values).
- **Gold answer:** `y_star = round(mean(visible evidence))` (0–100).

## Intended Use

- Research on anchoring bias in LLMs.
- Benchmarking susceptibility and relevance discrimination across models and suites.
- Not for training; evaluation only.

## Citation

```bibtex
@inproceedings{anchorbench2026,
  title     = {AnchorBench: A Theory-Grounded, Multi-Paradigm Benchmark for Anchoring Bias in Large Language Models},
  booktitle = {Proceedings of the Conference on Language Modeling (COLM)},
  year      = {2026},
}
```

## Dataset Structure

Released layout on the Hub — one file per suite, 14,400 rows in total:

```
data/external.jsonl              1,800 rows
data/history.jsonl               1,800
data/icl.jsonl                   1,800
data/rag.jsonl                   1,800
data/tool.jsonl                  1,800
data/external_uncertain.jsonl    5,400   (360 items x 15 conditions)
```

The internal repository additionally carries `itemspecs.jsonl` and the
ablation views for each suite; those are not part of this release. See the
[code repository](https://github.com/Yiderigun/LLM_anchoring) for them.

### Fields

| Field | Type | Description |
|---|---|---|
| `item_id` | string | Stable identifier. Encodes suite, domain, difficulty (`e`/`h`) and anchor offset, e.g. `EXT-pricing_wtp-e-off15-001`. |
| `suite` | string | `external`, `history`, `icl`, `rag`, `tool`, `external_uncertain`. |
| `condition` | string | One of the five matched conditions (15 for the uncertain suite). |
| `relevance` | string | `none` for control, else `irrelevant` / `plausible` / `self_generated`. |
| `anchor_polarity` | string \| null | `low` or `high`; null for control. |
| `anchor_value` | int \| null | The number placed in the prompt. Null for control **and for all of History** — see below. |
| `offset` | int \| null | Designed distance of the anchor from the evidence centre: 15, 25 or 40. |
| `difficulty` | string | `easy` (5 ratings, low noise) or `hard` (3 of 5 shown, higher noise). |
| `domain` | string | One of six business domains. |
| `prompt_text` | string | The exact prompt the models were given. |
| `y_star_evidence` | int | Gold answer: the rounded mean of the *full* five ratings. |
| `split` | string | Always `test`. |

**`anchor_value` is null for every History row, deliberately.** History's
anchor is the model's *own* Stage-1 answer, so it is a property of the
(item, model) run rather than of the item — the same item records 80 for
Qwen-7B and 81 for Llama-8B. Publishing one number would misrepresent what the
model saw. Take the realised value from the released results instead.

### Loading

```python
from datasets import load_dataset

ds = load_dataset("Yiderigun/LLM_anchoring", data_files="data/external.jsonl")["train"]

# UAI needs the control answer for the same item, so group by item_id.
row = ds[0]
print(row["condition"], row["anchor_value"], row["y_star_evidence"])
```

Scoring an anchored response, given the model's control answer `y_ctrl` for
the same `item_id`:

```
UAI = (y_anchored - y_ctrl) / (anchor_value - y_ctrl)
```

Items where `|anchor_value - y_ctrl| < 3` are excluded from UAI in the paper:
the near-zero denominator makes the ratio unstable. Conclusions are stable
across thresholds of 1, 3 and 5.

### Known properties worth stating

- **Eval-only.** No train or validation portion; every row is `split: test`.
- **Three prompts repeat inside `uncertain_p1_control`** (6 of 360 items,
  1.7%) with different gold answers. That is the design rather than a
  collision: at k=1 the prompt shows one visible rating while gold stays the
  mean of all five, so two items sharing a first rating give identical prompts
  with different answers. The irreducible uncertainty is the point of that
  condition.
- **Synthetic by construction.** Items are instantiated from author-designed
  templates; scenario text, evidence labels and anchor framings come from
  pools generated with LLM assistance and then verified by the authors, while
  the numeric evidence, anchor values and gold answers come from seeded
  sampling. AnchorBench is a controlled diagnostic, not a sample of organic
  user queries.

Version history and the mapping from dataset version to paper tables is in
[VERSIONS.md](VERSIONS.md).
