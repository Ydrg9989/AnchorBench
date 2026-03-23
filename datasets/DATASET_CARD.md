# AnchorBench

> **Dataset card for Hugging Face.** Copy this content into the README of your HF dataset repo (e.g. `Yiderigun/LLM_anchoring`) so it appears as the dataset card.

**A multi-paradigm benchmark for anchoring bias in large language models.**

## Dataset Description

AnchorBench measures how much LLM numeric estimates shift toward salient reference numbers delivered through six interface channels (prompt text, conversation history, in-context demos, retrieved documents, tool outputs in JSON, tool outputs via function calling). Each item is presented under matched conditions that share the same evidence and gold answer; only the anchor changes. The benchmark distinguishes **irrelevant** anchors (transparently arbitrary) from **plausible** anchors (weakly credible), so you can measure both raw susceptibility and relevance discrimination.

- **Homepage:** [Repository / paper link]
- **Paper:** AnchorBench: A Theory-Grounded, Multi-Paradigm Benchmark for Anchoring Bias in Large Language Models (COLM 2026)
- **Language:** English
- **Task:** Numeric estimation (0–100) under anchoring manipulations
- **License:** [To be specified]

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
| **Tool / Tool-Read** | (none) | Core-only |

**File:** `promptviews_ablation.jsonl` in each suite directory (where applicable).

## Suites (6)

| Suite | Anchor channel | Items | Core views | Ablation views | Total views |
|-------|----------------|-------|------------|----------------|-------------|
| **external_core** | Sentence in prompt | 360 | 1,800 | 1,440 | 3,240 |
| **history_core** | Prior turn in conversation | 120 | 600 | 120 | 720 |
| **icl_core** | Demo header metadata | 360 | 1,800 | 720 | 2,520 |
| **rag_core** | Retrieved document | 360 | 1,800 | 4,320 | 6,120 |
| **tool_core** | Tool output (function calling) | 360 | 1,800 | 0 | 1,800 |
| **tool_read_core** | Tool output (plain JSON) | 360 | 1,800 | 0 | 1,800 |

**Total core:** 1,920 items, 9,600 prompt views.
**Total with ablations:** 1,920 items, 16,200 prompt views.

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

```
external_core/
  promptviews_core.jsonl        # 1,800 lines (core)
  promptviews_ablation.jsonl    # 1,440 lines (placebo/authority)
  promptviews.jsonl             # 3,240 lines (all)
  itemspecs.jsonl               # 360 items

history_core/
  promptviews_core.jsonl        # 600 lines
  promptviews_ablation.jsonl    # 120 lines (control_twostage)
  promptviews.jsonl             # 720 lines
  itemspecs.jsonl               # 120 items

icl_core/
  promptviews_core.jsonl        # 1,800 lines
  promptviews_ablation.jsonl    # 720 lines (neutral)
  promptviews.jsonl             # 2,520 lines
  itemspecs.jsonl               # 360 items

rag_core/
  promptviews_core.jsonl        # 1,800 lines
  promptviews_ablation.jsonl    # 4,320 lines (order/disclaimer/authority)
  promptviews.jsonl             # 6,120 lines
  anchorbench_corpus.jsonl      # RAG document corpus
  itemspecs.jsonl               # 360 items

tool_core/
  promptviews_core.jsonl        # 1,800 lines
  promptviews.jsonl             # 1,800 lines
  itemspecs.jsonl               # 360 items

tool_read_core/
  promptviews_core.jsonl        # 1,800 lines
  promptviews.jsonl             # 1,800 lines
  itemspecs.jsonl               # 360 items
```

Total core: **9,600 lines** (5 conditions × ~1,920 items).
Total with ablations: **16,200 lines**.
