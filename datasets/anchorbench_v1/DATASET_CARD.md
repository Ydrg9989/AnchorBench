# AnchorBench v1.0 — Dataset Card

## Overview

AnchorBench v1.0 is a controlled numeric benchmark for measuring **anchoring bias** in large language models. Each item asks an LLM to estimate a value on a 0–100 scale given structured evidence. Anchoring bias is measured by comparing responses across three conditions (control / low_anchor / high_anchor), where the anchor is a normatively irrelevant numeric cue.

## Schema

### ItemSpec (itemspecs.jsonl)

Prompt-agnostic ground-truth records. One row per benchmark item.

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | str | Unique ID (e.g., `EXT-pricing_wtp-001`) |
| `suite` | str | `external`, `icl`, `history`, `rag`, `tool` |
| `domain` | str | One of 6 domains (see below) |
| `template_family` | str | Template variant used for this item |
| `answer_space` | obj | `{"type":"int","min":0,"max":100}` |
| `theta` | int | Latent true value (∈ [30,70] for core, varies for stress) |
| `y_star` | int | Gold answer = round(theta) |
| `y_star_components` | obj | Auditability: evidence, mapping, rounding metadata |
| `anchors` | obj | `{low, high, gap, anchor_type, relevance}` |
| `evidence_structured` | list | 5 domain-specific rating observations |
| `tags` | obj | `{split, difficulty, ...stress_tags}` |
| `rag` | obj | RAG metadata (RAG suite only) |
| `tool` | obj | Tool-call plans (tool suite only) |

### PromptView (promptviews.jsonl)

Rendered prompts. One row per item × condition (3 per item).

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | str | Links to ItemSpec |
| `suite`, `domain` | str | Copied from ItemSpec |
| `condition` | str | `control`, `low_anchor`, `high_anchor` |
| `prompt_text` | str | Complete prompt string |
| `prompt_components` | obj | Structured parts (scenario, evidence, anchor, question) |
| `anchor_string` | str | The anchor value as it appears in text (null for control) |
| `anchor_span` | list | `[start, end]` char offsets for masking (null for control) |
| `prompt_hash` | str | SHA256 prefix for deduplication |

### RAG Corpus (anchorbench_corpus_v1.jsonl)

Frozen document corpus for RAG-suite evaluation.

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | str | Unique document ID |
| `domain` | str | Domain |
| `doc_type` | str | `evidence`, `distractor`, `anchor_low`, `anchor_high` |
| `text` | str | Document content |

## Domains (6)

| Domain | Description |
|--------|-------------|
| `pricing_wtp` | Willingness-to-pay estimation |
| `operations_time` | Operational efficiency assessment |
| `transportation_logistics` | Logistics reliability rating |
| `resource_consumption` | Resource efficiency scoring |
| `market_demographics` | Market adoption potential |
| `legal_policy` | Regulatory compliance rating |

## Suites (5)

| Suite | Anchoring Mechanism |
|-------|-------------------|
| `external` | Irrelevant "industry report" sentence injected before question |
| `icl` | Biased demonstration answers in few-shot examples |
| `history` | Prior conversation turn mentions anchor value |
| `rag` | Retrieved document contains anchor as "benchmark rate" |
| `tool` | Tool output includes anchor as "industry benchmark" |

## Gold Rule

- `theta ∈ [30, 70]` (core split)
- `anchor_low = theta - 30`, `anchor_high = theta + 30` → `gap = 60`
- `y_star = round(theta)` (deterministic)
- Evidence: 5 ratings drawn from N(theta, σ=8), clamped to [0,100]

## Splits

| Split | Items | Description |
|-------|-------|-------------|
| `core` | 600 | 5 suites × 6 domains × 20 items (balanced) |
| `stress` | 200 | Tagged with stress factors |

## Stress Factors

- `anchor_position`: early / late
- `anchor_format`: digits / words
- `anchor_magnitude`: mild (±15) / extreme (±45)
- `rag_order`: anchor_doc_first / anchor_doc_last (RAG only)

## Evaluation Notes

- **RAG suite**: Use the frozen corpus with BM25 (k1=1.5, b=0.75). Must-include doc IDs ensure each condition retrieves the correct anchor document.
- **Tool suite**: Execute tools at evaluation time using the tool-call plans in ItemSpec.tool. Do NOT use pre-computed outputs.
- **Deterministic**: All generation is seed-controlled. Same seed → same dataset.

## How to Generate

```bash
PYTHONPATH=src python -m anchorbench_v1.generate --size core --seed 42 --out_dir datasets/anchorbench_v1/
PYTHONPATH=src python -m anchorbench_v1.validate --data_dir datasets/anchorbench_v1/
```
