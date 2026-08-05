# AnchorBench: Data Objects Reference

## ItemSpec

**Module:** `src/anchorbench/data/schema.py`

A prompt-agnostic ground-truth record for one benchmark item. Contains all information needed to render prompts and evaluate responses.

### Fields

| Field                | Type              | Description                                                |
|----------------------|-------------------|------------------------------------------------------------|
| `item_id`            | `str`             | Unique identifier (e.g., `EXT-pricing_wtp-e-off15-001`) |
| `suite`              | `str`             | Suite name: `external`, `history`, `icl`, `rag`, `tool` |
| `domain`             | `str`             | Domain ID (one of 6 domains)                               |
| `template_family`    | `str`             | Template variant identifier                                |
| `answer_space`       | `dict`            | Always `{"type": "int", "min": 0, "max": 100}`            |
| `theta`              | `int`             | Latent signal value, drawn from [30, 70]                   |
| `y_star`             | `int`             | Primary gold answer (= `y_star_evidence`)                   |
| `y_star_components`  | `dict`            | Derivation audit trail for the gold answer                 |
| `y_star_theta`       | `int`             | Gold answer from latent theta                              |
| `y_star_evidence`    | `int`             | Gold answer from `round(mean(visible evidence))`           |
| `difficulty`         | `str`             | `"easy"` or `"hard"`                                        |
| `anchors`            | `dict`            | Anchor values: `low`, `high`, `gap`, `offset`             |
| `evidence_structured`| `list[dict]`      | Evidence ratings (5 per item)                              |
| `tags`               | `dict`            | Metadata: `split`, `difficulty`, `icl_demos` (ICL only)    |
| `rag`                | `dict` or `None`  | RAG-specific: corpus size, retrieval method, doc roles     |
| `tool`               | `dict` or `None`  | Tool-specific: available tools, schema version             |
| `history`            | `dict` or `None`  | History-specific: subset indices, warmup cases             |
| `scenario_text`      | `str` or `None`   | LLM-generated scenario text (if enhanced)                  |
| `render_version`     | `str`             | Renderer version (`"2.0.0"`)                               |
| `generator_version`  | `str`             | Git hash at generation time                                |
| `seed`               | `int`             | Master seed used for generation                            |

### Evidence Entry Structure

Each evidence entry in `evidence_structured`:

```json
{
    "type": "rating",
    "value": 45,
    "value_raw": 45,
    "label": "Customer segment A survey score",
    "index": 0,
    "missing": false
}
```

For hard items, missing entries have `"value": null, "missing": true`.

### Anchor Dict Structure

```json
{
    "low": 25,
    "high": 75,
    "gap": 50,
    "offset": 25,
    "anchor_type": "numeric",
    "relevance": "stratified"
}
```

---

## PromptView

**Module:** `src/anchorbench/data/schema.py`

A rendered prompt for one item × condition, ready for LLM evaluation.

### Fields

| Field              | Type              | Description                                              |
|--------------------|-------------------|----------------------------------------------------------|
| `item_id`          | `str`             | Matches the parent ItemSpec                              |
| `suite`            | `str`             | Suite name                                               |
| `domain`           | `str`             | Domain ID                                                |
| `condition`        | `str`             | One of: `control`, `irrelevant_low/high`, `plausible_low/high` |
| `prompt_text`      | `str`             | Full prompt text to send to the LLM                      |
| `prompt_components`| `dict`            | Decomposed parts: `scenario`, `evidence`, `question`, etc. |
| `anchor_string`    | `str` or `None`   | The anchor value as it appears in the prompt             |
| `anchor_span`      | `list[int]` or `None` | Character offsets `[start, end]` of anchor in prompt |
| `anchor_relevance` | `str`             | `"none"`, `"irrelevant"`, or `"plausible"`               |
| `anchor_value`     | `int` or `None`   | Numeric anchor value (None for control)                  |
| `prompt_hash`      | `str`             | SHA-256 prefix for dedup (`sha256:...`)                  |
| `provenance`       | `dict`            | Rendering metadata (tool name for Tool suite, etc.)      |

### Condition Mapping

| Condition          | `anchor_relevance` | `anchor_value` |
|--------------------|--------------------|----------------|
| `control`          | `"none"`           | `None`         |
| `irrelevant_low`   | `"irrelevant"`     | `anchors["low"]` |
| `irrelevant_high`  | `"irrelevant"`     | `anchors["high"]` |
| `plausible_low`    | `"plausible"`      | `anchors["low"]` |
| `plausible_high`   | `"plausible"`      | `anchors["high"]` |

---

## RAGDoc

**Module:** `src/anchorbench/data/schema.py`

One document in the frozen RAG corpus (RAG suite only).

| Field          | Type            | Description                            |
|----------------|-----------------|----------------------------------------|
| `doc_id`       | `str`           | Unique document identifier             |
| `domain`       | `str`           | Domain ID                              |
| `doc_type`     | `str`           | `"core"`, `"filler"`, `"anchor_slot"`  |
| `text`         | `str`           | Document text                          |
| `role`         | `str`           | Same as `doc_type`                      |
| `relevance`    | `str`           | `"none"`, `"irrelevant"`, `"plausible"`|
| `anchor_value` | `int` or `None` | Anchor value embedded in the document  |

---

## Output File Formats

### `itemspecs.jsonl`

One JSON object per line. Each line is a serialized `ItemSpec.to_dict()`.

### `promptviews.jsonl`

One JSON object per line. Each line is a serialized `PromptView.to_dict()`.

### `manifest.json`

```json
{
    "version": "1.0.0",
    "benchmark_version": "2.0.0",
    "suite": "external",
    "generator_version": "abc1234",
    "seed": 42,
    "size": "pilot",
    "llm_enhanced": false,
    "timestamp": "2026-03-10T00:00:00Z",
    "counts": {
        "itemspecs": 180,
        "promptviews": 900,
        "suites": ["external"],
        "domains": ["pricing_wtp", "operations_time", ...],
        "difficulties": ["easy", "hard"],
        "anchor_offsets": [15, 25, 40],
        "conditions_per_item": 5
    },
    "files": {
        "itemspecs": "datasets/.../itemspecs.jsonl",
        "promptviews": "datasets/.../promptviews.jsonl"
    }
}
```

### Answer Format

Every prompt ends with the fixed instruction:

> Return only a single integer 0–100 on the last line.

This is enforced by the validator (`ANSWER_FORMAT_INSTRUCTION` in `schema.py`).
