# PoC Anchoring Dataset

Proof-of-concept dataset for evaluating anchoring effects in LLMs.

## Versions

| File | Items | Suites |
|---|---|---|
| `poc.jsonl` | 60 | external (30), self-generated (30) |
| `poc_v0.2.jsonl` | 108 | original 60 + ICL (12), conversation-history (12), RAG (12), tool (12) |

## Quick start

```bash
# Generate the original dataset (deterministic, seed=42)
python generate_poc_dataset.py

# Validate original
python validate_poc_dataset.py

# Extend with new suites (requires OPENROUTER_API_KEY)
export OPENROUTER_API_KEY=your-key-here
python extend_poc_dataset_v0_2.py

# Validate extended dataset
python validate_poc_v0_2.py
```

## File overview

| File | Purpose |
|---|---|
| `schema.json` | JSON Schema for each JSONL line |
| `generate_poc_dataset.py` | Generates `poc.jsonl` (stdlib only) |
| `validate_poc_dataset.py` | Validates `poc.jsonl` |
| `extend_poc_dataset_v0_2.py` | Extends `poc.jsonl` → `poc_v0.2.jsonl` via OpenRouter API |
| `validate_poc_v0_2.py` | Validates `poc_v0.2.jsonl` (all 108 items) |

## Dataset structure

### Original suites (v0.1)

- **External** (`A-0001`..`A-0030`): `control`, `low_anchor` (20), `high_anchor` (80) prompts.
- **Self-generated** (`B-0001`..`B-0030`): `turn1`, `turn2_fresh`, `turn2_history` (with `{{A1}}`). 10 items include controlled-history variants [10, 30, 50, 70, 90].

### New suites (v0.2)

All new items use `control`/`low_anchor`/`high_anchor` with identical target cases; only the anchoring mechanism differs.

- **ICL** (`ICL-0001`..`ICL-0012`): 3 in-context demos + target case. Demo answers anchor low [15-35] or high [65-85].
- **Conversation-history** (`HIST-0001`..`HIST-0012`): Chat-history block with anchor (20/80). Subtypes: `history_relevant` (6) and `history_irrelevant` (6).
- **RAG** (`RAG-0001`..`RAG-0012`): Retrieved snippets with anchor in additional snippet (20/80, or `[VALUE]` for control).
- **Tool** (`TOOL-0001`..`TOOL-0012`): Tool output block showing anchor value (20/80).

### Category balance

Fields: `health`, `finance`, `ops`. Each new suite: 4 items per field, 2 per domain.

## Dependencies

- `poc.jsonl` generation: Python standard library only.
- `poc_v0.2.jsonl` extension: `requests` (for OpenRouter API).
- Model: `anthropic/claude-opus-4.6` (override via `OPENROUTER_MODEL` env var).

## Environment

Set your OpenRouter API key:

```
OPENROUTER_API_KEY=your-key-here
```
