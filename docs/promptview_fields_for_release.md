# PromptView fields: prompt_hash, anchor_span, provenance

## What they are

| Field | Meaning | Example |
|-------|--------|--------|
| **prompt_hash** | Truncated SHA-256 of `prompt_text`. Used to verify determinism and detect duplicate prompts. | `"sha256:4b933057d5cfbdd3"` |
| **anchor_span** | Character offsets `[start, end]` in `prompt_text` where the anchor string appears. | `[342, 344]` for the substring `"55"` |
| **provenance** | Optional metadata about how the prompt was built (e.g. template, corpus_id). In current data it is usually `{}`. | `{}` or `{"corpus_id": "RAG-..."}` |

## Where they are used

- **Benchmark evaluation** (`scripts/eval/run_*.py`): Only `prompt_text`, `item_id`, `condition`, `domain`, `suite` (and gold from itemspecs) are used. **prompt_hash**, **anchor_span**, and **provenance** are **not** used for running models or computing UAI/TAR.
- **Validators** (`validators.py`): Require `prompt_hash` and use it to ensure the five conditions per item have distinct prompts (no duplicate hashes). They do not use `anchor_span` or `provenance`.
- **Mitigation / mechinterp** (`src/mitigation_eval/`, `src/mechinterp/`): Use **anchor_span** to locate the anchor in the prompt for strategies like late binding or counterfactual edits. They do not need `prompt_hash` or `provenance` from the JSONL.

## Can you remove them for the public benchmark?

| Field | Safe to remove? | Note |
|-------|------------------|------|
| **provenance** | **Yes** | Always empty or internal. Not needed for running the benchmark. |
| **prompt_hash** | **Yes** | Not needed for evaluation. Consumers can recompute from `prompt_text` if they need a hash. Validation of the *released* file can skip hash checks or recompute. |
| **anchor_span** | **Yes, if** you only care about core benchmark use (run model → parse answer → UAI/TAR). **Keep** if you want the public dataset to support mitigation and mechanistic interp out of the box (those need to know where the anchor is in the text). | If removed, mitigation/mechinterp code would need to derive span by searching for `anchor_string` in `prompt_text`. |

**Recommendation:** For a minimal public benchmark dataset you can remove all three. Use the export script below to generate a HF-ready JSONL without these fields.
