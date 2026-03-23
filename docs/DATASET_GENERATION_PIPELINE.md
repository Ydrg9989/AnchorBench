# Benchmark Dataset Generation Pipeline

This document identifies the **code used** for AnchorBench benchmark dataset generation and the **redundant or unused** code that can be removed.

---

## Entry point

- **CLI:** `PYTHONPATH=src python -m anchorbench_v1.generate --suite <suite> --size <smoke|pilot|core> --out_dir <path>`
- **Script:** `scripts/generate_all.sh [SIZE] [SEED]` — loops over suites and calls the above.

---

## Code used for benchmark generation

### Core (always used)

| Module | Role |
|--------|------|
| `src/anchorbench_v1/generate.py` | Orchestrates: load params → generate ItemSpecs → (optional LLM enhance) → render PromptViews → write JSONL + manifest. |
| `src/anchorbench_v1/itemspec_gen.py` | Generates ItemSpec records per suite (external, history, icl, rag, tool, tool_agentic, tool_read). |
| `src/anchorbench_v1/schema.py` | ItemSpec, PromptView, RAGDoc, write_jsonl, read_jsonl. |
| `src/anchorbench_v1/domains.py` | DOMAIN_IDS, DOMAINS, domain configs (evidence labels, templates, anchor preambles). |
| `src/anchorbench_v1/suites/__init__.py` | SUITE_RENDERERS dispatch. |
| `src/anchorbench_v1/suites/_shared.py` | CONDITIONS, format_evidence, resolve_templates. |
| `src/anchorbench_v1/suites/external.py` | render_external. |
| `src/anchorbench_v1/suites/history.py` | render_history. |
| `src/anchorbench_v1/suites/icl.py` | render_icl. |
| `src/anchorbench_v1/suites/rag.py` | render_rag, build_item_corpus, build_full_corpus. |
| `src/anchorbench_v1/suites/tool_agentic.py` | render_tool (agentic). |
| `src/anchorbench_v1/suites/tool_read.py` | render_tool_read. |
| `src/anchorbench_v1/suites/tool.py` | Shim re-exporting tool_agentic (backward compat). |

### Optional (only with `--llm_enhance`)

| Module | Role |
|--------|------|
| `src/anchorbench_v1/llm_enhance.py` | enhance_scenarios(): OpenRouter calls to generate scenario text per item; cached. |
| `src/anchorbench_v1/openrouter_client.py` | OpenRouterClient used by llm_enhance. |
| `src/anchorbench_v1/config/__init__.py` | get_role_config("bulk_writer") for llm_enhance. |

### Validation (after generation, not part of “generation” code)

| Module | Role |
|--------|------|
| `src/anchorbench_v1/validate.py` | CLI to validate generated datasets. |
| `src/anchorbench_v1/validators.py` | validate_all(), schema and pairing checks. |

---

## Redundant / unused for benchmark generation

### 1. Removed: `src/llm_anchoring/`

- **Status:** No Python source files; only `__pycache__` left (sources were removed earlier).
- **Role:** Old pipeline (template_gen, render_dataset, etc.). Replaced by `anchorbench_v1`.
- **Action:** Delete the entire `src/llm_anchoring/` directory.

### 2. Removed: `src/anchorbench_v1/stress.py`

- **Status:** Never imported by `generate.py` or `itemspec_gen.py`.
- **Role:** Stress-split generator (200 items with stress factors); separate from the main benchmark.
- **Action:** Remove; reintroduce later if you add a stress split to the pipeline.

### 3. Not used by pipeline: `src/anchorbench_v1/llm_generate.py`

- **Status:** Not imported by `generate.py` or `llm_enhance.py`. `llm_enhance` uses `openrouter_client` and `config` directly.
- **Role:** Standalone role-based generation helper (`generate(role, prompt, ...)`).
- **Action:** Optional removal if you do not use this API elsewhere.

### 4. Not used by pipeline: `src/anchorbench_v1/model_registry.py`

- **Status:** Standalone CLI (`python -m anchorbench_v1.model_registry --check`). Not called by generate.
- **Role:** Verifies OpenRouter model IDs in config.
- **Action:** Keep if you use it for ops; optional remove if you never run it.

### 5. Different dataset: `scripts/data_gen/gen_syn_anchors_local.py`

- **Status:** Generates **SynAnchors** (40 items, 10 topics × 4), not AnchorBench.
- **Output:** e.g. `data/processed/syn_anchors_v0/dataset.jsonl`.
- **Action:** Keep if you use SynAnchors; remove if you only care about AnchorBench benchmark generation.

---

## Summary

- **Pipeline:** `generate.py` → `itemspec_gen` + `domains` + `schema` + `suites/*`; optionally `llm_enhance` → `openrouter_client` + `config`.
- **Removed as redundant for benchmark generation:** `src/llm_anchoring/` (dead), `src/anchorbench_v1/stress.py` (unused).
- **Optional:** `llm_generate.py`, `model_registry.py` (not in pipeline); `scripts/data_gen/gen_syn_anchors_local.py` (other dataset).
