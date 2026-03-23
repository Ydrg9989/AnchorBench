# AnchorBench Evaluation Pipeline

This document describes how evaluation experiments are set up: data loading, inference, **parsing**, **metrics**, and result files.

---

## 1. Overview

- **Shared evaluation package:** `src/anchorbench_eval/` contains all parsing, metrics, backend, and orchestration logic.
- **Suite runners:** `scripts/eval/run_external.py`, `run_history.py`, `run_icl.py`, `run_rag.py`, `run_tool.py` — thin CLI wrappers (~50-70 lines each) that call into the shared package.
- **Unified metrics:** `scripts/eval/unified_metrics.py` re-exports from `anchorbench_eval.metrics` for backward compatibility.
- **Batch comparison:** `scripts/eval/recompute_all_unified.py` reads multiple `results.jsonl` files (no re-inference), runs unified metrics, and prints comparison tables + LaTeX.

**Two pipelines:**

| Pipeline | Purpose | Output |
|----------|---------|--------|
| **run_\*.py** | Run inference for one suite + one model | `results.jsonl`, `summary.json` |
| **unified_metrics.py** / **recompute_all_unified.py** | Compute/compare canonical metrics from existing results | Printed summary, `unified_summary.json` |

---

## 2. Architecture

```
src/anchorbench_eval/
├── __init__.py
├── parsing.py       # 3-tier parsing: structured → regex → LLM fallback
├── metrics.py       # UAI, TAR, Disc_delta, MAE, bootstrap CI, paired tests
├── backends.py      # HFBackend, VLLMBackend, APIBackend
├── evaluator.py     # prepare_items, run_single_stage, run_history_two_stage
└── io.py            # load_promptviews, load_itemspecs, load_records

scripts/eval/
├── run_external.py  # Thin wrapper → evaluator.run_single_stage
├── run_icl.py       # Thin wrapper → evaluator.run_single_stage (batched)
├── run_rag.py       # Thin wrapper → evaluator.run_single_stage (batched)
├── run_tool.py      # Thin wrapper → evaluator + chat-template tool messages
├── run_history.py   # Thin wrapper → evaluator.run_history_two_stage
├── unified_metrics.py        # Re-export from anchorbench_eval.metrics
└── recompute_all_unified.py  # Batch recompute from existing results
```

---

## 3. Experiment Settings

### 3.1 Inputs

- **Promptviews:** `promptviews.jsonl` — one JSON per prompt (item_id, condition, prompt_text, anchor_value, ...). Must have all 5 conditions per item.
- **Itemspecs:** `itemspecs.jsonl` — one JSON per item (item_id, y_star_evidence, difficulty, ...). Used for gold answer and metadata.

### 3.2 Model and decoding

- **Backends** (`src/anchorbench_eval/backends.py`):
  - **HFBackend** (default): HuggingFace Transformers — loads a causal LM, applies chat template, generates with `model.generate()`. Single-GPU or `device_map`; may underutilize GPU memory.
  - **VLLMBackend** (`--backend vllm`): vLLM — PagedAttention and continuous batching for high GPU utilization. Same chat/tool formatting as HF. Supports `--tensor_parallel_size` for multi-GPU and `--gpu_memory_utilization` (default 0.9).
  - **APIBackend**: OpenRouter/OpenAI-compatible API (used by API smoke tests).

- **Decoding:** temperature 0 (greedy), `--max_tokens` 512 by default.

- **History suite:** Control is single-turn; other conditions are **two-stage**. Uses `generate_chat()` for multi-turn.

### 3.3 Optional prompt suffixes

- `--request_final_line`: appends instruction to put the final estimate on the last line.
- `--request_reasoning`: asks for step-by-step reasoning.

### 3.4 Parsing options

- `--structured`: attempt structured JSON output when backend supports it.
- `--llm_fallback`: if the primary parser fails, use the same model (or `--fallback_model`) to extract the answer.

---

## 4. Parsing: 3-Tier Cascade

### Tier 1 — Structured output (`parse_structured`)

When `--structured` is enabled and the backend supports it, the model returns JSON `{"answer": int}`. If valid and in [0, 100], `parse_strategy="structured"`.

### Tier 2 — Deterministic regex (`parse_answer_int`)

**Location:** `src/anchorbench_eval/parsing.py`

**Order of rules:**

1. **Whole response is a single integer** — if `raw_text.strip()` matches `-?\d+` and value in [0, 100], return it.
2. **All candidate integers agree** — collect all integers in [0, 100]; if only one unique value, return it.
3. **CoT-style last number** — regex `(?:is|=|:\s*)\s*(\d{1,3})\s*(?:[.\s\n]|$)`, take last match.
4. **Last line has a single valid integer** — if exactly one integer in [0, 100] on the last line, return it.
5. **Echo filtering** — remove integers that also appear in `prompt_text`; if one candidate remains, return it.
6. Otherwise return `(None, False)`.

### Tier 3 — LLM self-extraction (`--llm_fallback`)

When both structured and regex fail:

- **Default:** use the same model to re-read its own output with an extraction prompt.
- **Override:** `--fallback_model` for speed/debug.
- Every parse result is logged with strategy: `"structured"`, `"regex"`, `"llm_fallback"`, or `"failed"`.

### Parse strategy taxonomy

| Strategy | Source | Official? |
|----------|--------|-----------|
| `structured` | Backend returned valid JSON with `answer` field | Yes |
| `regex` | Deterministic regex extraction from raw text | Yes (canonical) |
| `llm_fallback` | Same-model (or override) re-extraction | Logged, flagged |
| `failed` | All tiers failed | Excluded from metrics |

---

## 5. Result Record Schema

Each line of `results.jsonl` is one **item x condition** run:

```json
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "item_id": "EXT-pricing_wtp-e-off15-001",
  "suite": "external",
  "domain": "pricing_wtp",
  "difficulty": "easy",
  "condition": "plausible_low",
  "anchor_relevance": "plausible",
  "anchor_value": 55,
  "y_star_evidence": 77,
  "y_star_theta": 70,
  "answer_int": 72,
  "parsed_ok": true,
  "parse_strategy": "regex",
  "raw_text": "..."
}
```

**History-specific:** Rows may include `stage1_answer`, `stage1_raw_text`; `anchor_value` is the model's Stage 1 answer.

---

## 6. Evaluation Metrics (Unified)

**Source:** `src/anchorbench_eval/metrics.py` — `compute_unified_metrics(records, epsilon=3.0)`.

### 6.1 Definitions

- **Parse rate:** `#(parsed_ok and answer_int not null) / #records`.
- **MAE_control:** Mean absolute error on control condition: `mean(|answer_int - y_star_evidence|)`.
- **Acc10_control:** Fraction of control rows with error ≤ 10.
- **UAI (Unified Anchor Influence):** `(y_anchor - y_control) / (a - y_control)`, excluded when `|a - y_control| < epsilon`.
- **TAR (Toward-Anchor Rate):** 1 if `(y_anchor - y_control) * (a - y_control) > 0`, else 0.
- **Disc_delta:** `UAI_plaus - UAI_irr`.

### 6.2 History-specific (appendix)

- **ACR:** `(stage2 - stage1) / (y_star - stage1)` when `|y_star - stage1| ≥ 1`.
- **RR:** `1 - |stage2 - y_star| / |stage1 - y_star|` when `|stage1 - y_star| ≥ 1`.

### 6.3 Statistical helpers

- `bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05)` — bootstrap CI for the mean.
- `paired_wilcoxon(x, y)` — Wilcoxon signed-rank test.
- `bh_correction(p_values)` — Benjamini-Hochberg FDR correction.

---

## 7. How to Run Experiments

### Backend choice: HF vs vLLM

- **Default:** `--backend hf` uses HuggingFace Transformers (familiar, but often leaves GPU memory underutilized).
- **vLLM:** `--backend vllm` uses vLLM for higher throughput and full GPU utilization (PagedAttention, continuous batching). Install with `pip install vllm` (see `requirements.txt`).

**vLLM-specific options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism (e.g. 2 or 4 for 70B). |
| `--gpu_memory_utilization` | 0.9 | Fraction of GPU memory to use (0–1). |
| `--max_model_len` | 4096 | Max sequence length. |

**Example (vLLM, single GPU):**

```bash
PYTHONPATH=src python scripts/eval/run_external.py \
  --promptviews datasets/anchorbench_external_core/promptviews.jsonl \
  --itemspecs datasets/anchorbench_external_core/itemspecs.jsonl \
  --model_id meta-llama/Llama-3.2-3B-Instruct \
  --out_dir results/external_vllm \
  --backend vllm \
  --gpu_memory_utilization 0.95
```

**Example (vLLM, 70B with tensor parallelism):**

```bash
PYTHONPATH=src python scripts/eval/run_external.py \
  --promptviews datasets/anchorbench_external_core/promptviews.jsonl \
  --itemspecs datasets/anchorbench_external_core/itemspecs.jsonl \
  --model_id meta-llama/Llama-3.1-70B-Instruct \
  --out_dir results/external_70b \
  --backend vllm \
  --tensor_parallel_size 4 \
  --gpu_memory_utilization 0.9 \
  --max_model_len 4096
```

### Single suite, single model (HF)

```bash
PYTHONPATH=src python scripts/eval/run_external.py \
  --promptviews datasets/anchorbench_external_core/promptviews.jsonl \
  --itemspecs datasets/anchorbench_external_core/itemspecs.jsonl \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --out_dir results/external_core \
  --max_tokens 512
```

For CoT models with low parse rate:

```bash
  --llm_fallback --request_final_line
```

### Recompute all unified metrics (no new inference)

```bash
PYTHONPATH=src python scripts/eval/recompute_all_unified.py
```

### Run tests

```bash
PYTHONPATH=src python -m pytest tests/test_parsing.py tests/test_metrics.py -v
```

---

## 8. File Layout Summary

| Path | Role |
|------|------|
| `src/anchorbench_eval/parsing.py` | 3-tier parsing: parse_structured, parse_answer_int, LLMFallbackExtractor |
| `src/anchorbench_eval/metrics.py` | compute_unified_metrics, bootstrap_ci, paired_wilcoxon, bh_correction |
| `src/anchorbench_eval/backends.py` | HFBackend, APIBackend |
| `src/anchorbench_eval/evaluator.py` | prepare_items, run_single_stage, run_history_two_stage, write_and_summarize |
| `src/anchorbench_eval/io.py` | load_promptviews, load_itemspecs, load_records |
| `scripts/eval/run_*.py` | Thin CLI wrappers for each suite |
| `scripts/eval/unified_metrics.py` | Re-export from anchorbench_eval.metrics (backward compat) |
| `scripts/eval/recompute_all_unified.py` | Batch recompute from existing results.jsonl |
| `tests/test_parsing.py` | 37 tests for parsing (structured, regex, fallback, cascade) |
| `tests/test_metrics.py` | 14 tests for metrics (UAI, TAR, bootstrap, Wilcoxon, BH) |

---

## 9. Migration Notes

### What changed (refactor from previous version)

1. **Shared package created:** All parsing, metrics, IO, backend, and evaluation logic moved to `src/anchorbench_eval/`.
2. **Suite runners slimmed:** Each `scripts/eval/run_*.py` went from 230-440 lines to ~50-70 lines.
3. **Metric drift fixed:** The incorrect local `compute_metrics()` in `run_external.py` and `run_rag.py` (wrong UAI denominator `anchor_val - y_star` and epsilon=1) has been removed. All suites now use the canonical `compute_unified_metrics()`.
4. **3-tier parsing added:** Structured JSON output (Tier 1) and same-model LLM fallback (Tier 3) are now available via `--structured` and `--llm_fallback` flags.
5. **Statistical helpers added:** `bootstrap_ci`, `paired_wilcoxon`, `bh_correction` in `metrics.py`.
6. **`src/mitigation_eval/`** is untouched — the new package is independent.

### Backward compatibility

- Existing `results.jsonl` files are fully compatible with the new `compute_unified_metrics()`.
- `scripts/eval/unified_metrics.py` is a thin re-export; existing imports still work.
- CLI args are preserved; shell scripts (`run_*_gpus.sh`) keep working.
- Old `summary.json` files from External/RAG used the wrong formula; new runs produce correct ones.

### Deprecated

- Local `compute_metrics()` functions in the old `run_external.py` and `run_rag.py` — removed entirely.
- Direct imports from `mitigation_eval.runner` for parsing/IO in suite runners — replaced by `anchorbench_eval.*` imports.
