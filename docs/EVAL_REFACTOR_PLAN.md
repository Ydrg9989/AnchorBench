# Evaluation Pipeline Refactor Plan

## Motivation

The 5 suite runners (`scripts/eval/run_external.py`, `run_history.py`, `run_icl.py`,
`run_rag.py`, `run_tool.py`) duplicated ~80% of their code. Two of them
(`run_external.py`, `run_rag.py`) contained local `compute_metrics()` functions with
the **wrong UAI formula** (denominator `anchor_val - y_star` instead of the canonical
`anchor_val - y_control`, and epsilon=1 instead of 3).

## What Changed

All shared logic is now in `src/anchorbench_eval/`:

| Module | Responsibility |
|--------|---------------|
| `parsing.py` | 3-tier parsing: structured JSON → regex → LLM self-extraction fallback |
| `metrics.py` | Canonical UAI/TAR/Disc_delta/MAE + bootstrap CI + paired tests |
| `backends.py` | `HFBackend` and `APIBackend` with structured output support |
| `evaluator.py` | `prepare_items`, `run_single_stage`, `run_history_two_stage`, `write_and_summarize` |
| `io.py` | `load_promptviews`, `load_itemspecs`, `load_records` |

Suite runners in `scripts/eval/run_*.py` are now thin CLI wrappers (~50-70 lines each).

## Parsing Strategy (3-Tier)

1. **Structured** (Tier 1): If backend supports `json_schema`, request `{"answer": int}`. Logged as `parse_strategy="structured"`.
2. **Regex** (Tier 2): Deterministic `parse_answer_int()` — canonical parser for official paper numbers. Logged as `"regex"`.
3. **LLM fallback** (Tier 3, opt-in): Same model re-reads its own output. Logged as `"llm_fallback"`. Override with `--fallback_model`.

## Backward Compatibility

- `src/mitigation_eval/` is untouched.
- Existing `results.jsonl` files remain compatible with the new `compute_unified_metrics()`.
- `scripts/eval/unified_metrics.py` re-exports from `anchorbench_eval.metrics`.
- Shell scripts (`run_*_gpus.sh`) keep working (CLI args preserved).

## New CLI Flags

- `--structured` — attempt structured JSON output (auto-detect by default)
- `--llm_fallback` — enable LLM self-extraction as last resort
- `--fallback_model` — override extraction model (default: same model)
