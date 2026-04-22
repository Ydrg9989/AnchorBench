# Repository Structure

## Core Directories

```
src/
├── anchorbench_v1/                 # Benchmark data generation
│   ├── __init__.py
│   ├── schema.py                   # ItemSpec, PromptView, RAGDoc dataclasses
│   ├── domains.py                  # 6 domain configs (templates, preambles)
│   ├── itemspec_gen.py             # ItemSpec generators for all 5 suites
│   ├── generate.py                 # CLI: python -m anchorbench_v1.generate
│   ├── validate.py                 # CLI: python -m anchorbench_v1.validate
│   ├── validators.py               # Deterministic + optional LLM validators
│   ├── llm_enhance.py              # Optional LLM scenario enhancement
│   ├── openrouter_client.py        # OpenRouter API client (sync, for generation)
│   ├── config/
│   │   └── models.yaml             # Model role configs for LLM enhancement
│   └── suites/                     # Suite-specific prompt renderers
│       ├── _shared.py              # CONDITIONS, format_evidence, resolve_templates
│       ├── external.py             # External suite renderer
│       ├── history.py              # History suite renderer (two-stage)
│       ├── icl.py                  # ICL suite renderer (demo headers)
│       ├── icl_dist.py             # ICL demo-label bands × framing
│       ├── rag.py                  # RAG corpus renderer
│       ├── tool.py                 # Backward-compat shim → tool_agentic
│       ├── tool_agentic.py         # Tool function-calling renderer
│       └── tool_read.py            # Tool plain-JSON renderer
│
├── anchorbench_eval/               # Evaluation and metrics
│   ├── __init__.py
│   ├── constants.py                # SUITE_DATASETS, MODEL_SHORT, API_MODEL_IDS,
│   │                               #   DEFAULTS — loads from configs/benchmark.yaml
│   ├── backends.py                 # HFBackend, VLLMBackend inference backends
│   ├── evaluator.py                # Orchestrator: prepare_items, parse_response,
│   │                               #   build_record, run_single_stage,
│   │                               #   run_history_two_stage, write_and_summarize
│   ├── io.py                       # load_itemspecs, load_promptviews, load_records
│   ├── metrics.py                  # UAI, TAR, Disc_Δ, MAE, Acc10, bootstrap CIs
│   ├── parsing.py                  # 3-tier parsing cascade (XML → regex → LLM fallback)
│   └── runner_utils.py             # Shared CLI args, backend/suffix builders,
│                                   #   discover_results, fmt helpers
│
└── mitigation_eval/                # API evaluation support
    ├── __init__.py
    └── async_api.py                # AsyncOpenRouterClient (used by run_api_benchmark.py)

configs/
└── benchmark.yaml                  # Suite paths, model names, API IDs, defaults

scripts/
├── paper_model_ids.inc.sh          # Ten HF model ids (matches paper model table)
├── generate_all.sh                 # Generate all 5 suites
├── validate_all.sh                 # Validate all 5 suites
├── freeze_dataset.sh               # Archive validated dataset with checksums
├── smoke_test.sh                   # End-to-end smoke test (no GPU)
├── run_with_env.sh                 # Wrapper setting LD_LIBRARY_PATH for vLLM
├── run_full_benchmark.sh           # Main multi-GPU orchestration (open-weight)
├── run_model_vllm.sh               # Per-model vLLM driver (called by run_full_benchmark)
├── run_api_smoke.sh                # API model smoke test
├── run_icl_gpus.sh                 # Multi-GPU launcher: ICL
├── run_tool_gpus.sh                # Multi-GPU launcher: Tool
├── run_rag_gpus.sh                 # Multi-GPU launcher: RAG
├── run_history_gpus.sh             # Multi-GPU launcher: History
└── eval/                           # Suite evaluation runners + aggregation
    ├── run_external.py             # External suite runner
    ├── run_history.py              # History suite runner (two-stage)
    ├── run_icl.py                  # ICL suite runner
    ├── run_rag.py                  # RAG suite runner
    ├── run_tool.py                 # Tool suite runner
    ├── run_api_benchmark.py        # API model runner (all suites)
    ├── recompute_all_unified.py    # Batch recomputation from results.jsonl
    ├── export_latex_tables.py      # Generate paper LaTeX tables
    ├── parse_failure_report.py     # Parse failure diagnostics
    ├── reparse_with_fallback.py    # Re-parse with LLM fallback extractor
    └── repair_history_api.py       # Offline repair for API History records

datasets/                           # Generated benchmark data
├── anchorbench_external_core/      # 360 items × 5 conditions = 1,800 prompts
├── anchorbench_history_core/
├── anchorbench_icl_core/
├── anchorbench_icl_dist_core/      # ICL numeric positive control
├── anchorbench_rag_core/
└── anchorbench_tool_core/

results/                            # Evaluation outputs (gitignored)
├── full_benchmark/                 # Open-weight model results
│   ├── {suite}/{model_slug}/results.jsonl
│   └── unified_all_suites.json     # Aggregated metrics
└── api_benchmark/                  # API model results
    ├── {suite}/{model_slug}/results.jsonl
    └── unified_all_suites.json

outputs/                            # Paper-ready tables and figures

tests/
├── conftest.py                     # Shared fixtures (eliminates sys.path hacks)
├── test_generation.py              # 30 tests: reproducibility, gold, pairing, diversity
├── test_parsing.py                 # Answer parsing tests
├── test_metrics.py                 # Metric computation tests
└── test_smoke_pipeline.py          # End-to-end pipeline smoke test (no GPU)

COLM/                               # Camera-ready paper source (LaTeX)

docs/                               # Documentation
├── benchmark_spec_v1_frozen.md     # Frozen benchmark specification
├── data_generation_pipeline_readme.md
├── data_objects_reference.md
├── ENVIRONMENT.md                  # Conda/vLLM/cuDNN setup
├── REPRODUCIBILITY.md              # Full reproduction guide
└── REPO_STRUCTURE.md               # This file
```

## Canonical Workflow

```bash
# 1. Install
pip install -e .                    # or: pip install -e ".[all]"

# 2. Smoke test (no GPU)
bash scripts/smoke_test.sh

# 3. Generate datasets
bash scripts/generate_all.sh core 42

# 4. Validate
bash scripts/validate_all.sh core

# 5. Run open-weight models (requires 4× H100 GPUs)
bash scripts/run_full_benchmark.sh

# 6. Run API models
python scripts/eval/run_api_benchmark.py \
    --model_id openai/gpt-5.4-mini --suites external history icl rag tool

# 7. Recompute unified metrics
python scripts/eval/recompute_all_unified.py \
    --results_dir results/full_benchmark

# 8. Generate paper figures and tables
python COLM/scripts/generate_paper_figures.py
python scripts/eval/export_latex_tables.py
```

## Key Entry Points

| Task                      | Command                                                        |
|---------------------------|----------------------------------------------------------------|
| Smoke test                | `bash scripts/smoke_test.sh`                                   |
| Generate all suites       | `bash scripts/generate_all.sh core 42`                         |
| Generate one suite        | `anchorbench-generate --suite external --size core --seed 42`  |
| Validate datasets         | `bash scripts/validate_all.sh core`                            |
| Freeze for submission     | `bash scripts/freeze_dataset.sh core`                          |
| Run single model (vLLM)   | `bash scripts/run_model_vllm.sh <MODEL_ID> <GPU_ID>`           |
| Run all open-weight       | `bash scripts/run_full_benchmark.sh`                           |
| Run API models            | `python scripts/eval/run_api_benchmark.py ...`                 |
| Compute unified metrics   | `python scripts/eval/recompute_all_unified.py`                 |
| Generate paper tables     | `python scripts/eval/export_latex_tables.py`                   |
| Run tests                 | `pytest tests/ -v`                                             |
