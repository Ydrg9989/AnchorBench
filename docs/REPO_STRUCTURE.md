# Repository Structure

## Core Directories

```
src/
├── anchorbench_v1/                 # Benchmark generation package
│   ├── __init__.py                 # Package init, version
│   ├── schema.py                   # ItemSpec, PromptView, RAGDoc dataclasses
│   ├── domains.py                  # 6 domain configs (templates, preambles)
│   ├── itemspec_gen.py             # ItemSpec generators for all 5 suites
│   ├── generate.py                 # CLI entry point: python -m anchorbench_v1.generate
│   ├── validate.py                 # CLI entry point: python -m anchorbench_v1.validate
│   ├── validators.py               # Deterministic + optional LLM validators
│   ├── suites/                     # Suite-specific prompt renderers
│   │   ├── external.py             # External suite renderer
│   │   ├── history.py              # History suite renderer (two-stage)
│   │   ├── icl.py                  # ICL suite renderer (demo headers)
│   │   ├── rag.py                  # RAG renderer (corpus injection)
│   │   └── tool.py                 # Tool renderer (JSON tool output)
│   ├── llm_enhance.py              # Optional LLM scenario enhancement
│   ├── openrouter_client.py        # OpenRouter API client
│   └── config/
│       └── models.yaml             # Model role configs for LLM enhancement
│
├── mitigation_eval/                # Evaluation runner
│   ├── runner.py                   # HFRunner, parsing, inference
│   ├── cli.py                      # Evaluation CLI
│   └── async_api.py                # Async API client
│
└── mechinterp/                     # Mechanistic interpretability
    ├── activation_patching.py      # Causal tracing
    └── logit_lens.py               # Logit lens analysis

scripts/
├── generate_all.sh                 # Generate all 5 suites (canonical)
├── validate_all.sh                 # Validate all 5 suites (canonical)
├── freeze_dataset.sh               # Archive validated dataset
├── eval/                           # Suite evaluation runners
│   ├── run_external.py
│   ├── run_history.py
│   ├── run_icl.py
│   ├── run_rag.py
│   ├── run_tool.py
│   ├── run_tool_ecological.py
│   ├── unified_metrics.py          # Canonical UAI/TAR/Disc_delta
│   └── recompute_all_unified.py    # Batch recomputation
├── run_icl_gpus.sh                 # Multi-GPU launcher: ICL
├── run_tool_gpus.sh                # Multi-GPU launcher: Tool
├── run_rag_gpus.sh                 # Multi-GPU launcher: RAG
├── run_external_8B_7B.sh            # Multi-GPU launcher: External
└── run_all.sh                      # Orchestrated mechinterp + mitigation

datasets/                           # Generated benchmark data (gitignored JSONL)
├── anchorbench_external_pilot/
├── anchorbench_history_pilot/
├── anchorbench_icl_pilot/
├── anchorbench_rag_pilot/
└── anchorbench_tool_pilot/

results/                            # Evaluation outputs (gitignored)

docs/                               # Documentation
├── benchmark_spec_v1_frozen.md     # Frozen benchmark specification
├── freeze_decision_memo.md         # Decision memo
├── data_generation_pipeline_readme.md
├── data_objects_reference.md
├── full_benchmark_generation_plan.md
├── REPRODUCIBILITY.md
├── REPO_STRUCTURE.md               # This file
└── camera_ready_gap_report.md

paper/                              # COLM 2026 paper source (LaTeX)
```

## Canonical Workflow

```
1. pip install -r requirements.txt
2. bash scripts/generate_all.sh pilot 42
3. bash scripts/validate_all.sh pilot
4. bash scripts/freeze_dataset.sh pilot
5. bash scripts/run_{suite}_gpus.sh          # per suite
6. PYTHONPATH=src:scripts/eval python scripts/eval/recompute_all_unified.py
```

## Key Entry Points

| Task                      | Command                                                        |
|---------------------------|----------------------------------------------------------------|
| Generate all suites       | `bash scripts/generate_all.sh pilot 42`                        |
| Generate one suite        | `PYTHONPATH=src python -m anchorbench_v1.generate --suites external --size pilot --seed 42 --out_dir datasets/...` |
| Validate datasets         | `bash scripts/validate_all.sh pilot`                           |
| Freeze for submission     | `bash scripts/freeze_dataset.sh pilot`                         |
| Run evaluation            | `PYTHONPATH=src python scripts/eval/run_icl.py --model ... --promptviews ... --itemspecs ... --out_dir ...` |
| Compute unified metrics   | `PYTHONPATH=src:scripts/eval python scripts/eval/unified_metrics.py results/.../results.jsonl` |
