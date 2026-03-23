# Dataset and Results Cleanup Plan

This document defines what to **keep** (final benchmarking dataset + experiment results) and what to **remove** to reduce clutter.

---

## Datasets (`datasets/`)

### KEEP — Final benchmarking

| Directory | Purpose |
|-----------|---------|
| `anchorbench_external_core` | Main benchmark: 360 items, core + ablation (placebo/authority) |
| `anchorbench_history_core` | Main benchmark: 120 items, core + control_twostage |
| `anchorbench_icl_core` | Main benchmark: 360 items, core + neutral ablation |
| `anchorbench_rag_core` | Main benchmark: 360 items, core + order/disclaimer/authority ablations |
| `anchorbench_tool_core` | Main benchmark: 360 items, tool (agentic) |
| `anchorbench_tool_read_core` | Main benchmark: 360 items, tool (read-only JSON) |
| `anchorbench_*_smoke` | Smoke sets: external, history, icl, rag, tool (~6 items each). Used by `run_smoke_experiment.sh` |
| `DATASET_CARD.md` | Dataset documentation |
| `upload_hf.py` | Hugging Face upload helper |
| `hf_release/` | Release artifact (if you use it for HF) |

### REMOVE — Older / intermediate versions

| Directory | Reason |
|-----------|--------|
| `anchorbench_external_pilot12` | Small pilot (12 items); superseded by core for final numbers |
| `anchorbench_history_pilot12` | Same |
| `anchorbench_icl_pilot12` | Same |
| `anchorbench_rag_pilot12` | Same |
| `anchorbench_tool_pilot12` | Same |
| `anchorbench_tool_read_pilot12` | Same |
| `anchorbench_v1` | Old monolithic dataset; scripts use `*_core` / `*_smoke` now |
| `anchorbench_v2_external_pilot` | Old v2 pilot; superseded by *_core |
| `anchorbench_v2_history_pilot` | Same |
| `anchorbench_v2_icl_pilot` | Same |
| `anchorbench_v2_rag_pilot` | Same |
| `anchorbench_v2_tool_pilot` | Same |

---

## Results (`results/`)

### KEEP — Final experiment results

| Directory / File | Purpose |
|------------------|---------|
| `full_benchmark/` | Main benchmark: 5 suites × 10 models, core 5-condition results. Used for paper and `recompute_all_unified.py` |
| `smoke_experiment/` | Recent smoke test: 5 suites × 4 models on smoke datasets; used for quick validation |

Optional to keep (if you still need them for paper/figures):

| Item | Purpose |
|------|---------|
| `figure1_nai_vs_accuracy.png`, `figure2_dose_response.png`, `figure3_aalc_sweep.png` | Paper figures |
| `table1_nai.csv`, `table2_rr.csv`, `table3_stress.csv` | Paper tables |
| `summary.json` | Legacy summary |
| `dataset_and_experiment_summary.pdf` | Summary doc |

### REMOVE — Older / one-off runs

| Directory / File | Reason |
|------------------|--------|
| `external_v2_pilot` | Old pilot run |
| `external_v2_pilot_*` (all variants) | Old pilot runs (different models/configs) |
| `history_v2_pilot_*` | Old history pilot runs |
| `icl_v2_pilot/` | Old ICL pilot |
| `rag_v2_pilot/` | Old RAG pilot |
| `rag_v2_smoke_test/` | Old RAG smoke |
| `tool_v2_pilot/` | Old tool pilot |
| `tool_v2_smoke/`, `tool_v2_smoke_chat/` | Old tool smoke |
| `smoke_test/` | Older smoke (various models); superseded by `smoke_experiment` |
| `anchoring_eval_multi/`, `anchoring_eval_test/` | Old eval runs |
| `sanity/` | Sanity-check runs |
| `runs/` | Old generic runs |
| `external_70b/` | One-off 70B run |
| `mitigation/`, `mitigation_api_retest*`, `mitigation_smoke_fallback*` | Mitigation experiments (keep only if needed for paper) |
| `mechinterp/` | Mechanistic interp (keep only if needed) |
| `rerror_report/`, `syn_anchors/` | Other experiments |
| `xml_tag_test/` | Parsing test |
| `EXTERNAL_V2_PILOT_SUMMARY.md` | Summary for old pilot |
| `parsing_comparison.json`, `parsing_comparison_with_llm.json` | Parsing comparison (can keep if you use them) |
| `unified_all_suites.json` (loose in results/) | Duplicate; same content is in `full_benchmark/unified_all_suites.json` |

---

## Summary

- **Datasets:** Keep 6× `*_core`, 5× `*_smoke`, DATASET_CARD, upload_hf, hf_release. Remove all `*_pilot12`, `anchorbench_v1`, and `anchorbench_v2_*_pilot`.
- **Results:** Keep `full_benchmark/` and `smoke_experiment/`. Remove old pilot dirs, smoke_test, sanity, runs, and other one-off/experiment dirs unless you need them for the paper.

Run the cleanup script with `--dry-run` first to see what would be removed, then `--execute` to perform the cleanup.
