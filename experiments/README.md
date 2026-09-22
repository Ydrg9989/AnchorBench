# Experiment launchers

The shell scripts in this folder are the exact launchers that produced the
appendix experiments of the COLM 2026 paper. They are kept as **provenance**
for the numbers in `results/rebuttal/` and `results/revision/`, not as
supported entry points: they hard-code GPU indices, poll for `.done` flags,
assume the `LLM_anchoring` conda environment and expect to run inside tmux on
the machine the paper was produced on.

To evaluate a model or reproduce the main benchmark, use the `anchorbench`
CLI instead (see [docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md)). To
regenerate the appendix tables from the published results tarballs, run
`anchorbench tables --appendix`.

## Layout

```
experiments/
|-- run_stage3_reruns.sh      # D5 re-runs: tab:history_matched, tab:tool_plaintext
`-- rebuttal/                 # one launcher per appendix experiment, plus orchestration
```

Every launcher resolves the repository root from its own location, so the
folder can be run from anywhere inside the checkout.

## Which launcher produced what

Launchers named `_<x>_one_model.sh` run one model on one GPU; the `run_*.sh`
wrappers fan those out across four GPUs. Orchestration scripts
(`_chain_*.sh`, `_wait_then_chain.sh`, `_wrap_up*.sh`) sequence the chains and
run the analysis modules when every `.done` flag is present.

| Launcher | Runs | Writes to `results/rebuttal/` | Backs |
|---|---|---|---|
| `_b1_one_model.sh`, `_b1_api_model.sh`, `run_b1_cot_extended.sh`, `_chain_after_b1.sh` | `runners.rebuttal_cot`, `runners.api` | `cot_extended/` | `tab:cot_extended` |
| `_b2_one_model.sh` | `runners.rebuttal_spectrum` | `spectrum/` | `tab:plausibility_spectrum` |
| `_b3_one_model.sh` | `runners.external` on the weighted-mean split | `weighted_mean/` | `tab:weighted_mean` |
| `_c1_one_model.sh` | `runners.{external,history}` on the medical pilot | `medical/` | `tab:extension_pilot` |
| `_extension_pilot_one_model.sh`, `run_extension_pilot.sh` | `runners.{external,history}` on the other-domain pilot | `extension_pilot/` | `tab:extension_pilot` |
| `_d1_one_model.sh`, `run_p1_intensity_pathway.sh` | `runners.rebuttal_intensity` | `intensity/`, `intensity_rag/`, `intensity_history/` | `tab:intensity_pathway` |
| `_p2_one_model.sh`, `run_p2_rag_realism.sh` | `runners.rebuttal_rag_realism` | `rag_realism/` | `tab:rag_realism` |
| `_p3_one_model.sh`, `run_p3_tool_realism.sh` | `runners.rebuttal_tool_realism` | `tool_realism/` | `tab:tool_realism` |
| `_p4_one_model.sh`, `run_p4_task_spec.sh` | `runners.rebuttal_cot` (rule vs. judgment prompts) | `task_spec/` | `tab:task_spec` |
| `_p5_one_model.sh` | `runners.rebuttal_uncertain` | `uncertain/` | `tab:uncertain_k`, main-text Table 2 |
| `_large_api_full.sh`, `_large_ow_all_suites.sh`, `api_smoke_large.sh` | `runners.api`, per-suite runners with tensor parallelism | `large_api/`, `large_ow/`, `large_api_smoke/` | `tab:large_panel_results` |
| `_chain_p2_to_p5.sh`, `_wait_then_chain.sh`, `_wrap_up.sh`, `_wrap_up_p2p5.sh` | orchestration: sequence the per-model chains, poll `.done` flags, then run the analysis modules | `_chain_*` and `_wrap_up*` logs | — |
| `run_stage3_reruns.sh` | `anchorbench experiment` recipes `paper_history_matched`, `paper_tool_plaintext` | `results/history_matched/`, `results/tool_plaintext/` | D5 addendum in the ledger |

The table-to-module mapping for the analysis step is in
[docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md#appendix-tables-input-ed-by-the-paper).
