# Addendum: re-runs of the two unpreserved experiments

**These numbers do not replace anything in the paper.** They sit beside the
published values, which remain as printed.

`tab:tool_plaintext` and `tab:history_matched` were the only two appendix
tables that could not be regenerated: the runs behind them were never
preserved (D5 in [../../docs/RECONCILIATION.md](../../docs/RECONCILIATION.md)).
Both experiments were re-run on 2026-08-05 to restore reproducibility.

## What was run

| | |
|---|---|
| Recipes | `conf/experiment/paper_history_matched.yaml`, `conf/experiment/paper_tool_plaintext.yaml` |
| Command | `bash scripts/run_stage3_reruns.sh` |
| Hardware | 2 x H100 NVL, OLMo-32B at tensor-parallel 2 |
| Models | 10 open-weight (History), 5 (Tool plaintext) |
| Parse rates | 0.92-1.00, all cells |
| Git | commit at time of run, branch `freeze/camera-ready-2026-08` |

## Why the numbers differ from the published table

Three reasons, in decreasing order of importance:

1. **The original inputs were never preserved.** That is the defect being
   repaired. The published values came from runs whose configuration is not
   recorded anywhere, so this is an *independent* re-run under a documented
   configuration, not a reproduction attempt of a known recipe.
2. **Three bugs stood between the recipes and a working run**, all fixed
   before these numbers were produced (see the ledger): the data-group
   override was discarded by the experiment CLI, Gemma was misrouted to the
   hosted API by an hf_id prefix match, and the History condition set omitted
   the plain `control` baseline the comparison needs.
3. **vLLM is not bitwise reproducible at temperature 0**, which accounts for
   small residual differences but not the large ones.

## `tab:tool_plaintext`

Discrimination is close throughout; the control-condition MAE moves
substantially for two models.

| Model | Disc_struct | Disc_plain | MAE_struct | MAE_plain |
|---|---|---|---|---|
| Qwen-1.5B | 0.58 → **0.57** | 0.07 → **0.05** | 11.34 → **26.17** | 15.01 → **14.92** |
| Qwen-3B | 0.08 → **0.10** | 0.11 → **0.08** | 2.19 → **4.62** | 15.74 → **16.11** |
| Qwen-7B | 0.17 → **0.17** | 0.02 → **0.02** | 0.71 → **0.67** | 1.32 → **1.32** |
| Llama-3B | 0.23 → **0.19** | 0.10 → **0.15** | 2.28 → **2.33** | 9.49 → **10.83** |
| Llama-8B | 0.16 → **0.17** | 0.14 → **0.18** | 5.47 → **12.15** | 6.77 → **6.85** |

The table's claim — that plaintext rendering lowers discrimination relative to
structured tool messages — holds: Disc_plain < Disc_struct for 3 of 5 models
in the re-run, against 4 of 5 as published, and Qwen-7B reproduces exactly.

## `tab:history_matched`

Per-cell magnitudes differ considerably; the qualitative claim survives.

| Model | MAE_std | MAE_ts | Disc_std | Disc_match |
|---|---|---|---|---|
| Qwen-1.5B | 5.49 → **19.05** | 24.87 → **21.87** | 0.89 → **0.67** | 0.27 → **0.50** |
| Qwen-3B | 3.65 → **9.56** | 17.67 → **13.81** | 0.18 → **0.57** | 0.51 → **0.06** |
| Qwen-7B | 3.77 → **7.12** | 10.33 → **7.69** | 0.18 → **0.43** | 0.97 → **0.50** |
| Llama-1B | 15.71 → **15.96** | 20.87 → **16.04** | 0.01 → **-0.20** | 0.39 → **-0.15** |
| Llama-3B | 9.46 → **10.10** | 15.42 → **13.17** | 0.22 → **0.50** | 0.20 → **0.36** |
| Llama-8B | 7.02 → **6.59** | 8.05 → **8.47** | 0.08 → **0.13** | 0.33 → **0.28** |
| Gemma-1B | 14.71 → **15.81** | 27.45 → **26.14** | 0.21 → **0.31** | 0.09 → **0.66** |
| Gemma-4B | 12.03 → **15.25** | 17.32 → **14.91** | 0.24 → **0.27** | 0.57 → **0.46** |
| OLMo-13B | 5.40 → **4.30** | 9.86 → **10.23** | 0.29 → **0.36** | 0.82 → **0.54** |
| OLMo-32B | 4.13 → **12.51** | 8.51 → **8.52** | 0.12 → **1.06** | 0.69 → **0.67** |

**The main-text claim holds.** `findings.tex` states that matched-format
analysis "reduces Disc_Delta by up to 70% for some models". In the published
table the matched baseline lowers Disc for 3 of 10 models, with a largest
reduction of 70%. In the re-run it lowers Disc for 4 of 10, with a largest
reduction of 89%. The direction and the "for some models" framing are
unchanged; only the magnitudes move.

**What does not carry over** is any per-cell reading. OLMo-32B's Disc_std in
particular moves from 0.12 to 1.06, and Qwen-1.5B's MAE_std from 5.49 to
19.05. Anyone citing an individual number from this table should cite the
published value and note that an independent re-run did not reproduce it.

## Recommendation

Ship the paper unchanged and reference this addendum from the artifact
release. The published tables stay as printed; these serve as the
reproducibility record for two experiments that previously had none.
