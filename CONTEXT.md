# CONTEXT.md: the paper's terms as the code spells them

One line per term. Paper section refers to `sections/benchmark.tex` of the COLM 2026 camera-ready unless noted. When code and paper disagree on a name, the paper's word wins and the code name is listed after it.

## Objects

- **item**: one scenario with structured numeric evidence and a fixed gold answer; 360 per core suite (6 domains x 2 difficulties x 3 offsets x 10). Code: `ItemSpec` (`data/schema.py`), one line of `itemspecs.jsonl`.
- **prompt view**: one item rendered under one condition; the text a model actually saw. Code: `PromptView`, one line of `promptviews*.jsonl`. `promptviews_core.jsonl` holds the five core conditions.
- **record**: one model answer to one prompt view. Code: a dict in `results.jsonl` with `answer_int`, `parsed_ok`, `parse_strategy`, `anchor_value`, `y_star_evidence`, `condition`.
- **cell**: one (model, suite) pair; the unit of every table row and of the bootstrap. 70 cells = 14 models x 5 suites.
- **domain**: one of six business scenario families (`data/domains.py`, `DOMAINS`); the medical and other pilot domains are appendix-only.
- **difficulty**: `easy` (five ratings visible, sigma 8) or `hard` (two ratings hidden, sigma 15). `medium` exists in code and not in the paper.

## Suites (Sec. 3, "pathways")

- **pathway** / **suite**: how the anchor reaches the model. Five: **External** (a sentence in the prompt), **History** (the model's own Stage-1 answer), **ICL** (demonstration metadata; `icl`), **RAG** (one document of three), **Tool** (a tool-call response). Code: `suite` field; `conf/data/<suite>.yaml`; renderer in `data/suites/<suite>.py`; runner in `runners/<suite>.py`.
- **ICL-dist**: the diagnostic ICL variant whose demo answers carry the anchor distribution. Code: `icl_dist`, `suite = "icl"`, `variant = "dist"`.
- **unified key**: the capitalised spelling used in `unified_all_suites.json` (`External`, `History`, `Icl`, `Rag`, `Tool`). Code: `registry.Suite.unified_key`.

## Conditions (Sec. 3.2)

- **condition**: which anchor an item carries. Core five: `control`, `irrelevant_low`, `irrelevant_high`, `plausible_low`, `plausible_high`. Code: `CONDITIONS` (currently defined in five places; `data/suites/_shared.py` is the intended home).
- **relevance** (`anchor_relevance`): `irrelevant` (the number is a case ID or similar; any movement is bias), `plausible` (the number is framed as an estimate; some movement is rational), `none` for control. Paper subscripts: irr, pls.
- **direction**: `low` or `high`, whether the anchor sits below or above theta. Pooled within relevance for every reported UAI.
- **control_twostage**: the matched-format History baseline (Stage 1 asks a qualitative question). Code: `baseline_condition` argument of `compute_unified_metrics`.
- **extended conditions** (appendix only): `placebo_*`, `authority_*`, `neutral_*` (`EXTENDED_CONDITIONS`).

## Numbers on an item (Sec. 3.1)

- **theta**: the latent centre, an integer drawn uniformly from 30 to 70. Code: `ItemSpec.theta`, `THETA_MIN`, `THETA_MAX`.
- **evidence**: five ratings drawn from N(theta, sigma), rounded, clipped to 0 to 100. Code: `evidence_structured`, `EASY_SIGMA`, `HARD_SIGMA`.
- **gold answer** (y*): the rounded mean of the visible ratings. Code: `y_star_evidence` is the one field every metric uses; `y_star`, `y_star_theta` and `y_star_components` are legacy mirrors kept because the committed itemspecs are compared field by field. Function: `compute_gold_answer` (`data/itemspec_gen.py`).
- **anchor** (a): the inserted number. `a_low = max(0, theta - delta)`, `a_high = min(100, theta + delta)`. Code: `ItemSpec.anchors = {"low", "high", "offset"}`, `PromptView.anchor_value`. For History, a is the model's own Stage-1 answer.
- **offset** (delta): the anchor's distance from theta, one of 15, 25, 40. Code: `ANCHOR_OFFSETS`, `anchors["offset"]`, encoded in `item_id` as `-off15-`.
- **anchor sentence** / **preamble**: the one sentence that differs across conditions. Code: `resolve_anchor_preamble` in `_shared.py`, `prompt_components["anchor_sentence"]`.

## Metrics (Sec. 3.4, `eval/metrics.py`)

- **y_ctrl**: the model's control answer for the item; **y_anchor**: its answer under an anchored condition. Code: `answer_int` of the respective records.
- **UAI**, Unified Anchor Influence: `(y_anchor - y_ctrl) / (a - y_ctrl)`, the fraction of the control-to-anchor gap closed; 0 is no shift, 1 is full capitulation. Code: `uai_irr`, `uai_plaus`, plus `_low`/`_high` splits.
- **epsilon**: items with `|a - y_ctrl| < 3` are excluded from UAI (unstable denominator). Code: `EPSILON = 3.0` in `eval/metrics.py`; the only correct value. Anything using `1e-6` is a divergence (see the review).
- **TAR**, Toward-Anchor Rate: share of items whose answer moved toward the anchor, `1[(y_anchor - y_ctrl)(a - y_ctrl) > 0]`; covers excluded items too. Code: `tar_irr`, `tar_plaus`.
- **Disc_delta**, relevance discrimination: `UAI_pls - UAI_irr`. Code: `disc_delta`.
- **MAE_c**, **Acc_10**: control-only task accuracy; Acc_10 is the share within 10 points of gold. Code: `mae_control`, `acc10_control`.
- **parse rate**: parsed records over all records; unparsed records are excluded from every metric, never defaulted. Code: `parse_rate`, `parsed_ok`.
- **ACR**, **RR**: History-only revision metrics from Stage 1 to Stage 2. Code: `acr_mean`, `rr_mean`.
- **rational ceiling**: treating the anchor as one extra rating of weight w among n = 5 gives `UAI_max = w / (n + w)`, 0.167 at w = 1; for irrelevant anchors w = 0. Code: `rational_uai_max` (`analysis/bayesian_bound.py`).
- **bootstrap CI**, **Wilcoxon**, **BH**: per-cell bootstrap over items, paired signed-rank test across cells (two-sided in code), Benjamini-Hochberg over suites. Code: `bootstrap_ci`, `paired_wilcoxon`, `bh_correction`.

## Pipeline nouns

- **generate**: seed to itemspecs to prompt views (`anchorbench generate`, `data/generate.py`). Committed prompt views are ground truth; regeneration is a check.
- **runner**: an argv wrapper that loads prompt views, calls a **backend** (`HFBackend`, `VLLMBackend`, `OpenRouterBackend` behind the `Backend` protocol) through one of two **loops** (`run_single_stage`, `run_history_two_stage`) and writes `results.jsonl` + `summary.json`.
- **unified summary**: `unified_all_suites.json`, one row per run x suite x model, the sole input of Table 1, the figures and the claim verifier (`analysis/unified.py`).
- **claim**: one paper number pinned in `paper/verify.py`; a **known divergence** is a claim that disagrees by a recorded amount, with a row in `docs/RECONCILIATION.md` (the **ledger**).
- **tier**: a GPU grouping of models for launching (`conf/tier/`), not a paper concept. **Open-weight** and **API** are the two model panels the paper reports separately.
