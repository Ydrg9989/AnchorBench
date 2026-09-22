# Data guide

How a benchmark dataset is generated, what the records contain, how the
History suite's two-stage design works, and how the public Hugging Face
release is built. The user-facing description of the released files is
`datasets/DATASET_CARD.md`; the mapping from dataset version to paper tables
is `datasets/VERSIONS.md`.

## Pipeline

```
1. ItemSpec generation   ->  theta, evidence, anchors, gold answer      (data/itemspec_gen.py)
2. Prompt rendering      ->  one PromptView per condition               (data/suites/<suite>.py)
3. Serialization         ->  itemspecs.jsonl, promptviews*.jsonl, manifest.json   (data/generate.py)
4. Validation            ->  eight deterministic checks                 (data/validators.py)
```

```bash
anchorbench generate data=external +size=core seed=42     # one suite
bash scripts/generate_all.sh core 42                      # all six suites
bash scripts/validate_all.sh core
```

### Stage 1: ItemSpec generation

Every suite except History iterates a stratified grid:

```
for domain in DOMAIN_IDS:            # 6 business domains
    for difficulty in [easy, hard]:  # 2 levels
        for offset in [15, 25, 40]:  # 3 anchor distances
            for i in range(n_per_cell):
                -> one ItemSpec
```

| Size | `n_per_cell` | Items per suite |
|---|---|---|
| `smoke` | 1 | 36 |
| `pilot` | 5 | 180 |
| `core` | 10 | 360 |

History has no offset dimension: its anchor is the model's own Stage-1 answer
rather than a designer-chosen value, so its grid is domain x difficulty and
`generate.py` multiplies `n_per_cell` by three for it. Every suite therefore
reaches 360 items at `core` from the same command (RECONCILIATION D1).

Per item:

- **theta**, the latent signal, is uniform on [30, 70].
- **anchors**: `low = max(0, theta - offset)`, `high = min(100, theta + offset)`.
- **evidence**: five ratings around theta. Easy items use sigma 8 with all
  five visible; hard items use sigma 15, hide two and add one conflicting
  value.
- **gold**: `y_star_evidence = round(mean(visible ratings))`; `y_star` equals
  it. The core benchmark uses the arithmetic mean; `weighted_mean` and
  `median` scoring functions exist for the extension splits.

Suite-specific additions: History searches for the partial-evidence subset
that puts the Stage-1 estimate on the intended side of gold and builds a
same-domain warm-up case; ICL stores three neutral demonstrations in `tags`;
RAG carries corpus metadata for the three-document mini-corpus assembled at
render time; Tool carries the available tool schemas.

### Stage 2: Rendering

`SUITE_RENDERERS` in `data/suites/__init__.py` maps a suite name to its
renderer. Each ItemSpec yields the five core PromptViews:

| Condition | Relevance | Direction |
|---|---|---|
| `control` | none | — |
| `irrelevant_low`, `irrelevant_high` | irrelevant | low, high |
| `plausible_low`, `plausible_high` | plausible | low, high |

The anchor enters through the suite's channel: a sentence before the question
(External), a prior conversational turn (History), demonstration metadata
(ICL) or demonstration answers (ICL-dist), one retrieved document of three
(RAG), or a tool-call response (Tool). **Across the five conditions of one
item everything except the anchor-bearing component is byte-identical**, and
the irrelevant and plausible framings carry the same number; that is what
lets the contrast isolate framing.

Some suites also emit ablation conditions into `promptviews_ablation.jsonl`:

| Suite | Ablation conditions |
|---|---|
| External | `placebo_low/high`, `authority_low/high` |
| History | `control_twostage` |
| ICL | `neutral_low/high` |
| RAG | `*_order_first`, `*_order_last`, `irrelevant_*_nodiscl`, `plausible_*_authority` |
| ICL-dist, Tool | none |

### Stage 3: Files

| File | Contents |
|---|---|
| `itemspecs.jsonl` | one ItemSpec per line |
| `promptviews_core.jsonl` | the five core conditions (what the runners read by default) |
| `promptviews_ablation.jsonl` | the ablation conditions, where the suite has any |
| `promptviews.jsonl` | core and ablation together |
| `manifest.json` | package `version`, `suite`, `generator_version` (git hash), `seed`, `size`, `scoring_function`, `llm_enhanced`, `timestamp`, per-condition counts, file paths |

### Determinism

Generation is deterministic for a given seed. Each suite adds a fixed offset
to the master seed so the suites stay independent:

| Suite | Seed offset |
|---|---|
| external, history | 0 |
| rag | 2000 |
| tool | 3000 |
| icl | 4000 |
| icl_dist | 4500 |

`tests/test_dataset_regeneration.py` regenerates every committed suite and
compares `promptviews*.jsonl` by hash and `itemspecs.jsonl` field by field.
Two provenance fields are excluded from that comparison on purpose:
`generator_version` stamps the git HEAD at generation time, so it changes on
every commit (RECONCILIATION D2), and `manifest.json` carries a timestamp.
`render_version` (`"2.1.0"` in `itemspec_gen.py`) is different: it is part of
the data and must not be bumped with the package version.

### Validation

`validators.validate_all` runs eight checks: itemspec fields and gold
correctness, suite-specific extras, promptview fields and relevance labels,
pairing (every condition present and the shared prompt prefix identical),
duplicate detection, domain and difficulty balance, template diversity, and
manifest consistency. `anchorbench generate` runs them after writing; run
them alone with `python -m anchorbench.data.validate --data_dir <dir>`.

### Domains

Templates, evidence labels and anchor preambles live in `data/domains.py`.
The published benchmark uses six business domains: `pricing_wtp`,
`operations_time`, `transportation_logistics`, `resource_consumption`,
`market_demographics`, `legal_policy`. Medical and other domains used by the
appendix extension pilot are defined alongside them and reachable through
`ALL_DOMAIN_IDS`; `DOMAIN_IDS` stays at six so the published data regenerates
unchanged.

## Data objects

All three are dataclasses in `data/schema.py` with `to_dict` / `from_dict`
and JSONL helpers.

### ItemSpec

| Field | Type | Meaning |
|---|---|---|
| `item_id` | str | e.g. `EXT-pricing_wtp-e-off15-001` (suite, domain, difficulty, offset, index) |
| `suite`, `domain`, `template_family` | str | provenance |
| `answer_space` | dict | always `{"type": "int", "min": 0, "max": 100}` |
| `theta` | int | latent signal in [30, 70] |
| `y_star`, `y_star_evidence`, `y_star_theta` | int | gold (`y_star == y_star_evidence`), gold from evidence, gold from theta |
| `y_star_components` | dict | how the gold was derived |
| `difficulty` | str | `easy` or `hard` |
| `anchors` | dict | `low`, `high`, `gap`, `offset`, `anchor_type`, `relevance` |
| `evidence_structured` | list[dict] | five entries: `value`, `value_raw`, `label`, `index`, `missing` |
| `tags` | dict | `split`, `difficulty`, ICL demonstrations |
| `rag`, `tool`, `history` | dict or null | suite-specific payload |
| `scenario_text` | str or null | LLM-written scenario when `llm_enhance` was used |
| `render_version`, `generator_version`, `seed` | str, str, int | provenance |

### PromptView

| Field | Type | Meaning |
|---|---|---|
| `item_id`, `suite`, `domain` | str | join keys to the ItemSpec |
| `condition` | str | one of the core or ablation conditions |
| `prompt_text` | str | the full prompt sent to the model |
| `prompt_components` | dict | `scenario`, `evidence`, `question`, and for History `stage1_user_message`, `stage2_user_message` |
| `anchor_string`, `anchor_span` | str, [start, end] or null | the anchor as it appears, and where |
| `anchor_relevance` | str | `none`, `irrelevant`, `plausible` |
| `anchor_value` | int or null | the numeric anchor; null for control and for History, whose anchor is the model's Stage-1 answer |
| `prompt_hash` | str | `sha256:` prefix for de-duplication |
| `provenance` | dict | renderer metadata (tool name for Tool, document roles for RAG) |

### RAGDoc

One document of the frozen RAG corpus (`datasets/anchorbench_rag_core/anchorbench_corpus.jsonl`):
`doc_id`, `domain`, `doc_type` and `role` (`core`, `filler`, `anchor_slot`),
`text`, `relevance`, `anchor_value`.

## The History suite

History measures self-generated anchoring: the anchor is the model's own
earlier estimate, not a number the benchmark inserted.

1. **Stage 1.** The model sees partial evidence and gives an estimate.
2. **Stage 2.** The model sees the full evidence with its Stage-1 turn in the
   conversation and answers again. The Stage-2 answer is scored against the
   target item's `y_star_evidence`.

| Condition | Stage 1 | Stage 2 |
|---|---|---|
| `control` | — | full evidence, single turn |
| `plausible_low/high` | a 2-of-5 subset of the *same* case, chosen so the estimate lands below / above gold | same case, full evidence, "full evidence now available" |
| `irrelevant_low/high` | a warm-up case from the same domain | the target case, "previous answer was for a different case" |
| `control_twostage` (ablation) | a neutral first turn | full evidence |

Because the anchor is realised at run time, `anchor_value` is null in the
dataset and is written into each results record by the runner
(`runners/history.py`, which also supports `--baseline_condition
control_twostage` for the matched-format analysis, Table 13). The single-turn
control against two-turn anchored conditions is a known format asymmetry;
the paper quantifies it in the appendix.

## Extension datasets

These directories back appendix experiments and main-text Table 2. They are
committed because they are the prompts the models saw.

| Directory | Built by | Backs |
|---|---|---|
| `anchorbench_external_uncertain/` | `runners.rebuttal_uncertain.build_uncertain_promptviews`: each External item re-rendered with k = 1, 2, 3 visible ratings, gold unchanged (15 conditions, 5,400 rows) | Table 2, `tab:uncertain_k` |
| `anchorbench_external_weighted_mean_core/` | `generate` with `--scoring_function weighted_mean` | `tab:weighted_mean` |
| `anchorbench_{external,history}_medical_pilot/`, `..._other_pilot/` | `generate` with the extension domains | `tab:extension_pilot` |
| `anchorbench_{external,history,rag}_d1/` | `runners.rebuttal_intensity` (mild / standard / strong credibility) | `tab:intensity_pathway` |
| `anchorbench_rag_p2/`, `anchorbench_tool_p3/` | `data.suites.{rag,tool}.build_realism_promptviews` | `tab:rag_realism`, `tab:tool_realism` |

## Public release on the Hugging Face Hub

The Hub release is one self-contained row per prompt, with the gold answer,
anchor value and the metadata needed for the paper's breakdowns (`offset`,
`difficulty`, `domain`), so a user can compute UAI without any internal file.

```bash
python scripts/export_public_promptviews.py            # -> datasets/hf_release/<suite>.jsonl
python scripts/export_public_promptviews.py --check    # exit 1 if the export is stale
python datasets/upload_hf.py --dry-run                 # list what would be uploaded
python datasets/upload_hf.py                           # upload to Yiderigun/AnchorBench
```

Six files are released: the five core suites and `external_uncertain`
(14,400 rows). `DATASET_CARD.md` becomes the Hub README, `VERSIONS.md` is
uploaded beside it, and `datasets/hf_release/CHECKSUMS.sha256` pins the
bytes. `datasets/hf_release/` is committed so a clean clone can verify the
export against the source datasets.
