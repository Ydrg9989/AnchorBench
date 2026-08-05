# Dataset generation pipeline

How a benchmark dataset is produced. Each suite yields **ItemSpecs**
(ground-truth records) and **PromptViews** (rendered prompts), stratified by
domain, difficulty and condition.

```
1. ItemSpec generation   ->  theta, evidence, anchors, gold answer
2. Prompt rendering      ->  one prompt per condition
3. Serialization         ->  itemspecs.jsonl, promptviews*.jsonl, manifest.json
```

## Entry point

```bash
anchorbench generate data=external +size=core seed=42     # one suite
bash scripts/generate_all.sh core 42                      # all six
```

`generate_all.sh` loops the same command over
`external history icl icl_dist rag tool`.

## Stage 1 — ItemSpec generation

**Module:** `src/anchorbench/data/itemspec_gen.py`

Most suites iterate a stratified grid:

```
for domain in DOMAIN_IDS:            # 6 business domains
    for difficulty in [easy, hard]:  # 2 levels
        for offset in [15, 25, 40]:  # 3 anchor distances
            for i in range(n_per_cell):
                -> one ItemSpec
```

**History is the exception: it has no offset dimension.** Its anchor is the
model's own Stage-1 answer rather than a designer-specified value, so the grid
is domain x difficulty only and the elicitation target is fixed at
theta +/- 25. `generate.py` scales `n_per_cell` for such suites so every suite
still reaches 360 items at `--size core`; see
[RECONCILIATION.md](RECONCILIATION.md) D1 for why that matters.

Key computations:

- **theta** (latent signal): uniform on [30, 70]
- **anchor low / high**: `max(0, theta - offset)` / `min(100, theta + offset)`
- **evidence**: five ratings around theta — easy uses sigma=8 with all
  visible; hard uses sigma=15 with two hidden and one conflicting
- **y_star_evidence** (gold): `round(mean(visible ratings))`

Suite-specific additions: History searches a subset for plausible pressure and
builds same-domain warmup cases; ICL pre-generates three neutral
demonstrations into `tags`; RAG carries corpus metadata for the
three-document mini-corpus assembled at render time; Tool carries the
available tool schemas.

## Stage 2 — Prompt rendering

**Modules:** `src/anchorbench/data/suites/{external,history,icl,icl_dist,rag,tool}.py`,
dispatched through `SUITE_RENDERERS`.

Each ItemSpec yields five core PromptViews:

| Condition | Relevance | Direction |
|---|---|---|
| `control` | none | — |
| `irrelevant_low` | irrelevant | low |
| `irrelevant_high` | irrelevant | high |
| `plausible_low` | plausible | low |
| `plausible_high` | plausible | high |

The anchor enters through a suite-specific channel: a sentence before the
question (External), a prior conversational turn (History), demonstration
metadata (ICL), a retrieved document (RAG), or a tool-call response (Tool).

**The benchmark invariant:** across all five conditions of one item,
everything except the anchor-bearing component is identical. That is what
makes the irrelevant-vs-plausible contrast isolate framing alone.

Suites also emit ablation views (`promptviews_ablation.jsonl`) — placebo and
authority framings on External, `control_twostage` on History, neutral priming
on ICL, and position/disclaimer variants on RAG.

## Stage 3 — Serialization

**Module:** `src/anchorbench/data/schema.py`

| File | Contents |
|---|---|
| `itemspecs.jsonl` | one ItemSpec per line: ground truth, anchors, provenance |
| `promptviews_core.jsonl` | the five core conditions |
| `promptviews_ablation.jsonl` | suite-specific extra conditions |
| `promptviews.jsonl` | core + ablation together |
| `manifest.json` | seed, counts, `generator_version`, validation flag |

## Validation

```bash
bash scripts/validate_all.sh core
```

Eight deterministic checks in `src/anchorbench/data/validators.py`
(`validate_all`), covering the shared-prefix invariant, anchor placement, gold
correctness and duplicate detection.

## Deterministic regeneration

Generation is fully deterministic given the seed. Each suite offsets the
master seed so the suites stay independent:

| Suite | Seed offset |
|---|---|
| external | +0 |
| history | +0 |
| rag | +2000 |
| tool | +3000 |
| icl | +4000 |
| icl_dist | +4500 |

`tests/test_dataset_regeneration.py` asserts every committed suite still
regenerates byte-for-byte, so this is a checked property rather than a claim.

`itemspecs.jsonl` also carries `generator_version`, stamped from the current
git HEAD, so its *file hash* changes on every commit while its data does not.
Compare promptviews, or compare itemspecs ignoring that field.

## Domain configuration

Templates, evidence labels and anchor preambles live in
`src/anchorbench/data/domains.py`. The published benchmark uses six business
domains:

1. `pricing_wtp` — willingness to pay
2. `operations_time` — operational efficiency
3. `transportation_logistics` — logistics reliability
4. `resource_consumption` — resource efficiency
5. `market_demographics` — market adoption
6. `legal_policy` — regulatory compliance

`domains.py` additionally defines medical and other (legal-contract, consumer)
domains used by the extension pilot in the appendix. `DOMAINS` proper stays at
six entries so the published benchmark stays reproducible; the wider set is
reachable through `ALL_DOMAIN_IDS`.
