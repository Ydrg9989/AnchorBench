# AnchorBench v1 — Scaled-Up Anchoring Benchmark

Final benchmark dataset for measuring numeric anchoring bias in LLMs.

## Structure

```
benchmark/
  schema.json          # JSON schema (extends poc_dataset/schema.json)
  generate.py          # dataset generation entry-point
  validate.py          # schema + statistical validation
  templates/           # prompt templates by domain / subtype
  data/                # generated splits (gitignored; see manifest)
```

## Design targets

| Property         | PoC (v0.2) | AnchorBench v1 (target) |
|------------------|------------|-------------------------|
| Items            | 109        | ≥ 500                   |
| Domains          | 3          | ≥ 6                     |
| Anchor subtypes  | 2          | ≥ 4                     |
| Languages        | en         | en (+ multilingual TBD) |

## Generating the dataset

```bash
python benchmark/generate.py \
  --config  benchmark/config.yaml \
  --out     benchmark/data/ \
  --seed    42
python benchmark/validate.py benchmark/data/anchorbench_v1.jsonl
```
