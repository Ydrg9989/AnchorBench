# Reproducibility Guide

## Requirements

- Python 3.10+
- CUDA-capable GPU (for model evaluation only; generation is CPU-only)
- Dependencies: `pip install -r requirements.txt`

## Reproducing the Benchmark Dataset

### Quick Start

```bash
# 1. Generate all 5 suites (pilot size, seed 42)
bash scripts/generate_all.sh pilot 42

# 2. Validate all generated datasets
bash scripts/validate_all.sh pilot

# 3. Freeze dataset for release
bash scripts/freeze_dataset.sh pilot
```

### Step-by-Step

#### Generate a single suite

```bash
PYTHONPATH=src python -m anchorbench_v1.generate \
    --suites external --size pilot --seed 42 \
    --out_dir datasets/anchorbench_external_pilot/
```

Available `--suites`: `external`, `history`, `icl`, `rag`, `tool`

Available `--size`: `smoke` (6 items), `pilot` (180 items), `core` (360 items)

#### Validate

```bash
PYTHONPATH=src python -m anchorbench_v1.validate \
    --data_dir datasets/anchorbench_external_pilot/
```

### Deterministic Regeneration

All generation is deterministic given the same `--seed`. The only non-deterministic field in the output is `manifest.json:timestamp`. To verify:

```bash
# Compare two runs with the same seed
diff <(grep -v '"timestamp"' datasets/run1/manifest.json) \
     <(grep -v '"timestamp"' datasets/run2/manifest.json)

diff datasets/run1/itemspecs.jsonl datasets/run2/itemspecs.jsonl
diff datasets/run1/promptviews.jsonl datasets/run2/promptviews.jsonl
```

## Reproducing Paper Results

### Run a single model on one suite

```bash
PYTHONPATH=src python scripts/eval/run_icl.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --promptviews datasets/anchorbench_icl_pilot/promptviews.jsonl \
    --itemspecs datasets/anchorbench_icl_pilot/itemspecs.jsonl \
    --out_dir results/icl_pilot \
    --batch_size 32 --max_new_tokens 512
```

### Run all models on all suites

```bash
bash scripts/run_icl_gpus.sh
bash scripts/run_tool_gpus.sh
bash scripts/run_rag_gpus.sh
bash scripts/run_external_8B_7B.sh
```

### Recompute unified metrics

```bash
PYTHONPATH=src:scripts/eval python scripts/eval/recompute_all_unified.py
```

## Decoding Protocol

All evaluations use:
- `temperature = 0.0` (greedy decoding)
- `max_tokens = 512`
- No system prompt by default
- Answer format: integer 0–100 on the last line

## Parsing

Responses are parsed with a multi-stage regex parser (`parse_answer_int` in `src/mitigation_eval/runner.py`):
1. Look for a standalone integer on the last non-empty line
2. Fall back to the first integer in the response
3. Records with no parseable integer are marked `parsed_ok=False`

## Key Outputs

```
results/{suite}_pilot/{model_slug}/
    results.jsonl      — per-prompt raw outputs and parsed answers
    summary.json       — aggregated UAI, TAR, Disc_delta, parse rate
```

## Checksums

After freezing, checksums are stored in `datasets/anchorbench_pilot_checksums.sha256`. Verify with:

```bash
sha256sum -c datasets/anchorbench_pilot_checksums.sha256
```
