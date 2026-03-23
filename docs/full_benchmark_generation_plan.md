# AnchorBench: Full Benchmark Generation Plan

## Canonical Workflow

```
Step 1: Generate  →  bash scripts/generate_all.sh SIZE SEED
Step 2: Validate  →  bash scripts/validate_all.sh SIZE
Step 3: Freeze    →  bash scripts/freeze_dataset.sh SIZE
Step 4: Evaluate  →  bash scripts/run_{suite}_gpus.sh
```

## Size Presets

| Size  | Items per cell | Domains | Total items/suite | Total views/suite | Purpose           |
|-------|---------------|---------|-------------------|-------------------|-------------------|
| smoke | 1             | 1       | 6                 | 30                | Quick CI check    |
| pilot | 5             | 6       | 180               | 900               | Development runs  |
| core  | 10            | 6       | 360               | 1800              | Full experiments  |

For the paper submission, use **pilot** for all 5 suites.

## Per-Suite Counts (Pilot)

| Suite    | Domains | Difficulties | Offsets | n_per_cell | Items | Views |
|----------|---------|-------------|---------|------------|-------|-------|
| External | 6       | 2           | 3       | 5          | 180   | 900   |
| ICL      | 6       | 2           | 3       | 5          | 180   | 900   |
| RAG      | 6       | 2           | 3       | 5          | 180   | 900   |
| Tool     | 6       | 2           | 3       | 5          | 180   | 900   |
| History  | 6       | 2           | —       | 5          | 60    | 300   |
| **Total**|         |             |         |            | **780**| **3900**|

History uses 6×2×5 = 60 (no offset stratification; uses subset search instead).

## Seed Policy

Master seed: **42** (default)

Each suite adds a fixed offset to avoid RNG correlation:
- External: `seed + 0`
- History: `seed + 0` (different grid)
- RAG: `seed + 2000`
- Tool: `seed + 3000`
- ICL: `seed + 4000`

## Pre-Run Validation Checklist

After generation, the validator checks:

1. **Schema completeness**: All required fields present and typed correctly
2. **Paired-condition completeness**: Every item has exactly 5 conditions
3. **No duplicates**: Unique `item_id` in specs, unique `(item_id, condition)` in views
4. **Domain/difficulty balance**: Equal counts across domains and difficulties
5. **Gold-answer validity**: `y_star_evidence == round(mean(visible evidence))`
6. **Answer format instruction**: Present in every prompt
7. **Manifest consistency**: Counts match actual files
8. **Anchor placement**: Anchor strings present in anchored prompts
9. **Leakage check**: No theta/y_star leaked in prompt text
10. **Difficulty structure**: Easy items have no missing evidence; hard items have 2

## Commands

### Generate all suites (pilot, seed 42)

```bash
bash scripts/generate_all.sh pilot 42
```

### Validate

```bash
bash scripts/validate_all.sh pilot
```

### Freeze for submission

```bash
bash scripts/freeze_dataset.sh pilot
```

### Regeneration verification

To verify deterministic regeneration:

```bash
# Generate twice with same seed
bash scripts/generate_all.sh pilot 42
cp -r datasets/anchorbench_external_pilot datasets/anchorbench_external_pilot_copy

bash scripts/generate_all.sh pilot 42
diff datasets/anchorbench_external_pilot/itemspecs.jsonl \
     datasets/anchorbench_external_pilot_copy/itemspecs.jsonl
# Should report no differences (except manifest timestamp)
```
