# Runner — Anchoring Benchmark

Run the anchoring dataset against OpenRouter API models and local HF models, then compute metrics.

## Setup

```bash
pip install requests pyyaml numpy torch transformers

export OPENROUTER_API_KEY="your-key-here"
```

## Configuration

Edit `runner/config.models.yaml`:

```yaml
run_name: v02
dataset_path: poc_dataset/poc_v0.2.jsonl
output_dir: runner_outputs
n_samples_per_item: 5
decoding:
  temperature: 0.7
  top_p: 1.0
  max_tokens: 8
system_prompt: "You are a helpful assistant. Follow the instructions exactly."
openrouter_models:
  - openai/gpt-4o-mini
  - anthropic/claude-3.5-haiku
hf_models:
  - model_id: meta-llama/Llama-3.1-8B-Instruct
    dtype: bfloat16
    device_map: auto
```

## Running experiments

```bash
# OpenRouter API models (new suites only — skip already-run external/self_generated)
python runner/run_openrouter.py \
    --config runner/config.models.yaml \
    --run_name api_v02 \
    --suites icl,conversation_history,rag,tool

# Local HF models
python runner/run_hf.py \
    --config runner/config.models.yaml \
    --run_name hf_v02 \
    --suites icl,conversation_history,rag,tool

# All suites (including external/self_generated)
python runner/run_openrouter.py --config runner/config.models.yaml --run_name full_v02

# Dry run (no API calls / model loading)
python runner/run_openrouter.py --config runner/config.models.yaml --dry_run
python runner/run_hf.py --config runner/config.models.yaml --dry_run

# Quick test (2 items per suite)
python runner/run_openrouter.py --config runner/config.models.yaml \
    --run_name test --suites icl,rag --max_items 2
```

Both runners support **resume**: re-run the same command and completed records are skipped.

## Evaluation

```bash
# Single input file
python runner/eval_metrics.py \
    --dataset poc_dataset/poc_v0.2.jsonl \
    --inputs runner_outputs/api_v02.jsonl \
    --run_name v02

# Multiple input files (mixed backends)
python runner/eval_metrics.py \
    --dataset poc_dataset/poc_v0.2.jsonl \
    --inputs runner_outputs/api_v02.jsonl runner_outputs/hf_v02.jsonl \
    --run_name v02

# Include v0 results alongside v02
python runner/eval_metrics.py \
    --dataset poc_dataset/poc_v0.2.jsonl \
    --inputs runner_outputs/api_v02.jsonl runner_outputs/poc_v0_openrouter.jsonl \
    --run_name combined
```

Outputs: `runner_outputs/{run_name}_metrics.json` and `runner_outputs/{run_name}_metrics.csv`.

## File overview

| File | Purpose |
|---|---|
| `config.models.yaml` | Unified config for both backends |
| `utils.py` | Shared parsing, dataset I/O, retry, result formatting |
| `run_openrouter.py` | Run via OpenRouter API |
| `run_hf.py` | Run via local HF Transformers |
| `eval_metrics.py` | Compute all metrics (ΔEV, NAI, TVD, W1, Pull, Inertia, β) |

### Legacy files (v0 pipeline)

| File | Purpose |
|---|---|
| `config.example.yaml` / `config.yaml` | Old v0 config format |
| `run_open_models.py` | Old HF runner (v0, external/self_generated only) |
| `eval_poc.py` | Old eval (v0, external/self_generated metrics only) |

## Results JSONL schema

Each line:

```json
{"run_name":"...","model_id":"...","backend":"hf|openrouter","item_id":"...","suite":"...","subtype":"...","field":"...","domain":"...","condition":"...","sample_idx":0,"answer_int":42,"parsed_ok":true,"raw_text":"42"}
```

## Metrics

### External-like suites (external, icl, conversation_history, rag, tool)

- **ΔEV**: `EV_high − EV_low` (mean answer shift)
- **NAI**: `ΔEV / 60` (Normalized Anchoring Index)
- **TVD**: Total Variation Distance between high/low answer distributions
- **W1**: Wasserstein-1 distance between high/low distributions
- Bootstrap 95% CI over items (2000 resamples)

### Self-generated suite

- **delta_err**: `|EV_hist − y*₂| − |EV_fresh − y*₂|` (gold-based error shift)
- **Pull**: `(H−F)·(A1−F)` per-sample; PullRate = P(Pull > 0)
- **Inertia**: `|H−A1| − |F−A1|` per-sample; InertiaRate = P(Inertia < 0)

### Controlled-history

- OLS slope **β** from `EV_a = α + β·a` across anchors {10,30,50,70,90}
- **β_norm**: `β · 60/80` (normalized to [20,80] range)
