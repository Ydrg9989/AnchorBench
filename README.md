# LLM Anchoring

Investigating anchoring bias in large language models through controlled numeric-formatting experiments.

## Repository Layout

```
configs/
  data_gen.yaml                    # dataset generation config
  eval.yaml                        # evaluation config (grid, decoding, W&B)
data/interim/icl_anchor_v0/        # template specs, generated templates, LLM cache
data/processed/icl_anchor_v0/      # final dataset, splits, manifest
src/llm_anchoring/
  openrouter_client.py             # OpenRouter API with retry + disk cache
  data_gen/                        # template generation & rendering
  eval/                            # evaluation module
    inference.py                   # HF model loading + batch generation
    parse_utils.py                 # output parsing, format check, base_id
    parsers.py                     # strategies A/B/C for numeric extraction
    metrics.py                     # condition summaries, IAS, bootstrap CI
    wandb_logger.py                # W&B integration
scripts/data_gen/                  # data generation CLIs
scripts/eval/                      # evaluation CLIs
  run_inference.py                 # single-run inference
  compute_metrics.py               # metrics + W&B logging
  sweep.py                         # YAML-driven grid driver
  parse_ablation.py                # parsing-strategy ablation study
```

## Quickstart – Dataset Generation

### Prerequisites

```bash
pip install requests pyyaml
export OPENROUTER_API_KEY="sk-or-..."
```

### Step 1: Generate templates (calls OpenRouter)

```bash
python scripts/data_gen/gen_templates.py \
  --spec data/interim/icl_anchor_v0/template_specs.json \
  --out  data/interim/icl_anchor_v0/templates.jsonl \
  --model openai/gpt-4o-mini \
  --n_per_spec 3
```

### Step 2: Render dataset (pure Python, no API calls)

```bash
python scripts/data_gen/render_dataset.py \
  --templates data/interim/icl_anchor_v0/templates.jsonl \
  --spec      data/interim/icl_anchor_v0/template_specs.json \
  --out       data/processed/icl_anchor_v0/dataset.jsonl \
  --n_items_per_template 10 \
  --seed 42
```

The rendered dataset can be large; `data/processed/` should be gitignored. Keep `data/processed/icl_anchor_v0/manifest.json` committed for reproducibility tracking.

## Quickstart – Evaluation

### Prerequisites

```bash
pip install torch transformers numpy pyyaml wandb
```

### How to run the full experiment

The experiment matrix (3 models × 2 decoding regimes, 11 runs total) is defined
declaratively in `configs/eval.yaml` and executed with a single command:

```bash
python scripts/eval/sweep.py --config configs/eval.yaml
```

Add `--no_wandb` to skip W&B logging during development.

**What happens**: The sweep loads each model once, then runs all its preset/seed
combinations. For each run it generates outputs, parses them, computes
IAS/OutlierShift with bootstrap CI, saves artifacts, and logs to W&B.

**Interpreting results**: Greedy runs (temperature=0) produce deterministic outputs,
so IAS should be ~0 — this serves as a sanity baseline. Sampling runs
(temperature=0.7) reveal distributional anchoring through non-zero IAS.

### Experiment matrix (default config)

| Model | Preset | Seeds | n_samples | Runs |
|-------|--------|-------|-----------|------|
| Llama-3.2-1B-Instruct | sampling | 0,1,2 | 20 | 3 |
| Llama-3.2-3B-Instruct | greedy | 0 | 1 | 1 |
| Llama-3.2-3B-Instruct | sampling | 0,1,2 | 20 | 3 |
| Llama-3.1-8B-Instruct | greedy | 0 | 1 | 1 |
| Llama-3.1-8B-Instruct | sampling | 0,1,2 | 20 | 3 |
| **Total** | | | | **11** |

### Standalone single-run usage

For debugging or re-running a single configuration:

```bash
python scripts/eval/run_inference.py \
  --config configs/eval.yaml \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --decoding greedy --seed 0 --n_samples 1

python scripts/eval/compute_metrics.py \
  --generations results/runs/Llama-3.2-1B-Instruct_greedy_s0/generations.jsonl \
  --config configs/eval.yaml --no_wandb
```

### Output structure

```
results/runs/{model}_{preset}_s{seed}/
  generations.jsonl   # every raw generation (auditable)
  run_config.json     # exact config for this run
  summary.json        # IAS, CIs, condition-level metrics, anchor_collision_rate
  per_base.csv        # per base_id medians + IAS_i
```

### Parsing Ablation

After inference, run the parsing ablation to compare extraction strategies on the same generations:

```bash
python scripts/eval/parse_ablation.py \
  --generations results/eval/Llama-3.2-1B-Instruct/greedy_s42_n1/generations.jsonl
```

Outputs go to `parse_ablation/` next to `generations.jsonl`.

## Methodology: Parsing

**Primary method**: Hierarchical deterministic parser (Strategy B), applied to completion text only.

- Level 0: If the full output is a clean number (`^\d+\.\d{2}$`), accept directly.
- Level 1: Extract from `<value>...</value>` tag if present.
- Level 2: Find "answer" / "total" / "final" keyword lines; take the **last** number on that line.
- Level 3: Normalize currency symbols and commas; take the **last** number in the completion.

**Why last-number**: Chain-of-thought models echo prompt values early in their response.
Picking the first number is a known confound that captures echoed context values rather
than the model's computed answer.  The ablation quantifies this directly by comparing
`B_hier_last` vs `B_hier_first`.

**No new anchors**: The parser operates only on completion text and never sees the original
prompt. Strategy C (LLM-as-extractor) likewise receives completion-only input.

**Invalids**: Records where no number can be extracted are marked `parse_ok=False` and
excluded from anchoring metrics. Parse rate is reported per condition as a diagnostic.

**Robustness checks**: We ablate three strategies (strict-only, hier-last, hier-first) on
identical generations and report (a) Spearman correlation of per-base IAS, (b) fraction of
sign flips, and (c) MAE sensitivity.  If IAS conclusions are stable across parsers, the
finding is robust to parser choice.

## Key Metrics

- **IAS** (Induced Anchoring Shift): `median_pred(high_anchor) − median_pred(low_anchor)` per base task, paired.
- **OutlierShift**: `median_pred(outlier_anchor) − median_pred(low_anchor)`.
- **Bootstrap 95% CI** over base_id for aggregated IAS (2000 resamples).
- **Mitigation check**: placeholder_demo (digit-free ICL) should reduce IAS magnitude vs anchored demos.

## Design Principles

- **LLM produces only natural-language templates** — all numeric values, derived totals, and ground truth are computed by Python.
- **Digit-free templates** — enforced by regex (`\d`); invalid responses are automatically retried with corrective prompts.
- **Safe expression evaluation** — derived variables use an AST-based evaluator (no `eval`).
- **Disk-cached API calls** — keyed by SHA-256 of model + messages; safe to re-run without burning tokens.
- **Paired evaluation** — all anchoring metrics use deterministic `base_id = sha1(template_id + context + question + truth)` for within-task pairing.
- **W&B integration** — config, scalar summaries, per-base IAS tables, and dataset hash logged per run.
