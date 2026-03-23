# AnchorBench

**A Multi-Paradigm Benchmark for Anchoring Bias in Large Language Models**

AnchorBench maps four established human anchoring paradigms to five LLM interface channels — prompt text, conversation history, in-context demonstrations, retrieved documents, and tool outputs. Each item is presented under five matched conditions that share the same evidence, question, and gold answer, varying only the anchor. An explicit relevance axis distinguishes irrelevant anchors from plausibly informative ones, enabling measurement of both raw susceptibility and relevance discrimination.

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate all 5 suites (1,800 items, 9,000 prompts)
bash scripts/generate_all.sh core 42

# Validate
bash scripts/validate_all.sh core

# Freeze for release
bash scripts/freeze_dataset.sh core
```

## Environment

All experiments (evaluation and data generation) use the **conda environment `LLM_anchoring`**. This avoids CUDA/cuDNN conflicts (e.g. with vLLM). Use the wrapper for eval scripts:

```bash
conda activate LLM_anchoring
bash scripts/run_with_env.sh python scripts/eval/run_external.py ...
```

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for details and vLLM/cuDNN troubleshooting.

## Benchmark Design

### Suites

| Suite    | Anchor Channel               | Human Analogue               | Anchor Mechanism                          |
|----------|------------------------------|------------------------------|-------------------------------------------|
| External | Sentence before the question | Classic external anchor      | "A recent study found X"                  |
| History  | Prior conversation turn      | Self-generated anchor        | Partial-evidence estimate (stage 1 → 2)   |
| ICL      | Demo header metadata         | Numeric priming              | "Prior estimate for similar cases: X"     |
| RAG      | Retrieved document           | Ecological anchor (media)    | "A survey reported an index of X"         |
| Tool     | Tool output JSON field       | Ecological anchor (systems)  | `reference_value: X` or `request_id: X`  |

### Conditions (5 per item)

| Condition        | Anchor present? | Relevance  | Direction |
|------------------|:---------------:|------------|-----------|
| `control`        | No              | —          | —         |
| `irrelevant_low` | Yes             | Irrelevant | Low       |
| `irrelevant_high`| Yes             | Irrelevant | High      |
| `plausible_low`  | Yes             | Plausible  | Low       |
| `plausible_high` | Yes             | Plausible  | High      |

**Pairing invariant:** All 5 conditions for a given item share identical evidence, scenario text, question, and gold answer. Only the anchor-bearing component differs.

### Domains (6)

| Domain                         | Example task                              |
|--------------------------------|-------------------------------------------|
| Pricing / Willingness-to-Pay   | Estimate customer price tolerance (0–100) |
| Operations / Time Estimation   | Rate operational efficiency (0–100)       |
| Transportation / Logistics     | Assess delivery reliability (0–100)       |
| Resource Consumption           | Evaluate resource efficiency (0–100)      |
| Market / Demographics          | Estimate market adoption potential (0–100)|
| Legal / Policy Compliance      | Rate regulatory compliance (0–100)        |

### Difficulty

- **Easy:** 5 concordant evidence ratings (σ=8, no missing values)
- **Medium:** (extension) 5 ratings (σ=12, 1 missing, no conflict)
- **Hard:** 5 ratings (σ=15, 2 missing, 1 conflicting value)

### Dataset Counts

| Suite     | Domains | Difficulty | Offset strata | Items per cell | Total items | Prompts |
|-----------|--------:|-----------:|--------------:|---------------:|------------:|--------:|
| External  |       6 |          2 |             3 |             10 |         360 |   1,800 |
| History   |       6 |          2 |             1 |             30 |         360 |   1,800 |
| ICL       |       6 |          2 |             3 |             10 |         360 |   1,800 |
| RAG       |       6 |          2 |             3 |             10 |         360 |   1,800 |
| Tool      |       6 |          2 |             3 |             10 |         360 |   1,800 |
| **Total** |         |            |               |                |   **1,800** | **9,000** |

Anchor offsets: {15, 25, 40} points from θ (theta ∈ [30, 70]).

---

## Data Generation Pipeline

### Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  ItemSpec Gen    │────▸│  Prompt Render    │────▸│  Serialize    │
│  (itemspec_gen)  │     │  (suites/*)      │     │  (.jsonl)     │
└─────────────────┘     └──────────────────┘     └───────────────┘
    θ, evidence,            5 conditions            itemspecs.jsonl
    anchors, gold           per ItemSpec            promptviews.jsonl
    answer                                          manifest.json
```

**Stage 1 — ItemSpec generation:** For each cell in the (domain × difficulty × offset) grid, draw a latent value θ, generate 5 evidence ratings around θ, compute the gold answer as `round(mean(visible evidence))`, and construct low/high anchor values at the chosen offset.

**Stage 2 — Prompt rendering:** Each suite renderer takes an ItemSpec and produces 5 PromptView records (one per condition). The renderer injects the anchor through the suite's channel while keeping all other prompt components identical.

**Stage 3 — Serialization:** ItemSpecs and PromptViews are written as JSONL. A manifest records counts, domains, seed, and git hash for reproducibility.

### Generating Datasets

```bash
# All 5 suites (recommended):
bash scripts/generate_all.sh core 42

# Single suite:
PYTHONPATH=src python -m anchorbench_v1.generate \
    --suite external --size core --seed 42

# History requires --n_per_cell override (no offset dimension):
PYTHONPATH=src python -m anchorbench_v1.generate \
    --suite history --size core --seed 42 --n_per_cell 30

# Extension splits (separate from core, do not mix):
PYTHONPATH=src python -m anchorbench_v1.generate \
    --suite external --size pilot --seed 42 \
    --scoring_function weighted_mean

PYTHONPATH=src python -m anchorbench_v1.generate \
    --suite external --size pilot --seed 42 \
    --difficulties easy medium hard
```

**Size presets:**

| Size  | Items per cell | External/ICL/RAG/Tool | History (n_per_cell) | Use case         |
|-------|---------------:|----------------------:|---------------------:|------------------|
| smoke | 1              | 6                     | 2                    | Quick validation |
| pilot | 5              | 180                   | 60 (default) / 90*  | Development      |
| core  | 10             | 360                   | 120 (default) / 360* | Paper runs       |

\* Use `--n_per_cell 15` (pilot) or `--n_per_cell 30` (core) to match offset-suite totals. The `generate_all.sh` script handles this automatically.

### Validating Datasets

```bash
# All suites:
bash scripts/validate_all.sh core

# Single directory:
PYTHONPATH=src python -m anchorbench_v1.validate \
    --data_dir datasets/anchorbench_external_core/
```

Validation checks:
- Schema completeness (theta range, difficulty, anchors, evidence structure)
- Gold answer derivation (scoring-function-aware: mean, weighted_mean, median)
- Paired-condition completeness (all 5 conditions present per item)
- Paired-condition stem identity (evidence/scenario/question invariant across conditions)
- Domain/difficulty balance within each suite
- Template diversity (label families, scenario/question templates, phrasing indices)
- Duplicate detection (item IDs and prompt hashes)
- Answer-format instruction presence in every prompt
- Manifest count consistency

### Freezing for Release

```bash
bash scripts/freeze_dataset.sh core
# Creates datasets/anchorbench_frozen_core.tar.gz with SHA-256 checksums
```

---

## Output Data Format

### ItemSpec (`itemspecs.jsonl`)

Each line is a JSON object representing one benchmark item (prompt-agnostic ground truth):

| Field                | Type   | Description                                              |
|----------------------|--------|----------------------------------------------------------|
| `item_id`            | str    | Unique ID, e.g. `EXT-pricing_wtp-e-off15-001`           |
| `suite`              | str    | `external` / `history` / `icl` / `icl_dist` / `rag` / `tool` |
| `domain`             | str    | One of 6 evaluation domains                              |
| `difficulty`         | str    | `easy` or `hard`                                         |
| `theta`              | int    | Latent true value (30–70)                                |
| `y_star`             | int    | Gold answer = `round(aggregation(visible evidence))`     |
| `y_star_evidence`    | int    | Same as `y_star` (evidence-derived)                      |
| `anchors`            | dict   | `{low, high, offset, gap, anchor_type, relevance}`      |
| `evidence_structured`| list   | 5 evidence ratings with labels, values, missing flags    |
| `scoring_function`   | str    | `"mean"` (core), `"weighted_mean"`, `"median"`           |
| `evidence_label_family_idx` | int | Which label family was used (auditable)            |
| `scenario_template_idx`    | int | Which scenario template was used                   |
| `question_template_idx`    | int | Which question template was used                   |
| `anchor_phrasing_idx`      | int | Which phrasing variant was used                    |
| `seed`               | int    | RNG seed used for generation                             |

### PromptView (`promptviews.jsonl`)

Each line is a JSON object representing one concrete prompt (5 per ItemSpec):

| Field              | Type       | Description                                           |
|--------------------|------------|-------------------------------------------------------|
| `item_id`          | str        | Links back to the parent ItemSpec                     |
| `suite`            | str        | Suite name                                            |
| `domain`           | str        | Domain name                                           |
| `condition`        | str        | One of the 5 conditions                               |
| `prompt_text`      | str        | The full prompt text sent to the model                |
| `anchor_string`    | str\|null  | The anchor value as it appears in the prompt          |
| `anchor_value`     | int\|null  | Numeric anchor value (null for control)               |
| `anchor_relevance` | str        | `none` / `irrelevant` / `plausible`                   |
| `prompt_hash`      | str        | SHA-256 hash for determinism verification             |
| `prompt_components`| dict       | Decomposed prompt parts (scenario, evidence, question)|

---

## Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **UAI** (Unified Anchor Influence) | (response − gold) / (anchor − gold) | Fraction of available shift realized toward the anchor |
| **TAR** (Toward-Anchor Rate) | P(response shifts toward anchor) | Binary susceptibility indicator |
| **Disc_Δ** (Discrimination Delta) | UAI_plausible − UAI_irrelevant | Positive = model discriminates by relevance |

---

## Repository Layout

```
src/anchorbench_v1/           # Benchmark data generation
  schema.py                   #   ItemSpec, PromptView, RAGDoc dataclasses
  domains.py                  #   6 domains × 3 label families × 8 scenarios × 4 questions
  itemspec_gen.py             #   ItemSpec generators (shared backbone + per-suite)
  generate.py                 #   CLI: --suite, --size, --seed, --scoring_function
  validate.py                 #   Validation CLI
  validators.py               #   Deterministic validators + diversity checks
  suites/                     #   Per-suite prompt renderers
    _shared.py                #     CONDITIONS, format_evidence, resolve_templates
    external.py               #     External anchor renderer (9 conditions)
    history.py                #     History two-stage renderer (6 conditions)
    icl.py                    #     ICL demo-header renderer (7 conditions)
    icl_dist.py               #     ICL demo-label bands × framing (5 conditions)
    rag.py                    #     RAG corpus renderer (13+ conditions)
    tool_agentic.py           #     Tool function-calling renderer (5 conditions)
    tool_read.py              #     Tool plain-JSON renderer (5 conditions)
    tool.py                   #     Backward-compat shim → tool_agentic

tests/
  test_generation.py          # 30 tests: reproducibility, gold, pairing, diversity
  test_parsing.py             # Answer parsing tests
  test_metrics.py             # Metric computation tests

scripts/
  paper_model_ids.inc.sh      # Ten HF model ids (matches paper Table model-details)
  generate_all.sh             # Generate all 5 suites
  validate_all.sh             # Validate all 5 suites
  freeze_dataset.sh           # Archive with checksums
  eval/                       # Evaluation runners + unified metrics

datasets/                     # Generated benchmark data
  anchorbench_{suite}_{size}/ #   Per-suite output directories
    itemspecs.jsonl
    promptviews.jsonl
    manifest.json

docs/                         # Specification and reference docs
paper/                        # Paper source (LaTeX)
```

## Reproducing Paper Results

```bash
# Run evaluation for a single suite + model
PYTHONPATH=src python scripts/eval/run_external.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --promptviews datasets/anchorbench_external_core/promptviews.jsonl \
    --itemspecs datasets/anchorbench_external_core/itemspecs.jsonl \
    --out_dir results/external_core --batch_size 32 --max_new_tokens 512

# Multi-GPU batch runs (one model per GPU; see script headers for env overrides)
bash scripts/run_external_8B_7B.sh
bash scripts/run_icl_gpus.sh      # default data: anchorbench_icl_dist_core
bash scripts/run_history_gpus.sh  # default baseline: control_twostage (fair two-stage)
bash scripts/run_rag_gpus.sh
bash scripts/run_tool_gpus.sh

# Recompute unified metrics across all results
PYTHONPATH=src:scripts/eval python scripts/eval/recompute_all_unified.py
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the full reproduction guide.

## Documentation

| Document | Description |
|----------|-------------|
| [Benchmark Specification](docs/benchmark_spec_v1_frozen.md) | Frozen benchmark definition (conditions, invariants, metrics) |
| [Data Generation Pipeline](docs/data_generation_pipeline_readme.md) | Detailed pipeline walkthrough with code references |
| [Data Objects Reference](docs/data_objects_reference.md) | Complete ItemSpec / PromptView field reference |
| [Reproducibility Guide](docs/REPRODUCIBILITY.md) | Step-by-step reproduction instructions |
| [Repository Structure](docs/REPO_STRUCTURE.md) | Full file layout |

## License

[To be added before submission]
