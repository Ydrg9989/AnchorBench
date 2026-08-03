# AnchorBench

> A multi-paradigm benchmark for measuring numeric anchoring bias in
> Large Language Models. **14 models &times; 5 anchor pathways &times;
> 3 relevance conditions** = 9,000+ prompts, 70 evaluation cells, one
> command to reproduce.

| | |
| --- | --- |
| Paper | *AnchorBench: Measuring LLM Susceptibility to Numeric Anchoring Across Interface Paradigms* (COLM 2026) |
| License | Apache-2.0 |
| Python | &ge; 3.10 |
| Status | Release v2.0 |

AnchorBench evaluates how strongly an irrelevant or plausible numeric
"anchor" pulls a model's downstream estimate, across five interface
paradigms: **External** prompt context, **History** of prior
interactions, **In-Context Learning** demos, **Retrieval-Augmented
Generation** documents, and **Tool** outputs. The benchmark reports
five aligned metrics (UAI, TAR, Disc<sub>&Delta;</sub>, MAE, Acc<sub>10</sub>)
with bootstrap CIs and BH-corrected significance tests.

---

## Install

```bash
git clone https://github.com/<your-org>/anchorbench.git
cd anchorbench
pip install -e ".[all]"            # core + vllm + api + dev
```

Optional extras:

| Extra | Pulls in | When to use |
| --- | --- | --- |
| `[vllm]` | vLLM | open-weight models (Qwen, Llama, Gemma, OLMo) |
| `[api]`  | aiohttp | OpenRouter API models (GPT-5, Claude, Gemini, Grok) |
| `[dev]`  | pytest, ruff | running tests / linting |

Set `OPENROUTER_API_KEY` (see `.env.example`) before invoking the API tier.

---

## Quick start (3 commands)

```bash
# 1. Generate the External suite at "smoke" size (CPU only, ~10s)
anchorbench generate data=external +size=smoke

# 2. Evaluate Qwen-7B on it (GPU)
anchorbench eval data=external model=qwen_7b

# 3. Re-render every paper figure and table from existing results
anchorbench tables --paper
```

`anchorbench` is the single console script. It exposes Hydra-driven
subcommands (`eval`, `experiment`, `generate`, `tables`, `verify`,
`add-model`) so any cell, recipe, or override is one line.

---

## Reproduce the paper end-to-end

```bash
# Validate the environment and dry-run every planned cell
DRY_RUN=1 bash scripts/reproduce_paper.sh

# Full run: 14 models x 5 suites + figures + tables + verifier
bash scripts/reproduce_paper.sh
```

This script:

1. Regenerates any missing `datasets/anchorbench_*_core/` directories.
2. Runs the frozen `paper_main` recipe (70 evaluation cells).
3. Recomputes `unified_all_suites.json` for both the OW and API runs.
4. Regenerates Figures 4-5 and Tables 1-16 in `COLM/figures/` and `outputs/tables/`.
5. Runs `anchorbench verify`, which fails loudly if any numeric claim drifts.

Approximate cost: ~24h on 4xA100 + ~$300 OpenRouter spend at full size.

---

## Extending

Common extension points and where they live:

| You want to... | Look at | Doc |
| --- | --- | --- |
| Add a new model | `conf/model/*.yaml` | [docs/EXTENDING.md#add-a-model](docs/EXTENDING.md) |
| Add a new suite | `src/anchorbench/data/suites/` | [docs/EXTENDING.md#add-a-suite](docs/EXTENDING.md) |
| Add a metric | `src/anchorbench/eval/metrics.py` | [docs/EXTENDING.md#add-a-metric](docs/EXTENDING.md) |
| Add a named experiment | `conf/experiment/*.yaml` | [docs/EXTENDING.md#add-an-experiment](docs/EXTENDING.md) |
| One-shot model registration | `anchorbench add-model openai/gpt-5o` | -- |

---

## Repository structure

```
anchorbench/
|-- conf/                    # Hydra config tree (data, model, tier, decoding, experiment)
|-- src/anchorbench/         # single consolidated package
|   |-- data/                # ItemSpec + suite renderers + validators
|   |-- eval/                # backends, metrics, evaluator, IO
|   |-- inference/           # OpenRouter async client
|   |-- runners/             # per-suite runners (eval entry points)
|   |-- analysis/            # unified, gold-shift, sampling, mitigation
|   |-- paper/               # figure + table generators + verifier
|   `-- cli/                 # `anchorbench` Hydra-driven entrypoints
|-- scripts/                 # reproduce_paper.sh + thin wrappers
|-- docs/                    # ARCHITECTURE, REPRODUCIBILITY, EXTENDING
|-- datasets/                # gitignored (regenerated on demand)
|-- results/                 # gitignored (raw + unified outputs)
|-- COLM/                    # paper LaTeX source + figures
`-- tests/
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full data-flow
diagram and package map.

---

## Citation

```bibtex
@inproceedings{anchorbench2026,
  title     = {AnchorBench: Measuring LLM Susceptibility to Numeric
               Anchoring Across Interface Paradigms},
  author    = {AnchorBench Authors},
  booktitle = {Proceedings of the Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
