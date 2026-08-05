# AnchorBench

> A multi-pathway benchmark for the anchoring effect in large language
> models. **14 models &times; 5 anchor pathways &times; 3 relevance
> conditions** = 9,000 condition-controlled prompts per model, 70 evaluation
> cells, one command to reproduce.

| | |
| --- | --- |
| Paper | *AnchorBench: A Multi-Pathway Benchmark for the Anchoring Effect in LLMs* (COLM 2026) |
| Dataset | [Yiderigun/LLM_anchoring](https://huggingface.co/datasets/Yiderigun/LLM_anchoring) on Hugging Face |
| Raw results | Zenodo — DOI pending |
| Code license | Apache-2.0 · **Dataset** CC BY 4.0 |
| Python | &ge; 3.10 |
| Status | Release v2.0 |

AnchorBench measures how strongly an irrelevant or plausible numeric
"anchor" pulls a model's estimate, across five delivery pathways:
**External** prompt context, conversation **History**, **In-Context
Learning** demonstrations, **Retrieval-Augmented Generation** documents, and
**Tool** outputs. Because every item carries structured numeric evidence and
a deterministic gold answer, the benchmark measures not just whether outputs
shift but whether the shift is justified.

---

## Install

```bash
git clone https://github.com/Yiderigun/LLM_anchoring.git
cd LLM_anchoring
pip install -e ".[all]"            # core + vllm + api + dev
```

| Extra | Pulls in | When to use |
| --- | --- | --- |
| `[vllm]` | vLLM | open-weight models (Qwen, Llama, Gemma, OLMo) |
| `[api]`  | aiohttp | OpenRouter API models (GPT, Claude, Gemini, Grok) |
| `[dev]`  | pytest, ruff | tests and linting |

**API credentials.** Put `OPENROUTER_API_KEY` in
`~/.config/anchorbench/env`, not in the repo — `scripts/run_with_env.sh`
sources it from there. A key inside the working tree gets shipped by any
folder upload or tarball; `tests/test_no_secrets.py` fails if one reappears.
See `.env.example`.

---

## Quick start

```bash
# 1. Generate the External suite at "smoke" size (CPU only, ~10s)
anchorbench generate data=external +size=smoke

# 2. Evaluate Qwen-7B on it (GPU)
anchorbench eval data=external model=qwen_7b

# 3. Re-render every paper figure and table from existing results
anchorbench tables --paper
```

`anchorbench` is the single console script, exposing Hydra-driven
subcommands (`eval`, `experiment`, `generate`, `tables`, `verify`,
`add-model`), so any cell, recipe or override is one line.

---

## Reproduce the paper

```bash
DRY_RUN=1 bash scripts/reproduce_paper.sh    # validate + dry-run every cell
bash scripts/reproduce_paper.sh              # full run
```

The script regenerates any missing `datasets/anchorbench_*_core/`, runs the
frozen `paper_main` recipe (70 cells), recomputes `unified_all_suites.json`
for both tiers, regenerates the figures and main tables into `COLM/figures/`
and `outputs/tables/`, and finishes with `anchorbench verify`, which fails if
any numeric claim drifts.

Two things it does **not** cover:

```bash
anchorbench tables --appendix     # the 13 \input-ed appendix tables
bash scripts/run_stage3_reruns.sh # the two re-run experiments (addendum)
```

Approximate cost: ~24 h on 4x A100 plus roughly $300 of OpenRouter spend at
full size. The appendix tables additionally need `results/rebuttal/`, which
is published on Zenodo rather than committed.

---

## Verifying without re-running anything

Most of the repository can be checked on a clean clone in seconds, because
the two unified summaries and every generated table are committed:

```bash
pytest                                    # 239 tests
anchorbench verify --strict               # every numeric paper claim
python scripts/measure_paper_drift.py     # paper tables vs generator output
```

[docs/RECONCILIATION.md](docs/RECONCILIATION.md) is the ledger of every known
divergence between the paper, the committed artifacts and the current code,
with the command to re-measure each one.

---

## Extending

| You want to... | Look at | Doc |
| --- | --- | --- |
| Add a model | `conf/model/*.yaml` | [EXTENDING.md](docs/EXTENDING.md) |
| Add a suite | `src/anchorbench/data/suites/` | [EXTENDING.md](docs/EXTENDING.md) |
| Add a metric | `src/anchorbench/eval/metrics.py` | [EXTENDING.md](docs/EXTENDING.md) |
| Add an experiment | `conf/experiment/*.yaml` | [EXTENDING.md](docs/EXTENDING.md) |
| Register a model in one shot | `anchorbench add-model openai/gpt-5o` | — |

---

## Repository structure

```
LLM_anchoring/
|-- conf/                    # Hydra config tree (data, model, tier, decoding, experiment)
|-- src/anchorbench/         # single consolidated package
|   |-- data/                # ItemSpec generation, suite renderers, validators
|   |-- eval/                # backends, metrics, evaluator, IO
|   |-- inference/           # OpenRouter async client
|   |-- runners/             # per-suite runners
|   |-- analysis/            # unified metrics + the appendix analyses
|   |-- paper/               # figure/table generators + claim verifier
|   `-- cli/                 # `anchorbench` entry points
|-- datasets/                # committed: the exact prompts the models saw
|-- results/                 # bulk gitignored; unified summaries + tables committed
|-- COLM_camera_ready/       # camera-ready LaTeX source, figures and PDF
|-- scripts/                 # reproduce_paper.sh + thin wrappers
|-- docs/                    # ARCHITECTURE, RECONCILIATION, APPENDIX_TABLES, ...
`-- tests/
```

`datasets/` is **committed on purpose**: the suite renderers were
restructured after the paper's data was generated, so the committed prompts
are ground truth and regenerating them is a check rather than a build step.

Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the package map
and data flow, and [docs/APPENDIX_TABLES.md](docs/APPENDIX_TABLES.md) for
which module produces which appendix table.

---

## Citation

```bibtex
@inproceedings{borjigin2026anchorbench,
  title     = {AnchorBench: A Multi-Pathway Benchmark for the Anchoring
               Effect in {LLM}s},
  author    = {Borjigin, Yiderigun and Hermann, Alexander and
               Cyron, Christian and Aydin, Roland},
  booktitle = {Proceedings of the Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License

Code is Apache-2.0 (see [LICENSE](LICENSE)). The benchmark dataset is
released under CC BY 4.0; see
[datasets/DATASET_CARD.md](datasets/DATASET_CARD.md).
