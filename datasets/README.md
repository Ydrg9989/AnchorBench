# AnchorBench datasets (canonical layout)

## Benchmark data (paper / full eval)

| Suite    | Path |
|----------|------|
| External | `anchorbench_external_core/` |
| History  | `anchorbench_history_core/` |
| ICL      | `anchorbench_icl_core/` |
| RAG      | `anchorbench_rag_core/` (+ `anchorbench_corpus.jsonl`) |
| Tool     | `anchorbench_tool_core/` |
| Uncertain| `anchorbench_external_uncertain/` (backs Table 2; derived from external itemspecs) |

Each folder contains:

- `itemspecs.jsonl` — ground truth per item  
- `promptviews_core.jsonl` — **5 matched conditions** per item (recommended default)  
- `promptviews.jsonl` — all conditions including ablations (where applicable)  
- `manifest.json` — counts, seed, `generator_version`, validation flag  

**Sanity check:** open `promptviews_core.jsonl`, pick one `item_id`, grep the same id across conditions — scenario/evidence/question should match; only the anchor differs.

## Quick smoke eval (small slice of core)

No separate `*_smoke` trees in the repo root. Scripts use **`anchorbench_*_core` + `--max_items`** (same 5 conditions):

```bash
conda activate LLM_anchoring
cd /path/to/LLM_anchoring

bash scripts/run_smoke_sanity.sh
# vLLM: SMOKE_BACKEND=vllm SMOKE_MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_smoke_sanity.sh
```

**Slice:** external / icl / rag / tool → 6 items × 5 conditions; history → 2 items × 5.

Single-suite example (External, 6 items):

```bash
bash scripts/run_with_env.sh python scripts/eval/run_external.py \
  --promptviews datasets/anchorbench_external_core/promptviews_core.jsonl \
  --itemspecs datasets/anchorbench_external_core/itemspecs.jsonl \
  --max_items 6 \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --out_dir results/my_smoke_external \
  --max_tokens 256 --backend hf --seed 42
```

## Regenerate benchmark from code

```bash
conda activate LLM_anchoring
cd /path/to/LLM_anchoring
bash scripts/generate_all.sh core 42
# Tool-read (optional):
PYTHONPATH=src python -m anchorbench_v1.generate \
  --suite tool_read --size core --seed 42 \
  --out_dir datasets/anchorbench_tool_read_core
```

Optional tiny **`smoke`** / **`pilot`** sizes from `generate_all.sh` write new folders (e.g. `anchorbench_*_smoke`) for local dev only — they are **not** part of the canonical release layout.

## Archive

| Path | Contents |
|------|----------|
| `archive/smoke_benchmark_20260318/` | Former root-level `anchorbench_*_smoke` snapshots |
| `archive/legacy_pilot_smoke_20260318/` | Older pilot + duplicate smoke copies |
| `archive/README.md` | Short index |

Prior **smoke / pilot experiment outputs** live under **`results/archive/smoke_runs_20260318/`** (e.g. `smoke_experiment`, `smoke_retest`).

## Other

- `hf_release/` — prior Hugging Face export snapshots  
- `DATASET_CARD.md` — dataset documentation  

## Public release

The Hugging Face release is built from these directories, not shipped from
them directly:

```bash
python scripts/export_public_promptviews.py     # -> datasets/hf_release/*.jsonl
python datasets/upload_hf.py --dry-run          # inspect
python datasets/upload_hf.py                    # upload
```

`hf_release/` carries one self-contained row per prompt (prompt text, anchor
value, gold answer, metadata) so a user can run the benchmark and compute UAI
without any internal file. See [DATASET_CARD.md](DATASET_CARD.md) for the
field schema and [VERSIONS.md](VERSIONS.md) for the mapping from dataset
version to paper tables.

`tests/test_dataset_regeneration.py` asserts every released prompt is
byte-identical to the prompt the models were evaluated on.
