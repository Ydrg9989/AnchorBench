# AnchorBench datasets

The committed directories are the exact prompts the evaluated models saw.
They are ground truth: regenerating them is a check
(`tests/test_dataset_regeneration.py`), not a build step. How they are
generated and what each record contains is in
[docs/DATA.md](../docs/DATA.md).

## Core suites (Table 1 and the main results)

| Suite | Directory | Extra file |
|---|---|---|
| External | `anchorbench_external_core/` | — |
| History | `anchorbench_history_core/` | — |
| ICL | `anchorbench_icl_core/` | — |
| ICL-dist | `anchorbench_icl_dist_core/` | — |
| RAG | `anchorbench_rag_core/` | `anchorbench_corpus.jsonl`, the frozen 3-document corpus per item |
| Tool | `anchorbench_tool_core/` | — |

Each directory holds `itemspecs.jsonl` (ground truth), `promptviews_core.jsonl`
(the five matched conditions; the runners' default input),
`promptviews_ablation.jsonl` (extra conditions, where the suite has any),
`promptviews.jsonl` (both together) and `manifest.json`. Every core suite has
360 items and 1,800 core prompt views.

## Extension datasets (Table 2 and the appendix)

| Directory | Backs |
|---|---|
| `anchorbench_external_uncertain/` | Table 2 and `tab:uncertain_k` (k of 5 ratings visible) |
| `anchorbench_external_weighted_mean_core/` | `tab:weighted_mean` |
| `anchorbench_{external,history}_medical_pilot/`, `..._other_pilot/` | `tab:extension_pilot` |
| `anchorbench_{external,history,rag}_d1/` | `tab:intensity_pathway` |
| `anchorbench_rag_p2/`, `anchorbench_tool_p3/` | `tab:rag_realism`, `tab:tool_realism` |

## Regenerate and validate

```bash
bash scripts/generate_all.sh core 42        # all six core suites, into datasets/
bash scripts/validate_all.sh core
sha256sum -c anchorbench_core_checksums.sha256
```

`smoke` and `pilot` sizes write `anchorbench_<suite>_smoke/` and
`..._pilot/` for local development; they are not part of the release.

## Hugging Face release

```bash
python scripts/export_public_promptviews.py     # -> hf_release/<suite>.jsonl
python datasets/upload_hf.py --dry-run          # inspect
python datasets/upload_hf.py                    # upload
```

`hf_release/` (committed) is the flat, self-contained export that lives on the
Hub as [Yiderigun/AnchorBench](https://huggingface.co/datasets/Yiderigun/AnchorBench).
[DATASET_CARD.md](DATASET_CARD.md) is the Hub card with the field schema;
[VERSIONS.md](VERSIONS.md) ties each released version to the experiments and
paper tables it backs.
