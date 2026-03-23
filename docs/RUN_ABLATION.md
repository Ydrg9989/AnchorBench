# Ablation benchmark (vLLM, 4× GPU)

## What it runs

| Suites | Data |
|--------|------|
| external, icl, rag, history | `datasets/anchorbench_*_core/promptviews_ablation.jsonl` |

Same **10 open models** and **4-GPU tmux layout** as the main full benchmark (tool suite has no ablation runner in this pipeline).

## Start (recommended after main benchmark)

Archive any prior ablation results and run everything:

```bash
conda activate LLM_anchoring
cd /path/to/LLM_anchoring

bash scripts/run_ablation_full_4gpu.sh fresh
tmux attach -t ablation_benchmark
```

- **Session name:** `ablation_benchmark` (windows: `control`, `gpu0`–`gpu3`)
- **Results:** `results/ablation_full_benchmark/<suite>/<model_slug>/results.jsonl`
- **Logs:** `logs/ablation_<timestamp>/`
- **Archive:** `results/archive/ablation_full_benchmark_<timestamp>/` (on `fresh`)

## Resume only

```bash
bash scripts/run_ablation_full_4gpu.sh
```

Suites whose `results.jsonl` line count ≥ the ablation file line count are skipped.

## One model, one GPU

```bash
export ANCHORBENCH_FORCE_RERUN=1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_model_vllm_ablation.sh Qwen/Qwen2.5-3B-Instruct 0 512 0.92 4096 1
```

## Analysis

See `analysis/analyze_ablation_results.py` and `outputs/stats/ABLATION_STUDY_SUMMARY.md` for post-processing examples.
