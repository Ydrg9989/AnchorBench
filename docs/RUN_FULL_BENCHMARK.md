# Run full AnchorBench (open models, vLLM, 4× GPU)

## Why GPUs look idle

`scripts/run_model_vllm.sh` **skips** any suite whose `results.jsonl` already has **≥1800 lines**. If you previously finished the benchmark, tmux will exit each model in seconds and **nvidia-smi stays empty**.

## Option A — Fresh full run (tmux, 4 GPUs)

Archives existing `results/full_benchmark/` to **`results/archive/full_benchmark_<timestamp>/`**, then re-runs all 10 models:

```bash
conda activate LLM_anchoring
cd /path/to/LLM_anchoring

bash scripts/run_full_benchmark.sh fresh
```

(`force`, `archive`, and `new` are the same as `fresh`.)

Attach:

```bash
tmux attach -t benchmark
```

(Optional) Tune memory on dedicated H100s:

```bash
GPU_MEM_UTIL=0.92 GPU_MEM_UTIL_32B=0.88 bash scripts/run_full_benchmark.sh force
```

## Option B — One model, one GPU (debug / partial rerun)

Force re-run without wiping the whole tree:

```bash
conda activate LLM_anchoring
cd /path/to/LLM_anchoring

export ANCHORBENCH_VIEWS=promptviews_core.jsonl
export ANCHORBENCH_BATCH_SIZE=64
export ANCHORBENCH_FORCE_RERUN=1

CUDA_VISIBLE_DEVICES=0 bash scripts/run_model_vllm.sh Qwen/Qwen2.5-3B-Instruct 0 512 0.92 4096 1
```

(`GPU` argument is **device index within `CUDA_VISIBLE_DEVICES`**, usually `0` when you set one GPU.)

## Option C — Only incomplete suites

Omit `force` and `ANCHORBENCH_FORCE_RERUN`: anything already at 1800 lines is skipped; missing or short `results.jsonl` is re-run.

## Logs

- Per-GPU logs: `logs/benchmark_<timestamp>/gpu*.log`
- Results: `results/full_benchmark/`
