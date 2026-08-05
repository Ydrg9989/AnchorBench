# Environment: LLM_anchoring

All evaluation and data-generation experiments in this repo are intended to run in the **conda environment `LLM_anchoring`**. Agent mode and local runs should use this environment for consistency and to avoid CUDA/cuDNN conflicts.

## Using the conda environment

```bash
conda activate LLM_anchoring
```

Then run scripts as usual, e.g.:

```bash
python -m anchorbench.runners.external ...
```

## Recommended: run wrapper (avoids vLLM/cuDNN conflict)

If you see vLLM errors like:

```text
pydantic_core._pydantic_core.ValidationError: Assertion failed, Found 2 libcudnn.so.x in nvidia-cudnn-cuXX
```

use the project wrapper so the command runs in `LLM_anchoring` with env vars that reduce duplicate cuDNN visibility and use vLLM’s legacy engine:

```bash
bash scripts/run_with_env.sh python anchorbench.runners.external \
  --promptviews datasets/anchorbench_external_core/promptviews.jsonl \
  --itemspecs datasets/anchorbench_external_core/itemspecs.jsonl \
  --model_id meta-llama/Llama-3.1-70B-Instruct \
  --out_dir results/external_70b \
  --backend vllm \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.95
```

The wrapper:

- Activates `LLM_anchoring` when conda is available
- Sets `LD_LIBRARY_PATH` so `$CONDA_PREFIX/lib` is first (single set of CUDA/cuDNN libs)
- Sets `VLLM_USE_V1=0` so vLLM uses the legacy (V0) engine and avoids the V1 config assertion on duplicate libcudnn

Override the conda env name if needed:

```bash
ANCHORBENCH_CONDA_ENV=my_env bash scripts/run_with_env.sh python ...
```

## Agent / IDE

- **Terminal:** Run `conda activate LLM_anchoring` in the project terminal so all subsequent commands use this env.
- **Python interpreter:** Point the IDE/Cursor Python interpreter to the env, e.g.  
  `$CONDA_PREFIX/bin/python` with `CONDA_PREFIX` set to your env path (e.g. `.../miniforge3/envs/LLM_anchoring`).
- **Run scripts:** Prefer `bash scripts/run_with_env.sh python anchorbench.runners.* ...` so vLLM and path settings are applied.

## If the cuDNN error persists

1. **Single cuDNN in env:** Ensure only one cuDNN is visible:
   ```bash
   find $CONDA_PREFIX -name 'libcudnn*' 2>/dev/null
   ```
   If you see two (e.g. from `nvidia-cudnn-cuXX` and another package), uninstall the duplicate or set `LD_LIBRARY_PATH` to a single directory that contains one `libcudnn.so*` before running.

2. **Keep using V0:** Leave `VLLM_USE_V1=0` (the wrapper sets this by default).

3. **Conda cuDNN:** Prefer the CUDA/cuDNN stack from your conda env; avoid mixing system CUDA and conda CUDA in the same run.
