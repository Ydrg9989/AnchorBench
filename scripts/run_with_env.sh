#!/usr/bin/env bash
# Wrapper that activates the LLM_anchoring conda env and sets vLLM-safe
# environment variables before running any command.
#
# Usage:
#   bash scripts/run_with_env.sh python scripts/eval/run_external.py --backend vllm ...
#
# Override env name:
#   ANCHORBENCH_CONDA_ENV=my_env bash scripts/run_with_env.sh ...
set -euo pipefail

ENV_NAME="${ANCHORBENCH_CONDA_ENV:-LLM_anchoring}"

# Activate conda
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$ENV_NAME"
fi

# Ensure CONDA_PREFIX/lib is first to avoid duplicate cuDNN
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Use vLLM legacy engine to avoid V1 config assertion on duplicate libcudnn
export VLLM_USE_V1=0

# Load HF token from cache if not already set
if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
    if [ -f "$HF_TOKEN_FILE" ]; then
        export HF_TOKEN
        HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    fi
fi

# Load the OpenRouter key from outside the repo. Keeping it out of the working
# tree means a folder upload or tarball of the repo cannot leak it.
ANCHORBENCH_ENV="${ANCHORBENCH_ENV:-${HOME}/.config/anchorbench/env}"
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$ANCHORBENCH_ENV" ]; then
    # shellcheck disable=SC1090
    source "$ANCHORBENCH_ENV"
fi

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"

exec "$@"
