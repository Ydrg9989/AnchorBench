# Hugging Face Hub identifiers for the ten instruction-tuned models in the paper.
# Source of truth: COLM/sections/appendix/setup.tex (Table~\ref{tab:model-details}).
# Do not execute this file directly; it is sourced from run_*_gpus.sh scripts.
#
# shellcheck disable=SC2034
PAPER_ANCHORBENCH_MODEL_IDS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-1b-it"
  "google/gemma-3-4b-it"
  "allenai/OLMo-2-1124-13B-Instruct"
  "allenai/OLMo-2-0325-32B-Instruct"
)
