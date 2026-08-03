"""`anchorbench add-model` subcommand.

Appends a new model config to ``conf/model/<slug>.yaml`` and prints a
smoke-test command. Detects API vs open-weight models from the model
identifier prefix.

Usage::

    anchorbench add-model openai/gpt-5o
    anchorbench add-model meta-llama/Llama-4-8B-Instruct --short Llama-4-8B
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONF_MODEL_DIR = ROOT / "conf" / "model"

API_PREFIXES = ("openai/", "anthropic/", "google/", "x-ai/", "mistralai/", "meta/")


def _slugify_filename(model_id: str) -> str:
    last = model_id.split("/")[-1]
    s = re.sub(r"[^A-Za-z0-9]+", "_", last).strip("_").lower()
    return s


def _short_default(model_id: str) -> str:
    return model_id.split("/")[-1]


def main() -> int:
    p = argparse.ArgumentParser(prog="anchorbench add-model")
    p.add_argument("model_id", help="HF Hub or OpenRouter model identifier.")
    p.add_argument("--short", default=None,
                   help="Display name (default: last path segment).")
    p.add_argument("--name", default=None,
                   help="Long name (default: same as --short).")
    args = p.parse_args()

    model_id = args.model_id
    is_api = any(model_id.startswith(pre) for pre in API_PREFIXES)
    slug = model_id.replace("/", "_")
    fname = _slugify_filename(model_id)
    short = args.short or _short_default(model_id)
    name = args.name or short

    out_path = CONF_MODEL_DIR / f"{fname}.yaml"
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return 0

    if is_api:
        body = (
            f"name: {name}\n"
            f"hf_id: {model_id}\n"
            f"slug: {slug}\n"
            f"short: {short}\n"
            f"backend: openrouter\n"
            f"api_concurrency: 8\n"
        )
    else:
        body = (
            f"name: {name}\n"
            f"hf_id: {model_id}\n"
            f"slug: {slug}\n"
            f"short: {short}\n"
            f"backend: vllm\n"
            f"tensor_parallel_size: 1\n"
            f"gpu_memory_utilization: 0.9\n"
            f"max_model_len: 4096\n"
            f"dtype: bfloat16\n"
        )
    out_path.write_text(body)
    print(f"Created {out_path}")
    print()
    print("Smoke test:")
    print(f"  anchorbench eval data=external model={fname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
