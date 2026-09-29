"""`anchorbench add-model` subcommand.

Appends a new model config to ``conf/model/<slug>.yaml`` and prints a
smoke-test command. The backend is declared, never inferred from the
identifier: ``google/gemma-*`` is an open-weight model and
``google/gemini-*`` a hosted one, so a prefix rule mis-routes one of them
(see cli/cells.py and tests/test_model_routing.py).

Usage::

    anchorbench add-model meta-llama/Llama-4-8B-Instruct --short Llama-4-8B
    anchorbench add-model openai/gpt-5o --backend openrouter
"""

from __future__ import annotations

import argparse
import re
import sys

from anchorbench.paths import CONF_DIR

CONF_MODEL_DIR = CONF_DIR / "model"

BACKENDS = ("vllm", "hf", "openrouter")


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
    p.add_argument("--backend", choices=BACKENDS, default="vllm",
                   help="Where the model runs: vllm or hf load weights locally, "
                        "openrouter calls the hosted API (default: vllm).")
    args = p.parse_args()

    model_id = args.model_id
    slug = model_id.replace("/", "_")
    fname = _slugify_filename(model_id)
    short = args.short or _short_default(model_id)
    name = args.name or short

    out_path = CONF_MODEL_DIR / f"{fname}.yaml"
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return 0

    family = model_id.split("/")[0]
    meta = (
        "# Table metadata read by anchorbench.registry: edit family/family_latex,\n"
        "# params (open-weight) and latex (macro printing `short`) before adding the\n"
        "# model to conf/panel.yaml, which is what puts it into the paper tables.\n"
        f"family: {family}\n"
        f"latex: '{short}'\n"
    )
    if args.backend == "openrouter":
        body = (
            f"name: {name}\n"
            f"hf_id: {model_id}\n"
            f"slug: {slug}\n"
            f"short: {short}\n"
            f"backend: openrouter\n"
            f"api_concurrency: 8\n"
        ) + meta
    else:
        body = (
            f"name: {name}\n"
            f"hf_id: {model_id}\n"
            f"slug: {slug}\n"
            f"short: {short}\n"
            f"backend: {args.backend}\n"
            f"tensor_parallel_size: 1\n"
            f"gpu_memory_utilization: 0.9\n"
            f"max_model_len: 4096\n"
            f"dtype: bfloat16\n"
        ) + meta
    out_path.write_text(body)
    print(f"Created {out_path}")
    print()
    print("Smoke test:")
    print(f"  anchorbench eval data=external model={fname}")
    print("To include it in the paper tables, add the key to conf/panel.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
