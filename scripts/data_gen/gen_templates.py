#!/usr/bin/env python3
"""CLI: generate digit-free NL templates via OpenRouter from spec file."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llm_anchoring.data_gen.template_gen import generate_template

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate templates from specs via LLM")
    p.add_argument(
        "--spec",
        type=Path,
        default=Path("data/interim/icl_anchor_v0/template_specs.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/icl_anchor_v0/templates.jsonl"),
    )
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--n_per_spec", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("data/interim/icl_anchor_v0/cache"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = json.loads(args.spec.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    total, ok = 0, 0
    with args.out.open("w") as f:
        for spec in specs:
            for i in range(args.n_per_spec):
                total += 1
                logger.info(
                    "spec=%s  variant=%d/%d", spec["id"], i + 1, args.n_per_spec
                )
                record = generate_template(
                    spec, model=args.model, cache_dir=args.cache_dir
                )
                if record is None:
                    logger.warning("SKIP spec=%s variant=%d", spec["id"], i + 1)
                    continue
                f.write(json.dumps(record) + "\n")
                ok += 1

    logger.info("done: %d/%d templates written to %s", ok, total, args.out)


if __name__ == "__main__":
    main()
