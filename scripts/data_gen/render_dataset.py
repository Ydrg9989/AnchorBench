#!/usr/bin/env python3
"""CLI: render the final dataset from templates + specs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llm_anchoring.data_gen.render_dataset import render_items

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render dataset from templates")
    p.add_argument(
        "--templates",
        type=Path,
        default=Path("data/interim/icl_anchor_v0/templates.jsonl"),
    )
    p.add_argument(
        "--spec",
        type=Path,
        default=Path("data/interim/icl_anchor_v0/template_specs.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/icl_anchor_v0/dataset.jsonl"),
    )
    p.add_argument("--n_items_per_template", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo_shots", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs_list = json.loads(args.spec.read_text())
    spec_map = {s["id"]: s for s in specs_list}

    templates: list[dict] = []
    with args.templates.open() as f:
        for line in f:
            line = line.strip()
            if line:
                templates.append(json.loads(line))

    logger.info("loaded %d templates", len(templates))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total = 0
    with args.out.open("w") as f:
        for tpl in templates:
            spec = spec_map.get(tpl["template_id"])
            if spec is None:
                logger.warning("no spec for template_id=%s", tpl["template_id"])
                continue
            items = render_items(
                tpl, spec,
                n_items=args.n_items_per_template,
                rng=rng,
                demo_shots=args.demo_shots,
            )
            for item in items:
                f.write(json.dumps(item) + "\n")
                total += 1

    logger.info("wrote %d items to %s", total, args.out)
    _write_manifest(args, total)


def _write_manifest(args: argparse.Namespace, total: int) -> None:
    manifest_path = args.out.parent / "manifest.json"
    content = args.out.read_bytes()
    manifest = {
        "dataset_version": "icl_anchor_v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_items": total,
        "seed": args.seed,
        "demo_shots": args.demo_shots,
        "n_items_per_template": args.n_items_per_template,
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
