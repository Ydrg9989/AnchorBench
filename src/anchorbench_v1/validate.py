"""Validation CLI for AnchorBench v1.

Usage:
    PYTHONPATH=src python -m anchorbench_v1.validate \\
        --data_dir datasets/anchorbench_v1/

Runs deterministic validators on ItemSpecs and PromptViews.
Optionally runs LLM spot-check with --llm_spot_check.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .schema import ItemSpec, PromptView, read_jsonl
from .validators import validate_all, llm_spot_check

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Validate AnchorBench v1 dataset")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument(
        "--llm_spot_check",
        action="store_true",
        help="Run optional LLM-based spot-check (requires OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--spot_check_fraction",
        type=float,
        default=0.05,
        help="Fraction of items for LLM spot-check (default: 0.05)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for spot-check sampling")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    specs_path = data_dir / "itemspecs.jsonl"
    views_path = data_dir / "promptviews.jsonl"

    if not specs_path.exists():
        logger.error("itemspecs.jsonl not found in %s", data_dir)
        sys.exit(1)
    if not views_path.exists():
        logger.error("promptviews.jsonl not found in %s", data_dir)
        sys.exit(1)

    logger.info("Loading dataset from %s...", data_dir)
    specs = [ItemSpec.from_dict(d) for d in read_jsonl(specs_path)]
    views = [PromptView.from_dict(d) for d in read_jsonl(views_path)]
    logger.info("Loaded %d itemspecs, %d promptviews", len(specs), len(views))

    # Deterministic validation
    logger.info("Running deterministic validators...")
    errors = validate_all(specs, views)

    if errors:
        logger.error("FAILED: %d validation errors", len(errors))
        for err in errors[:50]:
            logger.error("  %s", err)
        if len(errors) > 50:
            logger.error("  ... and %d more", len(errors) - 50)
        sys.exit(1)
    else:
        logger.info("PASSED: All deterministic validators OK (%d specs, %d views)", len(specs), len(views))

    # Optional LLM spot-check
    if args.llm_spot_check:
        logger.info(
            "Running LLM spot-check (fraction=%.2f, seed=%d)...",
            args.spot_check_fraction, args.seed,
        )
        flags = llm_spot_check(
            specs, views,
            sample_fraction=args.spot_check_fraction,
            seed=args.seed,
        )
        if flags:
            logger.warning("LLM spot-check flagged %d items:", len(flags))
            for f in flags:
                logger.warning("  %s", json.dumps(f))
        else:
            logger.info("LLM spot-check: all sampled items OK")

    # Summary stats
    suites = set(s.suite for s in specs)
    domains = set(s.domain for s in specs)
    conditions = set(v.condition for v in views)
    splits = set(s.tags.get("split", "unknown") for s in specs)
    print(f"\nDataset summary:")
    print(f"  Items:      {len(specs)}")
    print(f"  PromptViews: {len(views)}")
    print(f"  Suites:     {sorted(suites)}")
    print(f"  Domains:    {sorted(domains)}")
    print(f"  Conditions: {sorted(conditions)}")
    print(f"  Splits:     {sorted(splits)}")


if __name__ == "__main__":
    main()
