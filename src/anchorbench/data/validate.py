"""Validation CLI for AnchorBench.

Usage (single suite):
    PYTHONPATH=src python -m anchorbench.data.validate \\
        --data_dir datasets/anchorbench_external_smoke/

Usage (all suites at once):
    bash scripts/validate_all.sh smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .schema import ItemSpec, PromptView, read_jsonl
from .validators import validate_all

logger = logging.getLogger(__name__)


def _load(data_dir: Path):
    specs_path = data_dir / "itemspecs.jsonl"
    views_path = data_dir / "promptviews.jsonl"
    manifest_path = data_dir / "manifest.json"

    if not specs_path.exists() or not views_path.exists():
        logger.error("Missing itemspecs.jsonl or promptviews.jsonl in %s", data_dir)
        sys.exit(1)

    specs = [ItemSpec.from_dict(d) for d in read_jsonl(specs_path)]
    views = [PromptView.from_dict(d) for d in read_jsonl(views_path)]
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    return specs, views, manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Validate AnchorBench dataset(s)")
    parser.add_argument("--data_dir", required=True, nargs="+",
                        help="One or more dataset directories to validate")
    args = parser.parse_args()

    total_errors = 0
    total_specs = 0
    total_views = 0

    for d in args.data_dir:
        data_dir = Path(d)
        specs, views, manifest = _load(data_dir)
        errors = validate_all(specs, views, manifest)

        if errors:
            logger.error("FAILED [%s]: %d errors", data_dir.name, len(errors))
            for e in errors[:50]:
                logger.error("  %s", e)
            if len(errors) > 50:
                logger.error("  ... and %d more", len(errors) - 50)
            total_errors += len(errors)
        else:
            logger.info("PASSED [%s]: %d specs, %d views", data_dir.name, len(specs), len(views))

        total_specs += len(specs)
        total_views += len(views)

    print(f"\nTotal: {total_specs} specs, {total_views} views, {total_errors} errors")
    sys.exit(1 if total_errors else 0)


if __name__ == "__main__":
    main()
