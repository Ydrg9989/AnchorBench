"""Migration helper: PoC v0.2 → AnchorBench v1 field mapping.

Maps naming conventions and template families from the PoC dataset.
Not a 1:1 migration — domains changed, but this helps reuse patterns.

Usage:
    PYTHONPATH=src python -m anchorbench_v1.migrate_v0_2 \\
        --input poc_dataset/poc_v0.2.jsonl \\
        --output datasets/anchorbench_v1/migrated_v0_2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Domain mapping: PoC field/domain → v1 domain (best-effort)
_DOMAIN_MAP = {
    ("health", "disease_prevalence_estimation"): "market_demographics",
    ("health", "treatment_success_risk"): "market_demographics",
    ("finance", "credit_default_risk"): "pricing_wtp",
    ("finance", "fraud_risk"): "legal_policy",
    ("ops", "project_delay_risk"): "operations_time",
    ("ops", "qa_defect_rate"): "resource_consumption",
}

# Suite mapping: PoC suite → v1 suite
_SUITE_MAP = {
    "external": "external",
    "self_generated": "external",  # closest analogue
    "icl": "icl",
    "conversation_history": "history",
    "rag": "rag",
    "tool": "tool",
}


def migrate_item(item: dict) -> dict:
    """Map a PoC v0.2 item to v1-compatible fields (best-effort)."""
    poc_field = item.get("field", "")
    poc_domain = item.get("domain", "")
    v1_domain = _DOMAIN_MAP.get((poc_field, poc_domain), poc_domain)
    v1_suite = _SUITE_MAP.get(item.get("suite", ""), item.get("suite", ""))

    gold = item.get("gold", {})
    theta = gold.get("theta", 50)
    y_star = gold.get("y_star") or gold.get("y_star_stage2") or round(theta)

    anchors = item.get("anchors", {})

    return {
        "item_id": item.get("item_id", ""),
        "suite": v1_suite,
        "domain": v1_domain,
        "template_family": item.get("template_id", ""),
        "answer_space": item.get("answer_space", {"type": "int", "min": 0, "max": 100}),
        "theta": theta,
        "y_star": y_star,
        "anchors": {
            "low": anchors.get("low"),
            "high": anchors.get("high"),
            "gap": (anchors.get("high", 0) or 0) - (anchors.get("low", 0) or 0),
            "anchor_type": "numeric",
            "relevance": "normatively_irrelevant",
        },
        "tags": {
            "split": "migrated_v0_2",
            "original_field": poc_field,
            "original_domain": poc_domain,
            "original_suite": item.get("suite", ""),
        },
        "meta": item.get("meta", {}),
        "_migration_note": "Best-effort mapping from PoC v0.2; review before use",
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Migrate PoC v0.2 to v1 format")
    parser.add_argument("--input", required=True, help="Path to poc_v0.2.jsonl")
    parser.add_argument("--output", required=True, help="Output path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    items = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    logger.info("Loaded %d items from %s", len(items), input_path)

    migrated = [migrate_item(item) for item in items]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for m in migrated:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    logger.info("Wrote %d migrated items to %s", len(migrated), output_path)

    # Summary
    suites = set(m["suite"] for m in migrated)
    domains = set(m["domain"] for m in migrated)
    print(f"\nMigration summary:")
    print(f"  Items: {len(migrated)}")
    print(f"  Suites: {sorted(suites)}")
    print(f"  Domains: {sorted(domains)}")


if __name__ == "__main__":
    main()
