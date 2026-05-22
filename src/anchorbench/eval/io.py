"""I/O utilities for AnchorBench evaluation data."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_promptviews(path: Path | str) -> dict[str, dict[str, dict]]:
    """Load promptviews.jsonl -> {item_id: {condition: view_dict}}."""
    views: dict[str, dict[str, dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            pv = json.loads(line.strip())
            views.setdefault(pv["item_id"], {})[pv["condition"]] = pv
    return views


def load_itemspecs(path: Path | str) -> dict[str, dict]:
    """Load itemspecs.jsonl -> {item_id: spec_dict}."""
    specs: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line.strip())
            specs[s["item_id"]] = s
    return specs


def load_records(path: Path | str) -> list[dict]:
    """Load JSONL records, skipping malformed lines."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"  WARNING: skipping malformed line {i + 1} in {path}",
                    file=sys.stderr,
                )
    return records
