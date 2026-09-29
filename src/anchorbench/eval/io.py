"""I/O utilities for AnchorBench evaluation data."""

from __future__ import annotations

import json
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
    """The records of a results.jsonl. A malformed line is an error, not a skip:
    every table downstream would silently lose that record."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: malformed JSON ({e.msg})") from None
    return records


def write_records(records: list[dict], path: Path | str) -> None:
    """One JSON object per line, UTF-8, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def splice_core_records(
    new_records: list[dict], core_path: Path | str, keep: set[str] | None = None,
) -> list[dict]:
    """The published core records (only the conditions in ``keep`` when given)
    followed by ``new_records``: how an appendix experiment gets its control
    and standard-anchor rows without re-running them.

    A missing core file is an error. Splicing nothing in used to yield a
    complete-looking run in which every curve mean was None.
    """
    core_path = Path(core_path)
    if not core_path.is_file():
        raise FileNotFoundError(f"core results missing: {core_path}")
    core = load_records(core_path)
    if keep is not None:
        core = [r for r in core if r["condition"] in keep]
    return core + list(new_records)
