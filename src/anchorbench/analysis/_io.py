"""File and number helpers shared by the analysis modules.

Each analysis module used to carry its own ``write_csv``, ``write_json``,
``_fmt`` and record loader: fifteen CSV writers in eight variants, fourteen
JSON writers in four, sixteen formatters in seven. The variants differed
only in defaults (two or three decimals, a sign, a LaTeX minus), so they
collapse into the few parameterised functions below. Every caller was
migrated with its regenerated output checked byte for byte against the
committed artifacts.

The LaTeX and Markdown writers stay in their modules: each renders a
different appendix table, so there is nothing shared to extract.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DASH = "---"


# -- numbers -----------------------------------------------------------------

def fmt(v: float | None, prec: int = 2, dash: str = DASH, *, signed: bool = False) -> str:
    """Fixed-point number, or ``dash`` for None and non-finite values."""
    if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
        return dash
    return f"{v:+.{prec}f}" if signed else f"{v:.{prec}f}"


def fmt_latex(v: float | None, prec: int = 2, dash: str = DASH) -> str:
    """Fixed-point number with a LaTeX minus sign (``$-$0.12``)."""
    if v is None:
        return dash
    s = f"{v:.{prec}f}"
    return f"$-${s[1:]}" if s.startswith("-") else s


def fmt_pct(v: float | None, dash: str = DASH) -> str:
    """A 0-1 fraction as an integer percentage with a LaTeX percent sign."""
    if v is None:
        return dash
    return f"{round(v * 100):d}\\%"


# -- files -------------------------------------------------------------------

def write_json(payload: Any, path: Path) -> None:
    """``json.dumps(payload, indent=2)`` to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", path)


def write_csv(
    rows: Iterable[Mapping[str, Any]],
    path: Path,
    *,
    fieldnames: list[str] | None = None,
) -> None:
    """Write dict rows as CSV.

    Columns are ``fieldnames`` if given, else the keys of the first row in
    order; extra keys in later rows are ignored. Without ``fieldnames`` an
    empty ``rows`` writes nothing, since there are no columns to name; with
    them the header is written even for no rows.
    """
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %s (%d rows)", path, len(rows))


def load_records(path: Path) -> list[dict]:
    """Records of a ``results.jsonl``; an empty list when the file is absent."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_summary(path: Path) -> dict | None:
    """A ``summary.json``, or None when the file is absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def safe_metrics(records: list[dict], baseline: str) -> dict:
    """Unified metrics, falling back to ``control`` when ``baseline`` is absent.

    Published History runs used ``control`` while later ones used
    ``control_twostage``; this lets one analysis read both.
    """
    from anchorbench.eval.metrics import compute_unified_metrics

    if not records:
        return {}
    present = {r.get("condition") for r in records}
    effective = baseline if baseline in present else "control"
    if effective not in present:
        log.warning("No suitable baseline in records (have %s); metrics may be partial",
                    sorted(present))
    return compute_unified_metrics(records, baseline_condition=effective)
