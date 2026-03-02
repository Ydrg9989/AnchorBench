"""Shared utilities for the anchoring runner pipeline."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, TextIO

# ── answer parsing ────────────────────────────────────────────────────

_INT_PAT = re.compile(r"(?<![\d.])-?\d+(?![\d.])")


def parse_answer_int(
    raw_text: str, prompt_text: str = "",
) -> tuple[int | None, bool]:
    """Extract a single integer in [0, 100] from model output.

    1) Strip → exact integer only ⇒ accept if in range.
    2) Regex-extract all ints, keep those in [0,100].
    3) One unique candidate ⇒ accept.
    4) Multiple ⇒ subtract integers found in *prompt_text* (anchor echo),
       then accept iff exactly one remains.
    5) Never coerce failures.
    """
    text = raw_text.strip()

    if re.fullmatch(r"-?\d+", text):
        v = int(text)
        return (v, True) if 0 <= v <= 100 else (None, False)

    all_ints = [int(m.group()) for m in _INT_PAT.finditer(text)]
    candidates = [c for c in all_ints if 0 <= c <= 100]
    if not candidates:
        return None, False

    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0], True

    if prompt_text:
        prompt_nums = {
            int(m.group())
            for m in _INT_PAT.finditer(prompt_text)
            if 0 <= int(m.group()) <= 100
        }
        filtered = [c for c in unique if c not in prompt_nums]
        if len(filtered) == 1:
            return filtered[0], True

    return None, False


def parse_answer(raw_text: str) -> tuple[int | None, bool]:
    """Backward-compatible wrapper (no prompt dedup)."""
    return parse_answer_int(raw_text)


# ── dataset I/O ───────────────────────────────────────────────────────

def load_dataset(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_completed(
    path: Path,
) -> tuple[set[tuple], dict[tuple, int | None]]:
    """Load already-written results for resume.

    completed key  : (model_id, backend, item_id, condition, sample_idx)
    turn1_answers  : (model_id, backend, item_id, sample_idx) → answer_int
    """
    completed: set[tuple] = set()
    turn1_answers: dict[tuple, int | None] = {}
    if not path.exists():
        return completed, turn1_answers
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            be = r.get("backend", "unknown")
            completed.add(
                (r["model_id"], be, r["item_id"], r["condition"], r["sample_idx"])
            )
            if r["condition"] == "turn1":
                turn1_answers[
                    (r["model_id"], be, r["item_id"], r["sample_idx"])
                ] = r["answer_int"]
    return completed, turn1_answers


# ── result writing ────────────────────────────────────────────────────

def make_result(
    run_name: str,
    model_id: str,
    backend: str,
    item: dict,
    condition: str,
    sample_idx: int,
    answer_int: int | None,
    parsed_ok: bool,
    raw_text: str,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "model_id": model_id,
        "backend": backend,
        "item_id": item["item_id"],
        "suite": item["suite"],
        "subtype": item["subtype"],
        "field": item["field"],
        "domain": item["domain"],
        "condition": condition,
        "sample_idx": sample_idx,
        "answer_int": answer_int,
        "parsed_ok": parsed_ok,
        "raw_text": raw_text,
    }


def append_jsonl(fh: TextIO, record: dict) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


# ── retry / backoff ──────────────────────────────────────────────────

def retry_with_backoff(
    fn,
    max_retries: int = 8,
    base_wait: float = 1.0,
    max_wait: float = 120.0,
    retryable_status: tuple[int, ...] = (429, 500, 502, 503, 504),
):
    """Call *fn*; retry on retryable HTTP status with exponential backoff.

    *fn* must return a ``requests.Response``.
    """
    import requests

    for attempt in range(max_retries):
        try:
            resp = fn()
            if resp.status_code == 200:
                return resp
            if resp.status_code in retryable_status:
                wait = min(base_wait * 2 ** attempt, max_wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            wait = min(base_wait * 2 ** attempt, max_wait)
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {max_retries} retries")


# ── suite / condition constants ──────────────────────────────────────

EXTERNAL_LIKE_SUITES = frozenset(
    {"external", "icl", "conversation_history", "rag", "tool"}
)
CTRL_LOW_HIGH = ("control", "low_anchor", "high_anchor")
