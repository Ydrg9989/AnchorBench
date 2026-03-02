#!/usr/bin/env python3
"""Generate stress-test items for anchoring mitigation evaluation.

T1: Anchor magnitude extremes (5/95 instead of 20/80)
T2: Anchor position (front vs end)
T3: Anchor format (verbal instead of numeric)
T4: RAG snippet ranking (anchor snippet in position 1 vs 3)

Generates 16 items (4 per test) from existing dataset items.

Usage:
  python mitigation/generate_stress.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
from utils import load_dataset

FOOTER = "Answer with ONLY one integer from 0 to 100. No symbols. No words."
OUT_PATH = Path("poc_dataset/stress_test.jsonl")


def _pick_items(items: list[dict], suite: str, n: int) -> list[dict]:
    return [it for it in items if it["suite"] == suite][:n]


# ── T1: Anchor magnitude extremes ────────────────────────────────────

def generate_t1(items: list[dict]) -> list[dict]:
    """Replace anchors 20→5 and 80→95 in external items."""
    source = _pick_items(items, "external", 4)
    new_items = []
    for i, it in enumerate(source):
        new = {
            "item_id": f"T1-{i+1:04d}",
            "suite": "external",
            "subtype": "stress_t1_magnitude",
            "field": it["field"],
            "domain": it["domain"],
            "template_id": f"stress_t1_{i+1}",
            "answer_space": it["answer_space"],
            "anchors": {"low": 5, "high": 95, "controlled_history": []},
            "gold": it["gold"],
            "prompts": {
                "control": it["prompts"]["control"],
                "low_anchor": it["prompts"]["low_anchor"].replace(
                    "lower than 20%", "lower than 5%"
                ),
                "high_anchor": it["prompts"]["high_anchor"].replace(
                    "higher or lower than 80%", "higher or lower than 95%"
                ),
            },
            "meta": {"language": "en", "version": "stress", "seed": 42},
        }
        new_items.append(new)
    return new_items


# ── T2: Anchor position ──────────────────────────────────────────────

def generate_t2(items: list[dict]) -> list[dict]:
    """Create anchor-first and anchor-last variants for external items."""
    source = _pick_items(items, "external", 4)
    new_items = []
    anchor_re = re.compile(
        r"Before you estimate, consider: is .+? higher or lower than \d+%\?\s*"
    )
    for i, it in enumerate(source):
        for pos in ("front", "end"):
            for cond in ("low_anchor", "high_anchor"):
                prompt = it["prompts"][cond]
                m = anchor_re.search(prompt)
                if not m:
                    continue
                anchor_text = m.group().strip()
                without = prompt[: m.start()] + prompt[m.end() :]

                if pos == "front":
                    restructured = anchor_text + "\n\n" + without
                else:
                    restructured = without.replace(
                        FOOTER, anchor_text + "\n" + FOOTER
                    )
                it_prompts = it["prompts"].copy()
                it_prompts[cond] = restructured

        new = {
            "item_id": f"T2-{i+1:04d}",
            "suite": "external",
            "subtype": "stress_t2_position",
            "field": it["field"],
            "domain": it["domain"],
            "template_id": f"stress_t2_{i+1}",
            "answer_space": it["answer_space"],
            "anchors": it["anchors"],
            "gold": it["gold"],
            "prompts": {
                "control": it["prompts"]["control"],
                "low_anchor": _move_anchor(it["prompts"]["low_anchor"], "end"),
                "high_anchor": _move_anchor(it["prompts"]["high_anchor"], "end"),
                "low_anchor_front": _move_anchor(it["prompts"]["low_anchor"], "front"),
                "high_anchor_front": _move_anchor(it["prompts"]["high_anchor"], "front"),
            },
            "meta": {"language": "en", "version": "stress", "seed": 42},
        }
        new_items.append(new)
    return new_items


def _move_anchor(prompt: str, position: str) -> str:
    anchor_re = re.compile(
        r"Before you estimate, consider: is .+? higher or lower than \d+%\?\s*"
    )
    m = anchor_re.search(prompt)
    if not m:
        return prompt
    anchor_text = m.group().strip()
    without = prompt[: m.start()] + prompt[m.end() :]
    if position == "front":
        return anchor_text + "\n\n" + without
    else:
        return without.replace(FOOTER, anchor_text + "\n" + FOOTER)


# ── T3: Anchor format (verbal) ───────────────────────────────────────

VERBAL = {
    "20": "twenty",
    "80": "eighty",
}


def generate_t3(items: list[dict]) -> list[dict]:
    """Replace numeric anchors with verbal equivalents."""
    source = _pick_items(items, "external", 4)
    new_items = []
    for i, it in enumerate(source):
        new = {
            "item_id": f"T3-{i+1:04d}",
            "suite": "external",
            "subtype": "stress_t3_format",
            "field": it["field"],
            "domain": it["domain"],
            "template_id": f"stress_t3_{i+1}",
            "answer_space": it["answer_space"],
            "anchors": it["anchors"],
            "gold": it["gold"],
            "prompts": {
                "control": it["prompts"]["control"],
                "low_anchor": it["prompts"]["low_anchor"].replace(
                    "lower than 20%", "lower than around twenty percent"
                ),
                "high_anchor": it["prompts"]["high_anchor"].replace(
                    "lower than 80%", "lower than around eighty percent"
                ),
            },
            "meta": {"language": "en", "version": "stress", "seed": 42},
        }
        new_items.append(new)
    return new_items


# ── T4: RAG snippet ranking ──────────────────────────────────────────

def generate_t4(items: list[dict]) -> list[dict]:
    """Swap RAG snippet order: anchor snippet first vs last."""
    source = _pick_items(items, "rag", 4)
    new_items = []

    snippet_re = re.compile(
        r"(Retrieved snippet \(relevant\):.*?\n\n)"
        r"(Retrieved snippet \(additional\):.*?\n\n)",
        re.DOTALL,
    )

    for i, it in enumerate(source):
        prompts_new = {"control": it["prompts"]["control"]}
        for cond in ("low_anchor", "high_anchor"):
            prompt = it["prompts"][cond]
            m = snippet_re.search(prompt)
            if m:
                relevant = m.group(1)
                additional = m.group(2)
                swapped = additional + relevant
                prompts_new[cond] = prompt[: m.start()] + swapped + prompt[m.end() :]
            else:
                prompts_new[cond] = prompt

        new = {
            "item_id": f"T4-{i+1:04d}",
            "suite": "rag",
            "subtype": "stress_t4_rank",
            "field": it["field"],
            "domain": it["domain"],
            "template_id": f"stress_t4_{i+1}",
            "answer_space": it["answer_space"],
            "anchors": it["anchors"],
            "gold": it["gold"],
            "prompts": prompts_new,
            "meta": {"language": "en", "version": "stress", "seed": 42},
        }
        new_items.append(new)
    return new_items


# ── main ──────────────────────────────────────────────────────────────

def main():
    items = load_dataset("poc_dataset/poc_v0.2.jsonl")

    stress_items = []
    stress_items.extend(generate_t1(items))
    stress_items.extend(generate_t2(items))
    stress_items.extend(generate_t3(items))
    stress_items.extend(generate_t4(items))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for it in stress_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Generated {len(stress_items)} stress-test items → {OUT_PATH}")
    for prefix in ("T1", "T2", "T3", "T4"):
        n = sum(1 for it in stress_items if it["item_id"].startswith(prefix))
        print(f"  {prefix}: {n} items")


if __name__ == "__main__":
    main()
