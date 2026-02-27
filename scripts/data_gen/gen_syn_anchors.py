#!/usr/bin/env python3
"""Generate a SynAnchors-style numerical anchoring dataset via OpenRouter.

Calls the LLM to draft ~6 candidates per topic (10 topics), validates each
against strict constraints, selects the best 4 per topic, and writes 40
items as JSONL.

Usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python scripts/data_gen/gen_syn_anchors.py \
    --model openai/gpt-4o-mini \
    --out data/processed/syn_anchors_v0/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from llm_anchoring.openrouter_client import chat, extract_content

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOPICS = [
    "Geography & distances",
    "Population & demographics",
    "Physics measurements",
    "Chemistry/material properties",
    "Weather/climate normals",
    "Transportation/travel time or distance",
    "Sports stats",
    "Economics/simple finance quantities",
    "Nutrition facts",
    "Computing/hardware specs",
]

ITEMS_PER_TOPIC = 4
CANDIDATES_PER_TOPIC = 6

SYSTEM_PROMPT = """\
You are a dataset engineer building a cognitive-anchoring benchmark for LLMs.
You produce STRICT JSON arrays with no markdown fences and no commentary outside the JSON.
Every item must be a numeric question with exactly one scalar answer."""

TOPIC_PROMPT_TEMPLATE = """\
Generate exactly {n_cand} candidate items for the topic: "{topic}".

Each item is a numeric question with a single scalar answer (one number) and a unit.
Two conditions exist for evaluation:
  (1) WITH-ANCHOR: prepend an anchor_text containing an irrelevant random number.
  (2) WITHOUT-ANCHOR: the same question alone.

Return a JSON array of {n_cand} objects. Each object must follow this schema exactly:
{{
  "topic": "{topic}",
  "question": "...",
  "unit": "...",
  "true_value": <float>,
  "answer_precision": "int|1dp|2dp",
  "anchor_value": <float>,
  "anchor_text": "...",
  "notes": "...",
  "difficulty": "easy|medium|hard"
}}

Strict constraints you MUST satisfy for every item:
1) Single scalar: the question must have ONE numeric answer only (no ranges, no multiple values).
2) Unit clarity: the question clearly states the unit; true_value uses that unit.
3) No leakage: anchor_text must NOT reveal, approximate, or hint at true_value.
4) Irrelevance: anchor_text must be about something incidental (receipt total, bus number,
   room number, jersey number, a random code, etc.) that is NOT connected to the quantity.
5) Anchor distance: anchor_value must be sufficiently different from true_value:
   abs(anchor_value - true_value) / max(abs(true_value), 1e-9) >= 0.30
6) The question must NOT contain anchor_value.
7) Keep values realistic (no absurd magnitudes).
8) Avoid iconic trivia that LLMs memorize too easily; prefer plausible but non-famous values.
9) Vary units and magnitudes across items.
10) Keep anchor_text to 1-2 sentences and clearly unrelated.
11) Keep questions neutral and factual in tone.

Return ONLY the JSON array, no markdown fences, no other text."""


def _parse_json_array(text: str) -> list[dict] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


REQUIRED_KEYS = {
    "topic", "question", "unit", "true_value", "answer_precision",
    "anchor_value", "anchor_text", "notes", "difficulty",
}


def validate_item(item: dict) -> list[str]:
    """Return list of error strings; empty means valid."""
    errors = []
    missing = REQUIRED_KEYS - set(item.keys())
    if missing:
        errors.append(f"missing keys: {missing}")
        return errors

    tv = item["true_value"]
    av = item["anchor_value"]
    if not isinstance(tv, (int, float)):
        errors.append(f"true_value not numeric: {tv}")
        return errors
    if not isinstance(av, (int, float)):
        errors.append(f"anchor_value not numeric: {av}")
        return errors

    denom = max(abs(tv), 1e-9)
    rel_dist = abs(av - tv) / denom
    if rel_dist < 0.30:
        errors.append(f"anchor too close: rel_dist={rel_dist:.3f}")

    q = item["question"].lower()
    av_str = str(av)
    av_int_str = str(int(av)) if av == int(av) else None
    if av_str in q or (av_int_str and av_int_str in q):
        errors.append("question contains anchor_value")

    if item["answer_precision"] not in ("int", "1dp", "2dp"):
        errors.append(f"bad answer_precision: {item['answer_precision']}")
    if item["difficulty"] not in ("easy", "medium", "hard"):
        errors.append(f"bad difficulty: {item['difficulty']}")

    at = item["anchor_text"].lower()
    tv_str_int = str(int(tv)) if tv == int(tv) else None
    tv_str = f"{tv:.1f}"
    if tv_str in at or (tv_str_int and tv_str_int in at and len(tv_str_int) >= 3):
        errors.append("anchor_text may leak true_value")

    return errors


def generate_topic_candidates(
    topic: str, model: str, cache_dir: Path
) -> list[dict]:
    """Call the LLM once for a topic, return validated candidates."""
    prompt = TOPIC_PROMPT_TEMPLATE.format(
        topic=topic, n_cand=CANDIDATES_PER_TOPIC
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    resp = chat(messages, model=model, temperature=0.7, cache_dir=cache_dir)
    text = extract_content(resp)
    items = _parse_json_array(text)
    if items is None:
        logger.error("Failed to parse JSON for topic '%s'", topic)
        return []

    valid = []
    for item in items:
        errs = validate_item(item)
        if errs:
            logger.warning("  reject: %s — %s", item.get("question", "?")[:60], errs)
        else:
            valid.append(item)
    return valid


def select_best(candidates: list[dict], n: int) -> list[dict]:
    """Select n items maximizing unit diversity."""
    if len(candidates) <= n:
        return candidates

    selected = []
    used_units: set[str] = set()
    for item in candidates:
        u = item["unit"].lower()
        if u not in used_units:
            selected.append(item)
            used_units.add(u)
            if len(selected) == n:
                return selected

    for item in candidates:
        if item not in selected:
            selected.append(item)
            if len(selected) == n:
                return selected
    return selected


def backfill_topic(
    topic: str, have: int, need: int, model: str, cache_dir: Path
) -> list[dict]:
    """Generate extra candidates for a topic that came up short."""
    shortfall = need - have
    prompt = TOPIC_PROMPT_TEMPLATE.format(
        topic=topic, n_cand=shortfall + 2
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt + "\n\nGenerate DIFFERENT items from any you have produced before."},
    ]
    resp = chat(
        messages, model=model, temperature=0.9,
        cache_dir=cache_dir, use_cache=False,
    )
    text = extract_content(resp)
    items = _parse_json_array(text)
    if items is None:
        return []
    return [it for it in items if not validate_item(it)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/syn_anchors_v0/dataset.jsonl"))
    p.add_argument("--cache_dir", type=Path,
                   default=Path("data/interim/syn_anchors_v0/cache"))
    args = p.parse_args()

    all_items: list[dict] = []

    for ti, topic in enumerate(TOPICS):
        logger.info("[%d/%d] Generating candidates for: %s", ti + 1, len(TOPICS), topic)
        cands = generate_topic_candidates(topic, args.model, args.cache_dir)
        logger.info("  %d valid candidates", len(cands))

        if len(cands) < ITEMS_PER_TOPIC:
            logger.info("  backfilling %d items...", ITEMS_PER_TOPIC - len(cands))
            extra = backfill_topic(topic, len(cands), ITEMS_PER_TOPIC,
                                   args.model, args.cache_dir)
            cands.extend(extra)
            logger.info("  now %d candidates", len(cands))

        selected = select_best(cands, ITEMS_PER_TOPIC)
        for si, item in enumerate(selected):
            item["id"] = f"num_{ti * ITEMS_PER_TOPIC + si + 1:04d}"
            item["topic"] = topic
        all_items.extend(selected)

    logger.info("Total items: %d (target: %d)", len(all_items), len(TOPICS) * ITEMS_PER_TOPIC)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")
    logger.info("Saved to %s", args.out)

    n_fail = 0
    for item in all_items:
        errs = validate_item(item)
        if errs:
            logger.error("FINAL VALIDATION FAIL: %s — %s", item["id"], errs)
            n_fail += 1
    if n_fail:
        logger.error("%d items failed final validation!", n_fail)
    else:
        logger.info("All %d items pass validation.", len(all_items))


if __name__ == "__main__":
    main()
