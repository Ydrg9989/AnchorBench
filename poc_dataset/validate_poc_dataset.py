#!/usr/bin/env python3
"""Validate poc.jsonl against the schema and additional integrity checks."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ANSWER_FMT = "Answer with ONLY one integer from 0 to 100. No symbols. No words."

ALLOWED_KEYS = {
    "item_id", "suite", "subtype", "field", "domain", "template_id",
    "answer_space", "anchors", "gold", "prompts", "meta",
}

FIELDS = ["health", "finance", "ops"]
DOMAINS = {
    "health": {"disease_prevalence_estimation", "treatment_success_risk"},
    "finance": {"credit_default_risk", "fraud_risk"},
    "ops": {"project_delay_risk", "qa_defect_rate"},
}

SUITE_SUBTYPE = {
    "external": "semantic_two_step",
    "self_generated": "draft_revise_evidence",
}


def fail(item_id: str, msg: str) -> None:
    print(f"FAIL [{item_id}]: {msg}", file=sys.stderr)
    sys.exit(1)


def collect_all_prompts(item: dict) -> list[str]:
    """Return every non-null prompt string in this item."""
    p = item["prompts"]
    texts = []
    for k in ("control", "low_anchor", "high_anchor",
              "turn1", "turn2_fresh", "turn2_history"):
        if p[k] is not None:
            texts.append(p[k])
    for obj in p["turn2_controlled_history"]:
        texts.append(obj["prompt"])
    return texts


def validate_item(item: dict) -> None:
    iid = item.get("item_id", "???")

    # only allowed keys
    extra = set(item.keys()) - ALLOWED_KEYS
    if extra:
        fail(iid, f"extra top-level keys: {extra}")

    # suite / subtype match
    suite = item["suite"]
    if suite not in SUITE_SUBTYPE:
        fail(iid, f"unknown suite '{suite}'")
    if item["subtype"] != SUITE_SUBTYPE[suite]:
        fail(iid, f"subtype mismatch for suite '{suite}'")

    # field / domain
    field = item["field"]
    if field not in FIELDS:
        fail(iid, f"unknown field '{field}'")
    if item["domain"] not in DOMAINS[field]:
        fail(iid, f"domain '{item['domain']}' invalid for field '{field}'")

    # answer_space
    asp = item["answer_space"]
    if asp != {"type": "int", "min": 0, "max": 100}:
        fail(iid, f"unexpected answer_space: {asp}")

    # ── prompt checks ──
    all_prompts = collect_all_prompts(item)
    for pt in all_prompts:
        if ANSWER_FMT not in pt:
            fail(iid, f"prompt missing answer-format line:\n{pt[:120]}...")
        if "theta" in pt.lower():
            fail(iid, "prompt contains the word 'theta'")
        if "y_star" in pt:
            fail(iid, "prompt contains 'y_star'")
        if re.search(r"\[(\d+,\s*){2,}", pt):
            fail(iid, "prompt appears to contain raw numeric signal list")

    # ── suite-specific checks ──
    p = item["prompts"]
    anch = item["anchors"]
    gold = item["gold"]

    if suite == "external":
        for k in ("control", "low_anchor", "high_anchor"):
            if p[k] is None:
                fail(iid, f"external item missing prompts.{k}")
        for k in ("turn1", "turn2_fresh", "turn2_history"):
            if p[k] is not None:
                fail(iid, f"external item has non-null prompts.{k}")
        if p["turn2_controlled_history"]:
            fail(iid, "external item has non-empty turn2_controlled_history")

        if anch["low"] != 20 or anch["high"] != 80:
            fail(iid, f"external anchors wrong: {anch}")
        if anch["controlled_history"]:
            fail(iid, "external item has non-empty controlled_history")

        if gold["y_star"] is None:
            fail(iid, "external item has null y_star")
        if gold["y_star_stage1"] is not None or gold["y_star_stage2"] is not None:
            fail(iid, "external item has non-null stage gold values")

    elif suite == "self_generated":
        for k in ("turn1", "turn2_fresh", "turn2_history"):
            if p[k] is None:
                fail(iid, f"self item missing prompts.{k}")
        for k in ("control", "low_anchor", "high_anchor"):
            if p[k] is not None:
                fail(iid, f"self item has non-null prompts.{k}")

        if "{{A1}}" not in p["turn2_history"]:
            fail(iid, "turn2_history missing '{{A1}}' placeholder")

        if anch["low"] is not None or anch["high"] is not None:
            fail(iid, "self item has non-null low/high anchors")

        if gold["y_star"] is not None:
            fail(iid, "self item has non-null y_star")
        if gold["y_star_stage1"] is None or gold["y_star_stage2"] is None:
            fail(iid, "self item has null stage gold values")

        ctrl = anch["controlled_history"]
        ctrl_prompts = p["turn2_controlled_history"]
        if ctrl:
            if sorted(ctrl) != [10, 30, 50, 70, 90]:
                fail(iid, f"controlled_history anchors wrong: {ctrl}")
            if len(ctrl_prompts) != 5:
                fail(iid, f"expected 5 controlled-history prompts, got {len(ctrl_prompts)}")
            got_anchors = sorted(obj["anchor"] for obj in ctrl_prompts)
            if got_anchors != [10, 30, 50, 70, 90]:
                fail(iid, f"controlled-history prompt anchors wrong: {got_anchors}")
        else:
            if ctrl_prompts:
                fail(iid, "controlled_history empty but prompts present")


def main() -> None:
    path = Path(__file__).resolve().parent / "poc.jsonl"
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"FAIL line {lineno}: invalid JSON – {e}", file=sys.stderr)
                sys.exit(1)
            items.append(obj)

    print(f"Loaded {len(items)} items from {path.name}")

    for item in items:
        validate_item(item)

    # ── balance checks ──
    suite_cnt = Counter(it["suite"] for it in items)
    for s, expected in [("external", 30), ("self_generated", 30)]:
        if suite_cnt[s] != expected:
            print(f"FAIL: expected {expected} {s} items, got {suite_cnt[s]}",
                  file=sys.stderr)
            sys.exit(1)

    field_cnt = Counter((it["suite"], it["field"]) for it in items)
    for s in ("external", "self_generated"):
        for fld in FIELDS:
            n = field_cnt[(s, fld)]
            if n != 10:
                print(f"FAIL: {s}/{fld} has {n} items, expected 10",
                      file=sys.stderr)
                sys.exit(1)

    domain_cnt = Counter((it["suite"], it["field"], it["domain"]) for it in items)
    for s in ("external", "self_generated"):
        for fld in FIELDS:
            for dom in DOMAINS[fld]:
                n = domain_cnt[(s, fld, dom)]
                if n != 5:
                    print(f"FAIL: {s}/{fld}/{dom} has {n} items, expected 5",
                          file=sys.stderr)
                    sys.exit(1)

    ctrl_items = [it for it in items if it["anchors"]["controlled_history"]]
    if len(ctrl_items) != 10:
        print(f"FAIL: expected 10 controlled-history items, got {len(ctrl_items)}",
              file=sys.stderr)
        sys.exit(1)

    ctrl_fields = Counter(it["field"] for it in ctrl_items)
    for fld in FIELDS:
        n = ctrl_fields[fld]
        if n < 3 or n > 4:
            print(f"FAIL: controlled-history {fld} has {n} items (need 3-4)",
                  file=sys.stderr)
            sys.exit(1)

    # ── summary ──
    print("\n=== Validation Summary ===")
    print(f"Total items: {len(items)}")
    for s in ("external", "self_generated"):
        print(f"  {s}: {suite_cnt[s]}")
        for fld in FIELDS:
            doms = "  ".join(
                f"{d}={domain_cnt[(s, fld, d)]}"
                for d in sorted(DOMAINS[fld])
            )
            print(f"    {fld}: {field_cnt[(s, fld)]}  ({doms})")
    print(f"  controlled-history items: {len(ctrl_items)}")
    print(f"    per field: { {f: ctrl_fields[f] for f in FIELDS} }")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
