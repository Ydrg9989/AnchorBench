#!/usr/bin/env python3
"""Validate poc_v0.2.jsonl — original items (A-/B-) + new items (ICL-/HIST-/RAG-/TOOL-)."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

FOOTER = "Answer with ONLY one integer from 0 to 100. No symbols. No words."

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

VALID_SUITES: dict[str, set[str]] = {
    "external":               {"semantic_two_step"},
    "self_generated":         {"draft_revise_evidence"},
    "icl":                    {"range_anchoring"},
    "conversation_history":   {"history_relevant", "history_irrelevant"},
    "rag":                    {"retrieval_snippet_anchor"},
    "tool":                   {"tool_output_anchor"},
}

NEW_SUITES = {"icl", "conversation_history", "rag", "tool"}

ID_PREFIX = {
    "external": "A-",
    "self_generated": "B-",
    "icl": "ICL-",
    "conversation_history": "HIST-",
    "rag": "RAG-",
    "tool": "TOOL-",
}

errors: list[str] = []


def err(item_id: str, msg: str) -> None:
    errors.append(f"[{item_id}] {msg}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _collect_prompts(item: dict) -> list[str]:
    p = item["prompts"]
    texts = []
    for k in ("control", "low_anchor", "high_anchor",
              "turn1", "turn2_fresh", "turn2_history"):
        if p.get(k) is not None:
            texts.append(p[k])
    for obj in p.get("turn2_controlled_history", []):
        texts.append(obj["prompt"])
    return texts


def _strip_mechanism(prompt: str, suite: str) -> str:
    """Remove the anchoring-mechanism block, returning target text + footer."""
    text = prompt

    if suite == "icl":
        marker = "Now estimate for a new case:\n"
        idx = text.find(marker)
        if idx >= 0:
            return text[idx + len(marker):]
        return text

    text = re.sub(r'^Chat history:\n.*?\n\n', '', text, flags=re.DOTALL)
    text = re.sub(
        r'^Retrieved snippet \(relevant\): .*?\n\nRetrieved snippet \(additional\): .*?\n\n',
        '', text, flags=re.DOTALL,
    )
    text = re.sub(r'^Tool output:\n\d+\n\n', '', text, flags=re.DOTALL)
    return text


def _extract_scenario_sentence(target_text: str) -> str | None:
    m = re.match(r'(.*?)\s*Five independent\s', target_text, re.DOTALL)
    return m.group(1).strip() if m else None


# ── Item-level validation ──────────────────────────────────────────────────────

def validate_item(item: dict) -> None:
    iid = item.get("item_id", "???")
    suite = item.get("suite", "")

    # -- keys --
    extra = set(item.keys()) - ALLOWED_KEYS
    if extra:
        err(iid, f"extra top-level keys: {extra}")
    missing = ALLOWED_KEYS - set(item.keys())
    if missing:
        err(iid, f"missing keys: {missing}")

    # -- suite / subtype --
    if suite not in VALID_SUITES:
        err(iid, f"unknown suite '{suite}'")
        return
    subtype = item.get("subtype", "")
    if subtype not in VALID_SUITES[suite]:
        err(iid, f"subtype '{subtype}' invalid for suite '{suite}'")

    # -- field / domain --
    field = item.get("field", "")
    if field not in FIELDS:
        err(iid, f"unknown field '{field}'")
    elif item.get("domain") not in DOMAINS.get(field, set()):
        err(iid, f"domain '{item.get('domain')}' invalid for field '{field}'")

    # -- id prefix --
    expected_pfx = ID_PREFIX.get(suite, "")
    if expected_pfx and not iid.startswith(expected_pfx):
        err(iid, f"item_id should start with '{expected_pfx}'")

    # -- answer_space --
    if item.get("answer_space") != {"type": "int", "min": 0, "max": 100}:
        err(iid, f"unexpected answer_space")

    # -- footer in every non-null prompt --
    for pt in _collect_prompts(item):
        if not pt.rstrip().endswith(FOOTER):
            err(iid, f"prompt does not end with footer: ...{pt[-60:]}")

    # -- suite-specific --
    if suite in NEW_SUITES:
        _validate_new(item)
    elif suite == "external":
        _validate_external(item)
    elif suite == "self_generated":
        _validate_self(item)


# ── Original-suite checks (carried over) ──────────────────────────────────────

def _validate_external(item: dict) -> None:
    iid = item["item_id"]
    p = item["prompts"]
    anch = item["anchors"]
    gold = item["gold"]

    for k in ("control", "low_anchor", "high_anchor"):
        if p.get(k) is None:
            err(iid, f"external missing prompts.{k}")
    for k in ("turn1", "turn2_fresh", "turn2_history"):
        if p.get(k) is not None:
            err(iid, f"external has non-null prompts.{k}")
    if p.get("turn2_controlled_history"):
        err(iid, "external has non-empty turn2_controlled_history")

    if anch.get("low") != 20 or anch.get("high") != 80:
        err(iid, f"external anchors wrong: {anch}")
    if anch.get("controlled_history"):
        err(iid, "external has non-empty controlled_history")

    if gold.get("y_star") is None:
        err(iid, "external has null y_star")
    if gold.get("y_star_stage1") is not None or gold.get("y_star_stage2") is not None:
        err(iid, "external has non-null stage gold values")


def _validate_self(item: dict) -> None:
    iid = item["item_id"]
    p = item["prompts"]
    anch = item["anchors"]
    gold = item["gold"]

    for k in ("turn1", "turn2_fresh", "turn2_history"):
        if p.get(k) is None:
            err(iid, f"self missing prompts.{k}")
    for k in ("control", "low_anchor", "high_anchor"):
        if p.get(k) is not None:
            err(iid, f"self has non-null prompts.{k}")

    if p.get("turn2_history") and "{{A1}}" not in p["turn2_history"]:
        err(iid, "turn2_history missing '{{A1}}' placeholder")

    if anch.get("low") is not None or anch.get("high") is not None:
        err(iid, "self has non-null low/high anchors")

    if gold.get("y_star") is not None:
        err(iid, "self has non-null y_star")
    if gold.get("y_star_stage1") is None or gold.get("y_star_stage2") is None:
        err(iid, "self has null stage gold values")

    ctrl = anch.get("controlled_history", [])
    ctrl_prompts = p.get("turn2_controlled_history", [])
    if ctrl:
        if sorted(ctrl) != [10, 30, 50, 70, 90]:
            err(iid, f"controlled_history anchors wrong: {ctrl}")
        if len(ctrl_prompts) != 5:
            err(iid, f"expected 5 controlled prompts, got {len(ctrl_prompts)}")


# ── New-suite checks ───────────────────────────────────────────────────────────

def _validate_new(item: dict) -> None:
    iid = item["item_id"]
    suite = item["suite"]
    p = item["prompts"]
    anch = item["anchors"]
    gold = item["gold"]

    for k in ("control", "low_anchor", "high_anchor"):
        if p.get(k) is None:
            err(iid, f"new item missing prompts.{k}")
            return
    for k in ("turn1", "turn2_fresh", "turn2_history"):
        if p.get(k) is not None:
            err(iid, f"new item has non-null prompts.{k}")
    if p.get("turn2_controlled_history"):
        err(iid, "new item has non-empty turn2_controlled_history")

    if anch.get("low") != 20 or anch.get("high") != 80:
        err(iid, f"new item anchors wrong: low={anch.get('low')} high={anch.get('high')}")
    if anch.get("controlled_history"):
        err(iid, "new item has non-empty controlled_history")

    if gold.get("y_star") is None:
        err(iid, "new item has null y_star")
    if gold.get("y_star_stage1") is not None or gold.get("y_star_stage2") is not None:
        err(iid, "new item has non-null stage gold values")

    # -- target-text equality across conditions --
    targets = {}
    for cond in ("control", "low_anchor", "high_anchor"):
        stripped = _strip_mechanism(p[cond], suite)
        targets[cond] = stripped.strip()

    if not (targets["control"] == targets["low_anchor"] == targets["high_anchor"]):
        err(iid, "target text differs across control/low/high conditions")

    # -- scenario-sentence digit check --
    footer_idx = targets["control"].rfind("\n" + FOOTER)
    target_body = targets["control"][:footer_idx] if footer_idx >= 0 else targets["control"]
    scn_sentence = _extract_scenario_sentence(target_body)
    if scn_sentence and re.search(r'\d', scn_sentence):
        err(iid, f"scenario sentence contains forbidden digits: {scn_sentence[:80]}")

    # -- ICL-specific checks --
    if suite == "icl":
        _validate_icl_demos(item)

    # -- HIST-specific checks --
    if suite == "conversation_history":
        _validate_hist(item)

    # -- RAG-specific checks --
    if suite == "rag":
        _validate_rag(item)

    # -- TOOL-specific checks --
    if suite == "tool":
        _validate_tool(item)


def _validate_icl_demos(item: dict) -> None:
    iid = item["item_id"]
    p = item["prompts"]

    for cond in ("control", "low_anchor", "high_anchor"):
        prompt = p[cond]

        marker = "Now estimate for a new case:\n"
        marker_idx = prompt.find(marker)
        if marker_idx < 0:
            err(iid, f"ICL {cond}: missing 'Now estimate for a new case:' marker")
            continue

        demo_section = prompt[:marker_idx]

        answers = re.findall(r'\n(\d+)\s*$', demo_section, re.MULTILINE)
        if len(answers) != 3:
            err(iid, f"ICL {cond}: found {len(answers)} demo answers, expected 3")
            continue

        for a_str in answers:
            v = int(a_str)
            if v < 0 or v > 100:
                err(iid, f"ICL {cond}: demo answer {v} out of [0,100]")

        demo_blocks = re.findall(
            r'Example \d+:\n(.*?)(?=\n\d+\s*$)', demo_section, re.DOTALL | re.MULTILINE
        )
        for di, db in enumerate(demo_blocks):
            scn = _extract_scenario_sentence(db.strip())
            if scn and re.search(r'\d', scn):
                err(iid, f"ICL {cond} demo {di+1} scenario has digits: {scn[:60]}")


def _validate_hist(item: dict) -> None:
    iid = item["item_id"]
    p = item["prompts"]
    subtype = item["subtype"]

    if "Chat history:" in p["control"]:
        err(iid, "HIST control should have no Chat history block")

    for cond in ("low_anchor", "high_anchor"):
        if "Chat history:" not in p[cond]:
            err(iid, f"HIST {cond} missing 'Chat history:' block")

        expected_anchor = "20" if cond == "low_anchor" else "80"
        hist_match = re.search(r'Chat history:\n(.*?)\n\n', p[cond], re.DOTALL)
        if hist_match:
            hist_text = hist_match.group(1)
            if expected_anchor not in hist_text:
                err(iid, f"HIST {cond}: history block missing anchor {expected_anchor}")
            forbidden = re.findall(r'\d+', hist_text)
            for d in forbidden:
                if d not in (expected_anchor, ):
                    err(iid, f"HIST {cond}: history block contains forbidden digit '{d}'")


def _validate_rag(item: dict) -> None:
    iid = item["item_id"]
    p = item["prompts"]

    for cond in ("control", "low_anchor", "high_anchor"):
        if "Retrieved snippet (relevant):" not in p[cond]:
            err(iid, f"RAG {cond}: missing 'Retrieved snippet (relevant):' block")
        if "Retrieved snippet (additional):" not in p[cond]:
            err(iid, f"RAG {cond}: missing 'Retrieved snippet (additional):' block")

    rel_match = re.search(r'Retrieved snippet \(relevant\): (.+?)(?:\n\n)', p["control"], re.DOTALL)
    if rel_match:
        rel_text = rel_match.group(1).strip()
        if re.search(r'\d', rel_text):
            err(iid, f"RAG relevant snippet contains digits: {rel_text[:60]}")

    add_ctrl = re.search(r'Retrieved snippet \(additional\): (.+?)(?:\n\n)', p["control"], re.DOTALL)
    if add_ctrl:
        add_text = add_ctrl.group(1).strip()
        if re.search(r'\d', add_text):
            err(iid, "RAG control additional snippet should have no digits")

    for cond, expected in [("low_anchor", "20"), ("high_anchor", "80")]:
        add_m = re.search(r'Retrieved snippet \(additional\): (.+?)(?:\n\n)', p[cond], re.DOTALL)
        if add_m:
            add_text = add_m.group(1).strip()
            if expected not in add_text:
                err(iid, f"RAG {cond}: additional snippet missing anchor {expected}")


def _validate_tool(item: dict) -> None:
    iid = item["item_id"]
    p = item["prompts"]

    if "Tool output:" in p["control"]:
        err(iid, "TOOL control should have no Tool output block")

    for cond, expected in [("low_anchor", "20"), ("high_anchor", "80")]:
        if "Tool output:" not in p[cond]:
            err(iid, f"TOOL {cond}: missing 'Tool output:' block")
        else:
            tool_m = re.search(r'Tool output:\n(\d+)', p[cond])
            if tool_m:
                if tool_m.group(1) != expected:
                    err(iid, f"TOOL {cond}: expected anchor {expected}, "
                        f"got {tool_m.group(1)}")
            else:
                err(iid, f"TOOL {cond}: could not parse Tool output value")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    path = Path(__file__).resolve().parent / "poc_v0.2.jsonl"
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                err(f"line-{lineno}", f"invalid JSON: {e}")
                continue
            items.append(obj)

    print(f"Loaded {len(items)} items from {path.name}")

    for item in items:
        validate_item(item)

    # -- item_id uniqueness --
    ids = [it.get("item_id", "???") for it in items]
    for dup_id, cnt in Counter(ids).items():
        if cnt > 1:
            err(dup_id, f"duplicate item_id (appears {cnt} times)")

    # ── Balance checks ─────────────────────────────────────────────────────────

    suite_cnt = Counter(it["suite"] for it in items)

    expected_counts = {
        "external": 30, "self_generated": 30,
        "icl": 12, "conversation_history": 12, "rag": 12, "tool": 12,
    }
    for s, exp in expected_counts.items():
        if suite_cnt.get(s, 0) != exp:
            err("BALANCE", f"expected {exp} {s} items, got {suite_cnt.get(s, 0)}")

    if len(items) != 108:
        err("BALANCE", f"expected 108 total items, got {len(items)}")

    for s in ("icl", "rag", "tool"):
        suite_items = [it for it in items if it["suite"] == s]
        fc = Counter(it["field"] for it in suite_items)
        for fld in FIELDS:
            if fc.get(fld, 0) != 4:
                err("BALANCE", f"{s}/{fld}: {fc.get(fld,0)} items, expected 4")
        dc = Counter((it["field"], it["domain"]) for it in suite_items)
        for fld in FIELDS:
            for dom in DOMAINS[fld]:
                if dc.get((fld, dom), 0) != 2:
                    err("BALANCE", f"{s}/{fld}/{dom}: {dc.get((fld,dom),0)} items, expected 2")

    hist_items = [it for it in items if it["suite"] == "conversation_history"]
    hf = Counter(it["field"] for it in hist_items)
    for fld in FIELDS:
        if hf.get(fld, 0) != 4:
            err("BALANCE", f"conversation_history/{fld}: {hf.get(fld,0)} items, expected 4")
    hs = Counter(it["subtype"] for it in hist_items)
    if hs.get("history_relevant", 0) != 6:
        err("BALANCE", f"history_relevant: {hs.get('history_relevant',0)}, expected 6")
    if hs.get("history_irrelevant", 0) != 6:
        err("BALANCE", f"history_irrelevant: {hs.get('history_irrelevant',0)}, expected 6")
    for fld in FIELDS:
        fld_hist = [it for it in hist_items if it["field"] == fld]
        rel = sum(1 for it in fld_hist if it["subtype"] == "history_relevant")
        irr = sum(1 for it in fld_hist if it["subtype"] == "history_irrelevant")
        if rel != 2:
            err("BALANCE", f"hist/{fld}/relevant: {rel}, expected 2")
        if irr != 2:
            err("BALANCE", f"hist/{fld}/irrelevant: {irr}, expected 2")

    hd = Counter((it["field"], it["domain"]) for it in hist_items)
    for fld in FIELDS:
        for dom in DOMAINS[fld]:
            if hd.get((fld, dom), 0) != 2:
                err("BALANCE", f"hist/{fld}/{dom}: {hd.get((fld,dom),0)}, expected 2")

    # ── Output ─────────────────────────────────────────────────────────────────

    if errors:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"VALIDATION FAILED — {len(errors)} error(s):", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(f"Total items: {len(items)}")
    for s in ("external", "self_generated", "icl", "conversation_history", "rag", "tool"):
        si = [it for it in items if it["suite"] == s]
        print(f"  {s}: {len(si)}")
        if s in NEW_SUITES:
            for fld in FIELDS:
                fi = [it for it in si if it["field"] == fld]
                doms = "  ".join(
                    f"{d}={sum(1 for it in fi if it['domain']==d)}"
                    for d in sorted(DOMAINS[fld])
                )
                print(f"    {fld}: {len(fi)}  ({doms})")
            if s == "conversation_history":
                for st in ("history_relevant", "history_irrelevant"):
                    print(f"    {st}: {sum(1 for it in si if it['subtype']==st)}")
    print(f"\nAll {len(items)} items passed validation.")


if __name__ == "__main__":
    main()
