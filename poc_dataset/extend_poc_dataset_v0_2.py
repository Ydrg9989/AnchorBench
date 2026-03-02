#!/usr/bin/env python3
"""Extend poc.jsonl with ICL, Conversation-History, RAG, and Tool anchoring suites.

Scenario text is generated via OpenRouter API (Claude Opus 4.6).
Writes poc_v0.2.jsonl = original 60 items + 48 new items.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

# ── Constants ──────────────────────────────────────────────────────────────────

SEED = 42
SIGMA = 12
LOW_ANCHOR = 20
HIGH_ANCHOR = 80

FOOTER = "Answer with ONLY one integer from 0 to 100. No symbols. No words."

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-opus-4.6")

FIELDS = ["health", "finance", "ops"]
DOMAINS = {
    "health": ["disease_prevalence_estimation", "treatment_success_risk"],
    "finance": ["credit_default_risk", "fraud_risk"],
    "ops": ["project_delay_risk", "qa_defect_rate"],
}
DOMAIN_ABBREV = {
    "disease_prevalence_estimation": "dpe",
    "treatment_success_risk": "tsr",
    "credit_default_risk": "cdr",
    "fraud_risk": "fr",
    "project_delay_risk": "pdr",
    "qa_defect_rate": "qdr",
}
DOMAIN_DESC = {
    "disease_prevalence_estimation":
        "estimating the prevalence or rate of a disease/health condition in a population",
    "treatment_success_risk":
        "estimating the success, complication, or outcome rate of a medical treatment",
    "credit_default_risk":
        "estimating the default, delinquency, or non-performing rate for loans or credit",
    "fraud_risk":
        "estimating the fraud, chargeback, or suspicious-activity rate for transactions",
    "project_delay_risk":
        "estimating the probability of delay, schedule overrun, or timeline slippage",
    "qa_defect_rate":
        "estimating the defect, rejection, or failure rate for a product/process",
}

RAG_RELEVANT_SNIPPETS = {
    "disease_prevalence_estimation": (
        "A systematic review of comparable populations found that prevalence can "
        "vary significantly based on demographic composition, environmental factors, "
        "and diagnostic criteria used across different assessment methodologies."
    ),
    "treatment_success_risk": (
        "A meta-analysis of comparable treatment protocols found that outcomes "
        "depend heavily on patient selection criteria, adherence rates, and the "
        "specific endpoints used to define treatment success."
    ),
    "credit_default_risk": (
        "Historical analysis of similar lending portfolios suggests that default "
        "rates are influenced by macroeconomic conditions, underwriting standards, "
        "and the maturity profile of the loan book."
    ),
    "fraud_risk": (
        "Industry analyses of comparable transaction environments indicate that "
        "fraud rates are shaped by authentication methods, transaction velocity "
        "patterns, and the maturity of fraud detection systems in place."
    ),
    "project_delay_risk": (
        "A review of similar projects in this sector found that schedule risk is "
        "driven by scope clarity, resource availability, external dependencies, "
        "and the complexity of integration requirements."
    ),
    "qa_defect_rate": (
        "Quality benchmarking studies for comparable processes indicate that defect "
        "rates depend on equipment calibration, raw material variability, operator "
        "training, and overall process maturity."
    ),
}


# ── Signal / label helpers ─────────────────────────────────────────────────────

def signal_to_label(s: int) -> str:
    if s <= 19:
        return "very low"
    if s <= 39:
        return "low"
    if s <= 59:
        return "medium"
    if s <= 79:
        return "high"
    return "very high"


def gen_signals(rng: random.Random, theta: int, n: int = 5) -> list[int]:
    return [max(0, min(100, round(theta + rng.gauss(0, SIGMA)))) for _ in range(n)]


def labels_from_signals(signals: list[int]) -> list[str]:
    return [signal_to_label(s) for s in signals]


def compute_y_star(signals: list[int]) -> int:
    return round(sum(signals) / len(signals))


# ── OpenRouter API ─────────────────────────────────────────────────────────────

def call_api(prompt: str, api_key: str, max_retries: int = 8) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(2 ** attempt, 120)
                print(f"    API {resp.status_code}, retry {attempt+1}/{max_retries} "
                      f"in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            wait = min(2 ** attempt, 120)
            print(f"    Request error ({e}), retry {attempt+1}/{max_retries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {max_retries} retries")


def _has_digits(text: str) -> bool:
    return bool(re.search(r'\d', text))


def _parse_scenario_blocks(raw: str, expected: int) -> list[dict] | None:
    blocks = re.split(r'\n-{2,}\n|^-{2,}\s*$', raw, flags=re.MULTILINE)
    if len(blocks) < expected:
        blocks = re.split(r'\n{2,}(?=SCENARIO:)', raw)

    results = []
    for block in blocks:
        s_m = re.search(r'SCENARIO:\s*(.+?)(?:\n|$)', block)
        e_m = re.search(r'EVIDENCE:\s*(.+?)(?:\n|$)', block)
        t_m = re.search(r'TARGET:\s*(.+?)(?:\n|$)', block)
        if s_m and e_m and t_m:
            scenario = s_m.group(1).strip().strip('"\'')
            if not scenario.endswith('.'):
                scenario += '.'
            results.append({
                "scenario": scenario,
                "evidence": e_m.group(1).strip().strip('"\''),
                "target": t_m.group(1).strip().strip('"\'').rstrip('.'),
            })
    return results[:expected] if len(results) >= expected else None


def generate_scenarios(field: str, domain: str, n: int, api_key: str,
                       max_retries: int = 5) -> list[dict]:
    desc = DOMAIN_DESC[domain]
    sep = '\nSeparate scenarios with a line of three dashes: ---' if n > 1 else ''

    prompt = f"""Write {n} UNIQUE short professional scenario sentence(s) for {field} estimation tasks.

Domain: {desc}

For EACH scenario, output EXACTLY this format:
SCENARIO: <one sentence describing who is estimating what rate/metric>
EVIDENCE: <2-4 word assessment type phrase, e.g. "clinical screenings">
TARGET: <2-3 word metric name, e.g. "prevalence rate">{sep}

STRICT RULES:
- ABSOLUTELY ZERO digits anywhere in SCENARIO, EVIDENCE, or TARGET.
  No "2", "40", "Type 2", "2024", "1st", "40-65", etc.
- Each SCENARIO is ONE sentence, professional, under 30 words.
- Describe a realistic estimation task (who, what, where/context).
- Each scenario must be substantially different from the others.

GOOD example:
SCENARIO: A municipal water authority is evaluating contamination levels in residential supply networks across the region.
EVIDENCE: water quality assessments
TARGET: contamination rate

BAD example (has digits):
SCENARIO: A hospital is estimating Type 2 diabetes rates among adults aged 40-65.
"""
    for retry in range(max_retries):
        raw = call_api(prompt, api_key)
        parsed = _parse_scenario_blocks(raw, n)
        if parsed is None:
            print(f"    Parse failed, retry {retry+1}/{max_retries}")
            continue

        all_clean = True
        for p in parsed:
            for key in ("scenario", "evidence", "target"):
                if _has_digits(p[key]):
                    print(f"    Digit in {key}: '{p[key][:60]}' retry {retry+1}")
                    all_clean = False
                    break
            if not all_clean:
                break
        if all_clean:
            return parsed
    raise RuntimeError(
        f"Failed to generate {n} digit-free scenarios for {field}/{domain} "
        f"after {max_retries} retries"
    )


# ── Prompt builders ────────────────────────────────────────────────────────────

def _target_text(scn: dict, labels: list[str]) -> str:
    """Build: scenario + evidence/labels + question (NO footer)."""
    labels_s = ", ".join(labels)
    return (
        f"{scn['scenario']} Five independent {scn['evidence']} provide the "
        f"following assessments of the {scn['target']}: {labels_s}. "
        f"Estimate the {scn['target']} as a percentage from 0 to 100."
    )


def build_icl_prompts(target_scn: dict, target_labels: list[str],
                      demo_scns: list[dict], rng: random.Random) -> dict:
    target_txt = _target_text(target_scn, target_labels)

    prompts = {}
    for condition in ("control", "low_anchor", "high_anchor"):
        if condition == "control":
            demo_thetas = [25, 50, 75]
        elif condition == "low_anchor":
            demo_thetas = [20, 25, 30]
        else:
            demo_thetas = [70, 75, 80]

        demo_parts = []
        for i, (dscn, dt) in enumerate(zip(demo_scns, demo_thetas)):
            d_signals = gen_signals(rng, dt, 5)
            d_labels = labels_from_signals(d_signals)
            d_answer = compute_y_star(d_signals)
            d_txt = _target_text(dscn, d_labels)
            demo_parts.append(f"Example {i+1}:\n{d_txt}\n{d_answer}")

        mechanism = "\n\n".join(demo_parts) + "\n\nNow estimate for a new case:\n"
        prompts[condition] = f"{mechanism}{target_txt}\n{FOOTER}"

    return prompts


def build_hist_prompts(target_scn: dict, target_labels: list[str],
                       subtype: str) -> dict:
    target_txt = _target_text(target_scn, target_labels)
    tgt = target_scn["target"]

    prompts = {}
    for condition in ("control", "low_anchor", "high_anchor"):
        if condition == "control":
            mechanism = ""
        else:
            anchor = LOW_ANCHOR if condition == "low_anchor" else HIGH_ANCHOR
            if subtype == "history_relevant":
                mechanism = (
                    f"Chat history:\n"
                    f"User: What is your initial sense of the {tgt}?\n"
                    f"Assistant: Based on comparable scenarios, I would estimate "
                    f"it is around {anchor}%.\n\n"
                )
            else:
                mechanism = (
                    f"Chat history:\n"
                    f"User: Can you confirm the case reference?\n"
                    f"Assistant: The reference identifier is RQ-{anchor}.\n"
                    f"User: Thanks.\n\n"
                )
        prompts[condition] = f"{mechanism}{target_txt}\n{FOOTER}"

    return prompts


def build_rag_prompts(target_scn: dict, target_labels: list[str],
                      domain: str) -> dict:
    target_txt = _target_text(target_scn, target_labels)
    relevant = RAG_RELEVANT_SNIPPETS[domain]

    prompts = {}
    for condition in ("control", "low_anchor", "high_anchor"):
        if condition == "control":
            additional = (
                "Retrieved snippet (additional): A preliminary review cites a "
                "baseline rate of [VALUE] percent for comparable contexts, though "
                "the exact figure was redacted from the public version."
            )
        elif condition == "low_anchor":
            additional = (
                "Retrieved snippet (additional): A comparable assessment "
                "conducted in a similar context reported a rate of approximately "
                "20 percent."
            )
        else:
            additional = (
                "Retrieved snippet (additional): A comparable assessment "
                "conducted in a similar context reported a rate of approximately "
                "80 percent."
            )

        mechanism = (
            f"Retrieved snippet (relevant): {relevant}\n\n{additional}\n\n"
        )
        prompts[condition] = f"{mechanism}{target_txt}\n{FOOTER}"

    return prompts


def build_tool_prompts(target_scn: dict, target_labels: list[str]) -> dict:
    target_txt = _target_text(target_scn, target_labels)

    prompts = {}
    for condition in ("control", "low_anchor", "high_anchor"):
        if condition == "control":
            mechanism = ""
        else:
            anchor = LOW_ANCHOR if condition == "low_anchor" else HIGH_ANCHOR
            mechanism = f"Tool output:\n{anchor}\n\n"
        prompts[condition] = f"{mechanism}{target_txt}\n{FOOTER}"

    return prompts


# ── Item builder ───────────────────────────────────────────────────────────────

def make_item(item_id: str, suite: str, subtype: str, field: str, domain: str,
              template_id: str, theta: int, gold_ys: int,
              prompts_dict: dict) -> dict:
    return {
        "item_id": item_id,
        "suite": suite,
        "subtype": subtype,
        "field": field,
        "domain": domain,
        "template_id": template_id,
        "answer_space": {"type": "int", "min": 0, "max": 100},
        "anchors": {
            "low": LOW_ANCHOR, "high": HIGH_ANCHOR, "controlled_history": [],
        },
        "gold": {
            "theta": theta, "y_star": gold_ys,
            "y_star_stage1": None, "y_star_stage2": None,
        },
        "prompts": {
            "control": prompts_dict["control"],
            "low_anchor": prompts_dict["low_anchor"],
            "high_anchor": prompts_dict["high_anchor"],
            "turn1": None,
            "turn2_fresh": None,
            "turn2_history": None,
            "turn2_controlled_history": [],
        },
        "meta": {"language": "en", "version": "0.2", "seed": SEED},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(SEED)
    src = Path(__file__).resolve().parent / "poc.jsonl"
    dst = Path(__file__).resolve().parent / "poc_v0.2.jsonl"

    existing = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                existing.append(json.loads(line))
    print(f"Loaded {len(existing)} existing items from {src.name}")

    suite_configs = [
        ("icl",                  "range_anchoring",           "ICL",  "icl"),
        ("conversation_history", None,                        "HIST", "hist"),
        ("rag",                  "retrieval_snippet_anchor",  "RAG",  "rag"),
        ("tool",                 "tool_output_anchor",        "TOOL", "tool"),
    ]

    new_items: list[dict] = []

    for suite_name, default_subtype, prefix, abbrev in suite_configs:
        idx = 0
        print(f"\n{'='*60}\nGenerating {suite_name} items (12 total)\n{'='*60}")

        for field in FIELDS:
            for domain in DOMAINS[field]:
                da = DOMAIN_ABBREV[domain]

                for local_i in range(2):
                    idx += 1
                    item_id = f"{prefix}-{idx:04d}"
                    tmpl_id = f"{field}_{da}_{abbrev}_{idx:02d}"

                    if suite_name == "conversation_history":
                        subtype = ("history_relevant" if local_i == 0
                                   else "history_irrelevant")
                    else:
                        subtype = default_subtype

                    theta = rng.randint(15, 85)
                    signals = gen_signals(rng, theta, 5)
                    labels = labels_from_signals(signals)
                    gold_ys = compute_y_star(signals)

                    n_scn = 4 if suite_name == "icl" else 1
                    print(f"  {item_id} [{field}/{domain}] theta={theta} "
                          f"labels={','.join(labels)} → generating {n_scn} scenario(s)...")
                    scns = generate_scenarios(field, domain, n_scn, api_key)
                    for s in scns:
                        print(f"    → {s['scenario'][:70]}...")

                    if suite_name == "icl":
                        prompts = build_icl_prompts(scns[0], labels, scns[1:4], rng)
                    elif suite_name == "conversation_history":
                        prompts = build_hist_prompts(scns[0], labels, subtype)
                    elif suite_name == "rag":
                        prompts = build_rag_prompts(scns[0], labels, domain)
                    else:
                        prompts = build_tool_prompts(scns[0], labels)

                    item = make_item(item_id, suite_name, subtype, field, domain,
                                     tmpl_id, theta, gold_ys, prompts)
                    new_items.append(item)

    with open(dst, "w", encoding="utf-8") as f:
        for item in existing:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        for item in new_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Wrote {len(existing) + len(new_items)} items to {dst.name}")
    print(f"  Original: {len(existing)}")
    print(f"  New:      {len(new_items)}")
    from collections import Counter
    for suite_name, _, prefix, _ in suite_configs:
        suite_items = [it for it in new_items if it["suite"] == suite_name]
        print(f"    {suite_name}: {len(suite_items)}")
        for fld in FIELDS:
            n = sum(1 for it in suite_items if it["field"] == fld)
            doms = "  ".join(
                f"{d}={sum(1 for it in suite_items if it['field']==fld and it['domain']==d)}"
                for d in DOMAINS[fld]
            )
            print(f"      {fld}: {n}  ({doms})")
        if suite_name == "conversation_history":
            sub_cnt = Counter(it["subtype"] for it in suite_items)
            print(f"      relevant={sub_cnt.get('history_relevant',0)}  "
                  f"irrelevant={sub_cnt.get('history_irrelevant',0)}")


if __name__ == "__main__":
    main()
