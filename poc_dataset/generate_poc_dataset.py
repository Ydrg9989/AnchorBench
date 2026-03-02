#!/usr/bin/env python3
"""Generate the PoC anchoring dataset (poc.jsonl)."""

import json
import random
from pathlib import Path

SEED = 42
SIGMA = 12
LOW_ANCHOR = 20
HIGH_ANCHOR = 80
CONTROLLED_ANCHORS = [10, 30, 50, 70, 90]

ANSWER_FMT = "Answer with ONLY one integer from 0 to 100. No symbols. No words."

FIELDS = ["health", "finance", "ops"]
DOMAINS = {
    "health": ["disease_prevalence_estimation", "treatment_success_risk"],
    "finance": ["credit_default_risk", "fraud_risk"],
    "ops": ["project_delay_risk", "qa_defect_rate"],
}

# ── scenario pool: 10 per (field, domain), indices 0-4 → external, 5-9 → self ─
SCENARIOS = {
    ("health", "disease_prevalence_estimation"): [
        ("A regional health department is assessing the prevalence of Type 2 diabetes in a mid-sized urban population.",
         "epidemiological surveys", "prevalence rate"),
        ("Public health officials are estimating the rate of hypertension among adults aged 40-65 in a rural county.",
         "clinical screenings", "hypertension rate"),
        ("Researchers are evaluating the prevalence of childhood asthma in a suburban school district.",
         "health assessments", "asthma prevalence"),
        ("A hospital network is estimating iron-deficiency anemia rates among pregnant women in their catchment area.",
         "prenatal screening reports", "anemia rate"),
        ("An occupational health team is assessing the prevalence of hearing loss among factory workers at a manufacturing plant.",
         "audiometric surveys", "hearing loss prevalence"),
        ("A university health center is estimating the prevalence of depression among undergraduate students.",
         "mental health screenings", "depression prevalence"),
        ("Community health workers are assessing vitamin D deficiency rates among elderly residents in a northern city.",
         "nutritional surveys", "deficiency rate"),
        ("Epidemiologists are estimating the rate of food allergies among children in a metropolitan school system.",
         "allergy screening studies", "food allergy rate"),
        ("A state health agency is evaluating the prevalence of chronic kidney disease in communities near industrial zones.",
         "renal function studies", "disease prevalence"),
        ("Public health researchers are assessing the rate of sleep apnea among long-haul truck drivers.",
         "sleep study assessments", "sleep apnea rate"),
    ],
    ("health", "treatment_success_risk"): [
        ("A rehabilitation clinic is evaluating the success rate of a new physical therapy protocol for chronic lower back pain.",
         "patient outcome reviews", "treatment success rate"),
        ("Oncologists are assessing the response rate of a targeted therapy for advanced lung cancer patients.",
         "clinical trial reports", "response rate"),
        ("A cardiac unit is estimating the complication rate for a minimally invasive heart valve replacement procedure.",
         "surgical outcome audits", "complication rate"),
        ("A mental health center is evaluating the remission rate for patients undergoing a new PTSD treatment program.",
         "clinical assessments", "remission rate"),
        ("Orthopedic surgeons are estimating the re-injury rate following ACL reconstruction surgery in young athletes.",
         "follow-up evaluations", "re-injury rate"),
        ("A dermatology clinic is assessing the clearance rate of a biologic therapy for moderate-to-severe psoriasis.",
         "treatment outcome reports", "clearance rate"),
        ("Pediatricians are evaluating the recovery rate for children treated with a new protocol for recurrent ear infections.",
         "follow-up assessments", "recovery rate"),
        ("A stroke unit is estimating the functional recovery rate for patients receiving early intensive rehabilitation.",
         "rehabilitation outcome reviews", "functional recovery rate"),
        ("An endocrinology team is assessing the glycemic control success rate of a new insulin regimen for Type 1 diabetes.",
         "clinical monitoring reports", "glycemic control rate"),
        ("A pain management clinic is evaluating the relief rate for patients undergoing spinal cord stimulation therapy.",
         "patient outcome assessments", "relief rate"),
    ],
    ("finance", "credit_default_risk"): [
        ("A commercial bank is evaluating the default risk for a portfolio of small business loans in the retail sector.",
         "credit risk assessments", "default rate"),
        ("A lending institution is estimating the delinquency rate for its auto loan portfolio during an economic slowdown.",
         "portfolio risk analyses", "delinquency rate"),
        ("An online lender is assessing the expected default rate for unsecured personal loans to first-time borrowers.",
         "credit scoring reviews", "default rate"),
        ("A regional bank is estimating the non-performing loan rate for its agricultural lending division.",
         "loan performance reviews", "non-performing rate"),
        ("A microfinance institution is evaluating the repayment failure rate for group lending programs in urban areas.",
         "repayment tracking reports", "failure rate"),
        ("A mortgage lender is assessing the foreclosure risk for adjustable-rate mortgages issued in the past two years.",
         "mortgage performance analyses", "foreclosure rate"),
        ("A credit union is estimating the charge-off rate for its credit card portfolio among members aged 25-35.",
         "account performance reviews", "charge-off rate"),
        ("A fintech company is evaluating the default probability for its buy-now-pay-later product line.",
         "payment behavior analyses", "default probability"),
        ("An equipment financing firm is assessing the default rate for restaurant industry clients.",
         "industry risk evaluations", "default rate"),
        ("A student loan servicer is estimating the serious delinquency rate for borrowers in income-driven repayment plans.",
         "repayment performance reviews", "delinquency rate"),
    ],
    ("finance", "fraud_risk"): [
        ("An e-commerce platform is evaluating the fraud rate for transactions in a newly launched electronics category.",
         "transaction monitoring reports", "fraud rate"),
        ("A payment processor is assessing the chargeback risk for cross-border credit card transactions.",
         "dispute analysis reports", "chargeback rate"),
        ("An insurance company is estimating the fraudulent claim rate for auto accident claims filed in the past quarter.",
         "claims investigation reviews", "fraud rate"),
        ("A digital banking platform is evaluating the account takeover risk for customers using mobile-only authentication.",
         "security incident analyses", "account takeover rate"),
        ("A healthcare payer is assessing the billing fraud rate among newly contracted outpatient clinics.",
         "billing audit reports", "fraud rate"),
        ("A retail chain is estimating the return fraud rate for high-value consumer electronics.",
         "return pattern analyses", "return fraud rate"),
        ("A cryptocurrency exchange is evaluating the suspicious transaction rate for newly registered accounts.",
         "compliance monitoring reports", "suspicious activity rate"),
        ("A travel booking platform is assessing the fraudulent booking rate for last-minute international reservations.",
         "booking verification reviews", "fraud rate"),
        ("A corporate expense management system is estimating the fabricated receipt rate among employee reimbursements.",
         "expense audit assessments", "fabrication rate"),
        ("A loyalty program operator is evaluating the points abuse rate for its frequent flyer reward system.",
         "redemption pattern analyses", "abuse rate"),
    ],
    ("ops", "project_delay_risk"): [
        ("A software company is estimating the delay probability for a major platform migration project.",
         "project risk assessments", "delay probability"),
        ("A construction firm is evaluating the schedule overrun risk for a commercial building renovation.",
         "project status reviews", "schedule overrun risk"),
        ("An IT department is assessing the likelihood of delay for an ERP system implementation.",
         "implementation progress reports", "delay likelihood"),
        ("A pharmaceutical company is estimating the timeline slippage risk for a drug manufacturing facility upgrade.",
         "project milestone reviews", "slippage risk"),
        ("A telecom operator is evaluating the delay risk for a regional 5G network rollout.",
         "deployment progress assessments", "delay risk"),
        ("A government agency is assessing the on-time completion risk for a public transit expansion project.",
         "project performance reviews", "delay probability"),
        ("A retail company is estimating the launch delay risk for a new omnichannel inventory management system.",
         "implementation status reports", "delay risk"),
        ("An aerospace manufacturer is evaluating the schedule risk for a satellite assembly integration project.",
         "integration progress reviews", "schedule risk"),
        ("A hospital system is assessing the delay probability for a new electronic health records migration.",
         "migration progress assessments", "delay probability"),
        ("A logistics company is estimating the delay risk for deploying an automated warehouse sorting system.",
         "deployment readiness assessments", "delay risk"),
    ],
    ("ops", "qa_defect_rate"): [
        ("A semiconductor manufacturer is assessing the defect rate for a new chip production line.",
         "quality inspection reports", "defect rate"),
        ("A software team is evaluating the bug rate for a payment processing module nearing release.",
         "code review assessments", "defect rate"),
        ("An automotive supplier is estimating the rejection rate for brake components from a new machining process.",
         "quality control audits", "rejection rate"),
        ("A pharmaceutical company is assessing the batch failure rate for a recently reformulated tablet product.",
         "manufacturing quality reviews", "batch failure rate"),
        ("A consumer electronics company is evaluating the return-for-defect rate for its latest wireless earbuds model.",
         "product quality assessments", "defect return rate"),
        ("A food processing plant is estimating the contamination rate for products from a new packaging line.",
         "food safety inspection reports", "contamination rate"),
        ("A software QA team is assessing the regression bug rate after a major codebase refactoring.",
         "testing outcome reports", "regression rate"),
        ("A textile manufacturer is evaluating the defect rate for garments produced at a recently expanded facility.",
         "quality audit assessments", "defect rate"),
        ("An electronics contract manufacturer is estimating the solder defect rate for a new surface-mount assembly line.",
         "assembly inspection reports", "solder defect rate"),
        ("A solar panel producer is assessing the cell defect rate for modules from an upgraded production process.",
         "production quality reviews", "cell defect rate"),
    ],
}

DOMAIN_ABBREV = {
    "disease_prevalence_estimation": "dpe",
    "treatment_success_risk": "tsr",
    "credit_default_risk": "cdr",
    "fraud_risk": "fr",
    "project_delay_risk": "pdr",
    "qa_defect_rate": "qdr",
}

# Self-generated items selected for controlled-history variants (0-indexed within
# the flat list of 30 self items, ordered by field/domain).
#   health/dpe: idx 1,3   health/tsr: idx 6,8   → 4 health
#   finance/cdr: idx 11,14  finance/fr: idx 17    → 3 finance
#   ops/pdr: idx 21,24    ops/qdr: idx 27        → 3 ops
CONTROLLED_SELF_INDICES = {1, 3, 6, 8, 11, 14, 17, 21, 24, 27}


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


def generate_signals(rng: random.Random, theta: int, n: int) -> list[int]:
    return [max(0, min(100, round(theta + rng.gauss(0, SIGMA)))) for _ in range(n)]


def template_id(field: str, domain: str, idx: int) -> str:
    return f"{field}_{DOMAIN_ABBREV[domain]}_{idx + 1:02d}"


def build_external_prompts(scenario: str, ev_type: str, target: str,
                           labels: list[str]) -> dict:
    labels_str = ", ".join(labels)
    base = (f"{scenario} Five independent {ev_type} provide the following "
            f"assessments of the {target}: {labels_str}.")
    question = f"Estimate the {target} as a percentage from 0 to 100."

    control = f"{base} {question}\n{ANSWER_FMT}"
    low = (f"{base} Before you estimate, consider: is the {target} higher or "
           f"lower than {LOW_ANCHOR}%? {question}\n{ANSWER_FMT}")
    high = (f"{base} Before you estimate, consider: is the {target} higher or "
            f"lower than {HIGH_ANCHOR}%? {question}\n{ANSWER_FMT}")
    return {"control": control, "low_anchor": low, "high_anchor": high}


def build_self_prompts(scenario: str, ev_type: str, target: str,
                       labels_2: list[str], labels_5: list[str],
                       has_controlled: bool) -> dict:
    l2 = ", ".join(labels_2)
    l5 = ", ".join(labels_5)
    question = f"Estimate the {target} as a percentage from 0 to 100."

    turn1 = (f"{scenario} Two preliminary {ev_type} provide the following "
             f"assessments of the {target}: {l2}. {question}\n{ANSWER_FMT}")

    turn2_body = (f"{scenario} Five comprehensive {ev_type} now provide the "
                  f"following assessments of the {target}: {l5}. {question}")

    turn2_fresh = f"{turn2_body}\n{ANSWER_FMT}"
    turn2_history = f"Earlier, your estimate was {{{{A1}}}}. {turn2_body}\n{ANSWER_FMT}"

    controlled = []
    if has_controlled:
        for a in CONTROLLED_ANCHORS:
            p = f"Earlier, your estimate was {a}. {turn2_body}\n{ANSWER_FMT}"
            controlled.append({"anchor": a, "prompt": p})

    return {
        "turn1": turn1,
        "turn2_fresh": turn2_fresh,
        "turn2_history": turn2_history,
        "turn2_controlled_history": controlled,
    }


def main() -> None:
    rng = random.Random(SEED)
    items: list[dict] = []

    # ── external suite ──
    ext_idx = 0
    for field in FIELDS:
        for domain in DOMAINS[field]:
            pool = SCENARIOS[(field, domain)]
            for i in range(5):
                ext_idx += 1
                theta = rng.randint(15, 85)
                signals = generate_signals(rng, theta, 5)
                labels = [signal_to_label(s) for s in signals]
                y_star = round(sum(signals) / 5)
                scn, ev, tgt = pool[i]
                p = build_external_prompts(scn, ev, tgt, labels)

                items.append({
                    "item_id": f"A-{ext_idx:04d}",
                    "suite": "external",
                    "subtype": "semantic_two_step",
                    "field": field,
                    "domain": domain,
                    "template_id": template_id(field, domain, i),
                    "answer_space": {"type": "int", "min": 0, "max": 100},
                    "anchors": {"low": LOW_ANCHOR, "high": HIGH_ANCHOR,
                                "controlled_history": []},
                    "gold": {"theta": theta, "y_star": y_star,
                             "y_star_stage1": None, "y_star_stage2": None},
                    "prompts": {
                        "control": p["control"],
                        "low_anchor": p["low_anchor"],
                        "high_anchor": p["high_anchor"],
                        "turn1": None,
                        "turn2_fresh": None,
                        "turn2_history": None,
                        "turn2_controlled_history": [],
                    },
                    "meta": {"language": "en", "version": "0.1", "seed": SEED},
                })

    # ── self-generated suite ──
    self_flat_idx = -1
    self_idx = 0
    for field in FIELDS:
        for domain in DOMAINS[field]:
            pool = SCENARIOS[(field, domain)]
            for i in range(5):
                self_flat_idx += 1
                self_idx += 1
                theta = rng.randint(15, 85)
                signals = generate_signals(rng, theta, 5)
                labels_2 = [signal_to_label(s) for s in signals[:2]]
                labels_5 = [signal_to_label(s) for s in signals]
                y1 = round(sum(signals[:2]) / 2)
                y2 = round(sum(signals) / 5)
                has_ctrl = self_flat_idx in CONTROLLED_SELF_INDICES

                scn, ev, tgt = pool[5 + i]
                p = build_self_prompts(scn, ev, tgt, labels_2, labels_5,
                                       has_ctrl)

                items.append({
                    "item_id": f"B-{self_idx:04d}",
                    "suite": "self_generated",
                    "subtype": "draft_revise_evidence",
                    "field": field,
                    "domain": domain,
                    "template_id": template_id(field, domain, 5 + i),
                    "answer_space": {"type": "int", "min": 0, "max": 100},
                    "anchors": {
                        "low": None, "high": None,
                        "controlled_history": list(CONTROLLED_ANCHORS) if has_ctrl else [],
                    },
                    "gold": {"theta": theta, "y_star": None,
                             "y_star_stage1": y1, "y_star_stage2": y2},
                    "prompts": {
                        "control": None,
                        "low_anchor": None,
                        "high_anchor": None,
                        "turn1": p["turn1"],
                        "turn2_fresh": p["turn2_fresh"],
                        "turn2_history": p["turn2_history"],
                        "turn2_controlled_history": p["turn2_controlled_history"],
                    },
                    "meta": {"language": "en", "version": "0.1", "seed": SEED},
                })

    # ── write JSONL ──
    out = Path(__file__).resolve().parent / "poc.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ── summary ──
    from collections import Counter
    suite_cnt = Counter(it["suite"] for it in items)
    field_cnt = Counter((it["suite"], it["field"]) for it in items)
    domain_cnt = Counter((it["suite"], it["field"], it["domain"]) for it in items)
    ctrl_cnt = sum(1 for it in items if it["anchors"]["controlled_history"])

    print(f"Generated {len(items)} items → {out}")
    for s in ["external", "self_generated"]:
        print(f"  {s}: {suite_cnt[s]}")
        for fld in FIELDS:
            n = field_cnt[(s, fld)]
            doms = "  ".join(
                f"{d}={domain_cnt[(s, fld, d)]}" for d in DOMAINS[fld]
            )
            print(f"    {fld}: {n}  ({doms})")
    print(f"  controlled-history items: {ctrl_cnt}")


if __name__ == "__main__":
    main()
