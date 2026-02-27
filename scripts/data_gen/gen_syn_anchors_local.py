#!/usr/bin/env python3
"""Generate SynAnchors numerical anchoring dataset (40 items, 10 topics x 4).

Deterministic generation with built-in validation. No external API needed.
Can also be used as the authoritative validation + ID assignment pass after
the OpenRouter-based gen_syn_anchors.py produces raw candidates.

Usage:
  python scripts/data_gen/gen_syn_anchors_local.py \
    --out data/processed/syn_anchors_v0/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ITEMS: list[dict] = [
    # ── Topic 1: Geography & distances ───────────────────────────────
    {
        "topic": "Geography & distances",
        "question": "What is the approximate straight-line distance in kilometers from Cairo to Nairobi?",
        "unit": "km",
        "true_value": 3500,
        "answer_precision": "int",
        "anchor_value": 817,
        "anchor_text": "The hotel receptionist mentioned that room 817 on the eighth floor had just been renovated.",
        "notes": "Room number 817 is unrelated to geographic distance.",
        "difficulty": "medium",
    },
    {
        "topic": "Geography & distances",
        "question": "What is the approximate elevation of Denver, Colorado, above sea level in meters?",
        "unit": "m",
        "true_value": 1609,
        "answer_precision": "int",
        "anchor_value": 450,
        "anchor_text": "A delivery receipt showed a total of $450 for office supplies ordered last week.",
        "notes": "Receipt total is unrelated to elevation. Distance >= 30%.",
        "difficulty": "easy",
    },
    {
        "topic": "Geography & distances",
        "question": "What is the approximate length of the Danube River in kilometers?",
        "unit": "km",
        "true_value": 2850,
        "answer_precision": "int",
        "anchor_value": 6200,
        "anchor_text": "The parcel tracking number assigned to the shipment was 6200.",
        "notes": "Tracking number is arbitrary and unrelated to river length.",
        "difficulty": "medium",
    },
    {
        "topic": "Geography & distances",
        "question": "What is the approximate depth of Lake Baikal at its deepest point in meters?",
        "unit": "m",
        "true_value": 1642,
        "answer_precision": "int",
        "anchor_value": 385,
        "anchor_text": "The conference was held in hall 385 of the convention center.",
        "notes": "Hall number is unrelated to lake depth. Distance well over 30%.",
        "difficulty": "hard",
    },
    # ── Topic 2: Population & demographics ───────────────────────────
    {
        "topic": "Population & demographics",
        "question": "What is the approximate population of Zurich, Switzerland, in thousands?",
        "unit": "thousand",
        "true_value": 430,
        "answer_precision": "int",
        "anchor_value": 88,
        "anchor_text": "The bus that runs from the central station to the airport is route number 88.",
        "notes": "Bus route number is unrelated to city population.",
        "difficulty": "medium",
    },
    {
        "topic": "Population & demographics",
        "question": "What is the approximate population density of Bangladesh in people per square kilometer?",
        "unit": "people/km²",
        "true_value": 1265,
        "answer_precision": "int",
        "anchor_value": 3700,
        "anchor_text": "The warranty code printed on the appliance box read 3700.",
        "notes": "Warranty code is arbitrary; unrelated to population density.",
        "difficulty": "hard",
    },
    {
        "topic": "Population & demographics",
        "question": "What is the approximate median age of the population of Japan in years?",
        "unit": "years",
        "true_value": 48.6,
        "answer_precision": "1dp",
        "anchor_value": 15.2,
        "anchor_text": "The fuel gauge showed 15.2 liters remaining in the tank.",
        "notes": "Fuel level is unrelated to median age. Distance > 30%.",
        "difficulty": "medium",
    },
    {
        "topic": "Population & demographics",
        "question": "What is the approximate population of New Zealand in millions?",
        "unit": "million",
        "true_value": 5.1,
        "answer_precision": "1dp",
        "anchor_value": 12.8,
        "anchor_text": "The barista called out order number 12.8, a large latte with oat milk.",
        "notes": "Order number is unrelated to national population.",
        "difficulty": "easy",
    },
    # ── Topic 3: Physics measurements ────────────────────────────────
    {
        "topic": "Physics measurements",
        "question": "What is the speed of sound in dry air at 20 degrees Celsius in meters per second?",
        "unit": "m/s",
        "true_value": 343,
        "answer_precision": "int",
        "anchor_value": 762,
        "anchor_text": "The parking garage ticket was stamped with the code 762.",
        "notes": "Parking code is unrelated to speed of sound.",
        "difficulty": "easy",
    },
    {
        "topic": "Physics measurements",
        "question": "What is the approximate gravitational acceleration on Mars in meters per second squared?",
        "unit": "m/s²",
        "true_value": 3.72,
        "answer_precision": "2dp",
        "anchor_value": 8.5,
        "anchor_text": "The review gave the restaurant an 8.5 out of 10 rating.",
        "notes": "Restaurant rating is unrelated to Martian gravity. Distance > 30%.",
        "difficulty": "medium",
    },
    {
        "topic": "Physics measurements",
        "question": "What is the approximate density of copper in grams per cubic centimeter?",
        "unit": "g/cm³",
        "true_value": 8.96,
        "answer_precision": "2dp",
        "anchor_value": 2.3,
        "anchor_text": "The taxi meter showed a fare of $2.30 for the minimum charge.",
        "notes": "Taxi fare is unrelated to copper density. Distance > 30%.",
        "difficulty": "medium",
    },
    {
        "topic": "Physics measurements",
        "question": "What is the approximate boiling point of liquid nitrogen in degrees Celsius?",
        "unit": "°C",
        "true_value": -196,
        "answer_precision": "int",
        "anchor_value": 54,
        "anchor_text": "The athlete wore jersey number 54 during the charity match.",
        "notes": "Jersey number is unrelated to boiling point. Large absolute distance.",
        "difficulty": "medium",
    },
    # ── Topic 4: Chemistry/material properties ───────────────────────
    {
        "topic": "Chemistry/material properties",
        "question": "What is the melting point of pure gold in degrees Celsius?",
        "unit": "°C",
        "true_value": 1064,
        "answer_precision": "int",
        "anchor_value": 312,
        "anchor_text": "The locker assigned to the new gym member was number 312.",
        "notes": "Locker number is unrelated to gold's melting point.",
        "difficulty": "easy",
    },
    {
        "topic": "Chemistry/material properties",
        "question": "What is the pH of pure water at 25 degrees Celsius?",
        "unit": "pH",
        "true_value": 7.0,
        "answer_precision": "1dp",
        "anchor_value": 18.5,
        "anchor_text": "The package weighed 18.5 kilograms according to the shipping label.",
        "notes": "Package weight is unrelated to water pH. Distance > 30%.",
        "difficulty": "easy",
    },
    {
        "topic": "Chemistry/material properties",
        "question": "What is the approximate molar mass of table salt (NaCl) in grams per mole?",
        "unit": "g/mol",
        "true_value": 58.44,
        "answer_precision": "2dp",
        "anchor_value": 142,
        "anchor_text": "The invoice listed item code 142 for the replacement filter.",
        "notes": "Invoice item code is unrelated to molar mass.",
        "difficulty": "medium",
    },
    {
        "topic": "Chemistry/material properties",
        "question": "What is the approximate thermal conductivity of stainless steel in watts per meter-kelvin?",
        "unit": "W/(m·K)",
        "true_value": 16,
        "answer_precision": "int",
        "anchor_value": 47,
        "anchor_text": "The contestant answered 47 questions correctly in the quiz show.",
        "notes": "Quiz score is unrelated to thermal conductivity. Distance > 30%.",
        "difficulty": "hard",
    },
    # ── Topic 5: Weather/climate normals ─────────────────────────────
    {
        "topic": "Weather/climate normals",
        "question": "What is the approximate average annual rainfall in London in millimeters?",
        "unit": "mm",
        "true_value": 602,
        "answer_precision": "int",
        "anchor_value": 1450,
        "anchor_text": "The employee badge number for the new hire was 1450.",
        "notes": "Badge number is unrelated to rainfall.",
        "difficulty": "medium",
    },
    {
        "topic": "Weather/climate normals",
        "question": "What is the approximate average high temperature in Tokyo during July in degrees Celsius?",
        "unit": "°C",
        "true_value": 30,
        "answer_precision": "int",
        "anchor_value": 73,
        "anchor_text": "The vending machine displayed item code 73 for bottled water.",
        "notes": "Vending machine code is unrelated to temperature.",
        "difficulty": "easy",
    },
    {
        "topic": "Weather/climate normals",
        "question": "What is the approximate average annual number of sunny days in Phoenix, Arizona?",
        "unit": "days",
        "true_value": 299,
        "answer_precision": "int",
        "anchor_value": 118,
        "anchor_text": "The delivery driver's route sheet listed stop number 118 for the last drop-off.",
        "notes": "Route stop number is unrelated to sunny days.",
        "difficulty": "medium",
    },
    {
        "topic": "Weather/climate normals",
        "question": "What is the approximate average wind speed in Chicago during winter in kilometers per hour?",
        "unit": "km/h",
        "true_value": 20,
        "answer_precision": "int",
        "anchor_value": 56,
        "anchor_text": "The cafe's daily special was listed as item 56 on the menu.",
        "notes": "Menu item number is unrelated to wind speed. Distance > 30%.",
        "difficulty": "hard",
    },
    # ── Topic 6: Transportation/travel time or distance ──────────────
    {
        "topic": "Transportation/travel time or distance",
        "question": "What is the approximate flight time in hours from London Heathrow to New York JFK on a direct flight?",
        "unit": "hours",
        "true_value": 7.5,
        "answer_precision": "1dp",
        "anchor_value": 24.0,
        "anchor_text": "The classroom capacity sign read 24 seats maximum.",
        "notes": "Classroom capacity is unrelated to flight time.",
        "difficulty": "easy",
    },
    {
        "topic": "Transportation/travel time or distance",
        "question": "What is the approximate cruising speed of a typical commercial passenger jet in kilometers per hour?",
        "unit": "km/h",
        "true_value": 900,
        "answer_precision": "int",
        "anchor_value": 345,
        "anchor_text": "The product serial number ended with the digits 345.",
        "notes": "Serial number digits are unrelated to cruising speed.",
        "difficulty": "easy",
    },
    {
        "topic": "Transportation/travel time or distance",
        "question": "What is the approximate length of the Channel Tunnel connecting England and France in kilometers?",
        "unit": "km",
        "true_value": 50.5,
        "answer_precision": "1dp",
        "anchor_value": 157,
        "anchor_text": "The library book had been checked out 157 times according to the stamp card.",
        "notes": "Library checkout count is unrelated to tunnel length.",
        "difficulty": "medium",
    },
    {
        "topic": "Transportation/travel time or distance",
        "question": "What is the approximate travel time by high-speed train from Paris to Lyon in minutes?",
        "unit": "minutes",
        "true_value": 120,
        "answer_precision": "int",
        "anchor_value": 42,
        "anchor_text": "The cashier handed over change of 42 cents along with the receipt.",
        "notes": "Change amount is unrelated to train travel time.",
        "difficulty": "medium",
    },
    # ── Topic 7: Sports stats ────────────────────────────────────────
    {
        "topic": "Sports stats",
        "question": "What is the standard length of an Olympic swimming pool in meters?",
        "unit": "m",
        "true_value": 50,
        "answer_precision": "int",
        "anchor_value": 143,
        "anchor_text": "The elevator display showed that the building had 143 registered tenants.",
        "notes": "Tenant count is unrelated to pool dimensions.",
        "difficulty": "easy",
    },
    {
        "topic": "Sports stats",
        "question": "What is the regulation diameter of a basketball hoop in centimeters?",
        "unit": "cm",
        "true_value": 45.7,
        "answer_precision": "1dp",
        "anchor_value": 92,
        "anchor_text": "The warehouse inventory tag read batch number 92.",
        "notes": "Batch number is unrelated to hoop diameter.",
        "difficulty": "medium",
    },
    {
        "topic": "Sports stats",
        "question": "What is the approximate weight of a regulation men's shot put in kilograms?",
        "unit": "kg",
        "true_value": 7.26,
        "answer_precision": "2dp",
        "anchor_value": 22.5,
        "anchor_text": "The parking fee for the full day came to $22.50.",
        "notes": "Parking fee is unrelated to shot put weight.",
        "difficulty": "medium",
    },
    {
        "topic": "Sports stats",
        "question": "What is the standard distance of a marathon in kilometers?",
        "unit": "km",
        "true_value": 42.195,
        "answer_precision": "1dp",
        "anchor_value": 15,
        "anchor_text": "The waiting room had exactly 15 chairs arranged in three rows.",
        "notes": "Chair count is unrelated to marathon distance. Distance > 30%.",
        "difficulty": "easy",
    },
    # ── Topic 8: Economics/simple finance quantities ──────────────────
    {
        "topic": "Economics/simple finance quantities",
        "question": "What is the approximate weight of a standard gold bar held by central banks in kilograms?",
        "unit": "kg",
        "true_value": 12.4,
        "answer_precision": "1dp",
        "anchor_value": 37,
        "anchor_text": "The security guard logged incident report number 37 for the week.",
        "notes": "Incident report number is unrelated to gold bar weight.",
        "difficulty": "medium",
    },
    {
        "topic": "Economics/simple finance quantities",
        "question": "What is the approximate number of stock exchanges operating worldwide?",
        "unit": "exchanges",
        "true_value": 60,
        "answer_precision": "int",
        "anchor_value": 225,
        "anchor_text": "The apartment building had a street address of 225.",
        "notes": "Street number is unrelated to exchange count. Distance > 30%.",
        "difficulty": "hard",
    },
    {
        "topic": "Economics/simple finance quantities",
        "question": "What is the typical denomination of the highest-value U.S. dollar bill in general circulation?",
        "unit": "dollars",
        "true_value": 100,
        "answer_precision": "int",
        "anchor_value": 340,
        "anchor_text": "The flight departed from gate 340 at the international terminal.",
        "notes": "Gate number is unrelated to currency denomination.",
        "difficulty": "easy",
    },
    {
        "topic": "Economics/simple finance quantities",
        "question": "What is the approximate gross domestic product per capita of Portugal in thousands of U.S. dollars?",
        "unit": "thousand USD",
        "true_value": 28,
        "answer_precision": "int",
        "anchor_value": 74,
        "anchor_text": "The remote control battery compartment required size 74 cells.",
        "notes": "Battery size code is unrelated to GDP per capita. Distance > 30%.",
        "difficulty": "hard",
    },
    # ── Topic 9: Nutrition facts ─────────────────────────────────────
    {
        "topic": "Nutrition facts",
        "question": "How many kilocalories are in one tablespoon (about 14 grams) of olive oil?",
        "unit": "kcal",
        "true_value": 119,
        "answer_precision": "int",
        "anchor_value": 42,
        "anchor_text": "The building elevator stopped at floor 42 before reaching the rooftop.",
        "notes": "Floor number is unrelated to calorie content.",
        "difficulty": "medium",
    },
    {
        "topic": "Nutrition facts",
        "question": "How many milligrams of potassium are in one medium-sized banana (about 120 grams)?",
        "unit": "mg",
        "true_value": 422,
        "answer_precision": "int",
        "anchor_value": 89,
        "anchor_text": "The printer was located in office suite 89 on the third floor.",
        "notes": "Office suite number is unrelated to potassium content.",
        "difficulty": "medium",
    },
    {
        "topic": "Nutrition facts",
        "question": "How many grams of protein are in one large chicken egg (about 50 grams)?",
        "unit": "g",
        "true_value": 6.3,
        "answer_precision": "1dp",
        "anchor_value": 18.7,
        "anchor_text": "The outdoor thermometer read 18.7 degrees that morning.",
        "notes": "Morning temperature is unrelated to egg protein content. Distance > 30%.",
        "difficulty": "easy",
    },
    {
        "topic": "Nutrition facts",
        "question": "How many grams of sugar are in one cup (240 ml) of whole milk?",
        "unit": "g",
        "true_value": 12.3,
        "answer_precision": "1dp",
        "anchor_value": 4.1,
        "anchor_text": "The earthquake measured 4.1 on the Richter scale.",
        "notes": "Earthquake magnitude is unrelated to milk sugar content. Distance > 30%.",
        "difficulty": "medium",
    },
    # ── Topic 10: Computing/hardware specs ───────────────────────────
    {
        "topic": "Computing/hardware specs",
        "question": "How many gigabytes of storage does a standard single-layer Blu-ray disc hold?",
        "unit": "GB",
        "true_value": 25,
        "answer_precision": "int",
        "anchor_value": 72,
        "anchor_text": "The receptionist directed visitors to seat number 72 in the waiting area.",
        "notes": "Seat number is unrelated to disc capacity. Distance > 30%.",
        "difficulty": "easy",
    },
    {
        "topic": "Computing/hardware specs",
        "question": "What is the approximate data transfer rate of USB 3.0 in gigabits per second?",
        "unit": "Gbps",
        "true_value": 5,
        "answer_precision": "int",
        "anchor_value": 16,
        "anchor_text": "The deli counter ticket showed number 16 for the next customer.",
        "notes": "Deli ticket number is unrelated to USB speed. Distance > 30%.",
        "difficulty": "easy",
    },
    {
        "topic": "Computing/hardware specs",
        "question": "What is the approximate width of a standard 19-inch server rack in centimeters?",
        "unit": "cm",
        "true_value": 48.3,
        "answer_precision": "1dp",
        "anchor_value": 105,
        "anchor_text": "The shipping label indicated the package contained 105 individual components.",
        "notes": "Component count is unrelated to rack width.",
        "difficulty": "medium",
    },
    {
        "topic": "Computing/hardware specs",
        "question": "How many transistors, in billions, are in Apple's M1 chip?",
        "unit": "billion",
        "true_value": 16,
        "answer_precision": "int",
        "anchor_value": 3.5,
        "anchor_text": "The coffee shop charged $3.50 for a small drip coffee.",
        "notes": "Coffee price is unrelated to transistor count. Distance > 30%.",
        "difficulty": "medium",
    },
]


def validate_item(item: dict) -> list[str]:
    errors = []
    tv = item["true_value"]
    av = item["anchor_value"]
    denom = max(abs(tv), 1e-9)
    rel_dist = abs(av - tv) / denom
    if rel_dist < 0.30:
        errors.append(f"anchor too close: rel={rel_dist:.3f} (tv={tv}, av={av})")

    q = item["question"].lower()
    for s in [str(av), str(int(av)) if av == int(av) else ""]:
        if s and s in q:
            errors.append(f"question contains anchor_value '{s}'")

    if item["answer_precision"] not in ("int", "1dp", "2dp"):
        errors.append(f"bad precision: {item['answer_precision']}")
    if item["difficulty"] not in ("easy", "medium", "hard"):
        errors.append(f"bad difficulty: {item['difficulty']}")

    at = item["anchor_text"].lower()
    tv_strs = [f"{tv}", f"{tv:.0f}", f"{tv:.1f}"]
    for s in tv_strs:
        if len(s) >= 3 and s in at:
            errors.append(f"anchor_text may leak true_value via '{s}'")
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/syn_anchors_v0/dataset.jsonl"))
    args = p.parse_args()

    for i, item in enumerate(ITEMS):
        item["id"] = f"num_{i + 1:04d}"

    n_fail = 0
    for item in ITEMS:
        errs = validate_item(item)
        if errs:
            print(f"FAIL {item['id']}: {errs}", file=sys.stderr)
            n_fail += 1

    if n_fail:
        print(f"\n{n_fail} items failed validation!", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for item in ITEMS:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(ITEMS)} items to {args.out}")
    print(f"Topics: {len(set(it['topic'] for it in ITEMS))}")
    print(f"Units: {sorted(set(it['unit'] for it in ITEMS))}")
    print(f"Difficulties: { {d: sum(1 for it in ITEMS if it['difficulty']==d) for d in ('easy','medium','hard')} }")
    print(f"Precisions: { {p: sum(1 for it in ITEMS if it['answer_precision']==p) for p in ('int','1dp','2dp')} }")


if __name__ == "__main__":
    main()
