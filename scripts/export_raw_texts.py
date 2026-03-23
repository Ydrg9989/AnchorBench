#!/usr/bin/env python3
"""Export raw_text from a results.jsonl to a readable file for sanity checking.

Usage:
    python scripts/export_raw_texts.py results/external_pilot_Qwen25_7B_512/results.jsonl
    # Writes results/external_pilot_Qwen25_7B_512/raw_texts_export.txt

    python scripts/export_raw_texts.py results/external_pilot_Llama31_8B_512/results.jsonl --out results/external_pilot_Llama31_8B_512/raw_texts_export.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export raw_text from results.jsonl to a text file")
    p.add_argument("results_jsonl", type=Path, help="Path to results.jsonl")
    p.add_argument("--out", type=Path, default=None, help="Output path (default: same dir as input, raw_texts_export.txt)")
    args = p.parse_args()

    in_path = args.results_jsonl
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    out_path = args.out or in_path.parent / "raw_texts_export.txt"

    count = 0
    with open(in_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                out.write(f"--- row {i + 1} | (parse error, skipping) ---\n\n")
                continue
            count += 1
            out.write(
                f"--- row {i + 1} | item_id={r.get('item_id')} | condition={r.get('condition')} | "
                f"answer_int={r.get('answer_int')} | parsed_ok={r.get('parsed_ok')} ---\n"
            )
            out.write((r.get("raw_text") or "(none)") + "\n\n")

    print(f"Wrote {out_path} ({count} records)")


if __name__ == "__main__":
    main()
