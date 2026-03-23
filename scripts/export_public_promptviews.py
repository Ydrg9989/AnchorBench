#!/usr/bin/env python3
"""Export promptviews for public/HF release.

  --minimal   : item_id, condition, prompt_text only (no ground truth).
  --hf        : item_id, condition, prompt_text, y_star (ground truth), anchor_value.
                Self-contained; reads itemspecs.jsonl in same dir for y_star.
  (default)   : drop prompt_hash, anchor_span, provenance, prompt_components.

  python scripts/export_public_promptviews.py --data_dir datasets/anchorbench_external_core --hf
  python scripts/export_public_promptviews.py --data_dir datasets --suites external history icl rag tool --size core --hf --out_dir datasets/hf_release
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Keys to drop for non-minimal export
DROP_KEYS = {"prompt_hash", "anchor_span", "provenance", "prompt_components"}
MINIMAL_KEYS = {"item_id", "condition", "prompt_text"}


def load_y_star_by_item(itemspecs_path: Path) -> dict[str, int]:
    out = {}
    if not itemspecs_path.is_file():
        return out
    with open(itemspecs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spec = json.loads(line)
            iid = spec.get("item_id")
            if iid is not None:
                out[iid] = spec.get("y_star_evidence", spec.get("y_star"))
    return out


def strip_record(obj: dict, minimal: bool, hf: bool, y_star_by_item: dict | None) -> dict:
    if hf:
        iid = obj.get("item_id")
        y_star = y_star_by_item.get(iid) if (y_star_by_item and iid is not None) else None
        return {
            "item_id": obj.get("item_id"),
            "condition": obj.get("condition"),
            "prompt_text": obj.get("prompt_text"),
            "y_star": y_star,
            "anchor_value": obj.get("anchor_value"),
        }
    if minimal:
        return {k: v for k, v in obj.items() if k in MINIMAL_KEYS}
    return {k: v for k, v in obj.items() if k not in DROP_KEYS}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export promptviews for public release (strip internal keys or --minimal: item_id, condition, prompt_text only)"
    )
    ap.add_argument("--data_dir", type=str, required=True, help="Single dataset dir or datasets/ when using --suites")
    ap.add_argument("--suites", type=str, nargs="*", help="If set, data_dir must be datasets/ and we use data_dir/anchorbench_{suite}_{size}/")
    ap.add_argument("--size", type=str, default="core", help="Size suffix when using --suites (default: core)")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory; default: data_dir")
    ap.add_argument("--minimal", action="store_true", help="Keep only item_id, condition, prompt_text (no ground truth)")
    ap.add_argument("--hf", action="store_true", help="HF release: item_id, condition, prompt_text, y_star, anchor_value (self-contained)")
    args = ap.parse_args()

    if args.minimal and args.hf:
        raise SystemExit("Use either --minimal or --hf, not both.")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    if args.suites:
        dirs = [data_dir / f"anchorbench_{s}_{args.size}" for s in args.suites]
    else:
        dirs = [data_dir]

    for d in dirs:
        if not d.is_dir():
            print(f"Skip (not a dir): {d}")
            continue
        src = d / "promptviews.jsonl"
        if not src.is_file():
            print(f"Skip (no promptviews.jsonl): {d}")
            continue
        y_star_by_item = load_y_star_by_item(d / "itemspecs.jsonl") if args.hf else None
        if args.hf and not y_star_by_item:
            print(f"Warning: no itemspecs in {d}, y_star will be null")
        out_dir = Path(args.out_dir) if args.out_dir else d
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("promptviews_public.jsonl" if not args.suites else f"{d.name}_promptviews_public.jsonl")

        count = 0
        with open(src) as f, open(out_path, "w") as g:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec = strip_record(rec, minimal=args.minimal, hf=args.hf, y_star_by_item=y_star_by_item)
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
        label = " (minimal)" if args.minimal else " (HF: y_star + anchor_value)" if args.hf else ""
        print(f"Wrote {count} records -> {out_path}{label}")


if __name__ == "__main__":
    main()
