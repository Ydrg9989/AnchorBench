#!/usr/bin/env python3
"""Upload the AnchorBench release to the Hugging Face Hub.

Generate the files first, from the repo root:

    python scripts/export_public_promptviews.py

then upload:

    python datasets/upload_hf.py --dry-run     # list what would go, change nothing
    python datasets/upload_hf.py

The suite list is explicit on purpose. This used to glob
``anchorbench_*_core`` for a file named ``promptviews_public.jsonl`` while the
exporter wrote ``{dir.name}_promptviews_public.jsonl``, so the glob matched
one directory by accident and the documented recipe uploaded external only.
The glob had also started catching ``_weighted_mean_core`` and
``_icl_dist_core``, which are ablation datasets that are not part of the
release.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "datasets" / "hf_release"
REPO_ID = "Yiderigun/LLM_anchoring"

# The five core suites plus the uncertain variant, which backs Table 2 in the
# main paper. Ablation datasets under datasets/ are deliberately excluded.
SUITES = ("external", "history", "icl", "rag", "tool", "external_uncertain")

# Shipped alongside the data so the Hub page documents itself.
EXTRA_FILES = ("DATASET_CARD.md", "VERSIONS.md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="List the uploads without contacting the Hub.")
    ap.add_argument("--repo_id", default=REPO_ID)
    args = ap.parse_args()

    uploads = []
    for suite in SUITES:
        src = RELEASE_DIR / f"{suite}.jsonl"
        if not src.exists():
            print(f"missing: {src.relative_to(ROOT)} -- run "
                  f"scripts/export_public_promptviews.py first", file=sys.stderr)
            return 1
        uploads.append((src, f"data/{suite}.jsonl"))

    for name in EXTRA_FILES:
        src = ROOT / "datasets" / name
        if src.exists():
            uploads.append((src, name))
        else:
            print(f"note: {name} not present, skipping")

    for src, target in uploads:
        size_kb = src.stat().st_size / 1024
        print(f"  {str(src.relative_to(ROOT)):55s} -> {target:32s} {size_kb:8.0f} KiB")

    if args.dry_run:
        print(f"\ndry run: {len(uploads)} file(s) would go to {args.repo_id}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    for src, target in uploads:
        print(f"uploading {target} ...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=target,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    print(f"\nDone. {len(uploads)} file(s) uploaded to {args.repo_id}.")
    print("Record the resulting revision hash in datasets/VERSIONS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
