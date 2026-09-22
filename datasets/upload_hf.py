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
DATASETS = ROOT / "datasets"
RELEASE_DIR = DATASETS / "hf_release"
REPO_ID = "Yiderigun/AnchorBench"

# The five core suites plus the uncertain variant, which backs Table 2 in the
# main paper. Ablation datasets under datasets/ are deliberately excluded.
SUITES = ("external", "history", "icl", "rag", "tool", "external_uncertain")

# Shipped alongside the data so the Hub page documents itself.
# DATASET_CARD.md becomes README.md so the Hub renders it as the card.
EXTRA_FILES = {"DATASET_CARD.md": "README.md", "VERSIONS.md": "VERSIONS.md"}


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

    corpus = DATASETS / "anchorbench_rag_core" / "anchorbench_corpus.jsonl"
    if corpus.exists():
        uploads.append((corpus, "data/rag_corpus.jsonl"))

    for name, target in EXTRA_FILES.items():
        src = ROOT / "datasets" / name
        if src.exists():
            uploads.append((src, target))
        else:
            print(f"note: {name} not present, skipping")

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()

    # Anything already on the Hub that the new layout does not produce. The
    # first release used anchorbench_<suite>_core/promptviews.jsonl; leaving
    # those beside data/<suite>.jsonl would publish two copies of the
    # benchmark with different schemas. It also drops tool_read, a suite
    # removed in v2.0 that appears nowhere in the paper.
    keep = {t for _, t in uploads}
    existing = {s.rfilename for s in
                api.repo_info(args.repo_id, repo_type="dataset").siblings}
    obsolete = sorted(f for f in existing - keep if not f.startswith("."))

    for src, target in uploads:
        print(f"  + {str(src.relative_to(ROOT)):52s} -> {target:32s} "
              f"{src.stat().st_size/1024:8.0f} KiB")
    for f in obsolete:
        print(f"  - {f}")

    if args.dry_run:
        print(f"\ndry run: {len(uploads)} added/updated, {len(obsolete)} removed "
              f"on {args.repo_id}")
        return 0

    ops = [CommitOperationAdd(path_in_repo=t, path_or_fileobj=str(s))
           for s, t in uploads]
    ops += [CommitOperationDelete(path_in_repo=f) for f in obsolete]

    # One commit, so the repo is never in a half-migrated state.
    info = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message="Release anchorbench-v2.0-core: full schema, add uncertain suite, drop tool_read",
    )
    print(f"\nDone. {len(uploads)} added/updated, {len(obsolete)} removed.")
    print(f"Revision: {info.oid}")
    print("Record that hash in datasets/VERSIONS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
