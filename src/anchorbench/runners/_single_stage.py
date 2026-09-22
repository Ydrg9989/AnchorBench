"""The one body shared by the single-stage suite runners.

External, ICL (and ICL-dist) and RAG are evaluated identically: load the
promptviews and itemspecs, keep every item that has all five conditions,
build a backend, run :func:`anchorbench.eval.evaluator.run_single_stage`, and
write ``results.jsonl`` plus ``summary.json``. Only the label in the log
lines differs. The three ``python -m anchorbench.runners.<suite>`` modules
exist because the CLI and the experiment launchers address a runner by suite
name; each is a few lines that call :func:`main` with its label.

History (two-stage protocol) and Tool (chat-template tool messages) have
their own runner modules because their loops genuinely differ.
"""

from __future__ import annotations

import argparse
import logging
import sys

from anchorbench.eval.evaluator import prepare_items, run_single_stage, write_and_summarize
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.runner_utils import (
    add_common_args,
    build_suffix,
    make_backend,
    make_fallback,
    model_output_dir,
)

log = logging.getLogger(__name__)


def main(label: str) -> None:
    """Parse the common runner arguments and evaluate one (model, suite) cell.

    ``label`` names the suite in log lines and in the summary header
    (``"External"``, ``"ICL"``, ``"RAG"``).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description=f"Run {label} inference and evaluation")
    add_common_args(p)
    args = p.parse_args()

    model_out = model_output_dir(args)

    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items(views, specs, args.max_items, args.seed)
    log.info("Loaded %d items (%d prompts)", len(items), len(items) * 5)

    if not items:
        log.error("No %s items found (need 5 conditions per item)", label)
        sys.exit(1)

    fallback = make_fallback(args)
    suffix = build_suffix(args)
    backend = make_backend(args)

    records = run_single_stage(
        backend, items, model_out / "results.jsonl",
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        prompt_suffix=suffix,
        use_llm_fallback=args.llm_fallback,
        fallback_extractor=fallback,
    )

    write_and_summarize(records, model_out, label=f"{label} | {args.model_id}")
