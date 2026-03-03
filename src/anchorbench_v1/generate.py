"""Main generation CLI for AnchorBench v1.

Usage (deterministic, no API):
    PYTHONPATH=src python -m anchorbench_v1.generate \\
        --size core --seed 42 --out_dir datasets/anchorbench_v1/

Usage (LLM-enhanced scenarios via OpenRouter):
    PYTHONPATH=src python -m anchorbench_v1.generate \\
        --size core --seed 42 --out_dir datasets/anchorbench_v1/ \\
        --llm_enhance --concurrency 8 --max_retries 5

Sizes:
    smoke  - 10 items (quick validation)
    core   - 600 items (5 suites × 6 domains × 20)
    stress - 200 stress-tagged items
    full   - core + stress (800)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

from . import __version__
from .domains import DOMAIN_IDS
from .itemspec_gen import SUITES, generate_itemspecs
from .schema import ItemSpec, PromptView, write_jsonl
from .stress import generate_stress_specs
from .suites import SUITE_RENDERERS
from .suites.rag import build_full_corpus

logger = logging.getLogger(__name__)


def _git_hash() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def render_all(specs: List[ItemSpec]) -> List[PromptView]:
    """Render PromptView records for all specs using suite-specific renderers."""
    views = []
    for spec in specs:
        renderer = SUITE_RENDERERS.get(spec.suite)
        if renderer is None:
            logger.warning("No renderer for suite %r, skipping %s", spec.suite, spec.item_id)
            continue
        views.extend(renderer(spec))
    return views


def generate_dataset(
    size: str = "core",
    seed: int = 42,
    out_dir: str = "datasets/anchorbench_v1",
    llm_enhance: bool = False,
    concurrency: int = 8,
    max_retries: int = 5,
) -> dict:
    """Generate AnchorBench v1 dataset artifacts.

    Args:
        size: Dataset size preset.
        seed: Master random seed for deterministic generation.
        out_dir: Output directory.
        llm_enhance: If True, use OpenRouter bulk_writer to generate unique
            scenario text per item. Requires OPENROUTER_API_KEY.
        concurrency: Max parallel API requests (only with --llm_enhance).
        max_retries: Retries per request on 429/5xx (only with --llm_enhance).

    Returns:
        Summary dict with counts and file paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_version = _git_hash()

    all_specs: List[ItemSpec] = []

    if size in ("smoke",):
        specs = generate_itemspecs(
            n_per_cell=1, seed=seed, generator_version=gen_version, split="dev",
        )
        by_suite: dict = {}
        for s in specs:
            by_suite.setdefault(s.suite, []).append(s)
        specs = []
        for suite_specs in by_suite.values():
            specs.extend(suite_specs[:2])
        all_specs.extend(specs)

    if size in ("core", "full"):
        specs = generate_itemspecs(
            n_per_cell=20, seed=seed, generator_version=gen_version, split="core",
        )
        all_specs.extend(specs)

    if size in ("stress", "full"):
        stress_specs = generate_stress_specs(
            n_items=200, seed=seed + 1000, generator_version=gen_version,
        )
        all_specs.extend(stress_specs)

    # LLM enhancement: generate unique scenarios via OpenRouter
    llm_calls = 0
    if llm_enhance:
        from .llm_enhance import enhance_scenarios

        # Only enhance suites that use scenario text (external, icl, history)
        enhanceable = [s for s in all_specs if s.suite in ("external", "icl", "history")]
        cache_path = str(out / "artifacts" / "scenarios_cache.jsonl")
        artifact_dir = str(out / "artifacts" / "openrouter_calls")
        llm_calls = enhance_scenarios(
            enhanceable,
            concurrency=concurrency,
            max_retries=max_retries,
            cache_path=cache_path,
            artifact_dir=artifact_dir,
        )

    # Render prompts (uses scenario_text if populated by LLM enhance)
    all_views = render_all(all_specs)

    # Write artifacts
    specs_path = out / "itemspecs.jsonl"
    views_path = out / "promptviews.jsonl"
    write_jsonl(all_specs, specs_path)
    write_jsonl(all_views, views_path)

    # Build RAG corpus
    rag_specs = [s for s in all_specs if s.suite == "rag"]
    if rag_specs:
        corpus = build_full_corpus(all_specs)
        corpus_path = out / "anchorbench_corpus_v1.jsonl"
        write_jsonl(corpus, corpus_path)
    else:
        corpus_path = None

    # Write manifest
    manifest = {
        "version": __version__,
        "generator_version": gen_version,
        "seed": seed,
        "size": size,
        "llm_enhanced": llm_enhance,
        "llm_calls_made": llm_calls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "itemspecs": len(all_specs),
            "promptviews": len(all_views),
            "suites": list(set(s.suite for s in all_specs)),
            "domains": list(set(s.domain for s in all_specs)),
        },
        "files": {
            "itemspecs": str(specs_path),
            "promptviews": str(views_path),
            "corpus": str(corpus_path) if corpus_path else None,
        },
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate AnchorBench v1 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sizes:\n"
            "  smoke   10 items (quick validation)\n"
            "  core    600 items (5 suites × 6 domains × 20)\n"
            "  stress  200 stress-tagged items\n"
            "  full    core + stress (800)\n"
            "\n"
            "LLM enhancement:\n"
            "  --llm_enhance uses the bulk_writer model (see config/models.yaml)\n"
            "  to generate unique scenario paragraphs per item via OpenRouter.\n"
            "  Requires OPENROUTER_API_KEY env var. Results are cached so\n"
            "  re-runs with the same output dir skip already-generated items.\n"
        ),
    )
    parser.add_argument(
        "--size",
        choices=["smoke", "core", "stress", "full"],
        default="core",
        help="Dataset size preset (default: core)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument(
        "--out_dir",
        default="datasets/anchorbench_v1",
        help="Output directory",
    )
    parser.add_argument(
        "--llm_enhance",
        action="store_true",
        help="Generate unique scenario text via OpenRouter (requires API key)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max parallel API requests (default: 8, use 1 to disable concurrency)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retries per request on 429/5xx (default: 5)",
    )
    args = parser.parse_args()

    mode = "LLM-enhanced" if args.llm_enhance else "deterministic"
    logger.info("Generating AnchorBench v1 (%s, seed=%d, mode=%s)...", args.size, args.seed, mode)

    manifest = generate_dataset(
        size=args.size,
        seed=args.seed,
        out_dir=args.out_dir,
        llm_enhance=args.llm_enhance,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )

    logger.info("Done! %d itemspecs, %d promptviews", manifest["counts"]["itemspecs"], manifest["counts"]["promptviews"])
    if manifest.get("llm_calls_made", 0) > 0:
        logger.info("LLM API calls made: %d", manifest["llm_calls_made"])
    logger.info("Files written to: %s", args.out_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
