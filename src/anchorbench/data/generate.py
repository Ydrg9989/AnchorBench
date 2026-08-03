"""AnchorBench dataset generation CLI.

Usage (core benchmark):
    PYTHONPATH=src python -m anchorbench.data.generate \
        --suite external --size pilot --seed 42

Extension splits (separate from core):
    PYTHONPATH=src python -m anchorbench.data.generate \
        --suite external --size pilot --seed 42 \
        --scoring_function weighted_mean

    PYTHONPATH=src python -m anchorbench.data.generate \
        --suite external --size pilot --seed 42 \
        --difficulties easy medium hard

Sizes:
    smoke  — ~6 items (1 domain, quick sanity check)
    pilot  — 180 items (6 domains × 2 difficulties × 3 offsets × 5)
    core   — 360 items (6 domains × 2 difficulties × 3 offsets × 10)

Extension scoring functions (generate as separate splits):
    mean          — core default: round(mean(visible))
    weighted_mean — round(weighted_mean(visible, weights))
    median        — round(median(visible))
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from . import __version__
from .domains import ALL_DOMAIN_IDS, DOMAIN_IDS, MEDICAL_DOMAIN_IDS
from .itemspec_gen import (
    generate_external_itemspecs,
    generate_history_itemspecs,
    generate_icl_dist_itemspecs,
    generate_icl_itemspecs,
    generate_rag_itemspecs,
    generate_tool_itemspecs,
)
from .schema import ItemSpec, PromptView, write_jsonl
from .suites import SUITE_RENDERERS
from .suites.rag import build_full_corpus
from .validators import validate_all

logger = logging.getLogger(__name__)

SUITE_NAMES = ["external", "history", "icl", "icl_dist", "rag", "tool"]

_ITEMSPEC_GENERATORS = {
    "external": generate_external_itemspecs,
    "history": generate_history_itemspecs,
    "icl": generate_icl_itemspecs,
    "icl_dist": generate_icl_dist_itemspecs,
    "rag": generate_rag_itemspecs,
    "tool": generate_tool_itemspecs,
}

_SIZE_PARAMS = {
    "smoke": {"n_per_cell": 1, "split": "dev"},
    "pilot": {"n_per_cell": 5, "split": "pilot"},
    "core":  {"n_per_cell": 10, "split": "core"},
}


def _git_hash() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def render_all(specs: list[ItemSpec]) -> list[PromptView]:
    """Render PromptView records for all specs using suite-specific renderers."""
    views: list[PromptView] = []
    for spec in specs:
        renderer = SUITE_RENDERERS.get(spec.suite)
        if renderer is None:
            logger.warning("No renderer for suite %r, skipping %s", spec.suite, spec.item_id)
            continue
        views.extend(renderer(spec))
    return views


def generate_suite_dataset(
    suite: str,
    size: str = "pilot",
    seed: int = 42,
    out_dir: str | None = None,
    llm_enhance: bool = False,
    concurrency: int = 8,
    max_retries: int = 5,
    domains: list[str] | None = None,
    n_per_cell: int | None = None,
    scoring_function: str = "mean",
    difficulties: list[str] | None = None,
    validate: bool = True,
) -> dict:
    """Generate a single-suite AnchorBench dataset.

    Args:
        suite: Suite name (external, history, icl, rag, tool, ...).
        size: "smoke" | "pilot" | "core".
        seed: Master random seed.
        out_dir: Output directory (auto-generated if None).
        llm_enhance: Generate unique scenario text via OpenRouter.
        scoring_function: Gold answer aggregation ("mean", "weighted_mean", "median").
        difficulties: Override difficulty levels (default: ["easy", "hard"]).
        validate: Run deterministic validators after generation.
    """
    if suite not in _ITEMSPEC_GENERATORS:
        raise ValueError(f"Unknown suite {suite!r}. Choose from: {SUITE_NAMES}")

    suffix = f"_{scoring_function}" if scoring_function != "mean" else ""
    out = Path(out_dir or f"datasets/anchorbench_{suite}_{size}{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    gen_version = _git_hash()

    params = _SIZE_PARAMS.get(size)
    if params is None:
        raise ValueError(f"Unknown size {size!r}. Choose from: {list(_SIZE_PARAMS)}")

    effective_domains = domains or DOMAIN_IDS
    if size == "smoke":
        effective_domains = effective_domains[:1]

    effective_n_per_cell = n_per_cell if n_per_cell is not None else params["n_per_cell"]
    effective_difficulties = difficulties or ["easy", "hard"]

    scoring_weights = None
    if scoring_function == "weighted_mean":
        scoring_weights = [1.0, 1.0, 1.5, 1.5, 2.0]

    gen_kwargs = dict(
        domains=effective_domains,
        n_per_cell=effective_n_per_cell,
        seed=seed,
        generator_version=gen_version,
        split=params["split"],
        scoring_function=scoring_function,
        scoring_weights=scoring_weights,
        difficulties=effective_difficulties,
    )

    all_specs = _ITEMSPEC_GENERATORS[suite](**gen_kwargs)

    llm_calls = 0
    if llm_enhance:
        from .llm_enhance import enhance_scenarios
        cache_path = str(out / "artifacts" / "scenarios_cache.jsonl")
        artifact_dir = str(out / "artifacts" / "openrouter_calls")
        llm_calls = enhance_scenarios(
            all_specs,
            concurrency=concurrency,
            max_retries=max_retries,
            cache_path=cache_path,
            artifact_dir=artifact_dir,
        )

    all_views = render_all(all_specs)

    # Write outputs
    specs_path = out / "itemspecs.jsonl"
    views_path = out / "promptviews.jsonl"
    write_jsonl(all_specs, specs_path)
    write_jsonl(all_views, views_path)

    n_items = len(all_specs)
    cond_per_item = len(all_views) // n_items if n_items else 0

    core_conditions = {"control", "irrelevant_low", "irrelevant_high",
                       "plausible_low", "plausible_high"}
    core_views = [v for v in all_views if v.condition in core_conditions]
    ablation_views = [v for v in all_views if v.condition not in core_conditions]

    core_path = out / "promptviews_core.jsonl"
    write_jsonl(core_views, core_path)

    if ablation_views:
        ablation_path = out / "promptviews_ablation.jsonl"
        write_jsonl(ablation_views, ablation_path)

    ablation_conditions = sorted(set(v.condition for v in ablation_views))

    manifest: dict = {
        "version": __version__,
        "suite": suite,
        "generator_version": gen_version,
        "seed": seed,
        "size": size,
        "scoring_function": scoring_function,
        "llm_enhanced": llm_enhance,
        "llm_calls_made": llm_calls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "itemspecs": n_items,
            "promptviews": len(all_views),
            "promptviews_core": len(core_views),
            "promptviews_ablation": len(ablation_views),
            "suites": [suite],
            "domains": sorted(set(s.domain for s in all_specs)),
            "difficulties": sorted(set(s.difficulty for s in all_specs)),
            "conditions_per_item": cond_per_item,
            "core_conditions": sorted(core_conditions),
            "ablation_conditions": ablation_conditions,
        },
        "files": {
            "itemspecs": str(specs_path),
            "promptviews": str(views_path),
            "promptviews_core": str(core_path),
        },
    }

    if ablation_views:
        manifest["files"]["promptviews_ablation"] = str(out / "promptviews_ablation.jsonl")

    if suite == "rag":
        corpus = build_full_corpus(all_specs)
        corpus_path = out / "anchorbench_corpus.jsonl"
        write_jsonl(corpus, corpus_path)
        manifest["counts"]["corpus_docs"] = len(corpus)
        manifest["files"]["corpus"] = str(corpus_path)

    # Validate
    if validate:
        errs = validate_all(all_specs, all_views, manifest)
        if errs:
            logger.warning("Validation found %d issue(s):", len(errs))
            for e in errs[:20]:
                logger.warning("  %s", e)
            manifest["validation_errors"] = errs[:50]
        else:
            logger.info("Validation passed: 0 errors")
        manifest["validation_passed"] = len(errs) == 0

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    """CLI entry point for AnchorBench dataset generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate AnchorBench dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--suite", required=True, choices=SUITE_NAMES,
                        help="Suite to generate")
    parser.add_argument("--size", choices=["smoke", "pilot", "core"], default="pilot",
                        help="Dataset size preset (default: pilot)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed (default: 42)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (auto if omitted)")
    parser.add_argument("--scoring_function", choices=["mean", "weighted_mean", "median"],
                        default="mean",
                        help="Gold answer aggregation (default: mean)")
    parser.add_argument("--difficulties", nargs="+",
                        choices=["easy", "medium", "hard"], default=None,
                        help="Difficulty levels (default: easy hard)")
    parser.add_argument("--llm_enhance", action="store_true",
                        help="Generate unique scenario text via OpenRouter")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max parallel API requests (default: 8)")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retries per request (default: 5)")
    parser.add_argument("--n_per_cell", type=int, default=None,
                        help="Override items per cell")
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help=(
            "Override domain set. Use 'medical' for the 3 rebuttal medical "
            "domains, 'all' for the union, or pass explicit domain IDs."
        ),
    )
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip deterministic validation")
    args = parser.parse_args()

    mode = "LLM-enhanced" if args.llm_enhance else "deterministic"
    sf = args.scoring_function
    logger.info(
        "Generating AnchorBench %s (%s, seed=%d, scoring=%s, mode=%s)...",
        args.suite, args.size, args.seed, sf, mode,
    )

    domains_arg = args.domains
    if domains_arg is not None and len(domains_arg) == 1:
        sentinel = domains_arg[0].lower()
        if sentinel == "medical":
            domains_arg = list(MEDICAL_DOMAIN_IDS)
        elif sentinel == "other":
            from anchorbench.data.domains import OTHER_DOMAIN_IDS
            domains_arg = list(OTHER_DOMAIN_IDS)
        elif sentinel == "all":
            domains_arg = list(ALL_DOMAIN_IDS)
        elif sentinel == "business":
            domains_arg = list(DOMAIN_IDS)

    manifest = generate_suite_dataset(
        suite=args.suite,
        size=args.size,
        seed=args.seed,
        out_dir=args.out_dir,
        llm_enhance=args.llm_enhance,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        n_per_cell=args.n_per_cell,
        scoring_function=sf,
        difficulties=args.difficulties,
        validate=not args.no_validate,
        domains=domains_arg,
    )

    logger.info(
        "Done! %d itemspecs, %d promptviews -> %s",
        manifest["counts"]["itemspecs"],
        manifest["counts"]["promptviews"],
        manifest["files"]["itemspecs"].rsplit("/", 1)[0],
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
