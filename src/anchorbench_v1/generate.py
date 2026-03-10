"""Main generation CLI for AnchorBench v1 and v2.

Usage (v1 deterministic, no API):
    PYTHONPATH=src python -m anchorbench_v1.generate \\
        --size core --seed 42 --out_dir datasets/anchorbench_v1/

Usage (v2 External-only pilot, deterministic):
    PYTHONPATH=src python -m anchorbench_v1.generate \\
        --v2 --suites external --size pilot --seed 42 \\
        --out_dir datasets/anchorbench_v2_external_pilot/

Usage (v2 External-only pilot, LLM-enhanced):
    PYTHONPATH=src python -m anchorbench_v1.generate \\
        --v2 --suites external --size pilot --seed 42 \\
        --out_dir datasets/anchorbench_v2_external_pilot/ \\
        --llm_enhance --concurrency 8

Sizes (v1):
    smoke  - 10 items (quick validation)
    core   - 600 items (5 suites × 6 domains × 20)
    stress - 200 stress-tagged items
    full   - core + stress (800)

Sizes (v2):
    smoke  - ~6 items (quick validation, 1 domain)
    pilot  - 90 items (3 domains × 2 difficulties × 3 offsets × 5)
    core   - 360 items (6 domains × 2 difficulties × 3 offsets × 10)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import __version__
from .domains import DOMAIN_IDS
from .itemspec_gen import (
    SUITES,
    generate_itemspecs,
    generate_external_v2_itemspecs,
    generate_history_v2_itemspecs,
    generate_rag_v2_itemspecs,
    generate_tool_v2_itemspecs,
    generate_icl_v2_itemspecs,
)
from .schema import ItemSpec, PromptView, write_jsonl
from .stress import generate_stress_specs
from .suites import SUITE_RENDERERS
from .suites.external import render_external_v2
from .suites.history import render_history_v2
from .suites.icl import render_icl_v2
from .suites.rag import build_full_corpus
from .suites.rag_v2 import build_full_corpus_v2, render_rag_v2
from .suites.tool_v2 import render_tool_v2

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


# ── v2 External-only generation ──────────────────────────────────────

_V2_SIZE_PARAMS = {
    "smoke": {"n_per_cell": 1, "domains": None, "split": "dev"},
    "pilot": {"n_per_cell": 5, "domains": None, "split": "pilot"},
    "core":  {"n_per_cell": 10, "domains": None, "split": "core"},
}


def generate_v2_dataset(
    size: str = "pilot",
    seed: int = 42,
    out_dir: str = "datasets/anchorbench_v2_external_pilot",
    llm_enhance: bool = False,
    concurrency: int = 8,
    max_retries: int = 5,
    domains: Optional[List[str]] = None,
) -> dict:
    """Generate External-suite v2 dataset with difficulty and stratified offsets.

    Args:
        size: "smoke" | "pilot" | "core"
        seed: Master random seed.
        out_dir: Output directory.
        llm_enhance: Use OpenRouter for unique scenario text.
        concurrency: Max parallel API requests.
        max_retries: Retries per request.
        domains: Override domain list (default: first 3 for pilot/smoke,
                 all 6 for core).

    Returns:
        Summary manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_version = _git_hash()

    params = _V2_SIZE_PARAMS.get(size)
    if params is None:
        raise ValueError(f"Unknown v2 size {size!r}. Choose from: {list(_V2_SIZE_PARAMS)}")

    effective_domains = domains or params["domains"]
    if effective_domains is None:
        effective_domains = DOMAIN_IDS[:3] if size in ("smoke", "pilot") else DOMAIN_IDS
    if size == "smoke":
        effective_domains = effective_domains[:1]

    all_specs = generate_external_v2_itemspecs(
        domains=effective_domains,
        n_per_cell=params["n_per_cell"],
        seed=seed,
        generator_version=gen_version,
        split=params["split"],
    )

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

    all_views: List[PromptView] = []
    for spec in all_specs:
        all_views.extend(render_external_v2(spec))

    specs_path = out / "itemspecs.jsonl"
    views_path = out / "promptviews.jsonl"
    write_jsonl(all_specs, specs_path)
    write_jsonl(all_views, views_path)

    difficulties = sorted(set(s.difficulty for s in all_specs))
    offsets = sorted(set(s.anchors.get("offset", 0) for s in all_specs))

    manifest = {
        "version": __version__,
        "benchmark_version": "2.0.0",
        "generator_version": gen_version,
        "seed": seed,
        "size": size,
        "llm_enhanced": llm_enhance,
        "llm_calls_made": llm_calls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "itemspecs": len(all_specs),
            "promptviews": len(all_views),
            "suites": ["external"],
            "domains": sorted(set(s.domain for s in all_specs)),
            "difficulties": difficulties,
            "anchor_offsets": offsets,
            "conditions_per_item": 5,
        },
        "files": {
            "itemspecs": str(specs_path),
            "promptviews": str(views_path),
        },
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ── v2 History generation ────────────────────────────────────────────

def generate_history_v2_dataset(
    size: str = "pilot",
    seed: int = 42,
    out_dir: str = "datasets/anchorbench_v2_history_pilot",
    llm_enhance: bool = False,
    concurrency: int = 8,
    max_retries: int = 5,
    domains: Optional[List[str]] = None,
) -> dict:
    """Generate History v2 dataset (same six domains as External v2; two-stage protocol).

    Args:
        size: "smoke" | "pilot" | "core"
        seed: Master random seed.
        out_dir: Output directory.
        llm_enhance: Use OpenRouter for unique scenario text (optional).
        concurrency: Max parallel API requests.
        max_retries: Retries per request.
        domains: Override domain list (default: all 6 for pilot/core, 1 for smoke).

    Returns:
        Summary manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_version = _git_hash()

    params = _V2_SIZE_PARAMS.get(size)
    if params is None:
        raise ValueError(f"Unknown v2 size {size!r}. Choose from: {list(_V2_SIZE_PARAMS)}")

    effective_domains = domains or params["domains"]
    if effective_domains is None:
        effective_domains = DOMAIN_IDS[:1] if size == "smoke" else DOMAIN_IDS
    if size == "smoke":
        effective_domains = effective_domains[:1]

    all_specs = generate_history_v2_itemspecs(
        domains=effective_domains,
        n_per_cell=params["n_per_cell"],
        seed=seed,
        generator_version=gen_version,
        split=params["split"],
    )

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

    all_views: List[PromptView] = []
    for spec in all_specs:
        all_views.extend(render_history_v2(spec))

    specs_path = out / "itemspecs.jsonl"
    views_path = out / "promptviews.jsonl"
    write_jsonl(all_specs, specs_path)
    write_jsonl(all_views, views_path)

    manifest = {
        "version": __version__,
        "benchmark_version": "2.0.0",
        "suite": "history_v2",
        "generator_version": gen_version,
        "seed": seed,
        "size": size,
        "llm_enhanced": llm_enhance,
        "llm_calls_made": llm_calls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "itemspecs": len(all_specs),
            "promptviews": len(all_views),
            "suites": ["history_v2"],
            "domains": sorted(set(s.domain for s in all_specs)),
            "difficulties": sorted(set(s.difficulty for s in all_specs)),
            "conditions_per_item": 5,
        },
        "files": {
            "itemspecs": str(specs_path),
            "promptviews": str(views_path),
        },
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ── v2 RAG generation ────────────────────────────────────────────────

def generate_rag_v2_dataset(
    size: str = "pilot",
    seed: int = 42,
    out_dir: str = "datasets/anchorbench_v2_rag_pilot",
    llm_enhance: bool = False,
    concurrency: int = 8,
    max_retries: int = 5,
    domains: Optional[List[str]] = None,
) -> dict:
    """Generate RAG v2 dataset (ecological anchoring via retrieved documents).

    Same six domains, evidence backbone, and difficulty axis as External v2.
    5 conditions per item: control, irrelevant_low/high, plausible_low/high.
    Frozen 3-document mini-corpus per item with deterministic retrieval.

    Args:
        size: "smoke" | "pilot" | "core"
        seed: Master random seed.
        out_dir: Output directory.
        llm_enhance: Use OpenRouter for unique scenario text (optional).
        concurrency: Max parallel API requests.
        max_retries: Retries per request.
        domains: Override domain list (default: all 6).

    Returns:
        Summary manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_version = _git_hash()

    params = _V2_SIZE_PARAMS.get(size)
    if params is None:
        raise ValueError(f"Unknown v2 size {size!r}. Choose from: {list(_V2_SIZE_PARAMS)}")

    effective_domains = domains or params["domains"]
    if effective_domains is None:
        effective_domains = DOMAIN_IDS[:1] if size == "smoke" else DOMAIN_IDS
    if size == "smoke":
        effective_domains = effective_domains[:1]

    all_specs = generate_rag_v2_itemspecs(
        domains=effective_domains,
        n_per_cell=params["n_per_cell"],
        seed=seed,
        generator_version=gen_version,
        split=params["split"],
    )

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

    all_views: List[PromptView] = []
    for spec in all_specs:
        all_views.extend(render_rag_v2(spec))

    specs_path = out / "itemspecs.jsonl"
    views_path = out / "promptviews.jsonl"
    write_jsonl(all_specs, specs_path)
    write_jsonl(all_views, views_path)

    corpus = build_full_corpus_v2(all_specs)
    corpus_path = out / "anchorbench_corpus_v2.jsonl"
    write_jsonl(corpus, corpus_path)

    manifest = {
        "version": __version__,
        "benchmark_version": "2.0.0",
        "suite": "rag_v2",
        "generator_version": gen_version,
        "seed": seed,
        "size": size,
        "llm_enhanced": llm_enhance,
        "llm_calls_made": llm_calls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "itemspecs": len(all_specs),
            "promptviews": len(all_views),
            "corpus_docs": len(corpus),
            "suites": ["rag_v2"],
            "domains": sorted(set(s.domain for s in all_specs)),
            "difficulties": sorted(set(s.difficulty for s in all_specs)),
            "conditions_per_item": 5,
        },
        "files": {
            "itemspecs": str(specs_path),
            "promptviews": str(views_path),
            "corpus": str(corpus_path),
        },
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ── v2 Tool generation ────────────────────────────────────────────────

def generate_tool_v2_dataset(
    size: str = "pilot",
    seed: int = 42,
    out_dir: str = "datasets/anchorbench_v2_tool_pilot",
    llm_enhance: bool = False,
    concurrency: int = 8,
    max_retries: int = 5,
    domains: Optional[List[str]] = None,
) -> dict:
    """Generate Tool v2 dataset (anchoring via structured tool outputs).

    Same six domains, evidence backbone, and difficulty axis as External v2.
    5 conditions per item: control, irrelevant_low/high, plausible_low/high.
    Deterministic tool outputs with condition-dependent anchor placement.

    Args:
        size: "smoke" | "pilot" | "core"
        seed: Master random seed.
        out_dir: Output directory.
        llm_enhance: Use OpenRouter for unique scenario text (optional).
        concurrency: Max parallel API requests.
        max_retries: Retries per request.
        domains: Override domain list (default: all 6).

    Returns:
        Summary manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_version = _git_hash()

    params = _V2_SIZE_PARAMS.get(size)
    if params is None:
        raise ValueError(f"Unknown v2 size {size!r}. Choose from: {list(_V2_SIZE_PARAMS)}")

    effective_domains = domains or params["domains"]
    if effective_domains is None:
        effective_domains = DOMAIN_IDS[:1] if size == "smoke" else DOMAIN_IDS
    if size == "smoke":
        effective_domains = effective_domains[:1]

    all_specs = generate_tool_v2_itemspecs(
        domains=effective_domains,
        n_per_cell=params["n_per_cell"],
        seed=seed,
        generator_version=gen_version,
        split=params["split"],
    )

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

    all_views: List[PromptView] = []
    for spec in all_specs:
        all_views.extend(render_tool_v2(spec))

    specs_path = out / "itemspecs.jsonl"
    views_path = out / "promptviews.jsonl"
    write_jsonl(all_specs, specs_path)
    write_jsonl(all_views, views_path)

    manifest = {
        "version": __version__,
        "benchmark_version": "2.0.0",
        "suite": "tool_v2",
        "generator_version": gen_version,
        "seed": seed,
        "size": size,
        "llm_enhanced": llm_enhance,
        "llm_calls_made": llm_calls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "itemspecs": len(all_specs),
            "promptviews": len(all_views),
            "suites": ["tool_v2"],
            "domains": sorted(set(s.domain for s in all_specs)),
            "difficulties": sorted(set(s.difficulty for s in all_specs)),
            "conditions_per_item": 5,
        },
        "files": {
            "itemspecs": str(specs_path),
            "promptviews": str(views_path),
        },
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ── v2 ICL generation ─────────────────────────────────────────────────

def generate_icl_v2_dataset(
    size: str = "pilot",
    seed: int = 42,
    out_dir: str = "datasets/anchorbench_v2_icl_pilot",
    llm_enhance: bool = False,
    concurrency: int = 8,
    max_retries: int = 5,
    domains: Optional[List[str]] = None,
) -> dict:
    """Generate ICL v2 dataset (anchoring via metadata priming).

    Same six domains, evidence backbone, and difficulty axis as External v2.
    5 conditions per item: control, irrelevant_low/high, plausible_low/high.
    Demo answers are neutral and identical across conditions; only demo
    header metadata varies.

    Args:
        size: "smoke" | "pilot" | "core"
        seed: Master random seed.
        out_dir: Output directory.
        llm_enhance: Use OpenRouter for unique scenario text (optional).
        concurrency: Max parallel API requests.
        max_retries: Retries per request.
        domains: Override domain list (default: all 6).

    Returns:
        Summary manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_version = _git_hash()

    params = _V2_SIZE_PARAMS.get(size)
    if params is None:
        raise ValueError(f"Unknown v2 size {size!r}. Choose from: {list(_V2_SIZE_PARAMS)}")

    effective_domains = domains or params["domains"]
    if effective_domains is None:
        effective_domains = DOMAIN_IDS[:1] if size == "smoke" else DOMAIN_IDS
    if size == "smoke":
        effective_domains = effective_domains[:1]

    all_specs = generate_icl_v2_itemspecs(
        domains=effective_domains,
        n_per_cell=params["n_per_cell"],
        seed=seed,
        generator_version=gen_version,
        split=params["split"],
    )

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

    all_views: List[PromptView] = []
    for spec in all_specs:
        all_views.extend(render_icl_v2(spec))

    specs_path = out / "itemspecs.jsonl"
    views_path = out / "promptviews.jsonl"
    write_jsonl(all_specs, specs_path)
    write_jsonl(all_views, views_path)

    manifest = {
        "version": __version__,
        "benchmark_version": "2.0.0",
        "suite": "icl_v2",
        "generator_version": gen_version,
        "seed": seed,
        "size": size,
        "llm_enhanced": llm_enhance,
        "llm_calls_made": llm_calls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "itemspecs": len(all_specs),
            "promptviews": len(all_views),
            "suites": ["icl_v2"],
            "domains": sorted(set(s.domain for s in all_specs)),
            "difficulties": sorted(set(s.difficulty for s in all_specs)),
            "conditions_per_item": 5,
        },
        "files": {
            "itemspecs": str(specs_path),
            "promptviews": str(views_path),
        },
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate AnchorBench dataset (v1 or v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "v1 Sizes:\n"
            "  smoke   10 items (quick validation)\n"
            "  core    600 items (5 suites × 6 domains × 20)\n"
            "  stress  200 stress-tagged items\n"
            "  full    core + stress (800)\n"
            "\n"
            "v2 Sizes (External-only):\n"
            "  smoke   ~6 items (1 domain, quick validation)\n"
            "  pilot   90 items (3 domains × 2 difficulties × 3 offsets × 5)\n"
            "  core    360 items (6 domains × 2 difficulties × 3 offsets × 10)\n"
            "\n"
            "LLM enhancement:\n"
            "  --llm_enhance uses the bulk_writer model (see config/models.yaml)\n"
            "  to generate unique scenario paragraphs per item via OpenRouter.\n"
            "  Requires OPENROUTER_API_KEY env var. Results are cached so\n"
            "  re-runs with the same output dir skip already-generated items.\n"
        ),
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use v2 generation (External suite with difficulty + stratified offsets)",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default=None,
        help="Comma-separated suite list (v1); v2: 'history'/'history_v2' for History, 'rag'/'rag_v2' for RAG, 'tool'/'tool_v2' for Tool, 'icl'/'icl_v2' for ICL, else External",
    )
    parser.add_argument(
        "--size",
        choices=["smoke", "core", "stress", "full", "pilot"],
        default="core",
        help="Dataset size preset (default: core)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: datasets/anchorbench_v1 or v2)",
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
    v2_size = args.size if args.size in ("smoke", "pilot", "core") else "pilot"
    v2_suites = (args.suites or "").strip().lower().replace(" ", "").split(",") if args.suites else []
    use_history_v2 = args.v2 and ("history" in v2_suites or "history_v2" in v2_suites)
    use_rag_v2 = args.v2 and ("rag" in v2_suites or "rag_v2" in v2_suites)
    use_tool_v2 = args.v2 and ("tool" in v2_suites or "tool_v2" in v2_suites)
    use_icl_v2 = args.v2 and ("icl" in v2_suites or "icl_v2" in v2_suites)

    if use_icl_v2:
        default_out = "datasets/anchorbench_v2_icl_pilot"
        out_dir = args.out_dir or default_out
        logger.info(
            "Generating AnchorBench v2 ICL (%s, seed=%d, mode=%s)...",
            v2_size, args.seed, mode,
        )
        manifest = generate_icl_v2_dataset(
            size=v2_size,
            seed=args.seed,
            out_dir=out_dir,
            llm_enhance=args.llm_enhance,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )
    elif use_tool_v2:
        default_out = "datasets/anchorbench_v2_tool_pilot"
        out_dir = args.out_dir or default_out
        logger.info(
            "Generating AnchorBench v2 Tool (%s, seed=%d, mode=%s)...",
            v2_size, args.seed, mode,
        )
        manifest = generate_tool_v2_dataset(
            size=v2_size,
            seed=args.seed,
            out_dir=out_dir,
            llm_enhance=args.llm_enhance,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )
    elif use_rag_v2:
        default_out = "datasets/anchorbench_v2_rag_pilot"
        out_dir = args.out_dir or default_out
        logger.info(
            "Generating AnchorBench v2 RAG (%s, seed=%d, mode=%s)...",
            v2_size, args.seed, mode,
        )
        manifest = generate_rag_v2_dataset(
            size=v2_size,
            seed=args.seed,
            out_dir=out_dir,
            llm_enhance=args.llm_enhance,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )
    elif use_history_v2:
        default_out = "datasets/anchorbench_v2_history_pilot"
        out_dir = args.out_dir or default_out
        logger.info(
            "Generating AnchorBench v2 History (%s, seed=%d, mode=%s)...",
            v2_size, args.seed, mode,
        )
        manifest = generate_history_v2_dataset(
            size=v2_size,
            seed=args.seed,
            out_dir=out_dir,
            llm_enhance=args.llm_enhance,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )
    elif args.v2 or args.size == "pilot":
        default_out = "datasets/anchorbench_v2_external_pilot"
        out_dir = args.out_dir or default_out
        logger.info(
            "Generating AnchorBench v2 External (%s, seed=%d, mode=%s)...",
            v2_size, args.seed, mode,
        )
        manifest = generate_v2_dataset(
            size=v2_size,
            seed=args.seed,
            out_dir=out_dir,
            llm_enhance=args.llm_enhance,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )
    else:
        out_dir = args.out_dir or "datasets/anchorbench_v1"
        logger.info(
            "Generating AnchorBench v1 (%s, seed=%d, mode=%s)...",
            args.size, args.seed, mode,
        )
        manifest = generate_dataset(
            size=args.size,
            seed=args.seed,
            out_dir=out_dir,
            llm_enhance=args.llm_enhance,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )

    logger.info(
        "Done! %d itemspecs, %d promptviews",
        manifest["counts"]["itemspecs"],
        manifest["counts"]["promptviews"],
    )
    if manifest.get("llm_calls_made", 0) > 0:
        logger.info("LLM API calls made: %d", manifest["llm_calls_made"])
    logger.info("Files written to: %s", out_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
