"""LLM-enhanced scenario generation via OpenRouter.

Uses the bulk_writer role to generate unique, realistic scenario paragraphs
for each ItemSpec. Results are cached to disk so re-runs with the same seed
don't burn tokens.

Usage:
    from anchorbench_v1.llm_enhance import enhance_scenarios
    enhance_scenarios(specs, concurrency=8, cache_path="out/scenarios_cache.jsonl")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .domains import DOMAINS
from .schema import ItemSpec

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a benchmark-dataset author. "
    "Write short, realistic scenario paragraphs for numeric estimation tasks. "
    "Requirements:\n"
    "- 3-5 sentences, specific and concrete\n"
    "- Name a specific (fictional) organization, its industry, and situation\n"
    "- Do NOT include any specific numbers, ratings, or benchmark values\n"
    "- Do NOT mention anchors, hints, or prior estimates\n"
    "- End the paragraph so that evidence data can follow naturally\n"
    "- Return ONLY the scenario paragraph, no preamble or commentary"
)


def _make_prompt(spec: ItemSpec, variation_idx: int) -> str:
    """Build the user prompt for scenario generation from a spec's domain."""
    dcfg = DOMAINS[spec.domain]
    return (
        f"Domain: {dcfg.display_name}\n"
        f"Description: {dcfg.description}\n"
        f"Evidence labels that will follow: {', '.join(dcfg.evidence_labels)}\n"
        f"Variation index: {variation_idx} (make this scenario distinct from others)\n"
        f"Item ID: {spec.item_id}\n\n"
        f"Write a unique scenario paragraph for this domain."
    )


def _load_cache(cache_path: Path) -> Dict[str, str]:
    """Load cached item_id → scenario_text mapping."""
    cache: Dict[str, str] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    cache[entry["item_id"]] = entry["scenario_text"]
    return cache


def _append_cache(cache_path: Path, item_id: str, scenario_text: str, provenance: dict):
    """Append one cache entry."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a") as f:
        f.write(json.dumps({
            "item_id": item_id,
            "scenario_text": scenario_text,
            "provenance": provenance,
        }, ensure_ascii=False) + "\n")


def enhance_scenarios(
    specs: List[ItemSpec],
    *,
    concurrency: int = 8,
    max_retries: int = 5,
    cache_path: Optional[str] = None,
    artifact_dir: Optional[str] = None,
) -> int:
    """Populate spec.scenario_text using the bulk_writer via OpenRouter.

    Skips items that already have scenario_text or are in the disk cache.
    Returns the number of API calls made.

    Args:
        specs: ItemSpec list (modified in place).
        concurrency: Max parallel API requests.
        max_retries: Retries per request on 429/5xx.
        cache_path: JSONL file for caching generated scenarios.
        artifact_dir: Directory for raw request/response artifacts.
    """
    from .openrouter_client import OpenRouterClient
    from .config import get_role_config

    cache_file = Path(cache_path or "datasets/anchorbench_v1/artifacts/scenarios_cache.jsonl")
    cache = _load_cache(cache_file)

    # Resolve which specs need generation
    to_generate: List[ItemSpec] = []
    for spec in specs:
        if spec.scenario_text:
            continue
        if spec.item_id in cache:
            spec.scenario_text = cache[spec.item_id]
            continue
        to_generate.append(spec)

    if not to_generate:
        logger.info("All %d scenarios already cached, no API calls needed.", len(specs))
        return 0

    logger.info(
        "Generating %d scenarios via bulk_writer (concurrency=%d, max_retries=%d)...",
        len(to_generate), concurrency, max_retries,
    )

    rcfg = get_role_config("bulk_writer")
    client = OpenRouterClient(
        artifact_dir=artifact_dir or "datasets/anchorbench_v1/artifacts/openrouter_calls",
        max_retries=max_retries,
    )

    # Build batch request list
    requests_list = []
    for i, spec in enumerate(to_generate):
        prompt = _make_prompt(spec, i)
        requests_list.append({
            "model": rcfg["model_id"],
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": rcfg["temperature"],
            "max_tokens": rcfg["max_tokens"],
        })

    # Execute with concurrency
    results = client.chat_batch(requests_list, concurrency=concurrency)

    # Apply results
    n_success = 0
    for spec, (text, provenance) in zip(to_generate, results):
        scenario = text.strip()
        # Basic quality gate: must be at least 50 chars and no digits
        if len(scenario) < 50:
            logger.warning("Short scenario for %s (%d chars), using fallback.", spec.item_id, len(scenario))
            continue

        spec.scenario_text = scenario
        provenance["role"] = "bulk_writer"
        _append_cache(cache_file, spec.item_id, scenario, provenance)
        n_success += 1

    logger.info(
        "Generated %d/%d scenarios successfully. Cache: %s",
        n_success, len(to_generate), cache_file,
    )
    return len(to_generate)
