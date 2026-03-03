"""Role-based LLM generation with provenance tracking.

Usage:
    from anchorbench_v1.llm_generate import generate
    text, prov = generate("bulk_writer", "Write a scenario about pricing...")
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .config import get_role_config
from .openrouter_client import OpenRouterClient

_client: Optional[OpenRouterClient] = None


def _get_client(artifact_dir: Optional[str] = None) -> OpenRouterClient:
    global _client
    if _client is None:
        _client = OpenRouterClient(
            artifact_dir=artifact_dir
            or "datasets/anchorbench_v1/artifacts/openrouter_calls"
        )
    return _client


def generate(
    role: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    schema: Optional[dict] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    artifact_dir: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Generate text for a given role. Returns (text, provenance).

    Provenance includes: model_id, role, temperature, max_tokens, seed,
    request_id, prompt_hash, usage.
    """
    rcfg = get_role_config(role)
    client = _get_client(artifact_dir)

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    extra_body = {}
    if schema:
        extra_body["response_format"] = {
            "type": "json_schema",
            "json_schema": schema,
        }

    text, prov = client.chat(
        model=rcfg["model_id"],
        messages=messages,
        temperature=temperature if temperature is not None else rcfg["temperature"],
        max_tokens=max_tokens if max_tokens is not None else rcfg["max_tokens"],
        seed=seed,
        extra_body=extra_body or None,
    )
    prov["role"] = role
    return text, prov


def generate_batch(
    role: str,
    prompts: List[str],
    *,
    system: Optional[str] = None,
    seed: Optional[int] = None,
    concurrency: int = 8,
    artifact_dir: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Batch-generate for a role with concurrency."""
    rcfg = get_role_config(role)
    client = _get_client(artifact_dir)

    requests_list = []
    for prompt in prompts:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        requests_list.append({
            "model": rcfg["model_id"],
            "messages": messages,
            "temperature": rcfg["temperature"],
            "max_tokens": rcfg["max_tokens"],
            "seed": seed,
        })

    results = client.chat_batch(requests_list, concurrency=concurrency)
    for _, prov in results:
        prov["role"] = role
    return results
