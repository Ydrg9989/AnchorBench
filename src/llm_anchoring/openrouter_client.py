"""Thin OpenRouter chat-completion client with retry and disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_CACHE_DIR = Path("data/interim/icl_anchor_v0/cache")
_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0


def _cache_key(model: str, messages: list[dict]) -> str:
    blob = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache(key: str, cache_dir: Path) -> dict | None:
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_cache(key: str, data: dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(data))


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Send a chat completion request to OpenRouter.

    Returns the full API response dict. Retries on 429/5xx with
    exponential backoff. Results are cached to disk by default.
    """
    key = _cache_key(model, messages)
    if use_cache:
        cached = _load_cache(key, cache_dir)
        if cached is not None:
            logger.debug("cache hit %s", key[:12])
            return cached

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    for attempt in range(_MAX_RETRIES):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            if use_cache:
                _save_cache(key, data, cache_dir)
            return data
        if resp.status_code not in _RETRYABLE:
            resp.raise_for_status()
        wait = _BACKOFF_BASE ** attempt
        logger.warning(
            "status %d, retry %d/%d in %.1fs",
            resp.status_code, attempt + 1, _MAX_RETRIES, wait,
        )
        time.sleep(wait)

    resp.raise_for_status()
    return {}  # unreachable, keeps mypy happy


def extract_content(response: dict[str, Any]) -> str:
    """Pull the assistant message text from an OpenRouter response."""
    return response["choices"][0]["message"]["content"]
