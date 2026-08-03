"""Minimal OpenRouter client with provenance logging and optional concurrency.

Supports:
- Explicit model IDs (no "latest" aliases)
- Full provenance per call (model_id, temperature, seed, request_id, usage, ...)
- Retry with exponential backoff on 429 / 5xx
- Concurrent batch requests via ThreadPoolExecutor
- Artifact saving for debugging
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRYABLE = {429, 500, 502, 503, 504}


class OpenRouterClient:
    """Thin OpenRouter API wrapper with retry, batching, and provenance logging."""

    def __init__(
        self,
        api_key: str | None = None,
        artifact_dir: str | None = None,
        max_retries: int = 5,
        timeout: int = 90,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise OSError(
                "Set OPENROUTER_API_KEY env var or pass api_key."
            )
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    # ── single request ───────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        seed: int | None = None,
        extra_body: dict | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Send one chat completion. Returns (text, provenance)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if extra_body:
            payload.update(extra_body)

        prompt_hash = hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode()
        ).hexdigest()[:16]

        text, provenance = self._request_with_retry(payload, prompt_hash)
        provenance.update(
            model_id=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            prompt_hash=prompt_hash,
        )
        return text, provenance

    # ── batch concurrent requests ────────────────────────────────────

    def chat_batch(
        self,
        requests_list: list[dict[str, Any]],
        concurrency: int = 8,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Run multiple chat() calls concurrently. Each dict in requests_list
        is passed as kwargs to chat(). Returns results in input order."""
        results: list[tuple[str, dict] | None] = [None] * len(requests_list)

        def _run(idx: int, kwargs: dict):
            return idx, self.chat(**kwargs)

        workers = max(1, min(concurrency, len(requests_list)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run, i, kw): i
                for i, kw in enumerate(requests_list)
            }
            for fut in as_completed(futures):
                idx, result = fut.result()
                results[idx] = result

        return results  # type: ignore[return-value]

    # ── internals ────────────────────────────────────────────────────

    def _request_with_retry(
        self, payload: dict, prompt_hash: str
    ) -> tuple[str, dict]:
        """Execute a single API request with exponential-backoff retry."""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    _API_URL,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code in _RETRYABLE:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Retry %d/%d (HTTP %d), waiting %.1fs",
                        attempt + 1, self.max_retries, resp.status_code, wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                provenance = {
                    "request_id": data.get("id", ""),
                    "usage": data.get("usage", {}),
                }
                self._save_artifact(prompt_hash, payload, data)
                return text, provenance

            except requests.RequestException as exc:
                last_exc = exc
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning("Request error (attempt %d): %s", attempt + 1, exc)
                time.sleep(wait)

        raise RuntimeError(
            f"OpenRouter request failed after {self.max_retries + 1} attempts: {last_exc}"
        )

    def _save_artifact(self, prompt_hash: str, payload: dict, response: dict) -> None:
        """Persist request/response pair to the artifact directory."""
        if not self.artifact_dir:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        path = self.artifact_dir / f"{prompt_hash}_{ts}.json"
        with open(path, "w") as f:
            json.dump({"request": payload, "response": response}, f, indent=2)

    # ── model listing (for registry checks) ──────────────────────────

    def list_models(self) -> list[str]:
        """Fetch available model IDs from OpenRouter."""
        resp = self._session.get(
            "https://openrouter.ai/api/v1/models", timeout=30
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
