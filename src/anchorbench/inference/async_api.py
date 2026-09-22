"""Async OpenRouter client with semaphore-based rate limiting.

This is the transport. :class:`anchorbench.inference.openrouter_backend.OpenRouterBackend`
wraps it in the synchronous :class:`anchorbench.eval.backends.Backend` interface
that the evaluation loop speaks, so runners never touch this class directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Statuses worth retrying with exponential backoff; anything else is final.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class AsyncOpenRouterClient:
    """Async OpenRouter client with configurable concurrency."""

    def __init__(
        self,
        api_key: str | None = None,
        max_concurrent: int = 20,
        max_retries: int = 5,
        base_wait: float = 1.0,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.base_wait = base_wait
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _single_request(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 8,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """One chat-completion request with retry and rate limiting.

        Returns ``{"raw_text", "usage", "status"}``; a failed request has an
        empty ``raw_text`` and a non-200 status (``-1`` once retries are spent),
        never an exception, except for authentication errors which are raised
        because retrying them cannot help.
        """
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with self.semaphore:
            session = await self._get_session()
            for attempt in range(self.max_retries):
                try:
                    async with session.post(OPENROUTER_URL, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            content = (data.get("choices", [{}])[0]
                                       .get("message", {}).get("content", "") or "")
                            return {
                                "raw_text": content,
                                "usage": data.get("usage", {}),
                                "status": 200,
                            }
                        if resp.status in (401, 403):
                            text = await resp.text()
                            log.error("API auth error %d: %s", resp.status, text[:200])
                            raise RuntimeError(
                                f"API authentication failed ({resp.status}): {text[:200]}"
                            )
                        if resp.status in _RETRY_STATUSES:
                            wait = min(self.base_wait * 2 ** attempt, 60.0)
                            log.warning("API %d, retrying in %.1fs", resp.status, wait)
                            await asyncio.sleep(wait)
                            continue
                        text = await resp.text()
                        log.error("API error %d: %s", resp.status, text[:200])
                        return {"raw_text": "", "usage": {}, "status": resp.status}
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    wait = min(self.base_wait * 2 ** attempt, 60.0)
                    log.warning("Request error: %s, retrying in %.1fs", e, wait)
                    await asyncio.sleep(wait)

            return {"raw_text": "", "usage": {}, "status": -1}

    async def query_messages_batch(
        self,
        model_id: str,
        messages_list: list[list[dict[str, str]]],
        max_tokens: int = 8,
        temperature: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Send one chat completion per conversation, concurrently, in order."""
        tasks = [
            self._single_request(model_id, messages, max_tokens, temperature)
            for messages in messages_list
        ]
        return await asyncio.gather(*tasks)

    async def query_batch(
        self,
        model_id: str,
        prompts: list[str],
        max_tokens: int = 8,
        temperature: float = 0.0,
        system_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """Send one single-turn user prompt per request, concurrently, in order."""
        messages_list = []
        for prompt in prompts:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)
        return await self.query_messages_batch(model_id, messages_list, max_tokens, temperature)
