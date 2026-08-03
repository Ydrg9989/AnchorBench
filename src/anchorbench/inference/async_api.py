"""Async OpenRouter client with semaphore-based rate limiting."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        prompt: str,
        max_tokens: int = 8,
        temperature: float = 0.0,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Single API request with retry and rate limiting."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

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
                            usage = data.get("usage", {})
                            return {
                                "raw_text": content,
                                "usage": usage,
                                "status": 200,
                            }
                        elif resp.status in (401, 403):
                            text = await resp.text()
                            log.error("API auth error %d: %s", resp.status, text[:200])
                            raise RuntimeError(f"API authentication failed ({resp.status}): {text[:200]}")
                        elif resp.status in (429, 500, 502, 503, 504):
                            wait = min(self.base_wait * 2 ** attempt, 60.0)
                            log.warning("API %d, retrying in %.1fs", resp.status, wait)
                            await asyncio.sleep(wait)
                        else:
                            text = await resp.text()
                            log.error("API error %d: %s", resp.status, text[:200])
                            return {"raw_text": "", "usage": {}, "status": resp.status}
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    wait = min(self.base_wait * 2 ** attempt, 60.0)
                    log.warning("Request error: %s, retrying in %.1fs", e, wait)
                    await asyncio.sleep(wait)

            return {"raw_text": "", "usage": {}, "status": -1}

    async def query_batch(
        self,
        model_id: str,
        prompts: list[str],
        max_tokens: int = 8,
        temperature: float = 0.0,
        system_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """Query all prompts concurrently, respecting semaphore."""
        tasks = [
            self._single_request(model_id, p, max_tokens, temperature, system_prompt)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)
