"""OpenRouter as a :class:`anchorbench.eval.backends.Backend`.

Hosted API models used to bypass the evaluation loop: ``runners/api.py`` and
four other runners each drove :class:`AsyncOpenRouterClient` directly and
re-implemented parse, build_record and the JSONL writer. This adapter puts
the API behind the same interface as the local HF and vLLM backends, so one
loop in :mod:`anchorbench.eval.evaluator` serves every model.

Concurrency is the client's semaphore; ``batch_size`` arguments are accepted
for interface compatibility and ignored. Token usage is accumulated in
:attr:`usage` and the per-request usage of the most recent call is left in
:attr:`last_usage`, which the evaluation loop copies into each record's
``api_usage`` field.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Coroutine
from typing import Any

from anchorbench.inference.async_api import AsyncOpenRouterClient

log = logging.getLogger(__name__)


def _run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine to completion from synchronous code.

    ``asyncio.run`` refuses to nest inside a running loop (a notebook, for
    example); fall back to a worker thread with its own loop in that case.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class OpenRouterBackend:
    """Synchronous ``Backend`` over the async OpenRouter client."""

    supports_structured: bool = False
    # OpenRouter models receive the plaintext rendering of the Tool suite,
    # which is what the paper's API tier was evaluated on.
    supports_tool_messages: bool = False

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        max_concurrent: int = 50,
        max_retries: int = 5,
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._max_concurrent = max_concurrent
        self._max_retries = max_retries
        self.usage: dict[str, int] = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.last_usage: list[dict[str, Any]] | None = None

    # -- transport ---------------------------------------------------------

    async def _query(
        self, messages_list: list[list[dict[str, str]]], max_tokens: int, temperature: float,
    ) -> list[dict[str, Any]]:
        client = AsyncOpenRouterClient(
            api_key=self._api_key,
            max_concurrent=self._max_concurrent,
            max_retries=self._max_retries,
        )
        try:
            return await client.query_messages_batch(
                self.model_id, messages_list, max_tokens=max_tokens, temperature=temperature,
            )
        finally:
            await client.close()

    def _complete(
        self, messages_list: list[list[dict[str, str]]], max_tokens: int, temperature: float,
    ) -> list[str]:
        if not messages_list:
            self.last_usage = []
            return []
        results = _run_sync(self._query(messages_list, max_tokens, temperature))
        self.last_usage = [r.get("usage", {}) or {} for r in results]
        self.usage["requests"] += len(results)
        for u in self.last_usage:
            self.usage["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
        failed = sum(1 for r in results if r.get("status") != 200)
        if failed:
            log.warning("%d/%d API requests failed for %s", failed, len(results), self.model_id)
        return [r.get("raw_text", "") for r in results]

    # -- Backend interface -------------------------------------------------

    @staticmethod
    def _user(prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def generate(
        self, prompt: str, *, max_tokens: int = 512,
        temperature: float = 0.0, structured: bool = False,
    ) -> str:
        return self._complete([self._user(prompt)], max_tokens, temperature)[0]

    def generate_chat(
        self, messages: list[dict], *, max_tokens: int = 512, temperature: float = 0.0,
    ) -> str:
        return self._complete([messages], max_tokens, temperature)[0]

    def generate_batch(
        self, prompts: list[str], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        return self._complete([self._user(p) for p in prompts], max_tokens, temperature)

    def generate_chat_batch(
        self, messages_list: list[list[dict]], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        return self._complete(list(messages_list), max_tokens, temperature)

    def generate_batch_tool(
        self, messages_list: list[list[dict]], tools: list[dict] | None = None,
        max_tokens: int = 64, temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        raise NotImplementedError(
            "OpenRouterBackend does not send structured tool messages; the Tool "
            "suite is evaluated on its plaintext rendering for API models "
            "(supports_tool_messages is False, and runners.tool honours it)."
        )

    def generate_for_extraction(self, raw_output: str, extraction_prompt: str) -> str:
        return self.generate(extraction_prompt, max_tokens=16, temperature=0.0)
