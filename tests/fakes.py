"""Test doubles that sit at the same seam as the real backends.

``FakeBackend`` satisfies :class:`anchorbench.eval.backends.Backend`, so the
evaluation loops run end to end in-process with no model, no GPU and no
network. Answers come from a policy function of the text the model would
see, so a test can make Stage 1 and Stage 2 of the History protocol answer
differently, or make every prompt fail.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Policy = Callable[[str], str]


def constant(answer: str) -> Policy:
    return lambda _text: answer


class FakeBackend:
    """Deterministic in-memory ``Backend``.

    ``policy`` maps the last user message (or the plain prompt) to the reply.
    Every call is appended to :attr:`calls` as ``(method, n_prompts)`` so a
    test can assert which interface method the loop used. With
    ``report_usage=True`` the backend fills :attr:`last_usage` like the
    OpenRouter adapter does.
    """

    supports_structured = False

    def __init__(
        self,
        policy: Policy | None = None,
        *,
        model_id: str = "fake/model",
        supports_tool_messages: bool = True,
        report_usage: bool = False,
        fail: bool = False,
    ) -> None:
        self.model_id = model_id
        self.policy = policy or constant("50")
        self.supports_tool_messages = supports_tool_messages
        self.report_usage = report_usage
        self.fail = fail
        self.calls: list[tuple[str, int]] = []
        self.last_usage: list[dict[str, Any]] | None = None

    # -- helpers -------------------------------------------------------------

    def _reply(self, texts: list[str], method: str) -> list[str]:
        self.calls.append((method, len(texts)))
        if self.fail:
            raise RuntimeError("fake backend failure")
        if self.report_usage:
            self.last_usage = [
                {"prompt_tokens": len(t.split()), "completion_tokens": 1} for t in texts
            ]
        return [self.policy(t) for t in texts]

    @staticmethod
    def _last_user(messages: list[dict]) -> str:
        users = [m["content"] for m in messages if m.get("role") == "user"]
        return users[-1] if users else ""

    # -- Backend interface -------------------------------------------------

    def generate(self, prompt: str, *, max_tokens: int = 512,
                 temperature: float = 0.0, structured: bool = False) -> str:
        return self._reply([prompt], "generate")[0]

    def generate_chat(self, messages: list[dict], *, max_tokens: int = 512,
                      temperature: float = 0.0) -> str:
        return self._reply([self._last_user(messages)], "generate_chat")[0]

    def generate_batch(self, prompts: list[str], *, max_tokens: int = 512,
                       temperature: float = 0.0, batch_size: int = 16) -> list[str]:
        return self._reply(list(prompts), "generate_batch")

    def generate_chat_batch(self, messages_list: list[list[dict]], *, max_tokens: int = 512,
                            temperature: float = 0.0, batch_size: int = 16) -> list[str]:
        return self._reply([self._last_user(m) for m in messages_list], "generate_chat_batch")

    def generate_batch_tool(self, messages_list: list[list[dict]], tools: list[dict] | None = None,
                            max_tokens: int = 64, temperature: float = 0.0,
                            batch_size: int = 16) -> list[str]:
        if not self.supports_tool_messages:
            raise NotImplementedError("this fake does not take tool messages")
        return self._reply([self._last_user(m) for m in messages_list], "generate_batch_tool")

    def generate_for_extraction(self, raw_output: str, extraction_prompt: str) -> str:
        return self._reply([extraction_prompt], "generate_for_extraction")[0]
