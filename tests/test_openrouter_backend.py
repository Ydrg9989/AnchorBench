"""The OpenRouter adapter, with the network replaced by a canned client."""

from __future__ import annotations

import pytest

from anchorbench.eval.backends import Backend
from anchorbench.inference import async_api
from anchorbench.inference.openrouter_backend import OpenRouterBackend


@pytest.fixture
def canned(monkeypatch):
    """Replace the HTTP round trip; echo the last user message with token counts."""
    sent: list[list[dict]] = []

    async def fake_query(self, model_id, messages_list, max_tokens=8, temperature=0.0):
        sent.extend(messages_list)
        return [
            {"raw_text": f"echo:{m[-1]['content']}",
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "status": 200}
            for m in messages_list
        ]

    monkeypatch.setattr(async_api.AsyncOpenRouterClient, "query_messages_batch", fake_query)
    return sent


def test_satisfies_the_backend_protocol():
    assert isinstance(OpenRouterBackend("openai/x", api_key="k"), Backend)
    assert OpenRouterBackend("openai/x", api_key="k").supports_tool_messages is False


def test_generate_batch_preserves_order_and_accumulates_usage(canned):
    b = OpenRouterBackend("openai/x", api_key="k")
    out = b.generate_batch(["a", "b", "c"])
    assert out == ["echo:a", "echo:b", "echo:c"]
    assert b.last_usage == [{"prompt_tokens": 10, "completion_tokens": 2}] * 3
    assert b.usage == {"requests": 3, "prompt_tokens": 30, "completion_tokens": 6}
    b.generate("d")
    assert b.usage["requests"] == 4


def test_chat_methods_send_the_messages_unchanged(canned):
    b = OpenRouterBackend("openai/x", api_key="k")
    convo = [{"role": "user", "content": "s1"}, {"role": "assistant", "content": "30"},
             {"role": "user", "content": "s2"}]
    assert b.generate_chat(convo) == "echo:s2"
    assert canned[-1] == convo
    assert b.generate_chat_batch([convo, convo]) == ["echo:s2", "echo:s2"]


def test_tool_messages_are_refused_explicitly():
    with pytest.raises(NotImplementedError):
        OpenRouterBackend("openai/x", api_key="k").generate_batch_tool([[{"role": "user", "content": "x"}]])


def test_empty_batch_makes_no_request(canned):
    b = OpenRouterBackend("openai/x", api_key="k")
    assert b.generate_batch([]) == [] and canned == [] and b.usage["requests"] == 0
