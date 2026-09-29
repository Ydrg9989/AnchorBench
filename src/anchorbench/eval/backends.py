"""Backend abstractions for AnchorBench evaluation.

HFBackend   — local HuggingFace Transformers inference
VLLMBackend — vLLM inference (PagedAttention, high GPU utilization)

Hosted models are served by
:class:`anchorbench.inference.openrouter_backend.OpenRouterBackend`, which
implements the same :class:`Backend` protocol, so the evaluation loop in
:mod:`anchorbench.eval.evaluator` does not know which kind it is driving.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


@runtime_checkable
class Backend(Protocol):
    """What the evaluation loop needs from a model, local or hosted.

    :class:`HFBackend`, :class:`VLLMBackend` and
    :class:`anchorbench.inference.openrouter_backend.OpenRouterBackend` all
    satisfy it, as does the ``FakeBackend`` the tests use. ``batch_size`` is
    a hint for backends that batch locally; hosted backends may ignore it.
    The loops only ever send batches, so there is no single-prompt method.
    """

    model_id: str

    @property
    def supports_tool_messages(self) -> bool:
        """Whether native tool-call messages can be sent (else use plaintext)."""
        ...

    def generate_batch(
        self, prompts: list[str], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]: ...

    def generate_chat_batch(
        self, messages_list: list[list[dict]], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]: ...

    def generate_batch_tool(
        self, messages_list: list[list[dict]], tools: list[dict] | None = None,
        max_tokens: int = 64, temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]: ...

def render_chat(tokenizer, messages: list[dict], tools: list[dict] | None = None) -> str:
    """The prompt string a chat template produces for ``messages``, with the
    tool schemas when given.

    A template that rejects the ``tools`` argument is an error here, not a
    reason to render without them: a Tool-suite prompt without the schema is
    a different experiment under the same suite name. Models without native
    tool support take the plaintext rendering (runners.tool --tool_plaintext).
    """
    if tools is None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
        )
    except TypeError as e:
        raise ValueError(
            "this model's chat template does not accept tool schemas; run the "
            "Tool suite with --tool_plaintext"
        ) from e


class HFBackend:
    """Local HuggingFace model backend."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        device_map: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        torch_dtype = getattr(torch, dtype, "auto")
        dm = device_map if device_map else device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=dm,
            trust_remote_code=True,
        )
        self.model.eval()
        self._torch = torch
        log.info("Loaded HF model %s on %s", model_id, self.model.device)

    @property
    def supports_tool_messages(self) -> bool:
        return True

    def _template(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        return render_chat(self.tokenizer, messages, tools)

    def _generate_texts(
        self, texts: list[str], max_tokens: int, temperature: float, batch_size: int,
    ) -> list[str]:
        """Left-padded batched generation over already-templated texts."""
        results: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, add_special_tokens=False,
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with self._torch.no_grad():
                out = self.model.generate(
                    **enc, **gen_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            results.extend(self.tokenizer.batch_decode(
                out[:, prompt_len:], skip_special_tokens=True,
            ))
        return results

    def generate_batch(
        self, prompts: list[str], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        texts = [self._template([{"role": "user", "content": p}]) for p in prompts]
        return self._generate_texts(texts, max_tokens, temperature, batch_size)

    def generate_chat_batch(
        self, messages_list: list[list[dict]], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        texts = [self._template(m) for m in messages_list]
        return self._generate_texts(texts, max_tokens, temperature, batch_size)

    def generate_batch_tool(
        self,
        messages_list: list[list[dict]],
        tools: list[dict] | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        batch_size: int = 16,
    ) -> list[str]:
        texts = [self._template(m, tools=tools) for m in messages_list]
        return self._generate_texts(texts, max_tokens, temperature, batch_size)

class VLLMBackend:
    """vLLM backend for high-throughput inference with full GPU utilization.

    Uses PagedAttention and continuous batching. Same chat/tool formatting as
    HFBackend (via Transformers tokenizer). Optional tensor parallelism for
    large models.
    """

    def __init__(
        self,
        model_id: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = 4096,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_id = model_id
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len or 4096
        self._dtype = dtype
        self._trust_remote_code = trust_remote_code

        # Tokenizer for chat/tool template (same as HF for consistency)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # vLLM engine
        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self._max_model_len,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )
        log.info(
            "Loaded vLLM model %s (tp=%s, gpu_util=%.2f)",
            model_id, tensor_parallel_size, gpu_memory_utilization,
        )

        self._sampling_params = SamplingParams(
            temperature=0.0, max_tokens=512,
        )

    @property
    def supports_tool_messages(self) -> bool:
        return True

    def _sampling(self, max_tokens: int, temperature: float = 0.0) -> Any:
        from vllm import SamplingParams
        return SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _prompt_for_user(self, prompt: str) -> str:
        return render_chat(self._tokenizer, [{"role": "user", "content": prompt}])

    def generate_batch(
        self, prompts: list[str], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        prompt_strings = [self._prompt_for_user(p) for p in prompts]
        sampling = self._sampling(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate(prompt_strings, sampling)
        return [o.outputs[0].text for o in outputs]

    def generate_chat_batch(
        self, messages_list: list[list[dict]], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        texts = [render_chat(self._tokenizer, m) for m in messages_list]
        sampling = self._sampling(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate(texts, sampling)
        return [o.outputs[0].text for o in outputs]

    def generate_batch_tool(
        self,
        messages_list: list[list[dict]],
        tools: list[dict] | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        batch_size: int = 16,
    ) -> list[str]:
        texts = [render_chat(self._tokenizer, messages, tools) for messages in messages_list]
        sampling = self._sampling(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate(texts, sampling)
        return [o.outputs[0].text for o in outputs]

