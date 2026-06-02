"""Independent async LLM client for judge workloads."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral
from pathlib import Path
from typing import Any

import httpx

from .schemas import LLMResponse


_tokenizer = None


def _init_tokenizer(model_path: str) -> None:
    global _tokenizer
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _tokenize_chat(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not messages:
        return payload
    tools = payload.get("tools")
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict):
        kwargs.update(chat_template_kwargs)
    if tools:
        kwargs["tools"] = tools
    input_ids = _plain_input_ids(_tokenizer.apply_chat_template(messages, **kwargs))
    consumed = {
        "messages",
        "tools",
        "temperature",
        "max_tokens",
        "max_new_tokens",
        "chat_template_kwargs",
    }
    out = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": payload.get("temperature", 0.0),
            "max_new_tokens": payload.get(
                "max_tokens", payload.get("max_new_tokens", 512)
            ),
        },
    }
    out.update({k: v for k, v in payload.items() if k not in consumed})
    return out


def _plain_input_ids(value: Any) -> list[int]:
    """Normalize tokenizer chat-template output into JSON-serializable ids."""
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, Mapping):
        value = value.get("input_ids")
    elif hasattr(value, "data") and isinstance(value.data, Mapping):
        value = value.data.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(x, Integral) for x in value):
        raise TypeError(
            "apply_chat_template must return input_ids as a list of integers; "
            f"got {type(value).__name__}"
        )
    return [int(x) for x in value]


class JudgeLLMClient:
    """Batch HTTP client for SGLang-compatible /generate endpoints.

    If ``tokenize_chat`` is true, chat ``messages`` are converted into
    ``input_ids`` locally using ``model_path``. Otherwise payloads are posted as
    provided, which is useful for servers accepting ``text`` or ``messages``.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 31877,
        endpoint: str = "/generate",
        base_url: str | None = None,
        api_key: str | None = None,
        model_path: str | None = None,
        tokenize_chat: bool = False,
        tokenizer_workers: int = 8,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.base_url = base_url or f"http://{host}:{port}{endpoint}"
        self.api_key = api_key
        self.model_path = model_path
        self.tokenize_chat = tokenize_chat
        self.tokenizer_workers = tokenizer_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._pool: ProcessPoolExecutor | None = None

    def __call__(
        self,
        payloads: dict[str, Any] | list[dict[str, Any]],
        *,
        max_concurrency: int = 8,
    ) -> LLMResponse | list[LLMResponse]:
        single = isinstance(payloads, dict)
        batch = [payloads] if single else payloads
        prepared = self._prepare_payloads(batch)
        result = asyncio.run(
            self._async_batch(prepared, max_concurrency=max_concurrency)
        )
        return result[0] if single else result

    def _prepare_payloads(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.tokenize_chat:
            return [self._normalize_sampling_params(p) for p in payloads]
        if not self.model_path:
            raise ValueError("model_path is required when tokenize_chat=true")
        model_path = Path(self.model_path)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                "model_path does not exist in the judge runner environment: "
                f"{self.model_path}"
            )
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.tokenizer_workers,
                initializer=_init_tokenizer,
                initargs=(self.model_path,),
            )
        return list(self._pool.map(_tokenize_chat, payloads, chunksize=64))

    @staticmethod
    def _normalize_sampling_params(payload: dict[str, Any]) -> dict[str, Any]:
        if "sampling_params" in payload:
            return payload
        if "temperature" not in payload and "max_tokens" not in payload:
            return payload
        out = dict(payload)
        temp = out.pop("temperature", 0.0)
        max_tokens = out.pop("max_tokens", out.pop("max_new_tokens", 512))
        out["sampling_params"] = {
            "temperature": temp,
            "max_new_tokens": max_tokens,
        }
        return out

    async def _async_batch(
        self,
        payloads: list[dict[str, Any]],
        *,
        max_concurrency: int,
    ) -> list[LLMResponse]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        limits = httpx.Limits(
            max_connections=max_concurrency,
            max_keepalive_connections=0,
        )
        timeout = httpx.Timeout(self.timeout, connect=10.0, pool=30.0)
        sem = asyncio.Semaphore(max_concurrency)
        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            limits=limits,
            http2=False,
        ) as client:
            return await asyncio.gather(
                *(self._post_one(client, sem, payload) for payload in payloads)
            )

    async def _post_one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        payload: dict[str, Any],
    ) -> LLMResponse:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(self.base_url, content=encoded)
                    resp.raise_for_status()
                    data = resp.json()
                    return self._normalize_response(data)
                except Exception as exc:  # noqa: BLE001 - record HTTP/runtime errors
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (2**attempt))
                        continue
                    return LLMResponse(
                        ok=False,
                        raw={},
                        error={
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "attempts": attempt + 1,
                            "timestamp": time.time(),
                        },
                    )
        raise AssertionError("unreachable")

    @staticmethod
    def _normalize_response(data: dict[str, Any]) -> LLMResponse:
        text = data.get("text")
        if text is None and isinstance(data.get("choices"), list) and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                text = choice.get("text")
                if text is None and isinstance(choice.get("message"), dict):
                    text = choice["message"].get("content")
        return LLMResponse(ok=True, text=text, raw=data, error=None)

    def __del__(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)


class OpenAIChatClient:
    """Batch HTTP client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.model = _env_or_config("MODEL", model)
        raw_base_url = _env_or_config("BASE_URL", base_url)
        self.api_key = _env_or_config("API_KEY", api_key)
        missing = [
            name
            for name, value in (
                ("MODEL", self.model),
                ("BASE_URL", raw_base_url),
                ("API_KEY", self.api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "External API judge backend requires MODEL, BASE_URL, and API_KEY. "
                "Set these environment variables first, or configure client.model, "
                "client.base_url, and client.api_key in the judge config. "
                f"Missing: {', '.join(missing)}"
            )
        self.base_url = _normalize_openai_chat_url(raw_base_url)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def __call__(
        self,
        payloads: dict[str, Any] | list[dict[str, Any]],
        *,
        max_concurrency: int = 8,
    ) -> LLMResponse | list[LLMResponse]:
        single = isinstance(payloads, dict)
        batch = [payloads] if single else payloads
        prepared = [self._prepare_payload(payload) for payload in batch]
        result = asyncio.run(
            self._async_batch(prepared, max_concurrency=max_concurrency)
        )
        return result[0] if single else result

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if messages is None:
            text = str(payload.get("text") or "")
            messages = [{"role": "user", "content": text}]

        out: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": payload.get("temperature", 0.0),
            "max_tokens": payload.get(
                "max_tokens", payload.get("max_new_tokens", 512)
            ),
        }
        for key in ("top_p", "presence_penalty", "frequency_penalty", "stop"):
            if key in payload:
                out[key] = payload[key]
        return out

    async def _async_batch(
        self,
        payloads: list[dict[str, Any]],
        *,
        max_concurrency: int,
    ) -> list[LLMResponse]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        limits = httpx.Limits(
            max_connections=max_concurrency,
            max_keepalive_connections=0,
        )
        timeout = httpx.Timeout(self.timeout, connect=10.0, pool=30.0)
        sem = asyncio.Semaphore(max_concurrency)
        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            limits=limits,
            http2=False,
        ) as client:
            return await asyncio.gather(
                *(self._post_one(client, sem, payload) for payload in payloads)
            )

    async def _post_one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        payload: dict[str, Any],
    ) -> LLMResponse:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(self.base_url, content=encoded)
                    resp.raise_for_status()
                    data = resp.json()
                    return JudgeLLMClient._normalize_response(data)
                except Exception as exc:  # noqa: BLE001 - record HTTP/runtime errors
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (2**attempt))
                        continue
                    return LLMResponse(
                        ok=False,
                        raw={},
                        error={
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "attempts": attempt + 1,
                            "timestamp": time.time(),
                        },
                    )
        raise AssertionError("unreachable")


def _env_or_config(env_name: str, config_value: str | None) -> str | None:
    value = os.environ.get(env_name)
    if value:
        return value
    return config_value


def _normalize_openai_chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"
