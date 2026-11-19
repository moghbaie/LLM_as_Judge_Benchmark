"""Async OpenRouter client: pinned provider, retries with backoff, usage and cost capture.

The API key is read from the OPENROUTER_API_KEY environment variable or Colab Secrets.
It is never written to disk or logs.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import asdict, dataclass, field

import httpx

BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401, 402, 403}   # bad key / out of credits / forbidden: stop the whole run


class FatalAPIError(RuntimeError):
    """Stop everything (bad key, no credits)."""


class CallFailed(RuntimeError):
    """This call failed after all retries (or with a non-retryable 4xx). The item is skipped, not scored."""


@dataclass
class Completion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None          # as billed by OpenRouter (usage.cost)
    provider: str | None = None            # provider that actually served the call
    model: str | None = None               # model id echoed by the API
    latency_s: float = 0.0
    finish_reason: str | None = None
    attempts: int = 1
    cached: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        try:  # Colab Secrets
            from google.colab import userdata  # type: ignore
            key = userdata.get("OPENROUTER_API_KEY")
        except Exception:
            key = None
    if not key:
        raise FatalAPIError("OPENROUTER_API_KEY not set (env var, or Colab Secrets with notebook access on)")
    return key


def build_body(model_id: str, messages: list[dict], *, provider: str | None, temperature: float | None,
               max_tokens: int, json_mode: bool, reasoning: dict | None, seed: int | None) -> dict:
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if temperature is not None:   # None = endpoint rejects temperature (e.g. Anthropic, OpenAI)
        body["temperature"] = temperature
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if provider:
        # one endpoint only; never silently fall back to another quantization
        body["provider"] = {"only": [provider], "allow_fallbacks": False, "require_parameters": True}
    if reasoning is not None:
        body["reasoning"] = reasoning
    if seed is not None:
        body["seed"] = seed
    return body


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, *, timeout: float = 300.0, max_retries: int = 6,
                 base_delay: float = 2.0, max_delay: float = 60.0, transport: httpx.AsyncBaseTransport | None = None):
        self._key = api_key or get_api_key()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._http = httpx.AsyncClient(
            base_url=BASE_URL, timeout=timeout, transport=transport,
            headers={"Authorization": f"Bearer {self._key}", "X-Title": "llm-judge-benchmark"})

    async def aclose(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(self.max_delay, float(retry_after))
            except ValueError:
                pass
        return min(self.max_delay, self.base_delay * 2 ** attempt) * (0.5 + random.random() / 2)

    async def complete(self, model_id: str, messages: list[dict], *, provider: str | None = None,
                       temperature: float | None = 0.0, max_tokens: int = 4096, json_mode: bool = True,
                       reasoning: dict | None = None, seed: int | None = None) -> Completion:
        body = build_body(model_id, messages, provider=provider, temperature=temperature,
                          max_tokens=max_tokens, json_mode=json_mode, reasoning=reasoning, seed=seed)
        last_err = "unknown"
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            retry_after = None
            try:
                r = await self._http.post("/chat/completions", json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = f"{type(e).__name__}: {e}"
            else:
                if r.status_code in FATAL_STATUS:
                    raise FatalAPIError(f"HTTP {r.status_code}: {r.text[:300]}")
                if r.status_code in RETRY_STATUS:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    retry_after = r.headers.get("retry-after")
                elif r.status_code >= 400:
                    raise CallFailed(f"HTTP {r.status_code} (not retried): {r.text[:300]}")
                else:
                    data = r.json()
                    if "error" in data:  # provider error delivered with HTTP 200
                        last_err = f"provider error: {str(data['error'])[:200]}"
                    elif not data.get("choices"):
                        last_err = "no choices in response"
                    else:
                        c = self._to_completion(data, time.perf_counter() - t0, attempt + 1)
                        if c.text.strip() or attempt >= 2:   # retry empty output twice, then accept it
                            return c
                        last_err = "empty output"
            if attempt < self.max_retries:
                await asyncio.sleep(self._delay(attempt, retry_after))
        raise CallFailed(f"gave up after {self.max_retries + 1} attempts: {last_err}")

    @staticmethod
    def _to_completion(data: dict, latency: float, attempts: int) -> Completion:
        ch = data["choices"][0]
        msg = ch.get("message") or {}
        u = data.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        return Completion(
            text=msg.get("content") or "",
            tokens_in=int(u.get("prompt_tokens") or 0),
            tokens_out=int(u.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            cost_usd=float(u["cost"]) if u.get("cost") is not None else None,
            provider=data.get("provider"),
            model=data.get("model"),
            latency_s=round(latency, 3),
            finish_reason=ch.get("finish_reason"),
            attempts=attempts,
        )

    # ------------------------------------------------------------------ no-cost helpers
    async def key_info(self) -> dict:
        """Credit limit and usage of the key (does not spend anything)."""
        r = await self._http.get("/key")
        if r.status_code in FATAL_STATUS:
            raise FatalAPIError(f"HTTP {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        return r.json().get("data", {})


def list_endpoints(model_id: str, transport: httpx.BaseTransport | None = None) -> list[dict]:
    """Public endpoint list for a model (no key needed)."""
    with httpx.Client(base_url=BASE_URL, timeout=30, transport=transport) as c:
        r = c.get(f"/models/{model_id}/endpoints")
        r.raise_for_status()
        return r.json()["data"]["endpoints"]


def check_pin(spec, transport: httpx.BaseTransport | None = None, tolerance: float = 0.01) -> dict:
    """Is the pinned endpoint still live, and do its prices match the registry?"""
    eps = list_endpoints(spec.model_id, transport)
    match = next((e for e in eps if e.get("tag") == spec.provider), None)
    if match is None:
        return {"model": spec.key, "ok": False, "problem": f"endpoint {spec.provider!r} not listed",
                "available": [e.get("tag") for e in eps]}
    pin = float(match["pricing"]["prompt"]) * 1e6
    pout = float(match["pricing"]["completion"]) * 1e6
    problems = []
    if abs(pin - spec.price_in) > tolerance or abs(pout - spec.price_out) > tolerance:
        problems.append(f"price changed: now {pin:.2f}/{pout:.2f}, registry {spec.price_in}/{spec.price_out}")
    sp = match.get("supported_parameters", [])
    needed = ["response_format"] + (["temperature"] if spec.temperature_supported else [])
    for p in needed:
        if p not in sp:
            problems.append(f"endpoint does not support {p}")
    if not spec.temperature_supported and "temperature" in sp:
        problems.append("endpoint now accepts temperature: set temperature_supported=True")
    return {"model": spec.key, "ok": not problems, "problem": "; ".join(problems) or None,
            "quantization": match.get("quantization"), "price_in": pin, "price_out": pout}
