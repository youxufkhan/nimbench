from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx

from nimbench.models import RequestSample

LOG = logging.getLogger("nimbench")

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_PROMPT = "Reply with one short word."
DEFAULT_TIMEOUT = 30.0
DEFAULT_TEMPERATURE = 0.0
FALLBACK_TEMPERATURE = 0.1
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "nimbench/0.1"


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute < 1:
            raise ValueError("--rpm must be >= 1")
        self.requests_per_minute = requests_per_minute
        self._interval = 60.0 / requests_per_minute
        self._next_allowed_at = 0.0

    def wait(self) -> float:
        now = time.monotonic()
        if self._next_allowed_at == 0.0:
            self._next_allowed_at = now
        delay = self._next_allowed_at - now
        if delay > 0:
            time.sleep(delay)
            now = time.monotonic()
        self._next_allowed_at = now + self._interval
        return max(delay, 0.0)


class APIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT,
        rate_limiter: RateLimiter | None = None,
    ):
        self.base_url = base_url.strip().rstrip("/")
        if not self.base_url:
            raise ValueError("Base URL cannot be empty.")
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limiter = rate_limiter

        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _request_with_retry(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None

        for attempt in range(2):
            try:
                if self.rate_limiter is not None:
                    delay = self.rate_limiter.wait()
                    if delay > 0:
                        LOG.info("rate limit wait %.2fs before %s %s", delay, method, url)

                response = self.client.request(method, url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code in RETRYABLE_STATUS and attempt == 0:
                    time.sleep(0.5)
                    continue
                # Store original response body on exception for classify_error_message
                setattr(exc, "_nimbench_body", exc.response.text)
                raise
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Request failed.")

    def discover_models(self) -> list[str]:
        payload = self._request_with_retry("GET", "models")
        return extract_model_ids(payload)

    def measure_once(self, model: str) -> RequestSample:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
            "max_tokens": 8,
            "temperature": DEFAULT_TEMPERATURE,
            "stream": False,
        }

        last_exc: Exception | None = None
        use_temperature_fallback = False

        for attempt in range(2):
            try:
                if self.rate_limiter is not None:
                    delay = self.rate_limiter.wait()
                    if delay > 0:
                        LOG.info("rate limit wait %.2fs before %s", delay, model)

                started = time.perf_counter()
                response = self.client.post(url, json=payload)
                response.raise_for_status()
                ended = time.perf_counter()

                latency_ms = (ended - started) * 1000.0
                resp_json = response.json()
                completion_tokens = extract_completion_tokens(resp_json)
                server_tps = extract_server_tokens_per_second(resp_json)

                return RequestSample(
                    latency_ms=latency_ms,
                    completion_tokens=completion_tokens,
                    server_tokens_per_second=server_tps,
                )
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                # Save body for classification
                setattr(exc, "_nimbench_body", exc.response.text)
                # Check temperature validation error
                if not use_temperature_fallback and is_temperature_validation_error(exc):
                    use_temperature_fallback = True
                    payload["temperature"] = FALLBACK_TEMPERATURE
                    LOG.info("retry %s with temperature=%.1f", model, FALLBACK_TEMPERATURE)
                    continue
                if exc.response.status_code in RETRYABLE_STATUS and attempt == 0:
                    time.sleep(0.5)
                    continue
                raise
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Request failed.")


def extract_model_ids(payload: Any) -> list[str]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            items = payload["data"]
        elif isinstance(payload.get("models"), list):
            items = payload["models"]
        else:
            items = []
    else:
        items = []

    models: list[str] = []
    seen: set[str] = set()
    for item in items:
        model_id: str | None = None
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            for key in ("id", "model", "name"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    model_id = value.strip()
                    break
        if model_id and model_id not in seen:
            models.append(model_id)
            seen.add(model_id)
    return models


def extract_completion_tokens(response: Any) -> int | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if isinstance(usage, dict):
        value = usage.get("completion_tokens")
        if isinstance(value, int) and value >= 0:
            return value
    stats = response.get("stats")
    if isinstance(stats, dict):
        value = stats.get("llm_output_token_length")
        if isinstance(value, int) and value >= 0:
            return value
        response_tokens = stats.get("response_tokens")
        if isinstance(response_tokens, dict):
            value = response_tokens.get("response_token_length")
            if isinstance(value, int) and value >= 0:
                return value
    return None


def extract_server_tokens_per_second(response: Any) -> float | None:
    if not isinstance(response, dict):
        return None
    stats = response.get("stats")
    if not isinstance(stats, dict):
        return None
    response_tokens = stats.get("response_tokens")
    if not isinstance(response_tokens, dict):
        return None
    value = response_tokens.get("tokens_per_second")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def is_temperature_validation_error(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code != 422:
        return False
    body = exc.response.text.lower()
    return "temperature" in body and (
        "greater_than" in body or "must be greater than 0" in body or "input should be greater than 0" in body
    )
