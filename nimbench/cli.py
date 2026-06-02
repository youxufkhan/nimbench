from __future__ import annotations

import argparse
import logging
import getpass
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_PROMPT = "Reply with one short word."
DEFAULT_TIMEOUT = 30.0
DEFAULT_REPEATS = 1
DEFAULT_REQUESTS_PER_MINUTE = 40
DEFAULT_TEMPERATURE = 0.0
FALLBACK_TEMPERATURE = 0.1
DEFAULT_SKIP_CACHE_FILENAME = "skip-cache.json"
DEFAULT_CACHE_DIR_NAME = "nimbench"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "nimbench/0.1"
NIMBENCH_CACHE_DIR_ENV = "NIMBENCH_CACHE_DIR"

CACHE_REASON_NOT_PROVISIONED = "not_provisioned"
CACHE_REASON_UNSUPPORTED_INPUT = "unsupported_input"
CACHE_REASON_DEGRADED = "degraded"
CACHE_REASON_TIMEOUT = "timeout"

CACHE_IMMEDIATE_REASONS = {
    CACHE_REASON_NOT_PROVISIONED,
    CACHE_REASON_UNSUPPORTED_INPUT,
}
CACHE_AFTER_REPEAT_REASONS = {
    CACHE_REASON_DEGRADED,
    CACHE_REASON_TIMEOUT,
}

CHAT_HINTS = (
    "chat",
    "coder",
    "deepseek",
    "dracarys",
    "gemma",
    "gpt-oss",
    "instruct",
    "kimi",
    "llama",
    "magistral",
    "mistral",
    "nemotron",
    "phi",
    "qwen",
    "sarvam",
    "seed",
    "starcoder",
    "step",
    "super",
    "thinking",
    "yi",
)

NON_CHAT_HINTS = (
    "audio",
    "classify",
    "classifier",
    "content-safety",
    "detect",
    "embed",
    "embedding",
    "fuyu",
    "gliner",
    "image",
    "guard",
    "moderation",
    "multimodal",
    "ocr",
    "pii",
    "rerank",
    "parse",
    "retriever",
    "reward",
    "safety",
    "speech",
    "vl",
    "topic-control",
    "translate",
    "vision",
)

LOG = logging.getLogger("nimbench")


@dataclass(slots=True)
class ModelResult:
    model: str
    samples_ms: list[float]
    errors: list[str]
    tokens_per_second_samples: list[float] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.samples_ms)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def median_ms(self) -> float | None:
        if not self.samples_ms:
            return None
        return float(statistics.median(self.samples_ms))

    @property
    def min_ms(self) -> float | None:
        if not self.samples_ms:
            return None
        return float(min(self.samples_ms))

    @property
    def max_ms(self) -> float | None:
        if not self.samples_ms:
            return None
        return float(max(self.samples_ms))

    @property
    def median_tokens_per_second(self) -> float | None:
        if not self.tokens_per_second_samples:
            return None
        return float(statistics.median(self.tokens_per_second_samples))

    @property
    def min_tokens_per_second(self) -> float | None:
        if not self.tokens_per_second_samples:
            return None
        return float(min(self.tokens_per_second_samples))

    @property
    def max_tokens_per_second(self) -> float | None:
        if not self.tokens_per_second_samples:
            return None
        return float(max(self.tokens_per_second_samples))

    def as_dict(self, rank: int | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "samples_ms": [round(sample, 3) for sample in self.samples_ms],
            "tokens_per_second_samples": [round(sample, 3) for sample in self.tokens_per_second_samples],
            "errors": list(self.errors),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "median_tokens_per_second": self.median_tokens_per_second,
            "min_tokens_per_second": self.min_tokens_per_second,
            "max_tokens_per_second": self.max_tokens_per_second,
        }
        if rank is not None:
            payload["rank"] = rank
        return payload


@dataclass(slots=True)
class RequestSample:
    latency_ms: float
    completion_tokens: int | None = None
    server_tokens_per_second: float | None = None


@dataclass(slots=True)
class SkipCacheEntry:
    reason: str
    last_seen_at: float
    last_error: str
    failure_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "last_seen_at": self.last_seen_at,
            "last_error": self.last_error,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkipCacheEntry":
        return cls(
            reason=str(payload.get("reason", "unknown")),
            last_seen_at=float(payload.get("last_seen_at", 0.0)),
            last_error=str(payload.get("last_error", "")),
            failure_count=max(1, int(payload.get("failure_count", 1))),
        )


def resolve_cache_dir(environ: Mapping[str, str] = os.environ) -> Path:
    override = environ.get(NIMBENCH_CACHE_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    xdg_cache_home = environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / DEFAULT_CACHE_DIR_NAME
    return Path.home() / ".cache" / DEFAULT_CACHE_DIR_NAME


def get_skip_cache_path(environ: Mapping[str, str] = os.environ) -> Path:
    return resolve_cache_dir(environ) / DEFAULT_SKIP_CACHE_FILENAME


def load_skip_cache(cache_path: Path) -> dict[str, SkipCacheEntry]:
    try:
        raw = cache_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        LOG.warning("ignore skip cache %s: %s", cache_path, exc)
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        LOG.warning("ignore skip cache %s: %s", cache_path, exc)
        return {}

    if isinstance(payload, dict) and isinstance(payload.get("models"), dict):
        items = payload["models"].items()
    elif isinstance(payload, dict):
        items = payload.items()
    else:
        return {}

    cache: dict[str, SkipCacheEntry] = {}
    for model, entry in items:
        if not isinstance(model, str) or not model.strip() or not isinstance(entry, dict):
            continue
        cache[model] = SkipCacheEntry.from_dict(entry)
    return cache


def save_skip_cache(cache_path: Path, cache: Mapping[str, SkipCacheEntry]) -> None:
    payload = {
        "version": 1,
        "models": {model: entry.to_dict() for model, entry in sorted(cache.items())},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(cache_path)


def record_skip_cache_failure(
    cache: dict[str, SkipCacheEntry],
    model: str,
    reason: str,
    error: str,
    *,
    now: float | None = None,
) -> None:
    current = cache.get(model)
    timestamp = time.time() if now is None else now
    failure_count = 1
    if current is not None and current.reason == reason:
        failure_count = current.failure_count + 1
    cache[model] = SkipCacheEntry(
        reason=reason,
        last_seen_at=timestamp,
        last_error=error,
        failure_count=failure_count,
    )


def should_skip_cached_entry(entry: SkipCacheEntry) -> bool:
    if entry.reason in CACHE_IMMEDIATE_REASONS:
        return True
    if entry.reason in CACHE_AFTER_REPEAT_REASONS:
        return entry.failure_count >= 2
    return False


@dataclass(slots=True)
class RateLimiter:
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE
    _next_allowed_at: float = 0.0
    _interval: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        if self.requests_per_minute < 1:
            raise ValueError("--rpm must be >= 1")
        self._interval = 60.0 / self.requests_per_minute

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nimbench",
        description="Benchmark NVIDIA NIM chat model latency.",
    )
    parser.add_argument("--api-key", help="NVIDIA API key.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Reserved. Benchmarking is sequential to preserve the 40 rpm cap.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Requests per model. Default: {DEFAULT_REPEATS}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after N successful benchmarks.",
    )
    parser.add_argument(
        "--pattern",
        help="Regex filter for model ids.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a table.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark every discovered model instead of only likely chat models.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore the local skip cache for this run and rebuild it from fresh results.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help=f"Request rate cap. Default: {DEFAULT_REQUESTS_PER_MINUTE} rpm.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )


def resolve_api_key(
    cli_value: str | None,
    environ: Mapping[str, str] = os.environ,
    prompt_fn: Callable[[str], str] = getpass.getpass,
) -> str:
    if cli_value:
        value = cli_value.strip()
        if value:
            return value
    env_value = environ.get("NVIDIA_API_KEY", "").strip()
    if env_value:
        return env_value
    try:
        value = prompt_fn("NVIDIA API key: ").strip()
        if value:
            return value
    except (EOFError, KeyboardInterrupt) as exc:  # pragma: no cover - terminal only
        raise RuntimeError("NVIDIA API key required.") from exc
    raise RuntimeError("NVIDIA API key required.")


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("Base URL cannot be empty.")
    return base_url


def join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=build_headers(api_key), method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        raise RuntimeError("Empty response body.")
    return json.loads(raw.decode("utf-8"))


def _remember_http_error_body(exc: HTTPError) -> str:
    cached = getattr(exc, "_nimbench_body", None)
    if isinstance(cached, str):
        return cached
    try:
        raw = exc.read()
    except Exception:  # pragma: no cover - urllib oddity
        raw = b""
    body = raw.decode("utf-8", "replace").strip()
    setattr(exc, "_nimbench_body", body)
    return body


def _http_error_body_contains(exc: HTTPError, needle: str) -> bool:
    return needle.lower() in _remember_http_error_body(exc).lower()


def request_json_with_retry(
    method: str,
    url: str,
    api_key: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    rate_limiter: RateLimiter | None = None,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            if rate_limiter is not None:
                delay = rate_limiter.wait()
                if delay > 0:
                    LOG.info("rate limit wait %.2fs before %s %s", delay, method, url)
            return request_json(method, url, api_key, payload=payload, timeout=timeout)
        except HTTPError as exc:
            last_exc = exc
            if exc.code in RETRYABLE_STATUS and attempt == 0:
                time.sleep(0.5)
                continue
            raise
        except URLError as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise
        except TimeoutError as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise
    if last_exc is not None:  # pragma: no cover - defensive
        raise last_exc
    raise RuntimeError("Request failed.")


def extract_model_ids(payload: Any) -> list[str]:
    items: Iterable[Any]
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


def discover_models(
    base_url: str,
    api_key: str,
    timeout: float,
    rate_limiter: RateLimiter | None = None,
) -> list[str]:
    payload = request_json_with_retry(
        "GET",
        join_url(base_url, "/models"),
        api_key,
        timeout=timeout,
        rate_limiter=rate_limiter,
    )
    models = extract_model_ids(payload)
    if not models:
        raise RuntimeError("No models discovered from /models.")
    return models


def build_chat_payload(model: str, temperature: float = DEFAULT_TEMPERATURE) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
        "max_tokens": 8,
        "temperature": temperature,
        "stream": False,
    }


def rank_model(model: str) -> tuple[int, str]:
    lowered = model.lower()
    if any(hint in lowered for hint in NON_CHAT_HINTS):
        return (2, model)
    if any(hint in lowered for hint in CHAT_HINTS):
        return (0, model)
    return (1, model)


def rank_models(models: Sequence[str]) -> list[str]:
    return sorted(models, key=rank_model)


def is_likely_chat_model(model: str) -> bool:
    lowered = model.lower()
    if any(hint in lowered for hint in NON_CHAT_HINTS):
        return False
    return any(hint in lowered for hint in CHAT_HINTS)


def select_models(models: Sequence[str], include_all: bool = False) -> list[str]:
    if include_all:
        return list(models)
    return [model for model in models if is_likely_chat_model(model)]


def classify_error_message(message: str) -> str | None:
    lowered = " ".join(message.lower().split())
    if "timed out" in lowered or "timeouterror" in lowered:
        return CACHE_REASON_TIMEOUT
    if "http 404" in lowered and (
        "not found" in lowered or "does not exist" in lowered or "page not found" in lowered
    ):
        return CACHE_REASON_NOT_PROVISIONED
    if "http 400" in lowered and "degraded function cannot be invoked" in lowered:
        return CACHE_REASON_DEGRADED
    if "http 400" in lowered and (
        "content cannot be a plain string" in lowered or "does not support text input" in lowered
    ):
        return CACHE_REASON_UNSUPPORTED_INPUT
    if "the model" in lowered and "does not exist" in lowered:
        return CACHE_REASON_NOT_PROVISIONED
    return None


def classify_result_for_cache(result: ModelResult) -> str | None:
    if result.success_count > 0 or not result.errors:
        return None
    categories = [classify_error_message(error) for error in result.errors]
    categories = [category for category in categories if category is not None]
    if not categories:
        return None
    unique = set(categories)
    if len(unique) != 1:
        return None
    category = unique.pop()
    if category in CACHE_IMMEDIATE_REASONS or category in CACHE_AFTER_REPEAT_REASONS:
        return category
    return None


def filter_cached_models(
    models: Sequence[str],
    skip_cache: Mapping[str, SkipCacheEntry],
) -> tuple[list[tuple[int, str]], list[tuple[int, str, SkipCacheEntry]]]:
    benchmarkable: list[tuple[int, str]] = []
    skipped: list[tuple[int, str, SkipCacheEntry]] = []
    for index, model in enumerate(models, start=1):
        cached_entry = skip_cache.get(model)
        if cached_entry is not None and should_skip_cached_entry(cached_entry):
            skipped.append((index, model, cached_entry))
            continue
        benchmarkable.append((index, model))
    return benchmarkable, skipped


def format_http_error(exc: HTTPError) -> str:
    detail = _remember_http_error_body(exc)
    if detail:
        detail = " ".join(detail.split())
        if len(detail) > 180:
            detail = detail[:177] + "..."
    if detail:
        return f"HTTP {exc.code}: {detail}"
    reason = getattr(exc, "reason", "")
    if reason:
        return f"HTTP {exc.code}: {reason}"
    return f"HTTP {exc.code}"


def format_request_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        return format_http_error(exc)
    if isinstance(exc, URLError):
        return f"URL error: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def is_temperature_validation_error(exc: HTTPError) -> bool:
    if exc.code != 422:
        return False
    body = _remember_http_error_body(exc).lower()
    return "temperature" in body and ("greater_than" in body or "must be greater than 0" in body or "input should be greater than 0" in body)


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


def outcome_label(result: ModelResult) -> str:
    if result.success_count > 0:
        return "ok"
    reason = classify_result_for_cache(result)
    if reason == CACHE_REASON_NOT_PROVISIONED:
        return "unavailable"
    if reason == CACHE_REASON_UNSUPPORTED_INPUT:
        return "unsupported"
    if reason == CACHE_REASON_DEGRADED:
        return "degraded"
    if reason == CACHE_REASON_TIMEOUT:
        return "timeout"
    return "fail"


def measure_once(
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    rate_limiter: RateLimiter | None = None,
) -> RequestSample:
    request_url = join_url(base_url, "/chat/completions")
    payload = build_chat_payload(model, temperature=DEFAULT_TEMPERATURE)

    last_exc: Exception | None = None
    use_temperature_fallback = False
    for attempt in range(2):
        try:
            if rate_limiter is not None:
                delay = rate_limiter.wait()
                if delay > 0:
                    LOG.info("rate limit wait %.2fs before %s", delay, model)
            started = time.perf_counter()
            response = request_json(
                "POST",
                request_url,
                api_key,
                payload=payload,
                timeout=timeout,
            )
            ended = time.perf_counter()
            latency_ms = (ended - started) * 1000.0
            completion_tokens = extract_completion_tokens(response)
            server_tps = extract_server_tokens_per_second(response)
            return RequestSample(
                latency_ms=latency_ms,
                completion_tokens=completion_tokens,
                server_tokens_per_second=server_tps,
            )
        except HTTPError as exc:
            last_exc = exc
            if not use_temperature_fallback and is_temperature_validation_error(exc):
                use_temperature_fallback = True
                payload = build_chat_payload(model, temperature=FALLBACK_TEMPERATURE)
                LOG.info("retry %s with temperature=%.1f", model, FALLBACK_TEMPERATURE)
                continue
            if exc.code in RETRYABLE_STATUS and attempt == 0:
                time.sleep(0.5)
                continue
            raise
        except URLError as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise
        except TimeoutError as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise
    if last_exc is not None:  # pragma: no cover - defensive
        raise last_exc
    raise RuntimeError("Request failed.")


def benchmark_model(
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    repeats: int,
    rate_limiter: RateLimiter | None = None,
) -> ModelResult:
    samples: list[float] = []
    errors: list[str] = []
    tokens_per_second_samples: list[float] = []
    for _ in range(repeats):
        try:
            sample = measure_once(model, base_url, api_key, timeout, rate_limiter=rate_limiter)
            samples.append(sample.latency_ms)
            if sample.server_tokens_per_second is not None:
                tokens_per_second_samples.append(sample.server_tokens_per_second)
            elif sample.completion_tokens and sample.latency_ms > 0:
                tokens_per_second_samples.append(sample.completion_tokens / (sample.latency_ms / 1000.0))
        except Exception as exc:
            errors.append(format_request_error(exc))
    return ModelResult(
        model=model,
        samples_ms=samples,
        errors=errors,
        tokens_per_second_samples=tokens_per_second_samples,
    )


def safe_benchmark_model(
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    repeats: int,
    rate_limiter: RateLimiter | None = None,
) -> ModelResult:
    try:
        return benchmark_model(model, base_url, api_key, timeout, repeats, rate_limiter=rate_limiter)
    except Exception as exc:  # pragma: no cover - extra safety
        return ModelResult(model=model, samples_ms=[], errors=[format_request_error(exc)])


def filter_models(models: Sequence[str], pattern: str | None, limit: int | None) -> list[str]:
    filtered = list(models)
    if pattern:
        regex = re.compile(pattern)
        filtered = [model for model in filtered if regex.search(model)]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        filtered = filtered[:limit]
    return filtered


def sort_results(results: Sequence[ModelResult]) -> tuple[list[ModelResult], list[ModelResult]]:
    successes = [result for result in results if result.success_count > 0]
    failures = [result for result in results if result.success_count == 0]
    successes.sort(key=lambda result: (result.median_ms or float("inf"), result.min_ms or float("inf"), result.model))
    failures.sort(key=lambda result: result.model)
    return successes, failures


def format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def summarize_failure_counts(results: Sequence[ModelResult]) -> dict[str, int]:
    counts = {
        "unavailable": 0,
        "unsupported": 0,
        "degraded": 0,
        "timeout": 0,
        "fail": 0,
    }
    _, failures = sort_results(results)
    for result in failures:
        label = outcome_label(result)
        counts[label] = counts.get(label, 0) + 1
    return counts


def render_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    lines = []
    header_line = "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    rule_line = "  ".join("-" * widths[index] for index in range(len(headers)))
    lines.append(header_line)
    lines.append(rule_line)
    for row in rows:
        lines.append("  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))
    return "\n".join(lines)


def render_text(
    results: Sequence[ModelResult],
    base_url: str,
    discovered_count: int,
    attempted_count: int | None = None,
    selected_count: int | None = None,
    skipped_non_chat_count: int = 0,
    skipped_cached_count: int = 0,
) -> str:
    successes, failures = sort_results(results)
    selected_count = discovered_count if selected_count is None else selected_count
    attempted_count = len(results) if attempted_count is None else attempted_count
    lines = [
        f"NIM bench: {selected_count} candidate model(s), {attempted_count} attempted, {discovered_count} discovered at {base_url}",
    ]
    if skipped_non_chat_count or skipped_cached_count:
        skip_bits = []
        if skipped_non_chat_count:
            skip_bits.append(f"{skipped_non_chat_count} non-chat model(s)")
        if skipped_cached_count:
            skip_bits.append(f"{skipped_cached_count} cached model(s)")
        lines.append(f"skipped {', '.join(skip_bits)} model(s)")

    if successes:
        rows = []
        for index, result in enumerate(successes, start=1):
            rows.append(
                [
                    str(index),
                    result.model,
                    format_ms(result.median_ms),
                    format_ms(result.min_ms),
                    format_ms(result.max_ms),
                    format_ms(result.median_tokens_per_second),
                    str(result.success_count),
                    str(result.error_count),
                ]
            )
        lines.append("")
        lines.append(
            render_table(
                rows,
                ["rank", "model", "median ms", "min ms", "max ms", "tok/s", "ok", "err"],
            )
        )
    else:
        lines.append("")
        lines.append("No successful runs.")

    if failures:
        rows = []
        for result in failures:
            rows.append([result.model, "; ".join(result.errors) or "unknown error"])
        lines.append("")
        lines.append(render_table(rows, ["failed model", "error"]))

    return "\n".join(lines)


def render_json(
    results: Sequence[ModelResult],
    base_url: str,
    discovered_count: int,
    attempted_count: int | None = None,
    selected_count: int | None = None,
    skipped_non_chat_count: int = 0,
    skipped_cached_count: int = 0,
    skip_cache_path: str | None = None,
) -> str:
    successes, failures = sort_results(results)
    selected_count = discovered_count if selected_count is None else selected_count
    attempted_count = len(results) if attempted_count is None else attempted_count
    payload = {
        "base_url": base_url,
        "discovered_count": discovered_count,
        "attempted_count": attempted_count,
        "selected_count": selected_count,
        "skipped_non_chat_count": skipped_non_chat_count,
        "skipped_cached_count": skipped_cached_count,
        "skip_cache_path": skip_cache_path,
        "failure_counts": summarize_failure_counts(results),
        "successes": [result.as_dict(rank=index) for index, result in enumerate(successes, start=1)],
        "failures": [result.as_dict() for result in failures],
    }
    return json.dumps(payload, indent=2)


def run(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    base_url = normalize_base_url(args.base_url)
    api_key = resolve_api_key(args.api_key)
    rate_limiter = RateLimiter(args.rpm)

    LOG.info("start nimbench")
    LOG.info("discover models from %s", base_url)
    discovered_models = discover_models(base_url, api_key, args.timeout, rate_limiter=rate_limiter)
    LOG.info("discovered %d model(s)", len(discovered_models))

    filtered_models = filter_models(discovered_models, args.pattern, None)
    models = select_models(filtered_models, include_all=args.all_models)
    models = rank_models(models)
    LOG.info("ranked %d candidate model(s)", len(models))
    if not args.all_models:
        skipped = len(filtered_models) - len(models)
        if skipped > 0:
            LOG.info("skipped %d obvious non-chat model(s); use --all-models to benchmark everything", skipped)

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        LOG.info("success cap %d model(s)", args.limit)

    if not models:
        raise RuntimeError("No models matched filters.")

    repeats = max(1, args.repeats)
    if args.concurrency != 1:
        LOG.info("ignore concurrency=%d; pace is fixed at %d rpm", args.concurrency, rate_limiter.requests_per_minute)

    cache_path = get_skip_cache_path()
    skip_cache: dict[str, SkipCacheEntry] = {}
    if args.refresh_cache:
        LOG.info("refresh skip cache requested; ignore %s", cache_path)
    else:
        skip_cache = load_skip_cache(cache_path)
        if skip_cache:
            LOG.info("loaded skip cache with %d model(s) from %s", len(skip_cache), cache_path)

    benchmark_models, cached_skips = filter_cached_models(models, skip_cache)
    results: list[ModelResult] = []
    successful_models = 0
    attempted_models = 0
    total_candidates = len(models)
    cache_dirty = args.refresh_cache
    cached_skip_count = len(cached_skips)
    skipped_non_chat_count = len(filtered_models) - len(models) if not args.all_models else 0

    if cached_skips:
        for index, model, entry in cached_skips:
            LOG.info(
                "[%d/%d] cached skip %s reason=%s failures=%d",
                index,
                total_candidates,
                model,
                entry.reason,
                entry.failure_count,
            )

    for index, model in benchmark_models:
        if args.limit is not None and successful_models >= args.limit:
            LOG.info("stop after %d successful benchmark(s)", successful_models)
            break

        attempted_models += 1
        LOG.info("[%d/%d] start %s", index, total_candidates, model)
        result = safe_benchmark_model(model, base_url, api_key, args.timeout, repeats, rate_limiter=rate_limiter)
        results.append(result)

        if result.success_count > 0:
            successful_models += 1
            if model in skip_cache:
                del skip_cache[model]
                cache_dirty = True
            LOG.info(
                "[%d/%d] ok %s median=%.1fms min=%.1fms max=%.1fms errors=%d",
                index,
                total_candidates,
                model,
                result.median_ms or 0.0,
                result.min_ms or 0.0,
                result.max_ms or 0.0,
                result.error_count,
            )
        else:
            reason = classify_result_for_cache(result)
            if reason is not None:
                record_skip_cache_failure(
                    skip_cache,
                    model,
                    reason,
                    "; ".join(result.errors) or "unknown error",
                )
                cache_dirty = True
            LOG.info(
                "[%d/%d] %s %s %s",
                index,
                total_candidates,
                outcome_label(result),
                model,
                "; ".join(result.errors) or "unknown error",
            )

    discovered_count = len(discovered_models)
    selected_count = len(models)
    output = (
        render_json(
            results,
            base_url,
            discovered_count,
            attempted_models,
            selected_count,
            skipped_non_chat_count=skipped_non_chat_count,
            skipped_cached_count=cached_skip_count,
            skip_cache_path=str(cache_path),
        )
        if args.json
        else render_text(
            results,
            base_url,
            discovered_count,
            attempted_models,
            selected_count,
            skipped_non_chat_count=skipped_non_chat_count,
            skipped_cached_count=cached_skip_count,
        )
    )
    print(output)

    if cache_dirty:
        try:
            save_skip_cache(cache_path, skip_cache)
            LOG.info("updated skip cache at %s with %d model(s)", cache_path, len(skip_cache))
        except OSError as exc:
            LOG.warning("could not write skip cache %s: %s", cache_path, exc)

    successes, _ = sort_results(results)
    done_line = f"done {len(successes)} success(es), {len(results) - len(successes)} fail(s)"
    if cached_skip_count:
        done_line += f", {cached_skip_count} cached skip(s)"
    LOG.info(done_line)
    return 0 if successes else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except Exception as exc:
        print(f"nimbench: {exc}", file=sys.stderr)
        return 1
