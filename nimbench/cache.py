from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from nimbench.models import ModelResult, SkipCacheEntry

LOG = logging.getLogger("nimbench")

DEFAULT_SKIP_CACHE_FILENAME = "skip-cache.json"
DEFAULT_CACHE_DIR_NAME = "nimbench"
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
    try:
        cache_path.parent.chmod(0o700)
    except OSError:
        pass

    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        tmp_path.chmod(0o600)
    except OSError:
        pass

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
