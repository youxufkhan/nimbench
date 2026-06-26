from __future__ import annotations

import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


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
    def from_dict(cls, payload: Mapping[str, Any]) -> SkipCacheEntry:
        return cls(
            reason=str(payload.get("reason", "unknown")),
            last_seen_at=float(payload.get("last_seen_at", 0.0)),
            last_error=str(payload.get("last_error", "")),
            failure_count=max(1, int(payload.get("failure_count", 1))),
        )
