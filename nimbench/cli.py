from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from urllib.parse import urlparse

import httpx

from nimbench.api import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, APIClient, RateLimiter
from nimbench.cache import (
    classify_result_for_cache,
    filter_cached_models,
    get_skip_cache_path,
    load_skip_cache,
    record_skip_cache_failure,
    save_skip_cache,
)
from nimbench.models import ModelResult
from nimbench.ui import (
    CONSOLE_STDERR,
    format_ms,
    get_progress_bar,
    outcome_label,
    render_json,
    render_rich_table,
    sort_results,
)

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
        default=1,
        help="Requests per model. Default: 1",
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
        default=40,
        help="Request rate cap. Default: 40 rpm.",
    )
    return parser.parse_args(argv)


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
    except (EOFError, KeyboardInterrupt) as exc:
        raise RuntimeError("NVIDIA API key required.") from exc
    raise RuntimeError("NVIDIA API key required.")


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("Base URL cannot be empty.")

    parsed = urlparse(base_url)
    if parsed.scheme == "http":
        hostname = parsed.hostname or ""
        if hostname not in ("localhost", "127.0.0.1", "::1"):
            CONSOLE_STDERR.print(
                "[bold yellow]Warning: Sending API keys over unencrypted "
                "HTTP to a remote host is insecure.[/bold yellow]"
            )
    return base_url


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


def format_http_status_error(exc: httpx.HTTPStatusError) -> str:
    detail = exc.response.text.strip()
    if detail:
        detail = " ".join(detail.split())
        if len(detail) > 180:
            detail = detail[:177] + "..."
        return f"HTTP {exc.response.status_code}: {detail}"
    return f"HTTP {exc.response.status_code}"


def format_request_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return format_http_status_error(exc)
    if isinstance(exc, httpx.RequestError):
        return f"Network error: {exc}"
    return f"{type(exc).__name__}: {exc}"


def benchmark_model(
    client: APIClient,
    model: str,
    repeats: int,
) -> ModelResult:
    samples: list[float] = []
    errors: list[str] = []
    tokens_per_second_samples: list[float] = []
    for _ in range(repeats):
        try:
            sample = client.measure_once(model)
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
    client: APIClient,
    model: str,
    repeats: int,
) -> ModelResult:
    try:
        return benchmark_model(client, model, repeats)
    except Exception as exc:
        return ModelResult(model=model, samples_ms=[], errors=[format_request_error(exc)])


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_url = normalize_base_url(args.base_url)
    api_key = resolve_api_key(args.api_key)
    rate_limiter = RateLimiter(args.rpm)

    if not args.json:
        CONSOLE_STDERR.print("[bold green]Starting nimbench...[/bold green]")
        CONSOLE_STDERR.print(f"Discovering models from {base_url}...")

    with APIClient(base_url, api_key, args.timeout, rate_limiter=rate_limiter) as client:
        try:
            discovered_models = client.discover_models()
        except Exception as exc:
            CONSOLE_STDERR.print(f"[bold red]Failed to discover models:[/bold red] {exc}")
            return 1

        if not args.json:
            CONSOLE_STDERR.print(f"Discovered {len(discovered_models)} model(s).")

        filtered_models = filter_models(discovered_models, args.pattern, None)
        models = select_models(filtered_models, include_all=args.all_models)
        models = rank_models(models)

        if not args.json:
            CONSOLE_STDERR.print(f"Ranked {len(models)} candidate model(s).")
            if not args.all_models:
                skipped = len(filtered_models) - len(models)
                if skipped > 0:
                    CONSOLE_STDERR.print(
                        f"[dim]Skipped {skipped} obvious non-chat model(s); "
                        "use --all-models to benchmark everything.[/dim]"
                    )

        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be >= 1")
            if not args.json:
                CONSOLE_STDERR.print(f"[dim]Success limit capped at {args.limit} model(s).[/dim]")

        if not models:
            CONSOLE_STDERR.print("[bold red]Error: No models matched filters.[/bold red]")
            return 1

        repeats = max(1, args.repeats)
        if args.concurrency != 1:
            if not args.json:
                CONSOLE_STDERR.print(
                    f"[dim]Ignore concurrency={args.concurrency}; "
                    f"pace is fixed at {rate_limiter.requests_per_minute} rpm.[/dim]"
                )

        cache_path = get_skip_cache_path()
        skip_cache = {}
        if args.refresh_cache:
            if not args.json:
                CONSOLE_STDERR.print(f"[dim]Refresh skip cache requested; ignoring {cache_path}[/dim]")
        else:
            skip_cache = load_skip_cache(cache_path)
            if skip_cache and not args.json:
                CONSOLE_STDERR.print(f"[dim]Loaded skip cache with {len(skip_cache)} model(s) from {cache_path}[/dim]")

        benchmark_models, cached_skips = filter_cached_models(models, skip_cache)
        results: list[ModelResult] = []
        successful_models = 0
        attempted_models = 0
        total_candidates = len(models)
        cache_dirty = args.refresh_cache
        cached_skip_count = len(cached_skips)
        skipped_non_chat_count = len(filtered_models) - len(models) if not args.all_models else 0

        if cached_skips and not args.json:
            for index, model, entry in cached_skips:
                CONSOLE_STDERR.print(
                    f"[{index}/{total_candidates}] [yellow]Cached skip:[/yellow] {model} "
                    f"[dim]reason={entry.reason} failures={entry.failure_count}[/dim]"
                )

        # Create progress bar for live feedback (if not JSON mode)
        if not args.json:
            progress_bar = get_progress_bar(len(benchmark_models))
            with progress_bar as progress:
                task = progress.add_task("[cyan]Benchmarking models...", total=len(benchmark_models))
                for index, model in benchmark_models:
                    if args.limit is not None and successful_models >= args.limit:
                        progress.console.print(
                            f"[dim]Stop limit of {args.limit} successful benchmark(s) reached.[/dim]"
                        )
                        break

                    attempted_models += 1
                    progress.update(
                        task,
                        description=f"[cyan]Benchmarking [bold]{model}[/bold] ({index}/{total_candidates})...",
                    )

                    result = safe_benchmark_model(client, model, repeats)
                    results.append(result)

                    if result.success_count > 0:
                        successful_models += 1
                        if model in skip_cache:
                            del skip_cache[model]
                            cache_dirty = True
                        progress.console.print(
                            f"[{index}/{total_candidates}] [green]OK[/green] {model} "
                            f"[dim]median={format_ms(result.median_ms)}ms "
                            f"min={format_ms(result.min_ms)}ms max={format_ms(result.max_ms)}ms "
                            f"tok/s={format_ms(result.median_tokens_per_second)}[/dim]"
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
                        progress.console.print(
                            f"[{index}/{total_candidates}] [red]{outcome_label(result).upper()}[/red] {model} "
                            f"[dim]errors={'; '.join(result.errors) or 'unknown error'}[/dim]"
                        )
                    progress.advance(task)
        else:
            # Simple silent loop for JSON output
            for index, model in benchmark_models:
                if args.limit is not None and successful_models >= args.limit:
                    break
                attempted_models += 1
                result = safe_benchmark_model(client, model, repeats)
                results.append(result)

                if result.success_count > 0:
                    successful_models += 1
                    if model in skip_cache:
                        del skip_cache[model]
                        cache_dirty = True
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

        discovered_count = len(discovered_models)
        selected_count = len(models)

        if args.json:
            output = render_json(
                results,
                base_url,
                discovered_count,
                attempted_models,
                selected_count,
                skipped_non_chat_count=skipped_non_chat_count,
                skipped_cached_count=cached_skip_count,
                skip_cache_path=str(cache_path),
            )
            print(output)
        else:
            render_rich_table(
                results,
                base_url,
                discovered_count,
                attempted_models,
                selected_count,
                skipped_non_chat_count=skipped_non_chat_count,
                skipped_cached_count=cached_skip_count,
            )

        if cache_dirty:
            try:
                save_skip_cache(cache_path, skip_cache)
                if not args.json:
                    CONSOLE_STDERR.print(
                        f"[dim]Updated skip cache at {cache_path} with {len(skip_cache)} model(s)[/dim]"
                    )
            except OSError as exc:
                CONSOLE_STDERR.print(f"[yellow]Warning: Could not write skip cache {cache_path}:[/yellow] {exc}")

        successes, _ = sort_results(results)
        if not args.json:
            done_line = f"\n[bold]Done:[/bold] {len(successes)} success(es), {len(results) - len(successes)} fail(s)"
            if cached_skip_count:
                done_line += f", {cached_skip_count} cached skip(s)"
            CONSOLE_STDERR.print(done_line)

        return 0 if successes else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except Exception as exc:
        print(f"nimbench: {exc}", file=sys.stderr)
        return 1
