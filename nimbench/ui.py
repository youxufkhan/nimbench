from __future__ import annotations

import json
from collections.abc import Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from nimbench.cache import classify_result_for_cache
from nimbench.models import ModelResult

CONSOLE_STDOUT = Console()
CONSOLE_STDERR = Console(stderr=True)


def sort_results(results: Sequence[ModelResult]) -> tuple[list[ModelResult], list[ModelResult]]:
    successes = [result for result in results if result.success_count > 0]
    failures = [result for result in results if result.success_count == 0]
    successes.sort(
        key=lambda result: (
            result.median_ms or float("inf"),
            result.min_ms or float("inf"),
            result.model,
        )
    )
    failures.sort(key=lambda result: result.model)
    return successes, failures


def outcome_label(result: ModelResult) -> str:
    if result.success_count > 0:
        return "ok"
    reason = classify_result_for_cache(result)
    if reason == "not_provisioned":
        return "unavailable"
    if reason == "unsupported_input":
        return "unsupported"
    if reason == "degraded":
        return "degraded"
    if reason == "timeout":
        return "timeout"
    return "fail"


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


def render_rich_table(
    results: Sequence[ModelResult],
    base_url: str,
    discovered_count: int,
    attempted_count: int | None = None,
    selected_count: int | None = None,
    skipped_non_chat_count: int = 0,
    skipped_cached_count: int = 0,
) -> None:
    successes, failures = sort_results(results)
    selected_count = discovered_count if selected_count is None else selected_count
    attempted_count = len(results) if attempted_count is None else attempted_count

    # Header Summary
    CONSOLE_STDOUT.print(
        f"\n[bold green]NIM Bench Summary[/bold green] | "
        f"[cyan]Base URL:[/cyan] {base_url} | "
        f"[cyan]Discovered:[/cyan] {discovered_count} | "
        f"[cyan]Attempted:[/cyan] {attempted_count} | "
        f"[cyan]Selected Candidates:[/cyan] {selected_count}"
    )

    if skipped_non_chat_count or skipped_cached_count:
        skips = []
        if skipped_non_chat_count:
            skips.append(f"{skipped_non_chat_count} non-chat")
        if skipped_cached_count:
            skips.append(f"{skipped_cached_count} cached-skipped")
        CONSOLE_STDOUT.print(f"[dim]Skipped: {', '.join(skips)}[/dim]")

    # Successful Runs Table
    if successes:
        table = Table(title="\n[bold]Successful Model Benchmarks[/bold]", header_style="bold blue")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Median Latency", justify="right")
        table.add_column("Min Latency", justify="right")
        table.add_column("Max Latency", justify="right")
        table.add_column("Speed (tok/s)", justify="right", style="bold magenta")
        table.add_column("Successes", justify="right")
        table.add_column("Errors", justify="right")

        for index, result in enumerate(successes, start=1):
            err_style = "red" if result.error_count > 0 else "dim green"
            table.add_row(
                str(index),
                result.model,
                f"{format_ms(result.median_ms)} ms",
                f"{format_ms(result.min_ms)} ms",
                f"{format_ms(result.max_ms)} ms",
                format_ms(result.median_tokens_per_second),
                f"[green]{result.success_count}[/green]",
                f"[{err_style}]{result.error_count}[/{err_style}]",
            )
        CONSOLE_STDOUT.print(table)
    else:
        CONSOLE_STDOUT.print("\n[bold yellow]No successful model benchmarks run.[/bold yellow]")

    # Failures Table
    if failures:
        fail_table = Table(title="\n[bold red]Failed Model Benchmarks[/bold red]", header_style="bold red")
        fail_table.add_column("Failed Model", style="yellow")
        fail_table.add_column("Errors / Status Reason", style="dim red")

        for result in failures:
            fail_table.add_row(
                result.model,
                "; ".join(result.errors) or "unknown error",
            )
        CONSOLE_STDOUT.print(fail_table)


def get_progress_bar(total: int) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE_STDERR,
    )
