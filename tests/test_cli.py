from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import nimbench.cli as cli


class DummyResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class CliTests(unittest.TestCase):
    def test_extract_model_ids_prefers_data_and_deduplicates(self):
        payload = {
            "data": [
                {"id": "meta/llama-3.2-1b-instruct"},
                {"model": "meta/llama-3.2-1b-instruct"},
                {"name": "mistralai/mistral-nemotron"},
                "openai/gpt-oss-20b",
            ]
        }
        self.assertEqual(
            cli.extract_model_ids(payload),
            ["meta/llama-3.2-1b-instruct", "mistralai/mistral-nemotron", "openai/gpt-oss-20b"],
        )

    def test_resolve_api_key_precedence(self):
        self.assertEqual(cli.resolve_api_key(" cli-key "), "cli-key")
        self.assertEqual(
            cli.resolve_api_key(None, environ={"NVIDIA_API_KEY": " env-key "}, prompt_fn=lambda _: "nope"),
            "env-key",
        )
        self.assertEqual(
            cli.resolve_api_key(None, environ={}, prompt_fn=lambda _: " prompted "),
            "prompted",
        )

    def test_filter_models_and_limit(self):
        models = ["a/one", "b/two", "a/three"]
        self.assertEqual(cli.filter_models(models, r"^a/", 1), ["a/one"])

    def test_rank_models_prefers_chat_like_ids(self):
        models = ["baai/bge-m3", "meta/llama-3.2-1b-instruct", "nvidia/gliner-pii"]
        self.assertEqual(
            cli.rank_models(models),
            ["meta/llama-3.2-1b-instruct", "baai/bge-m3", "nvidia/gliner-pii"],
        )

    def test_select_models_skips_specialized_models_by_default(self):
        models = [
            "meta/llama-3.2-1b-instruct",
            "nvidia/gliner-pii",
            "nvidia/nemoretriever-parse",
            "google/gemma-2-2b-it",
        ]
        self.assertEqual(
            cli.select_models(models),
            ["meta/llama-3.2-1b-instruct", "google/gemma-2-2b-it"],
        )
        self.assertEqual(cli.select_models(models, include_all=True), models)

    def test_classify_error_message_and_skip_policy(self):
        self.assertEqual(cli.classify_error_message("HTTP 404: 404 page not found"), cli.CACHE_REASON_NOT_PROVISIONED)
        self.assertEqual(
            cli.classify_error_message("HTTP 400: Content cannot be a plain string"),
            cli.CACHE_REASON_UNSUPPORTED_INPUT,
        )
        self.assertEqual(
            cli.classify_error_message("HTTP 400: DEGRADED function cannot be invoked"),
            cli.CACHE_REASON_DEGRADED,
        )
        self.assertEqual(
            cli.classify_error_message("TimeoutError: The read operation timed out"),
            cli.CACHE_REASON_TIMEOUT,
        )
        self.assertTrue(
            cli.should_skip_cached_entry(
                cli.SkipCacheEntry(cli.CACHE_REASON_NOT_PROVISIONED, 1.0, "HTTP 404: not found")
            )
        )
        self.assertFalse(
            cli.should_skip_cached_entry(cli.SkipCacheEntry(cli.CACHE_REASON_TIMEOUT, 1.0, "TimeoutError", 1))
        )
        self.assertTrue(
            cli.should_skip_cached_entry(cli.SkipCacheEntry(cli.CACHE_REASON_TIMEOUT, 1.0, "TimeoutError", 2))
        )

    def test_skip_cache_round_trip_and_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / cli.DEFAULT_SKIP_CACHE_FILENAME
            cache = {}
            cli.record_skip_cache_failure(
                cache,
                "cached/model",
                cli.CACHE_REASON_NOT_PROVISIONED,
                "HTTP 404: not found",
                now=123.0,
            )
            cli.save_skip_cache(cache_path, cache)

            loaded = cli.load_skip_cache(cache_path)
            self.assertIn("cached/model", loaded)
            self.assertEqual(loaded["cached/model"].reason, cli.CACHE_REASON_NOT_PROVISIONED)
            self.assertEqual(loaded["cached/model"].failure_count, 1)

            benchmarkable, skipped = cli.filter_cached_models(["cached/model", "live/model"], loaded)
            self.assertEqual(benchmarkable, [(2, "live/model")])
            self.assertEqual(skipped[0][1], "cached/model")

    def test_rate_limiter_sleeps_between_slots(self):
        clock = {"now": 100.0}
        sleeps: list[float] = []

        def fake_monotonic():
            return clock["now"]

        def fake_sleep(delay):
            sleeps.append(delay)
            clock["now"] += delay

        limiter = cli.RateLimiter(40)
        with patch("nimbench.cli.time.monotonic", side_effect=fake_monotonic), patch(
            "nimbench.cli.time.sleep", side_effect=fake_sleep
        ):
            first = limiter.wait()
            second = limiter.wait()

        self.assertEqual(first, 0.0)
        self.assertAlmostEqual(second, 1.5, places=6)
        self.assertEqual(len(sleeps), 1)

    def test_sort_results_orders_by_median(self):
        results = [
            cli.ModelResult("b", [20.0, 10.0], []),
            cli.ModelResult("a", [5.0], []),
            cli.ModelResult("c", [], ["x"]),
        ]
        successes, failures = cli.sort_results(results)
        self.assertEqual([item.model for item in successes], ["a", "b"])
        self.assertEqual([item.model for item in failures], ["c"])

    def test_render_text_includes_sorted_success_and_failed_section(self):
        results = [
            cli.ModelResult("slow", [50.0], []),
            cli.ModelResult("fast", [10.0], ["warn"], [25.0]),
            cli.ModelResult("dead", [], ["HTTP 500"]),
        ]
        text = cli.render_text(results, cli.DEFAULT_BASE_URL, 3)
        self.assertIn("fast", text)
        self.assertLess(text.index("fast"), text.index("slow"))
        self.assertIn("failed model", text)
        self.assertIn("tok/s", text)

    def test_render_json_shapes_payload(self):
        results = [
            cli.ModelResult("fast", [10.0], [], [25.0]),
            cli.ModelResult("dead", [], ["HTTP 500"]),
        ]
        payload = json.loads(cli.render_json(results, cli.DEFAULT_BASE_URL, 2))
        self.assertEqual(payload["discovered_count"], 2)
        self.assertEqual(payload["attempted_count"], 2)
        self.assertEqual(payload["skipped_cached_count"], 0)
        self.assertIn("timeout", payload["failure_counts"])
        self.assertEqual(payload["successes"][0]["model"], "fast")
        self.assertEqual(payload["successes"][0]["median_tokens_per_second"], 25.0)
        self.assertEqual(payload["failures"][0]["model"], "dead")

    def test_benchmark_model_collects_success_and_failure(self):
        calls = {"count": 0}

        def fake_urlopen(request, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise HTTPError(request.full_url, 503, "Service Unavailable", hdrs=None, fp=io.BytesIO(b""))
            return DummyResponse(b'{"ok": true}')

        with patch("nimbench.cli.urlopen", side_effect=fake_urlopen), patch("nimbench.cli.time.sleep"):
            result = cli.benchmark_model("meta/llama", cli.DEFAULT_BASE_URL, "key", 1.0, repeats=2)

        self.assertEqual(result.success_count, 2)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(calls["count"], 3)

    def test_benchmark_model_uses_temperature_fallback(self):
        calls = {"count": 0}

        def fake_urlopen(request, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise HTTPError(
                    request.full_url,
                    422,
                    "Unprocessable Entity",
                    hdrs=None,
                    fp=io.BytesIO(b'{"error":"body -> temperature Input should be greater than 0"}'),
                )
            return DummyResponse(b'{"usage":{"completion_tokens":4}}')

        with patch("nimbench.cli.urlopen", side_effect=fake_urlopen), patch("nimbench.cli.time.sleep"):
            result = cli.benchmark_model("google/gemma-2-2b-it", cli.DEFAULT_BASE_URL, "key", 1.0, repeats=1)

        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(calls["count"], 2)
        self.assertGreater(result.tokens_per_second_samples[0], 0)

    def test_benchmark_model_survives_failures(self):
        def fake_urlopen(request, timeout=None):
            raise HTTPError(request.full_url, 500, "Boom", hdrs=None, fp=io.BytesIO(b'{"error":"boom"}'))

        with patch("nimbench.cli.urlopen", side_effect=fake_urlopen), patch("nimbench.cli.time.sleep"):
            result = cli.benchmark_model("meta/llama", cli.DEFAULT_BASE_URL, "key", 1.0, repeats=1)

        self.assertEqual(result.success_count, 0)
        self.assertEqual(result.error_count, 1)
        self.assertIn("HTTP 500", result.errors[0])

    def test_run_skips_cached_models_before_benchmarking(self):
        benchmarked: list[str] = []

        def fake_safe_benchmark_model(model, base_url, api_key, timeout, repeats, rate_limiter=None):
            benchmarked.append(model)
            return cli.ModelResult(model, [10.0], [])

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / cli.DEFAULT_SKIP_CACHE_FILENAME
            cli.save_skip_cache(
                cache_path,
                {
                    "cached/model": cli.SkipCacheEntry(
                        cli.CACHE_REASON_NOT_PROVISIONED,
                        1.0,
                        "HTTP 404: not found",
                    )
                },
            )

            with patch.object(cli, "configure_logging", lambda: None), patch.object(
                cli, "discover_models", return_value=["cached/model", "live/model"]
            ), patch.object(cli, "filter_models", side_effect=lambda models, pattern, limit: list(models)), patch.object(
                cli, "select_models", side_effect=lambda models, include_all=False: list(models)
            ), patch.object(
                cli, "rank_models", side_effect=lambda models: list(models)
            ), patch.object(
                cli, "get_skip_cache_path", return_value=cache_path
            ), patch.object(
                cli, "safe_benchmark_model", side_effect=fake_safe_benchmark_model
            ), patch.object(
                cli, "save_skip_cache"
            ) as save_mock, patch(
                "sys.stdout", new=io.StringIO()
            ), patch(
                "sys.stderr", new=io.StringIO()
            ):
                exit_code = cli.run(["--api-key", "key", "--limit", "1"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(benchmarked, ["live/model"])
        save_mock.assert_not_called()

    def test_run_refresh_cache_rewrites_even_without_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / cli.DEFAULT_SKIP_CACHE_FILENAME
            cli.save_skip_cache(
                cache_path,
                {
                    "stale/model": cli.SkipCacheEntry(
                        cli.CACHE_REASON_NOT_PROVISIONED,
                        1.0,
                        "HTTP 404: not found",
                    )
                },
            )

            with patch.object(cli, "configure_logging", lambda: None), patch.object(
                cli, "discover_models", return_value=["live/model"]
            ), patch.object(cli, "filter_models", side_effect=lambda models, pattern, limit: list(models)), patch.object(
                cli, "select_models", side_effect=lambda models, include_all=False: list(models)
            ), patch.object(
                cli, "rank_models", side_effect=lambda models: list(models)
            ), patch.object(
                cli, "get_skip_cache_path", return_value=cache_path
            ), patch.object(
                cli, "safe_benchmark_model", return_value=cli.ModelResult("live/model", [10.0], [])
            ), patch.object(
                cli, "load_skip_cache"
            ) as load_mock, patch.object(
                cli, "save_skip_cache"
            ) as save_mock, patch(
                "sys.stdout", new=io.StringIO()
            ), patch(
                "sys.stderr", new=io.StringIO()
            ):
                exit_code = cli.run(["--api-key", "key", "--refresh-cache"])

        self.assertEqual(exit_code, 0)
        load_mock.assert_not_called()
        save_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
