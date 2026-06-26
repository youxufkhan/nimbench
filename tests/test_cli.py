from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import nimbench.api as api
import nimbench.cache as cache
import nimbench.cli as cli
import nimbench.models as models
import nimbench.ui as ui


def make_mock_response(status_code: int, json_data: dict, text: str = "") -> httpx.Response:
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=request,
        text=text or json.dumps(json_data),
    )


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
            api.extract_model_ids(payload),
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
        models_list = ["a/one", "b/two", "a/three"]
        self.assertEqual(cli.filter_models(models_list, r"^a/", 1), ["a/one"])

    def test_rank_models_prefers_chat_like_ids(self):
        models_list = ["baai/bge-m3", "meta/llama-3.2-1b-instruct", "nvidia/gliner-pii"]
        self.assertEqual(
            cli.rank_models(models_list),
            ["meta/llama-3.2-1b-instruct", "baai/bge-m3", "nvidia/gliner-pii"],
        )

    def test_select_models_skips_specialized_models_by_default(self):
        models_list = [
            "meta/llama-3.2-1b-instruct",
            "nvidia/gliner-pii",
            "nvidia/nemoretriever-parse",
            "google/gemma-2-2b-it",
        ]
        self.assertEqual(
            cli.select_models(models_list),
            ["meta/llama-3.2-1b-instruct", "google/gemma-2-2b-it"],
        )
        self.assertEqual(cli.select_models(models_list, include_all=True), models_list)

    def test_classify_error_message_and_skip_policy(self):
        self.assertEqual(
            cache.classify_error_message("HTTP 404: 404 page not found"),
            cache.CACHE_REASON_NOT_PROVISIONED,
        )
        self.assertEqual(
            cache.classify_error_message("HTTP 400: Content cannot be a plain string"),
            cache.CACHE_REASON_UNSUPPORTED_INPUT,
        )
        self.assertEqual(
            cache.classify_error_message("HTTP 400: DEGRADED function cannot be invoked"),
            cache.CACHE_REASON_DEGRADED,
        )
        self.assertEqual(
            cache.classify_error_message("TimeoutError: The read operation timed out"),
            cache.CACHE_REASON_TIMEOUT,
        )
        self.assertTrue(
            cache.should_skip_cached_entry(
                models.SkipCacheEntry(cache.CACHE_REASON_NOT_PROVISIONED, 1.0, "HTTP 404: not found")
            )
        )
        self.assertFalse(
            cache.should_skip_cached_entry(models.SkipCacheEntry(cache.CACHE_REASON_TIMEOUT, 1.0, "TimeoutError", 1))
        )
        self.assertTrue(
            cache.should_skip_cached_entry(models.SkipCacheEntry(cache.CACHE_REASON_TIMEOUT, 1.0, "TimeoutError", 2))
        )

    def test_skip_cache_round_trip_and_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / cache.DEFAULT_SKIP_CACHE_FILENAME
            cache_dict = {}
            cache.record_skip_cache_failure(
                cache_dict,
                "cached/model",
                cache.CACHE_REASON_NOT_PROVISIONED,
                "HTTP 404: not found",
                now=123.0,
            )
            cache.save_skip_cache(cache_path, cache_dict)

            loaded = cache.load_skip_cache(cache_path)
            self.assertIn("cached/model", loaded)
            self.assertEqual(loaded["cached/model"].reason, cache.CACHE_REASON_NOT_PROVISIONED)
            self.assertEqual(loaded["cached/model"].failure_count, 1)

            benchmarkable, skipped = cache.filter_cached_models(["cached/model", "live/model"], loaded)
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

        limiter = api.RateLimiter(40)
        with patch("nimbench.api.time.monotonic", side_effect=fake_monotonic), patch(
            "nimbench.api.time.sleep", side_effect=fake_sleep
        ):
            first = limiter.wait()
            second = limiter.wait()

        self.assertEqual(first, 0.0)
        self.assertAlmostEqual(second, 1.5, places=6)
        self.assertEqual(len(sleeps), 1)

    def test_sort_results_orders_by_median(self):
        results = [
            models.ModelResult("b", [20.0, 10.0], []),
            models.ModelResult("a", [5.0], []),
            models.ModelResult("c", [], ["x"]),
        ]
        successes, failures = ui.sort_results(results)
        self.assertEqual([item.model for item in successes], ["a", "b"])
        self.assertEqual([item.model for item in failures], ["c"])

    def test_render_json_shapes_payload(self):
        results = [
            models.ModelResult("fast", [10.0], [], [25.0]),
            models.ModelResult("dead", [], ["HTTP 500"]),
        ]
        payload = json.loads(ui.render_json(results, api.DEFAULT_BASE_URL, 2))
        self.assertEqual(payload["discovered_count"], 2)
        self.assertEqual(payload["attempted_count"], 2)
        self.assertEqual(payload["skipped_cached_count"], 0)
        self.assertIn("timeout", payload["failure_counts"])
        self.assertEqual(payload["successes"][0]["model"], "fast")
        self.assertEqual(payload["successes"][0]["median_tokens_per_second"], 25.0)
        self.assertEqual(payload["failures"][0]["model"], "dead")

    @patch("httpx.Client.request")
    def test_benchmark_model_collects_success_and_failure(self, mock_request):
        calls = {"count": 0}

        def fake_request(method, url, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                # Simulate a retryable status code error (e.g. 503)
                resp = make_mock_response(503, {}, "Service Unavailable")
                raise httpx.HTTPStatusError("Service Unavailable", request=resp.request, response=resp)
            return make_mock_response(200, {"usage": {"completion_tokens": 4}})

        mock_request.side_effect = fake_request

        with patch("nimbench.api.time.sleep") as mock_sleep:
            client = api.APIClient(api.DEFAULT_BASE_URL, "key", timeout=1.0)
            result = cli.benchmark_model(client, "meta/llama", repeats=2)

        self.assertEqual(result.success_count, 2)
        self.assertEqual(result.error_count, 0)
        # 1 retry on 503 (first call), then 2 successes = 3 total requests
        self.assertEqual(calls["count"], 3)
        mock_sleep.assert_called_with(0.5)

    @patch("httpx.Client.post")
    def test_benchmark_model_uses_temperature_fallback(self, mock_post):
        calls = {"count": 0}

        def fake_post(url, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                resp = make_mock_response(422, {}, '{"error":"body -> temperature Input should be greater than 0"}')
                raise httpx.HTTPStatusError("Unprocessable", request=resp.request, response=resp)
            return make_mock_response(200, {"usage": {"completion_tokens": 4}})

        mock_post.side_effect = fake_post

        with patch("nimbench.api.time.sleep"):
            client = api.APIClient(api.DEFAULT_BASE_URL, "key", timeout=1.0)
            result = cli.benchmark_model(client, "google/gemma-2-2b-it", repeats=1)

        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(calls["count"], 2)
        self.assertGreater(result.tokens_per_second_samples[0], 0)

    @patch("httpx.Client.post")
    def test_benchmark_model_survives_failures(self, mock_post):
        def fake_post(url, **kwargs):
            resp = make_mock_response(500, {}, '{"error":"boom"}')
            raise httpx.HTTPStatusError("Internal Error", request=resp.request, response=resp)

        mock_post.side_effect = fake_post

        with patch("nimbench.api.time.sleep"):
            client = api.APIClient(api.DEFAULT_BASE_URL, "key", timeout=1.0)
            result = cli.benchmark_model(client, "meta/llama", repeats=1)

        self.assertEqual(result.success_count, 0)
        self.assertEqual(result.error_count, 1)
        self.assertIn("HTTP 500", result.errors[0])

    def test_run_skips_cached_models_before_benchmarking(self):
        benchmarked: list[str] = []

        def fake_safe_benchmark_model(client, model, repeats):
            benchmarked.append(model)
            return models.ModelResult(model, [10.0], [])

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / cache.DEFAULT_SKIP_CACHE_FILENAME
            cache.save_skip_cache(
                cache_path,
                {
                    "cached/model": models.SkipCacheEntry(
                        cache.CACHE_REASON_NOT_PROVISIONED,
                        1.0,
                        "HTTP 404: not found",
                    )
                },
            )

            patch_discover = patch.object(
                api.APIClient, "discover_models", return_value=["cached/model", "live/model"]
            )
            with patch_discover, patch.object(
                cli, "filter_models", side_effect=lambda models, pattern, limit: list(models)
            ), patch.object(
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
            cache_path = Path(tmpdir) / cache.DEFAULT_SKIP_CACHE_FILENAME
            cache.save_skip_cache(
                cache_path,
                {
                    "stale/model": models.SkipCacheEntry(
                        cache.CACHE_REASON_NOT_PROVISIONED,
                        1.0,
                        "HTTP 404: not found",
                    )
                },
            )

            with patch.object(api.APIClient, "discover_models", return_value=["live/model"]), patch.object(
                cli, "filter_models", side_effect=lambda models, pattern, limit: list(models)
            ), patch.object(
                cli, "select_models", side_effect=lambda models, include_all=False: list(models)
            ), patch.object(
                cli, "rank_models", side_effect=lambda models: list(models)
            ), patch.object(
                cli, "get_skip_cache_path", return_value=cache_path
            ), patch.object(
                cli, "safe_benchmark_model", return_value=models.ModelResult("live/model", [10.0], [])
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
