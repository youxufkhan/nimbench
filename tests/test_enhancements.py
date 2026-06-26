from __future__ import annotations

import unittest

import nimbench.cache as cache
import nimbench.models as models
import nimbench.ui as ui


class EnhancementTests(unittest.TestCase):
    def test_summarize_failure_counts_uses_raw_keys(self):
        results = [
            models.ModelResult("m1", [], ["HTTP 404: Not Found"]),  # unavailable
            models.ModelResult("m2", [], ["HTTP 400: Content cannot be a plain string"]),  # unsupported
            models.ModelResult("m3", [], ["TimeoutError: timed out"]),  # timeout
            models.ModelResult("m4", [], ["boom"]),  # fail
        ]
        counts = ui.summarize_failure_counts(results)
        self.assertEqual(counts["unavailable"], 1)
        self.assertEqual(counts["unsupported"], 1)
        self.assertEqual(counts["timeout"], 1)
        self.assertEqual(counts["fail"], 1)

    def test_model_result_computes_stats_correctly(self):
        result = models.ModelResult(
            model="test-model",
            samples_ms=[10.0, 20.0, 30.0],
            errors=["err1"],
            tokens_per_second_samples=[50.0, 100.0, 150.0]
        )
        self.assertEqual(result.median_ms, 20.0)
        self.assertEqual(result.min_ms, 10.0)
        self.assertEqual(result.max_ms, 30.0)
        self.assertEqual(result.median_tokens_per_second, 100.0)
        self.assertEqual(result.min_tokens_per_second, 50.0)
        self.assertEqual(result.max_tokens_per_second, 150.0)
        self.assertEqual(result.success_count, 3)
        self.assertEqual(result.error_count, 1)

    def test_model_result_empty_samples(self):
        result = models.ModelResult("test-model", [], [])
        self.assertIsNone(result.median_ms)
        self.assertIsNone(result.min_ms)
        self.assertIsNone(result.max_ms)
        self.assertIsNone(result.median_tokens_per_second)
        self.assertIsNone(result.min_tokens_per_second)
        self.assertIsNone(result.max_tokens_per_second)
        self.assertEqual(result.success_count, 0)
        self.assertEqual(result.error_count, 0)

    def test_cache_classification_logic(self):
        self.assertEqual(
            cache.classify_error_message("http 400 degraded function cannot be invoked"),
            "degraded"
        )
        self.assertEqual(
            cache.classify_error_message("random unknown error"),
            None
        )


if __name__ == "__main__":
    unittest.main()
