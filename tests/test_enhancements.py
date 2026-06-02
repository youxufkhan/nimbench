from __future__ import annotations

import unittest
from unittest.mock import patch
import nimbench.cli as cli

class EnhancementTests(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(cli.format_duration(45), "45s")
        self.assertEqual(cli.format_duration(65), "1m 5s")
        self.assertEqual(cli.format_duration(120), "2m 0s")
        self.assertEqual(cli.format_duration(3661), "61m 1s") # Simple m s format as implemented

    def test_colorize_atty(self):
        with patch("sys.stdout.isatty", return_value=True):
            self.assertEqual(cli.colorize("test", cli.COLOR_RED), "\033[31mtest\033[0m")
        with patch("sys.stdout.isatty", return_value=False):
            self.assertEqual(cli.colorize("test", cli.COLOR_RED), "test")

    def test_summarize_failure_counts_uses_raw_keys(self):
        results = [
            cli.ModelResult("m1", [], ["HTTP 404: Not Found"]), # unavailable
            cli.ModelResult("m2", [], ["HTTP 400: Content cannot be a plain string"]), # unsupported
            cli.ModelResult("m3", [], ["TimeoutError: timed out"]), # timeout
            cli.ModelResult("m4", [], ["boom"]), # fail
        ]
        counts = cli.summarize_failure_counts(results)
        self.assertEqual(counts["unavailable"], 1)
        self.assertEqual(counts["unsupported"], 1)
        self.assertEqual(counts["timeout"], 1)
        self.assertEqual(counts["fail"], 1)

    def test_render_table_strips_ansi_for_width(self):
        headers = ["name", "status"]
        rows = [
            ["model-1", cli.colorize("ok", cli.COLOR_GREEN)],
            ["very-long-model-name", cli.colorize("fail", cli.COLOR_RED)],
        ]
        table = cli.render_table(rows, headers)
        lines = table.split("\n")
        # Header line: name (4) + 2 spaces + status (6)
        # model-1 (7) + padding(13) + 2 spaces + ok (2)
        # very-long-model-name (20) + 2 spaces + fail (4)
        self.assertIn("model-1             ", lines[2]) # 20 chars width

if __name__ == "__main__":
    unittest.main()
