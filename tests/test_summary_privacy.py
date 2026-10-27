"""Synthetic privacy regressions; no provider calls or real capture data."""
import json
import unittest

from llminsight.insight.summarizer import build_summary


class SummaryPrivacyTests(unittest.TestCase):
    def test_raw_operator_name_is_replaced_by_local_alias(self):
        metrics = {"efficiency": {"available": True, "top_optimization": [{
            "name": "/home/synthetic-user/private-model/kernel", "bound": "compute",
            "wasted_us": 12.5, "dur_us": 20.0, "mfu": 0.4,
        }]}}
        result = build_summary(metrics, [])
        row = result["efficiency"]["top_optimization"][0]
        self.assertEqual(row["name"], "operator_1")
        self.assertEqual(row["dur_us"], 20.0)
        self.assertNotIn("private-model", json.dumps(result))

    def test_recognizable_identifiers_are_redacted_in_nested_fields(self):
        markers = [r"C:\Users\synthetic user\private\trace.json", "/home/synthetic-user/private/trace.json",
                   r"\\synthetic-host\share\trace.json", "https://synthetic.invalid/private?q=synthetic",
                   "synthetic-user@example.invalid", "u12345678", "192.168.12.34"]
        metrics = {"efficiency": {"available": True, "mfu_estimate": {
            "available": True, "nested": [{"note": "source " + item} for item in markers],
        }}}
        result = build_summary(metrics, [])
        text = " ".join(item["note"] for item in result["efficiency"]["mfu_estimate"]["nested"])
        for marker in markers:
            self.assertNotIn(marker, text)
        self.assertTrue(result["efficiency"]["mfu_estimate"]["available"])

    def test_redacted_dictionary_keys_do_not_collapse_metric_rows(self):
        source = {"/home/synthetic-a/item": 1, "/home/synthetic-b/item": 2, "redacted_field_1": 3}
        result = build_summary({"efficiency": {"available": True, "unmodeled_flop_types": source}}, [])
        retained = result["efficiency"]["unmodeled_flop_types"]
        self.assertEqual(sorted(retained.values()), [1, 2, 3])
        self.assertNotIn("/home/", json.dumps(retained))

    def test_ordinary_metrics_and_rule_text_remain_useful(self):
        cards = [{"id": "synthetic_rule", "severity": "info", "category": "compute",
                  "title": "Check MatMulV3", "root_cause": "low MFU", "suggestion": "review shapes",
                  "expected_gain": "candidate", "confidence": 0.5}]
        result = build_summary({"efficiency": {"available": True, "matmul_mfu": 0.5}}, cards)
        self.assertEqual(result["efficiency"]["matmul_mfu"], 0.5)
        self.assertEqual(result["rule_cards"][0]["title"], "Check MatMulV3")


if __name__ == "__main__":
    unittest.main()
