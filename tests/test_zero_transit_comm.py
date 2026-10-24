"""Zero-payload collective diagnosis (通信消除 / barrier 占位).

Real 32-node GLM5.2 efsdp1 capture: 17 zero-byte allReduce ops (one per layer,
inside a 2-rank expert-data group) plus one 0-byte allReduce pair cost ~1 s of a
4.7 s step.  A bandwidth table cannot see them (Transit Size≈0 -> bandwidth N/A),
so LLMInsight now aggregates them (`communication.zero_transit`), renders a note
in the report, and raises a rule card when the share is material.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llminsight.metrics import compute_all
from llminsight.parser import load_profile
from llminsight.report import build_report_html
from llminsight.rules import run_rules

_STEP_HEADER = (
    "Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,"
    "Communication,Free,Stage,Bubble,Preparing\n"
)


def _comm_op(elapse_ms: float, wait_ms: float, sync_ms: float,
             transit_mb: float = 0.0, transit_ms: float = 0.0,
             bandwidth_gbps: float = 0.0) -> dict:
    return {
        "Communication Time Info": {
            "Elapse Time(ms)": elapse_ms,
            "Transit Time(ms)": transit_ms,
            "Wait Time(ms)": wait_ms,
            "Synchronization Time(ms)": sync_ms,
            "Start Timestamp(us)": 0,
        },
        "Communication Bandwidth Info": {
            "LOCAL": {
                "Transit Size(MB)": transit_mb,
                "Transit Time(ms)": transit_ms,
                "Bandwidth(GB/s)": bandwidth_gbps,
            },
        },
    }


def _write_synthetic(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    stage = 4_680_000.0  # step_trace_time.csv is in microseconds
    computing = 3_050_000.0
    comm_not_overlapped = 1_540_000.0
    free = stage - computing - comm_not_overlapped
    with open(root / "step_trace_time.csv", "w", encoding="utf-8") as fh:
        fh.write(_STEP_HEADER)
        fh.write(f"6,3,{computing},{comm_not_overlapped},2180000,3720000,{free},{stage},0,1700\n")
    comm = {
        "step3": {
            "collective": {
                # two zero-byte allReduce barriers (~51 ms each)
                "hcom_allReduce__636_1_1@a": _comm_op(51.0, 50.9, 51.0),
                "hcom_allReduce__636_2_1@a": _comm_op(51.0, 50.9, 51.0),
                # a real allGather that moves bytes
                "hcom_allGather__612_3_1@b": _comm_op(10.0, 4.0, 0.0,
                                                     transit_mb=100.0, transit_ms=0.5,
                                                     bandwidth_gbps=200.0),
                "Total Op Info": _comm_op(112.0, 105.8, 102.0),
            },
            "p2p": {},
        },
    }
    (root / "communication.json").write_text(json.dumps(comm), encoding="utf-8")


class ZeroTransitCommTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "rank_leaf"
        _write_synthetic(self.dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_zero_transit_aggregation(self) -> None:
        prof = load_profile(str(self.dir))
        metrics = compute_all(prof)
        comm = metrics["communication"]
        self.assertTrue(comm["available"])
        zt = comm["zero_transit"]
        self.assertEqual(zt["count"], 2)
        self.assertAlmostEqual(zt["elapse_ms"], 102.0, places=2)
        self.assertAlmostEqual(zt["mean_elapse_ms"], 51.0, places=2)
        self.assertIn("allReduce", zt["by_type"])
        # 102 ms of 112 ms comm elapse
        self.assertAlmostEqual(zt["pct_of_comm"], 91.07, places=1)
        # the real allGather must NOT be counted as zero-payload
        self.assertNotIn("allGather", zt["by_type"])

    def test_rule_card_and_report_note(self) -> None:
        prof = load_profile(str(self.dir))
        metrics = compute_all(prof)
        cap = metrics.get("meta", {}).get("config")
        cards = run_rules(metrics, cap)
        card = next((c for c in cards if c.get("id") == "comm_zero_transit"), None)
        self.assertIsNotNone(card, [c.get("id") for c in cards])
        self.assertIn("零载荷集合通信", card["title"])
        self.assertEqual(card["category"], "通信")
        html = build_report_html(metrics, cards)
        self.assertIn("零载荷集合通信", html)
        self.assertIn("barrier", html)

    def test_no_card_when_negligible(self) -> None:
        # only a tiny zero-payload op (<3% of step, <5% of comm) -> no card
        tiny = Path(self._tmp.name) / "tiny_leaf"
        tiny.mkdir(parents=True)
        with open(tiny / "step_trace_time.csv", "w", encoding="utf-8") as fh:
            fh.write(_STEP_HEADER)
            fh.write("6,3,3000000,1000000,1000000,2000000,680000,4680000,0,1700\n")
        comm = {"step3": {"collective": {
            "hcom_allReduce__1_1_1@a": _comm_op(50.0, 49.0, 50.0),
            "hcom_allGather__2_2_1@b": _comm_op(1950.0, 900.0, 0.0,
                                               transit_mb=40000.0, transit_ms=1950.0,
                                               bandwidth_gbps=205.0),
        }, "p2p": {}}}
        (tiny / "communication.json").write_text(json.dumps(comm), encoding="utf-8")
        prof = load_profile(str(tiny))
        metrics = compute_all(prof)
        cards = run_rules(metrics, metrics.get("meta", {}).get("config"))
        self.assertIsNone(next((c for c in cards if c.get("id") == "comm_zero_transit"), None))


if __name__ == "__main__":
    unittest.main()
