"""MC2 fused-op accounting + uncollected-link-bytes regression tests.

Three defects this file pins down (all found on a real Level-1 MC2 capture):

1. ``communication.json``'s per-op ``Elapse Time`` was summed and then divided by
   the step time to produce "零载荷集合通信占 step X%" plus a "最多回收 step 的 X%"
   promise.  A sum of per-op intervals is not wall-clock (parallel communicators
   overlap), so that share can exceed the step-level not-overlapped
   communication by an order of magnitude.  The recoverable bound must come from
   ``step_trace_time.csv``, not from the sum.

2. On a capture where HCCL link bytes were never recorded (profiling level too
   low), every op has ``Transit Size == 0``.  That absence was read as "every
   collective is a zero-payload barrier", and the per-communicator role was
   pinned to ``barrier``.  Missing bandwidth data must fail closed.

3. MC2 fused dispatch/combine kernels appear twice in ``op_statistic.csv`` (the
   ``*Mc2AicpuKernel`` AI_CPU launcher and the real MIX_AIC kernel, same wall
   clock), so ``hotspots.total_us`` exceeded the step time and ``attribution``
   dumped the fused GEMM into ``Elementwise/Other`` while reporting
   ``MoE-Dispatch (alltoallv) = 0``.

All fixtures are synthetic — no captured data ships with this repository.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llminsight.metrics import compute_all
from llminsight.metrics.matrix_power import compute_matrix_power
from llminsight.parser import load_profile
from llminsight.report import build_report_html
from llminsight.rules import run_rules

_STEP_HEADER = (
    "Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,"
    "Communication,Free,Stage,Bubble,Preparing\n"
)
_OP_HEADER = (
    "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),"
    "Avg Time(us),Max Time(us),Ratio(%)\n"
)
_KERNEL_COLUMNS = [
    "Device_id", "Type", "Name", "Accelerator Core", "Duration(us)",
    "Input Shapes", "Input Data Types", "Output Shapes", "Output Data Types",
]

# Ops of the synthetic capture (µs).  The MC2 pair is deliberately duplicated
# across the AI_CPU launcher and the MIX_AIC kernel with the same duration: that
# is what a real MC2 record looks like.
_OP_ROWS = [
    # OP Type, Core Type, Count, Total Time(us)
    ("MatMulV3", "AI_CORE", 10, 600_000.0),
    ("AlltoAllvGroupedMatMul", "MIX_AIC", 4, 200_000.0),
    ("AlltoAllvGroupedMatMulMc2AicpuKernel", "AI_CPU", 4, 200_000.0),
    ("GroupedMatMulAlltoAllv", "MIX_AIC", 4, 100_000.0),
    ("GroupedMatMulAlltoAllvMc2AicpuKernel", "AI_CPU", 4, 100_000.0),
    ("HcclLaunchAicpuKernel", "AI_CPU", 8, 60_000.0),
    ("Cast", "AI_VECTOR_CORE", 20, 40_000.0),
]


def _comm_op(elapse_ms: float, wait_ms: float,
             transit_mb: float = 0.0, transit_ms: float = 0.0) -> dict:
    return {
        "Communication Time Info": {
            "Elapse Time(ms)": elapse_ms,
            "Transit Time(ms)": transit_ms,
            "Wait Time(ms)": wait_ms,
            "Synchronization Time(ms)": wait_ms,
            "Start Timestamp(us)": 0,
        },
        "Communication Bandwidth Info": {
            "LOCAL": {
                "Transit Size(MB)": transit_mb,
                "Transit Time(ms)": transit_ms,
                "Bandwidth(GB/s)": (transit_mb / transit_ms) if transit_ms else 0.0,
            },
        },
    }


def _write_capture(root: Path, *, link_bytes_recorded: bool) -> None:
    """4.0 s step: computing 2.5 s, not-overlapped comm 0.4 s, free rest."""
    root.mkdir(parents=True, exist_ok=True)
    stage, computing, comm_no = 4_000_000.0, 2_500_000.0, 400_000.0
    with open(root / "step_trace_time.csv", "w", encoding="utf-8") as fh:
        fh.write(_STEP_HEADER)
        fh.write(f"1,0,{computing},{comm_no},900000,1300000,"
                 f"{stage - computing - comm_no},{stage},0,0\n")

    with open(root / "op_statistic.csv", "w", encoding="utf-8") as fh:
        fh.write(_OP_HEADER)
        total = sum(r[3] for r in _OP_ROWS)
        for op_type, core, count, us in _OP_ROWS:
            fh.write(f"1,{op_type},{core},{count},{us},1,{us / count},{us / count},"
                     f"{us / total * 100:.3f}\n")

    with open(root / "kernel_details.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(_KERNEL_COLUMNS) + "\n")
        for op_type, core, count, us in _OP_ROWS:
            for _ in range(count):
                fh.write(f"1,{op_type},{op_type}_0,{core},{us / count},"
                         "\"128,64;7,64,128\",DT_BF16;DT_BF16,\"128,128\",DT_BF16\n")

    # Two communicators.  With link bytes recorded, one of them is a genuine
    # zero-payload barrier and the other really moves data; without them, both
    # look identical (everything is 0 bytes).
    if link_bytes_recorded:
        barrier_a = _comm_op(60.0, 59.5)
        barrier_b = _comm_op(40.0, 39.8)
        payload = _comm_op(300.0, 90.0, transit_mb=800.0, transit_ms=60.0)
    else:
        barrier_a = _comm_op(60.0, 59.5)
        barrier_b = _comm_op(40.0, 39.8)
        payload = _comm_op(300.0, 90.0)
    comm = {
        "step": {
            "collective": {
                "hcom_allReduce__612_1_1@a": barrier_a,
                "hcom_allReduce__612_2_1@a": barrier_b,
                "hcom_allGather__920_3_1@b": payload,
                "Total Op Info": _comm_op(400.0, 189.3),
            },
            "p2p": {},
        }
    }
    (root / "communication.json").write_text(json.dumps(comm), encoding="utf-8")


class _Base(unittest.TestCase):
    link_bytes_recorded = True

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "rank_leaf"
        _write_capture(self.dir, link_bytes_recorded=self.link_bytes_recorded)
        self.prof = load_profile(str(self.dir))
        self.metrics = compute_all(self.prof)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class LinkBytesNotCollectedTests(_Base):
    """A capture whose HCCL link bytes were never recorded must fail closed."""

    link_bytes_recorded = False

    def test_zero_transit_is_unavailable_not_a_fact(self) -> None:
        comm = self.metrics["communication"]
        self.assertFalse(comm["transit_collected"])
        zt = comm["zero_transit"]
        self.assertFalse(zt["available"])
        self.assertEqual(zt.get("count", 0), 0)
        self.assertIn("未采集", zt["reason"])

    def test_no_group_is_labelled_barrier(self) -> None:
        comm = self.metrics["communication"]
        for group in comm["by_group"]:
            self.assertNotEqual(group["role"], "barrier", group)
        for role in (comm.get("wait_attribution") or {}).get("by_role") or []:
            self.assertNotEqual(role["role"], "barrier", role)

    def test_no_zero_payload_rule_card_or_report_claim(self) -> None:
        cards = run_rules(self.metrics, self.metrics.get("meta", {}).get("config"))
        self.assertIsNone(next((c for c in cards if c.get("id") == "comm_zero_transit"), None))
        html = build_report_html(self.metrics, cards)
        self.assertNotIn("零载荷集合通信", html)


class LinkBytesCollectedTests(_Base):
    """With real byte counts the zero-payload diagnosis still works, but it must
    no longer publish a step-time share derived from the summed intervals."""

    def test_zero_transit_still_detected(self) -> None:
        comm = self.metrics["communication"]
        self.assertTrue(comm["transit_collected"])
        zt = comm["zero_transit"]
        self.assertTrue(zt["available"])
        self.assertEqual(zt["count"], 2)
        self.assertAlmostEqual(zt["elapse_ms"], 100.0, places=3)

    def test_card_does_not_claim_summed_elapse_is_step_time(self) -> None:
        cards = run_rules(self.metrics, self.metrics.get("meta", {}).get("config"))
        card = next((c for c in cards if c.get("id") == "comm_zero_transit"), None)
        self.assertIsNotNone(card)
        # 100 ms summed elapse over a 4.0 s step would be 2.5 %; the card must not
        # publish that (or any) summed-elapse step share as recoverable gain.
        self.assertNotIn("pct_of_step", card["evidence"])
        self.assertIn("未掩盖通信", card["expected_gain"])
        self.assertIn("非 wall-clock", card["title"])
        # unbounded "recover X% of the step" promise removed
        self.assertNotIn("最多回收 step 的", card["expected_gain"])


class HotspotsDuplicateWrapperTests(_Base):
    def test_mc2_launcher_wrapper_is_not_double_counted(self) -> None:
        hot = self.metrics["hotspots"]
        types = {o["type"] for o in hot["ops"]}
        self.assertNotIn("AlltoAllvGroupedMatMulMc2AicpuKernel", types)
        self.assertNotIn("GroupedMatMulAlltoAllvMc2AicpuKernel", types)
        # the real MIX_AIC kernels stay visible
        self.assertIn("AlltoAllvGroupedMatMul", types)
        self.assertIn("GroupedMatMulAlltoAllv", types)
        # denominator must exclude the duplicate launcher wall clock
        self.assertAlmostEqual(hot["total_us"], 600_000 + 200_000 + 100_000 + 60_000 + 40_000, places=1)
        self.assertAlmostEqual(hot["dedup"]["excluded_us"], 300_000.0, places=1)
        # ratio is recomputed against the de-duplicated total, not the raw sum
        row = next(o for o in hot["ops"] if o["type"] == "AlltoAllvGroupedMatMul")
        self.assertAlmostEqual(row["ratio"], 200_000 / 1_000_000 * 100, places=2)


class AttributionMc2Tests(_Base):
    def test_fused_mc2_gets_its_own_module(self) -> None:
        attr = self.metrics["attribution"]
        modules = {m["module"]: m["us"] for m in attr["modules"]}
        fused_key = "MoE fused dispatch/combine (GEMM+comm)"
        self.assertIn(fused_key, modules)
        self.assertAlmostEqual(modules[fused_key], 300_000.0, places=1)
        self.assertAlmostEqual(attr["moe_focus"]["fused_dispatch_combine_us"], 300_000.0, places=1)
        # nothing of the fused MC2 time may leak into the residual bucket
        self.assertAlmostEqual(modules["Elementwise/Other"], 40_000.0, places=1)
        self.assertIn("MC2", attr["note"])


class HiddenOverheadTests(_Base):
    def test_format_init_is_not_added_to_device_total(self) -> None:
        ho = self.metrics["hidden_overhead"]
        bucket = next(b for b in ho["buckets"] if b["key"] == "format_init")
        self.assertFalse(bucket.get("additive", True))
        self.assertIn("Computing", bucket["detail"])
        # device total = not-overlapped comm + free only (format_init lives inside Computing)
        us = self.metrics["overview"]["us"]
        self.assertAlmostEqual(
            ho["device_total_us"], us["comm_not_overlapped"] + us["free"], places=1)

    def test_comm_wait_total_is_not_presented_as_wall_clock(self) -> None:
        ho = self.metrics["hidden_overhead"]
        bucket = next(b for b in ho["buckets"] if b["key"] == "comm_not_overlapped")
        self.assertIn("Σ", bucket["detail"])
        self.assertIn("不等于 step 墙钟", bucket["detail"])


class MatrixPowerMc2Tests(_Base):
    def test_mc2_fused_rows_are_marked_fused_and_have_no_mbu(self) -> None:
        result = compute_matrix_power(self.prof, self.metrics["efficiency"])
        self.assertTrue(result["available"])
        fused = [g for g in result["groups"] if g["operator"].startswith(("AlltoAllv", "GroupedMatMulAlltoAllv"))]
        self.assertEqual(len(fused), 2)
        for group in fused:
            self.assertTrue(group["fused_window"], group["label"])
            self.assertIsNone(group["mbu"], group["label"])
            self.assertIsNone(group["mfu"], group["label"])
        rows = [r for r in result["rows"] if r["operator"] == "AlltoAllvGroupedMatMul"]
        self.assertTrue(rows and all(r["fused_window"] for r in rows))


if __name__ == "__main__":
    unittest.main()
