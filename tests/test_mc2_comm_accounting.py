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

# A Level-1 capture (no op attributes) publishes the raw kernel symbol as OP Type
# instead of the profiler label, and records no AI_CPU Mc2AicpuKernel launcher at
# all.  Exact type matching then loses the whole fused window.
_MANGLED_OP_ROWS = [
    ("MatMulV3", "AI_CORE", 10, 600_000.0),
    ("_ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel31Ascend950"
     "AllToAllVGroupedMatmulINS0_30AllToAllVAllGatherProblemShape"
     "EN7Catlass4Gemm5BlockEPh", "MIX_AIC", 4, 200_000.0),
    ("_ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel25"
     "GroupedMatmulAllToAllVTlaINS0_30AllToAllVAllGatherProblemShape"
     "EN7Catlass4Gemm5BlockEPh", "MIX_AIC", 4, 100_000.0),
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


def _write_capture(root: Path, *, link_bytes_recorded: bool,
                   mangled_mc2: bool = False) -> None:
    """4.0 s step: computing 2.5 s, not-overlapped comm 0.4 s, free rest."""
    root.mkdir(parents=True, exist_ok=True)
    op_rows = _MANGLED_OP_ROWS if mangled_mc2 else _OP_ROWS
    stage, computing, comm_no = 4_000_000.0, 2_500_000.0, 400_000.0
    with open(root / "step_trace_time.csv", "w", encoding="utf-8") as fh:
        fh.write(_STEP_HEADER)
        fh.write(f"1,0,{computing},{comm_no},900000,1300000,"
                 f"{stage - computing - comm_no},{stage},0,0\n")

    with open(root / "op_statistic.csv", "w", encoding="utf-8") as fh:
        fh.write(_OP_HEADER)
        total = sum(r[3] for r in op_rows)
        for op_type, core, count, us in op_rows:
            fh.write(f"1,{op_type},{core},{count},{us},1,{us / count},{us / count},"
                     f"{us / total * 100:.3f}\n")

    with open(root / "kernel_details.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(_KERNEL_COLUMNS) + "\n")
        for op_type, core, count, us in op_rows:
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
    mangled_mc2 = False

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "rank_leaf"
        _write_capture(self.dir, link_bytes_recorded=self.link_bytes_recorded,
                       mangled_mc2=self.mangled_mc2)
        self.prof = load_profile(str(self.dir))
        self.metrics = compute_all(self.prof)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class FusedA2aVisibilityTests(_Base):
    """The a2a inside an MC2 fused kernel must not read as "EP is off".

    ``communication.json`` never records it, so the communication section used to
    publish ``a2a.available = false`` with no explanation, and the EP guess said
    "no reliable EP world-size signal" even though the fused kernel proves EP is on.
    """

    def test_communication_a2a_points_at_the_fused_window(self) -> None:
        comm = self.metrics["communication"]
        a2a = comm["a2a"]
        self.assertFalse(a2a["available"])           # nothing in communication.json
        self.assertGreater(a2a["fused_window_us"], 0)  # ... but the window is real
        self.assertIn("融合", a2a["note"])

    def test_ep_guess_cites_the_fused_all_to_all(self) -> None:
        config = self.metrics["meta"]["config"]
        basis = config["guesses"]["ep_world_size"]["basis"]
        self.assertIn("MC2 融合 all-to-all", basis)
        self.assertEqual(config["guesses"]["ep_world_size"]["guess"], None)


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


class FusedMc2EstimateTests(unittest.TestCase):
    """Level-1 MC2 rows carry no shapes → estimate MFU from function + anchors.

    The fused window's GEMM work is reconstructed from (a) the same layer's plain
    expert grouped GEMM (per-expert weight [K,N] + routed-row count M) and (b) the
    function contract (each call = one two-projection expert GEMM group).  The
    result is an ESTIMATED equivalent MFU of the fused window (the denominator still
    contains the internal a2o), never a FORMULA value.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "rank_leaf"
        self.dir.mkdir(parents=True, exist_ok=True)
        header = "Device_id,Type,Name,Accelerator Core,Duration(us),Input Shapes," \
                 "Input Data Types,Output Shapes,Output Data Types\n"
        dispatch = ("_ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel31Ascend950"
                    "AllToAllVGroupedMatmulINS0_30AllToAllVAllGatherProblemShapeEPh")
        combine = ("_ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel25"
                   "GroupedMatmulAllToAllVTlaINS0_30AllToAllVAllGatherProblemShapeEPh")
        with open(self.dir / "kernel_details.csv", "w", encoding="utf-8") as fh:
            fh.write(header)
            # Plain expert grouped GEMM: activation rows 1024, per-expert weight [4,2048,4096]
            for _ in range(2):
                fh.write(f'1,GroupedMatmul,aclnnGroupedMatmulV5,MIX_AIC,500000,'
                         f'"1024,2048;4,2048,4096",DT_BF16;DT_BF16,"4,2048,4096",DT_BF16\n')
            # Fused MC2 kernels: NO shapes/dtypes at all (Level-1 capture)
            for _ in range(2):
                fh.write(f'1,{dispatch},{dispatch}_0,MIX_AIC,150000,,,,\n')
            for _ in range(2):
                fh.write(f'1,{combine},{combine}_0,MIX_AIC,100000,,,,\n')
        self.prof = load_profile(str(self.dir))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_equivalent_mfu_is_estimated_from_anchors(self) -> None:
        from llminsight.config import SETTINGS
        from llminsight.metrics.matrix_power import compute_matrix_power

        result = compute_matrix_power(self.prof, {})
        est = result["fused_mc2_estimate"]
        self.assertIsNotNone(est)
        self.assertEqual(est["rows"], 1024)
        self.assertEqual(tuple(est["contracts"][0]), (2048, 4096))
        self.assertEqual(est["pairs"], 2)
        self.assertEqual(est["dtype"], "BF16")
        # 2 calls x 2·M·K·N over 0.5 s of fused window
        expected_flops = 2 * (2.0 * 1024 * 2048 * 4096)
        self.assertAlmostEqual(est["total_flops"], expected_flops, delta=expected_flops * 1e-9)
        peak = SETTINGS.chip.peak_cube_flops("BF16")
        expected_mfu = expected_flops / (peak * 0.5)
        self.assertAlmostEqual(est["equivalent_mfu"], expected_mfu, places=9)
        self.assertFalse(est["above_peak"])

        fused = [g for g in result["groups"] if g["fused_window"]]
        self.assertEqual(len(fused), 2)
        for group in fused:
            self.assertEqual(group["dtype"], "BF16")
            self.assertEqual(group["evidence_level"], "ESTIMATED")
            self.assertAlmostEqual(group["mfu"], expected_mfu, places=6)
            self.assertIsNone(group["mbu"])
            self.assertIn("功能约定", group["estimate_basis"])
        # work is split by each kernel's share of the fused window (a2a not separable)
        by_dur = {g["dur_us"]: g["flops"] for g in fused}
        self.assertAlmostEqual(by_dur[300000.0], expected_flops * 0.6, delta=1.0)
        self.assertAlmostEqual(by_dur[200000.0], expected_flops * 0.4, delta=1.0)

    def test_no_anchor_keeps_fused_mfu_unavailable(self) -> None:
        import pandas as pd
        from types import SimpleNamespace
        from llminsight.metrics.matrix_power import compute_matrix_power

        # same capture but with the plain expert GEMM removed -> nothing anchors
        kd = pd.read_csv(self.dir / "kernel_details.csv", low_memory=False)
        plain = SimpleNamespace(kernel_details=kd[kd["Type"] != "GroupedMatmul"])
        result = compute_matrix_power(plain, {})
        self.assertIsNone(result["fused_mc2_estimate"])
        for group in (g for g in result["groups"] if g["fused_window"]):
            self.assertIsNone(group["mfu"])
            self.assertNotEqual(group["evidence_level"], "ESTIMATED")

    def test_anchor_prefers_the_expert_weight_over_a_3d_activation(self) -> None:
        """A 3-D activation repeats a 2-D operand's tail; that must not become K,N."""
        import pandas as pd
        from llminsight.metrics.matrix_power import _expert_gemm_anchors

        frame = pd.DataFrame([{
            "Type": "GroupedMatmul",
            "Input Shapes": '"1024,2048;4,2048,4096"',
            "Output Shapes": '"4,1024,2048;4,512,1024"',
            "Input Data Types": "DT_BF16;DT_BF16",
        }])
        anchors = _expert_gemm_anchors(frame)
        self.assertEqual(anchors["rows"], 1024)
        # [4,1024,2048] trails the activation shape [1024,2048] -> skipped as weight
        self.assertEqual(anchors["contracts"], [(512, 1024)])

    def test_other_fused_families_never_borrow_the_mc2_anchor(self) -> None:
        """B4: MegaMoE / attention fused windows are not MC2 and keep their own path."""
        import pandas as pd
        from types import SimpleNamespace
        from llminsight.metrics.matrix_power import compute_matrix_power

        kd = pd.read_csv(self.dir / "kernel_details.csv", low_memory=False)
        extra = pd.DataFrame([{c: None for c in kd.columns}])
        extra.loc[0, ["Type", "Name", "Accelerator Core", "Duration(us)"]] = \
            ["mega_moe_bf16", "mega_moe_bf16_0", "MIX_AIC", 900_000.0]
        merged = pd.concat([kd, extra], ignore_index=True)
        result = compute_matrix_power(SimpleNamespace(kernel_details=merged), {})
        est = result["fused_mc2_estimate"]
        self.assertIsNotNone(est)
        # only the two MC2 fused calls are paired; the MegaMoE call is not counted
        self.assertEqual(est["pairs"], 2)
        self.assertEqual(est["dispatch_calls"], 2)
        self.assertEqual(est["combine_calls"], 2)
        mega = [g for g in result["groups"] if "mega_moe" in g["operator"].lower()]
        self.assertTrue(mega)
        for group in mega:
            self.assertIsNone(group["flops"])
            self.assertNotEqual(group["evidence_level"], "ESTIMATED")

    def test_own_shapes_anchor_uses_the_fused_dtype(self) -> None:
        """N4: a shape-bearing MC2 capture anchors on itself, including fp16 dtype."""
        import pandas as pd
        from types import SimpleNamespace
        from llminsight.config import SETTINGS
        from llminsight.metrics.matrix_power import compute_matrix_power, _expert_gemm_anchors

        header = ["Type", "Name", "Accelerator Core", "Duration(us)", "Input Shapes",
                  "Input Data Types", "Output Shapes", "Output Data Types"]
        rows = []
        for _ in range(2):
            rows.append(["AlltoAllvGroupedMatMul", "mc2", "MIX_AIC", 100_000.0,
                         "1024,2048;4,2048,4096", "DT_FLOAT16;DT_FLOAT16", "1024,4096", "DT_FLOAT16"])
            rows.append(["GroupedMatMulAlltoAllv", "mc2b", "MIX_AIC", 100_000.0,
                         "1024,4096;4,4096,2048", "DT_FLOAT16;DT_FLOAT16", "1024,2048", "DT_FLOAT16"])
        frame = pd.DataFrame(rows, columns=header)
        anchors = _expert_gemm_anchors(frame)
        self.assertTrue(anchors.get("from_fused_rows"))
        self.assertEqual(anchors["dtype"], "FP16")
        result = compute_matrix_power(SimpleNamespace(kernel_details=frame), {})
        est = result["fused_mc2_estimate"]
        self.assertEqual(est["dtype"], "FP16")
        peak = SETTINGS.chip.peak_cube_flops("FP16")
        self.assertAlmostEqual(est["equivalent_mfu"],
                               est["total_flops"] / (peak * est["total_dur_us"] * 1e-6), places=9)

    def test_anchor_implying_above_peak_fails_closed(self) -> None:
        """A wrong anchor (too many rows / wrong operand) must not publish an MFU."""
        from types import SimpleNamespace
        from llminsight.metrics.matrix_power import compute_matrix_power

        kd = self.prof.kernel_details.copy()
        # 1000x more routed rows than the window could physically compute
        kd.loc[kd["Type"] == "GroupedMatmul", "Input Shapes"] = '"1024,2048;4,2048,4096"'
        bumped = kd.copy()
        bumped["Input Shapes"] = bumped["Input Shapes"].where(
            bumped["Type"] != "GroupedMatmul", '"1024000000,2048;4,2048,4096"')
        result = compute_matrix_power(SimpleNamespace(kernel_details=bumped), {})
        est = result["fused_mc2_estimate"]
        self.assertIsNotNone(est)
        self.assertTrue(est["above_peak"])
        for group in (g for g in result["groups"] if g["fused_window"]):
            self.assertIsNone(group["mfu"])
            self.assertIsNone(group["flops"])
            self.assertNotEqual(group["evidence_level"], "ESTIMATED")
            self.assertIn("超过物理峰值", group["estimate_basis"])


class RecomputeConsistencyTests(unittest.TestCase):
    """B2: a caller-count "off" must not coexist with a published replay lever."""

    OVERVIEW = {"available": True, "step_count": 1,
                "us": {"stage": 2_000_000.0, "computing": 1_500_000.0,
                       "comm_not_overlapped": 100_000.0, "free": 100_000.0,
                       "overlapped": 300_000.0, "communication": 400_000.0,
                       "bubble": 0.0, "preparing": 0.0}}

    def _eff(self, candidates, share=0.05):
        return {
            "available": True, "efficiency_reliable": False, "peak_inconsistent": False,
            "recompute_estimate": {
                "available": True, "likely_granularity": "selective", "confidence": "medium",
                "flops_share": share, "flops_share_lo": share * 0.8, "flops_share_hi": share * 1.2,
                "candidates": candidates,
            },
            "attention_training_flops": {"available": False},
            "matmul_training_flops": {"available": False},
            "chip": {"name": "Ascend 950DT", "cube_bf16_tflops": 432.0},
        }

    def _capture(self, value):
        return {"capture": {"recompute": {"value": value, "recompute_forward_multiplier": 0.0}},
                "model": {}, "flags": {}, "env": {}, "guesses": {}}

    def test_identified_replay_outranks_a_caller_count_off(self) -> None:
        from llminsight.metrics import core
        candidates = [{"type": "RmsNorm", "method": "forward_backward_excess", "count": 10,
                       "forward_count": 20, "backward_count": 10,
                       "duration_us": 300_000.0, "evidence": "fwd/bwd 20/10"}]
        res = core.theoretical(None, self.OVERVIEW, self._eff(candidates), self._capture("off"))
        rc = res["recompute"]
        self.assertEqual(rc["granularity"], "selective")
        self.assertEqual(rc["method"], "measured_candidates")
        self.assertAlmostEqual(rc["overhead_us"], 300_000.0, places=1)
        self.assertIn("recompute_off", [w["id"] for w in res["whatif"]])

    def test_no_identified_replay_publishes_no_lever(self) -> None:
        from llminsight.metrics import core
        res = core.theoretical(None, self.OVERVIEW, self._eff([]), self._capture("off"))
        self.assertIsNone(res["recompute"])
        self.assertNotIn("recompute_off", [w["id"] for w in res["whatif"]])


class MatrixPowerMc2Tests(_Base):
    def test_mc2_fused_rows_are_marked_fused_and_have_no_mbu(self) -> None:
        result = compute_matrix_power(self.prof, self.metrics["efficiency"])
        self.assertTrue(result["available"])
        fused = [g for g in result["groups"] if g["operator"].startswith(("AlltoAllv", "GroupedMatMulAlltoAllv"))]
        self.assertEqual(len(fused), 2)
        for group in fused:
            self.assertTrue(group["fused_window"], group["label"])
            # MBU needs bytes inside a window that also carries the a2a -> unknown
            self.assertIsNone(group["mbu"], group["label"])
            # MFU may be estimated from the capture's own shapes, but never FORMULA
            self.assertNotEqual(group["evidence_level"], "FORMULA", group["label"])
        rows = [r for r in result["rows"] if r["operator"] == "AlltoAllvGroupedMatMul"]
        self.assertTrue(rows and all(r["fused_window"] for r in rows))


class RecomputeSingleSourceTests(_Base):
    """The ledger bucket must mirror theoretical.recompute instead of re-deriving.

    Re-deriving produced a model projection (computing × r_re) in the ledger while
    the What-if lever published the measured replay duration — a 32x disagreement
    for the same "重计算开销" label plus a full-vs-selective contradiction.
    """

    def test_hidden_overhead_mirrors_the_shared_recompute_view(self) -> None:
        from llminsight.metrics import core
        view = {"granularity": "selective", "overhead_us": 123_456.0,
                "footprint_us": 987_654.0, "overhead_pct": 3.09,
                "method": "measured_candidates"}
        ho = core.hidden_overhead(
            self.prof, core.overview(self.prof), {"capture": {"recompute": {"value": "selective"}}},
            recompute_view=view)
        bucket = next(b for b in ho["buckets"] if b["key"] == "recompute")
        self.assertAlmostEqual(bucket["us"], 123_456.0, places=1)
        self.assertIn("Selective", bucket["label"])
        self.assertIn("保守下界", bucket["detail"])
        self.assertIn("足迹上界", bucket["detail"])
        # non-additive: recompute already lives inside Computing
        self.assertFalse(bucket.get("additive", True))

    def test_ledger_labels_a_model_projection_as_projection(self) -> None:
        """B1: `computing × r_re` is a projection, never "实测可识别重放算子耗时"."""
        from llminsight.metrics import core
        view = {"granularity": "full", "overhead_us": 625_000.0, "footprint_us": None,
                "method": "model_projection"}
        ho = core.hidden_overhead(
            self.prof, core.overview(self.prof), {"capture": {"recompute": {"value": "full"}}},
            recompute_view=view)
        bucket = next(b for b in ho["buckets"] if b["key"] == "recompute")
        self.assertAlmostEqual(bucket["us"], 625_000.0, places=1)
        self.assertIn("模型投影", bucket["detail"])
        self.assertIn("而非实测", bucket["detail"])
        self.assertNotIn("实测可识别重放算子耗时", bucket["detail"])
        self.assertIn("模型投影", bucket["source"])

    def test_no_quantified_replay_means_no_ledger_bucket(self) -> None:
        """B2: without a number there is no bucket (the card only shows full|selective)."""
        from llminsight.metrics import core
        view = {"granularity": "off", "overhead_us": 0.0, "method": "model_projection"}
        ho = core.hidden_overhead(
            self.prof, core.overview(self.prof), {"capture": {"recompute": {"value": "off"}}},
            recompute_view=view)
        self.assertFalse([b for b in ho["buckets"] if b["key"] == "recompute"])

    def test_display_name_keeps_known_families_apart(self) -> None:
        from llminsight.parser.names import op_display_name
        dispatch = ("_ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel31Ascend950"
                    "AllToAllVGroupedMatmulINS0_30AllToAllVAllGatherProblemShapeEPh")
        combine = ("_ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel25"
                   "GroupedMatmulAllToAllVTlaINS0_30AllToAllVAllGatherProblemShapeEPh")
        self.assertIn("alltoallvgroupedmatmul", op_display_name(dispatch).lower())
        self.assertIn("groupedmatmulalltoallv", op_display_name(combine).lower())
        self.assertNotEqual(op_display_name(dispatch), op_display_name(combine))
        self.assertEqual(op_display_name("MatMulV3"), "MatMulV3")
        # already-truncated symbols (e.g. a timeline payload capped at 90 chars) must
        # still resolve to the canonical family instead of leaking a raw prefix
        self.assertIn("alltoallvgroupedmatmul", op_display_name(dispatch[:90]).lower())


class MangledMc2SymbolTests(_Base):
    """A Level-1 capture names the MC2 fused kernels with the raw C++ symbol.

    ``_ZN7Catccos…Ascend950AllToAllVGroupedMatmul…EPh`` never equals the profiler
    label, so exact OP-Type matching lost the whole fused window: it fell into
    ``Elementwise/Other`` while ``moe_focus.fused_dispatch_combine_us`` read 0 —
    the same failure the plain-label fix (#70) removed for op-attr captures.
    """

    mangled_mc2 = True

    def test_attribution_matches_fused_window_by_symbol_marker(self) -> None:
        attr = self.metrics["attribution"]
        modules = {m["module"]: m["us"] for m in attr["modules"]}
        fused_key = "MoE fused dispatch/combine (GEMM+comm)"
        self.assertIn(fused_key, modules)
        self.assertAlmostEqual(modules[fused_key], 300_000.0, places=1)
        self.assertAlmostEqual(attr["moe_focus"]["fused_dispatch_combine_us"], 300_000.0,
                               places=1)
        # residual keeps only the genuinely unattributed compute (Cast)
        self.assertAlmostEqual(modules["Elementwise/Other"], 40_000.0, places=1)
        # total must not double count the moved rows
        self.assertAlmostEqual(
            attr["total_us"],
            sum(m["us"] for m in attr["modules"]), places=1)

    def test_matrix_power_marks_mangled_fused_rows(self) -> None:
        result = compute_matrix_power(self.prof, self.metrics["efficiency"])
        fused = [g for g in result["groups"]
                 if "alltoallvgroupedmatmul" in g["operator"].lower()
                 or "groupedmatmulalltoallv" in g["operator"].lower()]
        self.assertEqual(len(fused), 2)
        for group in fused:
            self.assertTrue(group["fused_window"], group["label"])
            self.assertIsNone(group["mbu"], group["label"])   # bytes unseparable
            # this synthetic capture writes shapes for every row, so the fused
            # windows get the ESTIMATED path (never FORMULA)
            self.assertNotEqual(group["evidence_level"], "FORMULA", group["label"])

    def test_fused_markers_do_not_swallow_plain_expert_gemm(self) -> None:
        from llminsight.metrics.efficiency import _is_mc2_grouped_gemm_type
        self.assertFalse(_is_mc2_grouped_gemm_type("GroupedMatmul"))
        self.assertFalse(_is_mc2_grouped_gemm_type("MatMulV3"))
        self.assertTrue(_is_mc2_grouped_gemm_type("AlltoAllvGroupedMatMul"))
        self.assertTrue(_is_mc2_grouped_gemm_type(
            "_ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel25"
            "GroupedMatmulAllToAllVTlaEPh"))


if __name__ == "__main__":
    unittest.main()
