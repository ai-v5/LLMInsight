"""Regression tests for whole-model FLOP-coverage edge cases surfaced by the
kimir3 (KDA / delta-attention) profile analysis.

Covers two fixes:
1. CUSTOM_MODEL_COMPUTE_MARKERS missed the CamelCase `ChunkKdaFwd` type, so that
   kernel was excluded from the model-compute denominator, the counter-calibrated
   MFU estimate, and the "覆盖不足" diagnosis.
2. The 算子极致优化 (op_ceiling) what-if lever and compute-bound headroom were
   disabled whenever efficiency_reliable=False (formula coverage <90%), even
   though they only rely on the formula-covered matmul/FA/FAG rows — hiding a
   real lever on KDA-kind profiles that do have a counter-calibrated estimate.
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from llminsight.metrics import compute_all
from llminsight.parser import load_profile


def _write_synthetic_profile(root: Path) -> None:
    """Minimal torch_npu-style profile:

    - one formula-covered MatMulV3 row (keeps formula FLOPs > 0),
    - two KDA custom kernels (ChunkKdaFwd — CamelCase — and
      chunk_kda_bwd_kernel_intra) with MAC/vector counter ratios so the
      same-capture counter calibration can produce estimates.
    This forces flop_model_complete=False (custom rows dominate the model
    denominator) while keeping mfu_estimate.available=True.
    """
    kd = root / "kernel_details.csv"
    kd_cols = [
        "Name", "Type", "Accelerator Core", "Duration(us)",
        "Input Shapes", "Input Data Types",
        "Output Shapes", "Output Data Types",
        "aic_mac_ratio", "aic_mte2_ratio", "aic_mte3_ratio",
        "aiv_vec_ratio", "cube_utilization(%)",
    ]
    kd_rows = [
        ["aclnnMatmul_Test", "MatMulV3", "AI_CORE", "397.0",
         "4096,4096;4096,4096", "DT_BF16;DT_BF16", "4096,4096", "DT_BF16",
         "0.9", "0.1", "0.0", "0.0", "90"],
        ["aclnnChunkKdaFwd_KdaChunkForward_ChunkKdaFwd", "ChunkKdaFwd", "MIX_AIC",
         "20000.0",
         "1,8,4,16;1,8,4,16;1,8,4,16;1,8,4,16;1,4,8;1,8;1,8",
         "DT_BF16;DT_BF16;DT_BF16;DT_BF16;FLOAT;FLOAT;FLOAT",
         "1,8,4,16", "DT_BF16",
         "0.06", "0.2", "0.0", "0.32", "40"],
        ["aclnnChunkKdaBwd_ChunkKdaBwdIntra", "chunk_kda_bwd_kernel_intra",
         "MIX_AIC", "20000.0",
         "1,8,4,16;1,8,4,16;1,8,4,16;1,8,4,16;1,4,8;1,8;1,8",
         "DT_BF16;DT_BF16;DT_BF16;DT_BF16;FLOAT;FLOAT;FLOAT",
         "1,8,4,16", "DT_BF16",
         "0.28", "0.3", "0.0", "0.1", "40"],
        # A KDA-marker custom kernel with NO recorded shapes: the semantic FLOP
        # formula cannot apply, so it stays unmodeled (and keeps formula coverage
        # <90%) while its MAC/vector counters still keep the estimate available.
        ["aclnnKdaUnknown_UnmodeledKernel", "kda_unmodeled_custom",
         "MIX_AIC", "15000.0",
         "", "", "", "",
         "0.1", "0.1", "0.0", "0.1", "40"],
    ]
    with kd.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(kd_cols)
        writer.writerows(kd_rows)

    st = root / "step_trace_time.csv"
    st.write_text(
        "Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,"
        "Communication,Free,Stage,Bubble,Communication(Not Overlapped and "
        "Exclude Receive),Preparing\n"
        "0,1,3000000.0,200000.0,100000.0,300000.0,50000.0,3250000.0,0.0,"
        "200000.0,1000.0\n",
        encoding="utf-8",
    )


class WholeModelCoverageTests(unittest.TestCase):
    def _build(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_synthetic_profile(root)
            prof = load_profile(str(root))
            return compute_all(prof)

    def test_chunkkdafwd_has_semantic_flops_and_counter_estimate(self) -> None:
        m = self._build()
        eff = m["efficiency"]
        # Formula coverage: ChunkKdaFwd + chunk_kda_bwd are now formula-backed
        # (semantic KDA chunk FLOPs), so they leave the unmodeled list; the
        # shape-less kda_unmodeled_custom kernel stays unmodeled.
        self.assertNotIn("ChunkKdaFwd", eff["unmodeled_flop_types"])
        self.assertNotIn("chunk_kda_bwd_kernel_intra", eff["unmodeled_flop_types"])
        self.assertIn("kda_unmodeled_custom", eff["unmodeled_flop_types"])
        # Modeled duration now includes the two formula-backed KDA rows
        # (397.0 + 20000.0 + 20000.0), not the shape-less custom row.
        self.assertAlmostEqual(float(eff["model_compute_modeled_us"]), 40397.0, delta=1.0)
        self.assertGreater(float(eff["model_compute_total_us"]), 55000.0)
        # Counter calibration still estimates the KDA kernels' MFU.
        by_type = {t["type"]: t for t in eff["by_type"]}
        fwd = by_type.get("ChunkKdaFwd")
        self.assertIsNotNone(fwd)
        self.assertEqual(fwd.get("mfu_estimate_basis"), "same_capture_counter_calibration")
        bwd = by_type.get("chunk_kda_bwd_kernel_intra")
        self.assertIsNotNone(bwd.get("mfu_estimated"))

    def test_kda_flop_formulas(self) -> None:
        from llminsight.metrics.efficiency import _estimate_kda_flops
        # chunk fwd: 2·T·H·(4·BT·K + 2·K·V) with T=8 H=4 K=16 V=16
        f = _estimate_kda_flops("ChunkKdaFwd",
                                [[1, 8, 4, 16]] * 4, [[1, 8, 4, 16]])
        self.assertAlmostEqual(float(f), 294912.0, delta=1.0)
        # bwd multipliers
        f2 = _estimate_kda_flops("chunk_kda_bwd_kernel_wy_dqkg_fused",
                                 [[1, 8, 4, 16]] * 5, [])
        self.assertAlmostEqual(float(f2) / float(f), 2.0, delta=1e-6)
        f3 = _estimate_kda_flops("chunk_kda_bwd_kernel_dAv",
                                 [[1, 8, 4, 16]] * 3, [])
        self.assertAlmostEqual(float(f3) / float(f), 0.5, delta=1e-6)
        f4 = _estimate_kda_flops("chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
                                 [[1, 8, 4, 16]] * 4, [])
        self.assertAlmostEqual(float(f4) / float(f), 2.0, delta=1e-6)
        f5 = _estimate_kda_flops("recompute_w_u_fwd_kda_kernel",
                                 [[1, 8, 4, 16]] * 3, [])
        self.assertAlmostEqual(float(f5) / float(f), 0.5, delta=1e-6)
        # causal conv1d: depthwise W=4 -> 2·T·D·W
        fc = _estimate_kda_flops("causal_conv1d_fwd_kernel",
                                 [[1, 8, 1024], [1024, 4]], [[1, 8, 1024]])
        self.assertAlmostEqual(float(fc), 2 * 8 * 1024 * 4, delta=1.0)
        fcb = _estimate_kda_flops("causal_conv1d_bwd_kernel",
                                  [[1, 8, 1024], [1, 8, 1024], [1024, 4], [1, 8, 1024]], [])
        self.assertAlmostEqual(float(fcb), 2 * float(fc), delta=1.0)
        # gate/cumsum: 2·T·H·K
        fg = _estimate_kda_flops("kda_gate_bwd_kernel",
                                 [[1, 8, 4, 16], [4], [64], [1, 8, 4, 16]], [])
        self.assertAlmostEqual(float(fg), 2 * 8 * 4 * 16, delta=1.0)
        # shape-less -> None (falls back to counter proxy)
        self.assertIsNone(_estimate_kda_flops("ChunkKdaFwd", [], []))

    def test_estimate_mode_still_surfaces_op_ceiling_lever_and_compute_headroom(self) -> None:
        m = self._build()
        eff = m["efficiency"]
        # KDA custom rows dominate the denominator -> not formula-complete, but
        # the counter calibration keeps the whole-model estimate available.
        self.assertFalse(eff["flop_model_complete"])
        self.assertFalse(eff["efficiency_reliable"])
        self.assertTrue(eff["mfu_estimate"]["available"])
        # The op_ceiling lever must still be offered (it only covers the
        # formula-backed matmul/FA/FAG rows), and the compute-bound footnote
        # must report a positive headroom instead of collapsing to "0".
        theo = m["theoretical"]
        lever_ids = {w.get("id") for w in theo.get("whatif", [])}
        self.assertIn("op_ceiling", lever_ids, theo.get("whatif"))
        cb = theo.get("compute_bound") or {}
        self.assertGreater(cb.get("headroom_us") or 0.0, 0.0, cb)
        self.assertLess(cb.get("ideal_matmul_us") or 0.0,
                        m["overview"]["us"]["computing"])

    def test_mc2_aicpu_wrappers_use_mix_lane(self) -> None:
        from llminsight.metrics.efficiency import (
            _is_custom_model_compute_type,
            _is_kda_model_compute_type,
        )
        from llminsight.metrics.smart_timeline import _stream_from_meta, _stream_from_name

        names = (
            "AlltoAllvGroupedMatMulMc2AicpuKernel",
            "GroupedMatMulAlltoAllvMc2AicpuKernel",
        )
        for name in names:
            # Name fallback is used for trace events because efficiency excludes
            # AI_CPU dispatch rows from the kernel index.
            self.assertEqual(_stream_from_name(name), "mix")
            # Metadata path is used by the msprof/task_time timeline exporter.
            self.assertEqual(_stream_from_meta(name, "AI_CPU"), "mix")

        # The paired MIX_AIC task is a fused model-compute family; it must not
        # disappear from whole-model MFU coverage just because its FLOPs are
        # counter-calibrated rather than shape-formula backed.
        self.assertTrue(_is_custom_model_compute_type("AlltoAllvGroupedMatMul"))
        self.assertTrue(_is_custom_model_compute_type("GroupedMatMulAlltoAllv"))
        self.assertFalse(_is_kda_model_compute_type("AlltoAllvGroupedMatMul"))
        self.assertFalse(_is_kda_model_compute_type("GroupedMatMulAlltoAllv"))

        # Ordinary alltoall/HCCL dispatch remains Communication; this guard keeps
        # the narrow MC2 exception from swallowing real collectives.
        self.assertEqual(_stream_from_name("hcom_alltoallv_AicpuKernel"), "comm")


if __name__ == "__main__":
    unittest.main()
