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

    def test_chunkkdafwd_camelcase_is_model_compute_and_gets_counter_estimate(self) -> None:
        m = self._build()
        eff = m["efficiency"]
        # Custom KDA kernels enter the model-compute denominator (unmodeled list,
        # no semantic formula yet).
        self.assertIn("ChunkKdaFwd", eff["unmodeled_flop_types"])
        self.assertIn("chunk_kda_bwd_kernel_intra", eff["unmodeled_flop_types"])
        # CamelCase ChunkKdaFwd now gets the same-capture counter-calibrated
        # MFU estimate as the underscore-named KDA kernels (was None before).
        by_type = {t["type"]: t for t in eff["by_type"]}
        fwd = by_type.get("ChunkKdaFwd")
        self.assertIsNotNone(fwd)
        self.assertIsNotNone(fwd.get("mfu_estimated"),
                             "ChunkKdaFwd should carry a counter-calibrated MFU estimate")
        self.assertEqual(fwd.get("mfu_estimate_basis"), "same_capture_counter_calibration")
        bwd = by_type.get("chunk_kda_bwd_kernel_intra")
        self.assertIsNotNone(bwd.get("mfu_estimated"))

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


if __name__ == "__main__":
    unittest.main()
