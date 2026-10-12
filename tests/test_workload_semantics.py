"""Semantic accounting regressions using synthetic profiler records only."""
import sqlite3
import os
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from llminsight.metrics.efficiency import compute_efficiency
from llminsight.parser.derive import derive_model


def kernel(name="train", typ="MatMulV3", phase="model"):
    return {"Name": name, "Type": typ, "Workload Phase": phase,
            "Accelerator Core": "AI_CORE", "Duration(us)": 400.0,
            "Input Shapes": "4096,4096;4096,4096", "Input Data Types": "BF16;BF16",
            "Output Shapes": "4096,4096", "Output Data Types": "BF16",
            "aic_mac_ratio": 0.8, "aiv_vec_ratio": 0.0}


class WorkloadTests(unittest.TestCase):
    def test_unpaired_gemm_does_not_prove_recompute(self):
        result = compute_efficiency(SimpleNamespace(kernel_details=pd.DataFrame([kernel()])))
        self.assertFalse(result["recompute_estimate"]["available"])

    def test_optimizer_gemm_does_not_become_model_work_or_recompute(self):
        train = compute_efficiency(SimpleNamespace(kernel_details=pd.DataFrame([kernel()])))
        mixed = compute_efficiency(SimpleNamespace(kernel_details=pd.DataFrame([
            kernel(), kernel("optimizer", phase="optimizer")
        ])))
        self.assertEqual(train["useful_flops_estimated_total"], mixed["useful_flops_estimated_total"])
        self.assertEqual(train["recompute_estimate"], mixed["recompute_estimate"])
        self.assertEqual(mixed["workload_accounting"]["optimizer_kernel_count"], 1)

    def test_unmodeled_attention_really_contributes_to_proxy(self):
        attention = kernel("attention", "SparseFlashMla")
        attention["Input Shapes"] = ""
        attention["Output Shapes"] = ""
        result = compute_efficiency(SimpleNamespace(kernel_details=pd.DataFrame([kernel(), attention])))
        self.assertGreater(result["mfu_estimate"]["custom_estimated_flops"], 0)
        self.assertGreater(result["useful_flops_estimated_total"], result["useful_flops_modeled_partial"])

    def test_short_unknown_compute_cannot_be_declared_formula_complete(self):
        anchor = kernel()
        anchor["Duration(us)"] = 4000
        fused = kernel("mega_moe_bf16", "mega_moe_bf16")
        fused.update({"Input Shapes": "", "Output Shapes": "", "Accelerator Core": "MIX_AIC"})
        result = compute_efficiency(SimpleNamespace(kernel_details=pd.DataFrame([anchor, fused])))
        self.assertGreater(result["flop_coverage_pct"], 90)
        self.assertFalse(result["flop_model_complete"])
        self.assertIsNone(result["useful_flops_total"])
        self.assertGreater(result["useful_flops_estimated_total"], result["useful_flops_modeled_partial"])

    def test_mhc_compute_is_not_silently_excluded_from_model_work(self):
        custom = kernel("MhcPreBackward", "MhcPreBackward")
        custom.update({"Input Shapes": "", "Output Shapes": "", "Accelerator Core": "MIX_AIC"})
        result = compute_efficiency(SimpleNamespace(kernel_details=pd.DataFrame([kernel(), custom])))
        self.assertIn("MhcPreBackward", result["unmodeled_flop_types"])
        self.assertGreater(result["mfu_estimate"]["custom_estimated_flops"], 0)

    def test_mega_moe_is_fused_model_work_without_invented_shapes(self):
        fused = kernel("mega_moe_bf16", "mega_moe_bf16")
        fused.update({"Input Shapes": "", "Output Shapes": "", "Accelerator Core": "MIX_AIC"})
        result = compute_efficiency(SimpleNamespace(kernel_details=pd.DataFrame([kernel(), fused])))
        self.assertIn("mega_moe_bf16", result["unmodeled_flop_types"])
        self.assertGreater(result["fused_moe"]["phases"][0]["estimated_flops"], 0)
        self.assertIsNone(result["fused_moe"]["phases"][0]["communication_us"])

    def test_fused_proxy_requires_formula_anchor_and_consistent_peak(self):
        fused = kernel("mega_moe_bf16", "mega_moe_bf16")
        fused.update({"Input Shapes": "", "Output Shapes": "", "Accelerator Core": "MIX_AIC"})
        invalid_peak = kernel()
        invalid_peak["Duration(us)"] = 0.01
        for records in ([fused], [invalid_peak, fused]):
            result = compute_efficiency(SimpleNamespace(kernel_details=pd.DataFrame(records)))
            self.assertIsNone(result["fused_moe"]["phases"][0]["estimated_flops"])
            self.assertIsNone(next(x for x in result["by_type"] if x["type"] == "mega_moe_bf16")["mfu_estimated"])

    def test_fused_proxy_uses_recorded_active_time_not_full_fused_duration(self):
        from llminsight.config import SETTINGS
        fused = kernel("mega_moe_bf16", "mega_moe_bf16")
        fused.update({"Input Shapes": "", "Output Shapes": "", "Accelerator Core": "MIX_AIC",
                      "aic_mac_time(us)": 50, "aiv_vec_time(us)": 10})
        result = compute_efficiency(SimpleNamespace(kernel_details=pd.DataFrame([kernel(), fused])))
        expected = (50e-6 * SETTINGS.chip.peak_cube_flops("BF16")
                    + 10e-6 * SETTINGS.chip.peak_vector_flops("BF16")) * result["mfu_estimate"]["counter_scale"]
        observed = result["fused_moe"]["phases"][0]
        self.assertEqual(observed["mac_active_us"], 50)
        self.assertAlmostEqual(observed["estimated_flops"] / expected, 1, places=5)

    def test_hidden_size_uses_moe_input_not_expanded_norm(self):
        rows = [{"Name": "RmsNorm", "Type": "RmsNorm", "Input Shapes": "1,8,32;32", "Output Shapes": "1,8,32"},
                {"Name": "RmsNorm", "Type": "RmsNorm", "Input Shapes": "1,8,128;128", "Output Shapes": "1,8,128"}]
        profile = SimpleNamespace(kernel_details=pd.DataFrame(rows), meta={
            "framework_semantics": {"mega_moe_input_shapes": ["8,32;8,2;8,2;3,16,32;3,32,8"]}})
        self.assertEqual(derive_model(profile)["hidden_size"]["value"], 32)

    def test_optimizer_phase_uses_launch_time_across_worker_threads(self):
        from llminsight.parser.semantics import annotate_workload
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "ascend_pytorch_profiler_0.db"
            with closing(sqlite3.connect(db)) as con:
                con.executescript("""
                    CREATE TABLE STRING_IDS(id INTEGER, value TEXT);
                    INSERT INTO STRING_IDS VALUES(1, 'Optimizer.step#Test.step');
                    CREATE TABLE PYTORCH_API(startNs TEXT, endNs TEXT, globalTid INTEGER,
                        name INTEGER, inputShapes INTEGER, inputDtypes INTEGER);
                    INSERT INTO PYTORCH_API VALUES('1000000','2000000',4294967297,1,NULL,NULL);
                    CREATE TABLE CANN_API(startNs INTEGER, globalTid INTEGER, connectionId INTEGER);
                    INSERT INTO CANN_API VALUES(1500000,4294967298,7);
                    CREATE TABLE TASK(deviceId INTEGER, streamId INTEGER, taskId INTEGER,
                        startNs INTEGER, connectionId INTEGER);
                    INSERT INTO TASK VALUES(0,3,9,2500499,7);
                """)
                con.commit()
            frame = pd.DataFrame([{"Device_id": 0, "Stream ID": 3, "Task ID": 9, "Start Time(us)": 2500.5}])
            result = annotate_workload(os.path.relpath(td), frame)
            self.assertTrue(result["optimizer_window_count"])
            self.assertEqual(frame.iloc[0]["Workload Phase"], "optimizer")

    def test_long_wait_spans_bins_and_notify_record_is_not_wait(self):
        from llminsight.metrics.smart_timeline import _wait_xfer_from_task_slices
        with tempfile.TemporaryDirectory() as td:
            Path(td, "task_time.csv").write_text(
                "kernel_type,task_start(us),task_time(us)\n"
                "NOTIFY_WAIT_SQE,1000,2500\nNOTIFY_RECORD_SQE,1000,4000\nUBDMA,2500,1000\n")
            wait, transfer = _wait_xfer_from_task_slices(td, 1000, 1000, 4)
            self.assertEqual(wait, [1000, 1000, 500, 0])
            self.assertEqual(transfer, [0, 500, 500, 0])

    def test_hbm_pre_window_peak_is_not_first_bucket_peak(self):
        from llminsight.metrics.smart_timeline import _hbm_capacity_lane
        profile = SimpleNamespace(memory_record=pd.DataFrame({
            "Timestamp(us)": [500, 999, 1100, 2200], "Total Reserved(MB)": [10000, 1, 2, 3]}))
        self.assertEqual(_hbm_capacity_lane(profile, 1000, 1000, 2)["abs"], [2, 3])
        profile.memory_record = pd.DataFrame({
            "Timestamp(us)": [999, 1100, 2200], "Total Reserved(MB)": [10000, 2, 3]})
        self.assertEqual(_hbm_capacity_lane(profile, 1000, 1000, 2)["abs"], [10000, 3])

    def test_zero_recompute_saving_lower_bound_survives_whatif(self):
        from llminsight.metrics.core import _whatif_realistic
        result = _whatif_realistic(10000, 8000, 1000, 1000, 2000, 1000,
                                  0, 0.3, False, False, {},
                                  recompute={"us": 100, "us_lo": 0, "us_hi": 100, "flops_share": 0.1})
        lever = next(x for x in result["levers"] if x["id"] == "recompute_off")
        self.assertEqual(lever["recoverable_lo_us"], 0)


if __name__ == "__main__":
    unittest.main()
