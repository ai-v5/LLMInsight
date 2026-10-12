from types import SimpleNamespace

import pandas as pd
import pytest

from llminsight.metrics.matrix_power import compute_matrix_power


def test_matrix_power_separates_dtype_and_keeps_fused_rows():
    profile = SimpleNamespace(
        kernel_details=pd.DataFrame([
            {
                "Name": "matmul_bf16", "Type": "MatMulV3", "Duration(us)": 10.0,
                "Input Shapes": "4,8;8,16", "Input Data Types": "DT_BF16;DT_BF16",
                "Output Shapes": "4,16", "Output Data Types": "DT_BF16",
            },
            {
                "Name": "matmul_fp32", "Type": "MatMulV3", "Duration(us)": 20.0,
                "Input Shapes": "4,8;8,16", "Input Data Types": "FLOAT;FLOAT",
                "Output Shapes": "4,16", "Output Data Types": "FLOAT",
            },
            {
                "Name": "kda", "Type": "chunk_kda_fwd_kernel_intra_sub_chunk_fused", "Duration(us)": 30.0,
                "Input Shapes": "1,64,2,8;1,64,2,8;1,64,2,8", "Input Data Types": "BF16;BF16;BF16",
                "Output Shapes": "1,64,2,8", "Output Data Types": "BF16",
            },
        ])
    )

    result = compute_matrix_power(profile)

    assert result["available"] is True
    assert {item["label"] for item in result["groups"]} == {
        "MatMulV3@BF16", "MatMulV3@FP32", "chunk_kda_fwd_kernel_intra_sub_chunk_fused@BF16",
    }
    assert result["kernel_count"] == 3
    assert all("shape" in row and row["shape"] for row in result["rows"])
    assert all(row["mfu"] is not None for row in result["rows"])


def mega_profile():
    base = {"Input Shapes": "N/A", "Input Data Types": "N/A", "Output Shapes": "N/A",
            "Accelerator Core": "MIX_AIC", "Duration(us)": 100.0,
            "aic_mac_time(us)": 40.0, "aic_mac_ratio": 0.9}
    return SimpleNamespace(kernel_details=pd.DataFrame([
        dict(base, Type="mega_moe_bf16", Name="forward"),
        dict(base, Type="mega_moe_backward_bf16", Name="backward"),
        dict(base, Type="mega_moe_bf16", Name="wrapper", **{"Accelerator Core": "AI_CPU"}),
    ]))


def calibration():
    return {"peak_inconsistent": False,
            "chip": {"effective_peak_tflops": 432.0, "cube_bf16_tflops": 432.0},
            "mfu_estimate": {"counter_reference_proxy_flops": 1000.0, "counter_scale": 0.5}}


def test_mega_moe_is_visible_without_calibration():
    result = compute_matrix_power(mega_profile())
    assert result["available"] is True
    assert result["kernel_count"] == 2
    assert all(row["mfu"] is None for row in result["rows"])


def test_mega_moe_matrix_proxy_keeps_fusion_window_and_excludes_wrapper():
    result = compute_matrix_power(mega_profile(), calibration())
    assert result["kernel_count"] == 2
    assert {g["label"] for g in result["groups"]} == {"mega_moe_bf16@BF16", "mega_moe_backward_bf16@BF16"}
    for row in result["rows"]:
        assert row["evidence_level"] == "PROXY"
        assert row["mfu"] == pytest.approx(0.2)
        assert row["mac_active_us"] == 40.0
        assert row["mbu"] is None
        assert row["formula_available"] is False
    assert all(g["evidence_level"] == "PROXY" and g["mfu"] == pytest.approx(0.2) for g in result["groups"])


@pytest.mark.parametrize("missing", ["anchor", "peak", "counter", "dtype"])
def test_mega_moe_matrix_proxy_requires_evidence(missing):
    profile, eff = mega_profile(), calibration()
    if missing == "anchor":
        eff["mfu_estimate"]["counter_reference_proxy_flops"] = 0
    elif missing == "peak":
        eff["peak_inconsistent"] = True
    elif missing == "counter":
        profile.kernel_details["aic_mac_time(us)"] = "N/A"
        profile.kernel_details["aic_mac_ratio"] = "N/A"
    else:
        profile.kernel_details["Type"] = "mega_moe"
    result = compute_matrix_power(profile, eff)
    assert result["kernel_count"] == 2
    assert all(row["mfu"] is None and row["evidence_level"] == "UNAVAILABLE" for row in result["rows"])


def test_mega_moe_matrix_proxy_ratio_fallback_and_partial_group():
    profile = mega_profile()
    profile.kernel_details["Type"] = "mega_moe_bf16"
    profile.kernel_details["aic_mac_time(us)"] = "N/A"
    profile.kernel_details["aic_mac_ratio"] = [0.6, "N/A", "N/A"]
    result = compute_matrix_power(profile, calibration())
    assert result["rows"][0]["mfu"] == pytest.approx(0.3)
    assert result["rows"][1]["mfu"] is None
    assert result["groups"][0]["mfu"] is None  # missing work must not disappear from the group


def test_mega_moe_zero_mac_counter_is_not_missing():
    profile = mega_profile()
    profile.kernel_details["aic_mac_time(us)"] = 0.0
    result = compute_matrix_power(profile, calibration())
    assert all(row["mfu"] == 0.0 and row["flops"] == 0.0 for row in result["rows"])
    assert all(group["mfu"] == 0.0 and group["flops"] == 0.0 for group in result["groups"])


def test_mega_moe_dtype_uses_mangled_function_identifier():
    profile = mega_profile()
    profile.kernel_details["Type"] = ["_Z13mega_moe_bf16Pv", "_Z22mega_moe_bf16_backwardPv", "mega_moe_bf16"]
    result = compute_matrix_power(profile, calibration())
    assert all(row["dtype"] == "BF16" and row["dtype_basis"] == "kernel_symbol" for row in result["rows"])
    assert all(row["mfu"] == pytest.approx(0.2) for row in result["rows"])
