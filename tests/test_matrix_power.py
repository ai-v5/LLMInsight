from types import SimpleNamespace

import pandas as pd

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
