"""Targeted regression guards for fail-closed profiling interpretation.

The fixtures are synthetic on purpose: real profiles and launch logs stay under
``secret/`` and must never become test data.
"""
from __future__ import annotations

from datetime import datetime
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llminsight.insight.summarizer import build_summary
from llminsight.config import SETTINGS
from llminsight.metrics.core import _single_card, theoretical
from llminsight.metrics.efficiency import (
    ATTENTION_TYPES,
    MATMUL_TYPES,
    _estimate_matmul_flops,
    _estimate_sparse_flash_attention_flops,
    compute_efficiency,
)
from llminsight.parser.shapes import parse_shapes
from llminsight.parser.derive import derive_capture, derive_config, derive_guesses, derive_model
from llminsight.parser.profile import ProfileData
from llminsight.report import _sec_header
from llminsight.rules import run_rules


def _profile(kernel_details: pd.DataFrame, communication=None) -> ProfileData:
    return ProfileData(
        data_dir="synthetic",
        step_trace=pd.DataFrame(),
        op_statistic=pd.DataFrame(),
        api_statistic=pd.DataFrame(),
        kernel_details=kernel_details,
        operator_details=pd.DataFrame(),
        communication=communication or [],
        communication_raw={},
        communication_matrix={"step": {"p2p": {}, "collective": {}}},
        trace_path=None,
        meta={"profile_scope": "single_rank_or_matrix_missing", "multi_card": True},
    )


def _sparse_rows() -> pd.DataFrame:
    sparse_in = (
        "1,8,4,16;1,8,1,16;1,8,1,16;1,8,1,4;"
        ";1;1;1,8,4,4;1,8,1,4"
    )
    sparse_out = "1,8,4,16;1,1,8,4;1,1,8,4"
    grad_in = (
        "1,8,4,16;1,8,1,16;1,8,1,16;1,8,1,4;"
        "1,8,4,16;1,8,4,16;1,1,8,4;1,1,8,4;"
        "1;1;1,8,4,4;1,8,1,4"
    )
    grad_out = "1,8,4,16;1,8,1,16;1,8,1,16;1,8,4,4;1,8,1,4"
    rows = []
    for typ, count, shapes_in, shapes_out in (
        ("SparseFlashAttention", 2, sparse_in, sparse_out),
        ("SparseFlashAttentionGrad", 1, grad_in, grad_out),
    ):
        for _ in range(count):
            rows.append({
                "Type": typ,
                "Name": typ,
                "Duration(us)": 100.0,
                "Accelerator Core": "MIX_AIC",
                "Input Shapes": shapes_in,
                "Input Data Types": "DT_BF16",
                "Output Shapes": shapes_out,
                "Output Data Types": "DT_BF16",
            })
    rows.append({
        "Type": "GroupedMatmul",
        "Name": "GroupedMatmul",
        "Duration(us)": 20.0,
        "Accelerator Core": "AI_CORE",
        "Input Shapes": "32,64;8,64,128",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Output Shapes": "32,128",
        "Output Data Types": "DT_BF16",
    })
    return pd.DataFrame(rows)


def check_sparse_attention_and_capture() -> None:
    prof = _profile(
        _sparse_rows(),
        communication=[{
            "type": "alltoallv", "elapse_ms": 1.0, "wait_ms": 0.5,
            "transit_ms": 0.5, "links": {},
        }],
    )
    assert "SparseFlashAttention" in ATTENTION_TYPES
    assert "SparseFlashAttentionGrad" in ATTENTION_TYPES
    assert "BatchMatMulV3" in MATMUL_TYPES

    capture = derive_capture(prof)
    assert capture["recompute"]["value"] == "full", capture["recompute"]
    assert capture["single_card"]["value"] is False, capture["single_card"]
    assert _single_card(prof) is False

    model = derive_model(prof)
    assert model["architecture"]["value"] == "MoE + Sparse Attention", model
    guesses = derive_guesses(prof, model)
    assert guesses["ep_world_size"]["guess"] is None, guesses["ep_world_size"]
    assert guesses["num_layers"]["guess"] is None, guesses["num_layers"]

    old_sparse_mode = SETTINGS.sparse_attention_mode
    SETTINGS.sparse_attention_mode = 3
    try:
        eff = compute_efficiency(prof)
    finally:
        SETTINGS.sparse_attention_mode = old_sparse_mode
    # For S=8, topK=4, causal valid pairs are 1+2+3+4+4+4+4+4 = 26.
    # Hq=4, Dqk=16+4, Dv=16 -> fwd=7488, bwd=19136 useful cube FLOPs.
    expected_sparse = 2 * 7488.0 + 19136.0
    expected_grouped = 2.0 * 32 * 64 * 128
    assert eff["flop_model_complete"] is True, eff
    assert eff["model_mfu_compute"] is not None, eff
    assert math.isclose(
        eff["useful_flops_total"], expected_sparse + expected_grouped,
        rel_tol=0, abs_tol=0,
    ), eff
    assert "SparseFlashAttention" not in eff["unmodeled_flop_types"], eff
    assert any("有效稀疏 Cube FLOPs" in x for x in eff["flop_model_notes"]), eff
    sparse_types = {
        row["type"]: row for row in eff["by_type"]
        if row["type"] in {"SparseFlashAttention", "SparseFlashAttentionGrad"}
    }
    assert sparse_types and all(row["mbu"] is None for row in sparse_types.values()), eff
    assert not any(
        row["type"] in sparse_types for row in eff["top_optimization"]
    ), eff["top_optimization"]
    assert eff["op_ceiling_opt"]["by_class"]["attention"]["n"] == 0, eff
    assert eff["op_ceiling_opt"]["by_class"]["attention_grad"]["n"] == 0, eff

    ov = {
        "available": True,
        "us": {
            "stage": 1000.0, "computing": 700.0,
            "comm_not_overlapped": 200.0, "free": 100.0,
            "communication": 300.0, "overlapped": 100.0,
        },
        "ratios": {
            "comm_not_overlapped_pct": 20.0, "free_pct": 10.0,
            "overlap_rate_pct": 33.3,
        },
    }
    cfg = derive_config(prof)
    theo = theoretical(prof, ov, eff, cfg)
    assert theo["step_mfu"] is not None and theo["step_hfu"] is not None, theo
    # Sparse attention contributes useful FLOPs/MFU, but not a roofline What-if:
    # its repeated Gather/Scatter traffic is not reconstructible from tensor shapes.
    assert "op_ceiling" not in {x["id"] for x in theo["whatif"]}, theo["whatif"]
    assert "recompute_off" in {x["id"] for x in theo["whatif"]}, theo["whatif"]

    metrics = {
        "meta": {**prof.meta, "config": cfg},
        "overview": ov,
        "hotspots": {"available": True, "ops": []},
        "efficiency": eff,
        "communication": {
            "available": True,
            "by_type": [{
                "type": "alltoallv", "count": 1, "elapse_ms": 1.0,
                "wait_pct": 50.0,
            }],
        },
        "hidden_overhead": {"available": False, "buckets": []},
        "attribution": {"available": False},
        "memory": {"available": False},
        "theoretical": theo,
    }
    cards = run_rules(metrics, cfg)
    card_text = str(cards)
    assert "flop_model_incomplete" not in {x["id"] for x in cards}, cards
    assert "EP64" not in card_text, card_text
    summary = build_summary(metrics, cards, cfg)
    assert summary["model"]["arch"] == "MoE + Sparse Attention", summary["model"]
    assert summary["model"]["unknown_fields"]["ep_world_size"] == "未知"


def check_peak_inconsistency_fails_closed() -> None:
    kd = pd.DataFrame([{
        "Type": "BatchMatMulV3",
        "Name": "synthetic_bmm",
        "Duration(us)": 1.0,
        "Accelerator Core": "AI_CORE",
        "Input Shapes": "32,4096,192;32,192,512",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Output Shapes": "32,4096,512",
        "Output Data Types": "DT_BF16",
    }])
    eff = compute_efficiency(_profile(kd))
    assert eff["peak_inconsistent"] is True, eff["chip"]
    assert eff["matmul_mfu"] is None, eff
    assert eff["matmul_mfu_raw"] > 1.0, eff
    assert eff["useful_flops_total"] is None, eff
    assert eff["top_optimization"] == [] and eff["scatter"] == [], eff
    assert all(x["mfu"] is None for x in eff["by_type"]), eff


def check_small_peak_noise_is_accepted() -> None:
    flops = 2.0 * 32 * 4096 * 512 * 192
    duration_us = flops / (432e12 * 1.01) * 1e6
    kd = pd.DataFrame([{
        "Type": "BatchMatMulV3",
        "Name": "synthetic_bmm_noise",
        "Duration(us)": duration_us,
        "Accelerator Core": "AI_CORE",
        "Input Shapes": "32,4096,192;32,192,512",
        "Input Data Types": "DT_BF16;DT_BF16",
        "Output Shapes": "32,4096,512",
        "Output Data Types": "DT_BF16",
    }])
    eff = compute_efficiency(_profile(kd))
    assert eff["peak_inconsistent"] is False, eff["chip"]
    assert math.isclose(eff["matmul_mfu"], 1.01, rel_tol=1e-4), eff


def check_output_aware_gemm_and_sparse_formulas() -> None:
    # addmm weight-gradient form: A^T[K,M] @ B[K,N] -> [M,N].  Ignoring the
    # output would infer M=N=4096 and over-count this synthetic shape by 2x.
    gemm_in = parse_shapes("4096,2048;4096,2048;2048,2048")
    gemm_out = parse_shapes("2048,2048")
    expected_gemm = 2.0 * 2048 * 2048 * 4096
    assert _estimate_matmul_flops(gemm_in, gemm_out) == expected_gemm
    assert _estimate_matmul_flops(gemm_in) is None

    rows = _sparse_rows()
    fwd = rows[rows["Type"] == "SparseFlashAttention"].iloc[0]
    grad = rows[rows["Type"] == "SparseFlashAttentionGrad"].iloc[0]
    assert _estimate_sparse_flash_attention_flops(
        parse_shapes(fwd["Input Shapes"]), parse_shapes(fwd["Output Shapes"]), False,
        sparse_mode=3, sparse_block_size=1,
    ) == 7488.0
    assert _estimate_sparse_flash_attention_flops(
        parse_shapes(grad["Input Shapes"]), parse_shapes(grad["Output Shapes"]), True,
        sparse_mode=3, sparse_block_size=1,
    ) == 19136.0
    assert _estimate_sparse_flash_attention_flops(
        parse_shapes(fwd["Input Shapes"]), parse_shapes(fwd["Output Shapes"]), False,
    ) is None


def check_report_uses_generic_semantics() -> None:
    meta = {
        "profile_scope": "single_rank_or_matrix_missing",
        "config": {
            "model": {"architecture": "MoE + Sparse Attention"},
            "guesses": {
                "ep_world_size": {"label": "未知"},
                "num_experts": {"label": "未知"},
                "global_batch_size": {"label": "未知"},
            },
            "capture": {
                "single_card": {"value": False},
                "recompute": {"value": "full"},
                "blocking": {"value": False},
            },
        },
        "settings": {"chip": {"name": "synthetic"}, "data_dir": "synthetic"},
    }
    html = _sec_header(
        meta,
        {"step": 1, "device_id": 0},
        {},
        {"chip": {"name": "synthetic"}},
        datetime(2026, 1, 1),
    )
    assert "MoE + Sparse Attention" in html
    assert "MLA + MoE" not in html
    assert "EP64" not in html
    assert "多卡训练（单 rank 采集）" in html


if __name__ == "__main__":
    check_sparse_attention_and_capture()
    check_peak_inconsistency_fails_closed()
    check_small_peak_noise_is_accepted()
    check_output_aware_gemm_and_sparse_formulas()
    check_report_uses_generic_semantics()
    print("accuracy guards: OK")
