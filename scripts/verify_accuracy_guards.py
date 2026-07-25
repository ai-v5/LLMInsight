"""Targeted regression guards for fail-closed profiling interpretation.

The fixtures are synthetic on purpose: real profiles and launch logs stay under
``secret/`` and must never become test data.
"""
from __future__ import annotations

from datetime import datetime
import math
import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llminsight.insight.summarizer import build_summary
from llminsight.config import SETTINGS
from llminsight.metrics.core import _recompute_overhead, _single_card, theoretical
from llminsight.metrics.efficiency import (
    ATTENTION_TYPES,
    MATMUL_TYPES,
    _estimate_matmul_flops,
    _estimate_sparse_flash_attention_flops,
    compute_efficiency,
)
from llminsight.metrics.smart_timeline import _apply_chip, compute_msprof_smart_timeline
from llminsight.metrics import smart_timeline as smart_timeline_module
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


def _dsa_indexer_rows() -> pd.DataFrame:
    """Small, public analogue of the LI + fused KL-loss training schema."""
    li_in = "1,8,4,8;1,8,1,8;1,8,4;;;"
    li_out = "1,8,1,4;1,8,1,4"
    slig_in = (
        "1,8,8,16;1,8,1,16;1,8,4,8;1,8,1,8;1,8,4;"
        "1,8,1,4;1,1,8,8;1,1,8,8;1,8,8,4;1,8,1,4;;"
    )
    slig_out = "1,8,4,8;1,8,1,8;1,8,4;1"
    rows = []
    for typ, count, shapes_in, shapes_out in (
        ("LightningIndexer", 2, li_in, li_out),
        ("SparseLightningIndexerGradKLLoss", 1, slig_in, slig_out),
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
    return pd.DataFrame(rows)


def _matmul_phase_rows() -> pd.DataFrame:
    """A terminal projection plus a checkpointed weight-only projection."""
    rows = []
    for shapes_in, shapes_out in (
        ("4,8;16,8", "4,16"),
        ("4,16;16,8", "4,8"),
    ):
        rows.append({
            "Type": "MatMulV3", "Name": "terminal_projection",
            "Duration(us)": 100.0, "Accelerator Core": "AI_CORE",
            "Input Shapes": shapes_in, "Input Data Types": "DT_BF16;DT_BF16",
            "Output Shapes": shapes_out, "Output Data Types": "DT_BF16",
        })
    for _ in range(2):
        rows.append({
            "Type": "MatMulV3", "Name": "checkpointed_projection_fwd",
            "Duration(us)": 100.0, "Accelerator Core": "AI_CORE",
            "Input Shapes": "4,8;12,8", "Input Data Types": "DT_BF16;DT_BF16",
            "Output Shapes": "4,12", "Output Data Types": "DT_BF16",
        })
    rows.append({
        "Type": "MatMulV3", "Name": "checkpointed_projection_dw",
        "Duration(us)": 100.0, "Accelerator Core": "AI_CORE",
        "Input Shapes": "4,12;12,8", "Input Data Types": "DT_BF16;DT_BF16",
        "Output Shapes": "4,8", "Output Data Types": "DT_BF16",
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
    training_flops = eff["attention_training_flops"]
    assert training_flops["available"] is True, training_flops
    assert training_flops["forward_flops_per_call"] == 7488.0, training_flops
    assert training_flops["backward_flops_per_call"] == 19136.0, training_flops
    expected_bwd_fwd = 19136.0 / 7488.0
    assert math.isclose(
        training_flops["backward_forward_flop_ratio"], expected_bwd_fwd,
        rel_tol=1e-12,
    ), training_flops
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
            "stage": 1.0, "computing": 0.7,
            "comm_not_overlapped": 0.2, "free": 0.1,
            "communication": 0.3, "overlapped": 0.1,
        },
        "ratios": {
            "comm_not_overlapped_pct": 20.0, "free_pct": 10.0,
            "overlap_rate_pct": 33.3,
        },
    }
    cfg = derive_config(prof)
    theo = theoretical(prof, ov, eff, cfg)
    assert theo["step_mfu"] is not None and theo["step_hfu"] is not None, theo
    expected_attention_recompute = 7488.0
    expected_matmul_recompute = expected_grouped / 4.0
    expected_recompute_share = (
        expected_attention_recompute + expected_matmul_recompute
    ) / (expected_sparse + expected_grouped)
    expected_hfu = (expected_sparse + expected_grouped) / (432e12 * 1e-6)
    assert theo["step_hfu"] == round(expected_hfu, 4), theo
    assert theo["step_mfu"] == round(expected_hfu * (1 - expected_recompute_share), 4), theo
    assert theo["step_mfu_lo"] <= theo["step_mfu"] <= theo["step_mfu_hi"], theo
    assert theo["recompute"]["flops_share"] == round(expected_recompute_share, 4), theo
    assert theo["recompute"]["flop_ratio_source"] == \
        "component_weighted_matmul_phase_split_sparse_attention_exact", theo
    unclosed = _recompute_overhead(
        0.7,
        capture["recompute"],
        bwd_fwd_flop_ratio=expected_bwd_fwd,
        recompute_forward_multiplier=1.0,
        component_flops={
            "matmul_flops": expected_grouped,
            "attention_flops": expected_sparse,
            "model_flops": expected_grouped + expected_sparse + 1.0,
            "attention_recompute_flops": expected_attention_recompute,
        },
    )
    assert unclosed["flop_ratio_source"] == \
        "fallback_training_band_component_mismatch", unclosed
    assert unclosed["flops_share"] == 0.25, unclosed
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


def check_dsa_indexer_flops_and_phase_split() -> None:
    rows = pd.concat([_sparse_rows(), _dsa_indexer_rows()], ignore_index=True)
    prof = _profile(rows)
    old_sparse_mode = SETTINGS.sparse_attention_mode
    SETTINGS.sparse_attention_mode = 3
    try:
        eff = compute_efficiency(prof)
    finally:
        SETTINGS.sparse_attention_mode = old_sparse_mode

    li_per_call = 2.0 * 36 * 4 * 8
    slig_per_call = 2.0 * 26 * (8 * (16 + 4) + 3 * 4 * 8)
    sparse_total = 2 * 7488.0 + 19136.0
    grouped_total = 2.0 * 32 * 64 * 128
    expected_total = sparse_total + grouped_total + 2 * li_per_call + slig_per_call
    assert eff["flop_model_complete"] is True, eff
    assert eff["useful_flops_total"] == expected_total, eff
    by_type = {row["type"]: row for row in eff["by_type"]}
    for typ in ("LightningIndexer", "SparseLightningIndexerGradKLLoss"):
        assert by_type[typ]["mfu"] is not None, by_type[typ]
        assert by_type[typ]["mbu"] is None, by_type[typ]
    assert eff["unmodeled_flop_types"] == {}, eff["unmodeled_flop_types"]

    phase = eff["attention_training_flops"]
    assert phase["available"] is True, phase
    assert phase["indexer_forward_count"] == 2, phase
    assert phase["indexer_grad_loss_count"] == 1, phase
    assert phase["indexer_recompute_forward_flops"] == li_per_call, phase
    assert phase["recompute_forward_flops"] == 7488.0 + li_per_call, phase
    # SLIG is real auxiliary-loss/backward work even though checkpoint execution
    # places it in the grad-enabled recompute forward.
    assert phase["indexer_model_flops"] == li_per_call + slig_per_call, phase

    ov = {
        "available": True,
        "us": {
            "stage": 1.0, "computing": 0.7,
            "comm_not_overlapped": 0.2, "free": 0.1,
            "communication": 0.3, "overlapped": 0.1,
        },
        "ratios": {
            "comm_not_overlapped_pct": 20.0, "free_pct": 10.0,
            "overlap_rate_pct": 33.3,
        },
    }
    theo = theoretical(prof, ov, eff, derive_config(prof))
    expected_recompute = 7488.0 + li_per_call + grouped_total / 4.0
    expected_hfu = expected_total / (432e12 * 1e-6)
    expected_model_mfu = expected_hfu * (1.0 - expected_recompute / expected_total)
    assert theo["step_hfu"] == round(expected_hfu, 4), theo
    assert theo["step_mfu"] == round(expected_model_mfu, 4), theo
    assert theo["step_mfu_lo"] <= theo["step_mfu"] <= theo["step_mfu_hi"], theo
    assert theo["recompute"]["flops_share"] == round(
        expected_recompute / expected_total, 4
    ), theo

    # A known DSA type with unsupported shapes must enter the coverage denominator
    # and fail closed instead of leaving a misleading 100% model-FLOP coverage.
    broken_rows = rows.copy()
    broken_rows.loc[
        broken_rows["Type"] == "SparseLightningIndexerGradKLLoss", "Output Shapes"
    ] = "1,8,4,7;1,8,1,8;1,8,4;1"
    SETTINGS.sparse_attention_mode = 3
    try:
        broken = compute_efficiency(_profile(broken_rows))
    finally:
        SETTINGS.sparse_attention_mode = old_sparse_mode
    assert broken["flop_model_complete"] is False, broken
    assert broken["useful_flops_total"] is None, broken
    assert "SparseLightningIndexerGradKLLoss" in broken["unmodeled_flop_types"], broken


def check_matmul_phase_split() -> None:
    eff = compute_efficiency(_profile(_matmul_phase_rows()))
    phase = eff["matmul_training_flops"]
    terminal_f = 2.0 * 4 * 16 * 8
    checkpoint_f = 2.0 * 4 * 12 * 8
    assert phase["available"] is True, phase
    assert phase["executed_flops"] == 2 * terminal_f + 3 * checkpoint_f, phase
    assert phase["recompute_forward_flops"] == checkpoint_f, phase
    assert phase["exact_group_count"] == 2, phase


def check_smart_timeline_cube_breakdown() -> None:
    geom = {
        "available": True,
        "t0_us": 0.0,
        "span_us": 2_000_000.0,
        "span_s": 2.0,
        "bins": 2,
        "bin_us": 1_000_000.0,
        "flops_sum": [0.4e12, 0.9e12],
        "fa_flops_sum": [0.2e12, 0.3e12],
        "bytes_sum": [0.0, 0.0],
        "comm_occ": [0.0, 0.0],
        "vec_occ": [0.0, 0.0],
        "slices": [],
        "total_slices": 0,
        "shown_slices": 0,
        "modeled_dev_us": 0.0,
    }
    eff = {
        "chip": {"effective_peak_tflops": 1.0, "hbm_tbps": 1.0},
        "kernel_index": {},
        "attention_total_us": 1.0,
        "attention_flop_coverage_pct": 100.0,
        "unmodeled_flop_types": {},
        "peak_inconsistent": False,
    }
    tl = _apply_chip(geom, eff)
    util = {row["key"]: row for row in tl["utilization"]}
    assert util["cube_total"]["series"] == [0.6, 1.0], util
    assert util["cube_total"]["abs"] == [0.6, 1.2], util
    assert util["cube_gemm"]["series"] == [0.4, 0.9], util
    assert util["cube_attention"]["series"] == [0.2, 0.3], util

    incomplete = _apply_chip(
        geom, {**eff, "attention_flop_coverage_pct": 50.0}
    )
    incomplete_util = {row["key"]: row for row in incomplete["utilization"]}
    assert incomplete_util["cube_total"]["available"] is False, incomplete_util
    assert incomplete_util["cube_gemm"]["available"] is True, incomplete_util
    assert incomplete_util["cube_attention"]["available"] is False, incomplete_util

    kd = pd.DataFrame([
        {"Name": "gemm", "Type": "MatMulV3", "Accelerator Core": "AI_CORE",
         "Start Time(us)": 0.0, "Duration(us)": 100.0},
        {"Name": "fag", "Type": "SparseFlashAttentionGrad",
         "Accelerator Core": "MIX_AIC", "Start Time(us)": 100.0,
         "Duration(us)": 100.0},
        {"Name": "other", "Type": "OtherCube", "Accelerator Core": "AI_CORE",
         "Start Time(us)": 200.0, "Duration(us)": 100.0},
    ])
    old_bins = SETTINGS.smart_timeline_bins
    SETTINGS.smart_timeline_bins = 3
    try:
        msprof = compute_msprof_smart_timeline(
            SimpleNamespace(kernel_details=kd, meta={}), {"kernel_index": {}}
        )
    finally:
        SETTINGS.smart_timeline_bins = old_bins
    ms_util = {row["key"]: row for row in msprof["utilization"]}
    assert ms_util["cube_total"]["series"] == [1.0, 1.0, 1.0], ms_util
    assert ms_util["cube_gemm"]["series"] == [1.0, 0.0, 0.0], ms_util
    assert ms_util["cube_attention"]["series"] == [0.0, 1.0, 0.0], ms_util


def check_smart_timeline_cache_tracks_flop_model() -> None:
    prof = SimpleNamespace(
        kernel_details=pd.DataFrame([{"Input Shapes": "1"}]),
        trace_path=__file__,
    )
    unmodeled = {
        "available": True,
        "kernel_index": {
            "fag": {
                "type": "SparseFlashAttentionGrad",
                "core": "MIX_AIC",
                "dtype": "DT_BF16",
                "flops_per_us": None,
                "bytes_per_us": None,
            },
        },
    }
    modeled = {
        **unmodeled,
        "kernel_index": {
            "fag": {
                **unmodeled["kernel_index"]["fag"],
                "flops_per_us": 31_040_109.949,
            },
        },
    }

    cache_keys = []
    old_cached_json = smart_timeline_module.cached_json

    def capture_key(key, builder):
        cache_keys.append(key)
        return {"available": False, "reason": "synthetic cache-key guard"}

    smart_timeline_module.cached_json = capture_key
    try:
        smart_timeline_module.compute_smart_timeline(prof, unmodeled)
        smart_timeline_module.compute_smart_timeline(prof, modeled)
    finally:
        smart_timeline_module.cached_json = old_cached_json

    assert cache_keys[0] != cache_keys[1], cache_keys


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
    check_dsa_indexer_flops_and_phase_split()
    check_matmul_phase_split()
    check_smart_timeline_cube_breakdown()
    check_smart_timeline_cache_tracks_flop_model()
    check_peak_inconsistency_fails_closed()
    check_small_peak_noise_is_accepted()
    check_output_aware_gemm_and_sparse_formulas()
    check_report_uses_generic_semantics()
    print("accuracy guards: OK")
