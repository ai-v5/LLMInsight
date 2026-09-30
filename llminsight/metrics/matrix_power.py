"""Dtype-aware matrix compute metrics for MatMul and matrix-heavy fused kernels."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import SETTINGS, dtype_bytes
from ..parser.shapes import numel, parse_dtypes, parse_shapes
from .efficiency import (
    ATTENTION_TYPES,
    ATTENTION_GRAD_TYPES,
    DSA_INDEXER_TYPES,
    LIGHTNING_INDEXER_FWD_TYPES,
    LIGHTNING_INDEXER_KL_GRAD_TYPES,
    MATMUL_TYPES,
    SPARSE_ATTENTION_TYPES,
    _estimate_attention_flops,
    _estimate_kda_flops,
    _estimate_lightning_indexer_flops,
    _estimate_matmul_flops,
    _estimate_sparse_flash_attention_flops,
    _estimate_sparse_lightning_indexer_grad_kl_loss_flops,
    _is_kda_model_compute_type,
)

_FUSED_MARKERS = (
    "flashattention", "flash_attn", "fusedattention", "fused_attention",
    "chunk_kda", "chunkkda", "kda_", "gated_delta", "chunk_gla",
    "causal_conv1d", "_attn_", "attention",
)


def _canonical_dtype(value: str) -> str:
    d = (value or "").strip().upper()
    if d.startswith("DT_"):
        d = d[3:]
    if "FLOAT8" in d or "FP8" in d or "HIF8" in d:
        return "FP8"
    if "FLOAT4" in d or "FP4" in d:
        return "FP4"
    if "BFLOAT16" in d or d == "BF16":
        return "BF16"
    if "FLOAT16" in d or d in {"FP16", "HALF"}:
        return "FP16"
    if "FLOAT32" in d or d in {"FP32", "FLOAT"}:
        return "FP32"
    return d or "UNKNOWN"


def _matrix_dtype(input_shapes: List[List[int]], input_dtypes: List[str], output_dtypes: List[str]) -> str:
    for index, shape in enumerate(input_shapes):
        if len(shape) >= 2 and index < len(input_dtypes):
            candidate = _canonical_dtype(input_dtypes[index])
            if candidate not in {"INT64", "INT32", "UINT64", "UINT32", "BOOL", "UNKNOWN"}:
                return candidate
    for value in input_dtypes + output_dtypes:
        candidate = _canonical_dtype(value)
        if candidate != "UNKNOWN":
            return candidate
    return "UNKNOWN"


def _is_matrix_candidate(op_type: str, name: str) -> bool:
    if op_type in MATMUL_TYPES or op_type in ATTENTION_TYPES:
        return True
    if _is_kda_model_compute_type(op_type):
        return True
    text = f"{op_type} {name}".lower()
    return "matmul" in text or "gemm" in text or any(marker in text for marker in _FUSED_MARKERS)


def _flops(op_type: str, shapes_in: List[List[int]], shapes_out: List[List[int]]) -> Optional[float]:
    if op_type in MATMUL_TYPES:
        return _estimate_matmul_flops(shapes_in, shapes_out)
    lowered = op_type.lower()
    if "matmul" in lowered or "gemm" in lowered:
        return _estimate_matmul_flops(shapes_in, shapes_out)
    if op_type in SPARSE_ATTENTION_TYPES:
        return _estimate_sparse_flash_attention_flops(
            shapes_in, shapes_out, op_type in ATTENTION_GRAD_TYPES,
            sparse_mode=SETTINGS.sparse_attention_mode,
            sparse_block_size=(1 if SETTINGS.chip.name == "Ascend 950DT" else None),
        )
    if op_type in LIGHTNING_INDEXER_FWD_TYPES:
        return _estimate_lightning_indexer_flops(
            shapes_in, shapes_out, sparse_mode=SETTINGS.sparse_attention_mode,
        )
    if op_type in LIGHTNING_INDEXER_KL_GRAD_TYPES:
        return _estimate_sparse_lightning_indexer_grad_kl_loss_flops(
            shapes_in, shapes_out, sparse_mode=SETTINGS.sparse_attention_mode,
        )
    if op_type in ATTENTION_TYPES:
        return _estimate_attention_flops(shapes_in, shapes_out, op_type in ATTENTION_GRAD_TYPES)
    if _is_kda_model_compute_type(op_type):
        return _estimate_kda_flops(op_type, shapes_in, shapes_out)
    return None


def _bytes(shapes_in: List[List[int]], input_dtypes: List[str], shapes_out: List[List[int]], output_dtypes: List[str], op_type: str) -> float:
    total = 0.0
    for index, shape in enumerate(shapes_in):
        dtype = input_dtypes[index] if index < len(input_dtypes) else (input_dtypes[-1] if input_dtypes else "BF16")
        total += numel(shape) * dtype_bytes(dtype)
    for index, shape in enumerate(shapes_out):
        dtype = output_dtypes[index] if index < len(output_dtypes) else (output_dtypes[-1] if output_dtypes else "BF16")
        total += numel(shape) * dtype_bytes(dtype)
    if op_type in SPARSE_ATTENTION_TYPES or op_type in DSA_INDEXER_TYPES:
        return 0.0
    return total


def _shape_text(shapes_in: List[List[int]], shapes_out: List[List[int]]) -> str:
    def one(shape: List[int]) -> str:
        return "x".join(str(x) for x in shape)
    left = " · ".join(one(s) for s in shapes_in[:8]) or "—"
    right = " · ".join(one(s) for s in shapes_out[:4])
    return f"{left} → {right}" if right else left


def compute_matrix_power(prof) -> Dict[str, Any]:
    kd = prof.kernel_details
    if kd.empty:
        return {"available": False, "reason": "kernel_details.csv missing"}
    required = {"Type", "Name", "Duration(us)", "Input Shapes", "Input Data Types"}
    if not required.issubset(set(kd.columns)):
        return {"available": False, "reason": "kernel_details.csv lacks matrix fields"}
    has_out_shapes = "Output Shapes" in kd.columns
    has_out_dtypes = "Output Data Types" in kd.columns
    rows: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, Any]] = {}
    # ``iterrows`` boxes every scalar into a Series and is prohibitively slow on
    # long captures.  Record dictionaries keep the same readable access pattern
    # while making this extra view a small fraction of load time.  Select only
    # columns used by this view so the 50-column counter payload is not duplicated.
    matrix_columns = ["Type", "Name", "Duration(us)", "Input Shapes", "Input Data Types"]
    if has_out_shapes:
        matrix_columns.append("Output Shapes")
    if has_out_dtypes:
        matrix_columns.append("Output Data Types")
    for record in kd[matrix_columns].to_dict(orient="records"):
        op_type = str(record.get("Type") or "").strip()
        name = str(record.get("Name") or "").strip()
        raw_duration = record.get("Duration(us)")
        try:
            duration = float(raw_duration) if raw_duration == raw_duration else 0.0
        except (TypeError, ValueError):
            duration = 0.0
        if duration <= 0 or not _is_matrix_candidate(op_type, name):
            continue
        shapes_in = parse_shapes(record.get("Input Shapes"))
        shapes_out = parse_shapes(record.get("Output Shapes")) if has_out_shapes else []
        input_dtypes = parse_dtypes(record.get("Input Data Types"))
        output_dtypes = parse_dtypes(record.get("Output Data Types")) if has_out_dtypes else []
        dtype = _matrix_dtype(shapes_in, input_dtypes, output_dtypes)
        flops = _flops(op_type, shapes_in, shapes_out)
        bytes_moved = _bytes(shapes_in, input_dtypes, shapes_out, output_dtypes, op_type)
        peak = SETTINGS.chip.peak_cube_flops(dtype)
        duration_s = duration * 1e-6
        mfu = flops / (peak * duration_s) if flops and peak and duration_s else None
        mbu_raw = bytes_moved / duration_s / SETTINGS.chip.hbm_bandwidth if bytes_moved and duration_s else None
        mbu = min(mbu_raw, 1.0) if mbu_raw is not None else None
        label = f"{op_type}@{dtype}"
        group = grouped.setdefault(label, {"label": label, "operator": op_type, "dtype": dtype,
                                           "count": 0, "dur_us": 0.0, "flops": 0.0,
                                           "bytes": 0.0, "peak_time": 0.0})
        group["count"] += 1
        group["dur_us"] += duration
        group["flops"] += flops or 0.0
        group["bytes"] += bytes_moved
        if flops:
            group["peak_time"] += peak * duration_s
        rows.append({"operator": op_type, "name": name, "dtype": dtype, "label": label,
                     "shape": _shape_text(shapes_in, shapes_out), "dur_us": round(duration, 3),
                     "flops": round(flops, 1) if flops is not None else None,
                     "mfu": round(mfu, 4) if mfu is not None else None,
                     "mbu": round(mbu, 4) if mbu is not None else None,
                     "mbu_raw": round(mbu_raw, 4) if mbu_raw is not None else None,
                     "formula_available": flops is not None})
    if not rows:
        return {"available": False, "reason": "no matrix or matrix-heavy fused kernels"}
    total_us = sum(item["dur_us"] for item in rows)
    groups: List[Dict[str, Any]] = []
    for item in grouped.values():
        duration_s = item["dur_us"] * 1e-6
        mfu = item["flops"] / item["peak_time"] if item["peak_time"] else None
        mbu_raw = item["bytes"] / duration_s / SETTINGS.chip.hbm_bandwidth if item["bytes"] and duration_s else None
        groups.append({"label": item["label"], "operator": item["operator"], "dtype": item["dtype"],
                       "count": item["count"], "dur_us": round(item["dur_us"], 1),
                       "dur_pct": round(item["dur_us"] / total_us * 100.0, 3) if total_us else 0.0,
                       "mfu": round(mfu, 4) if mfu is not None else None,
                       "mbu": round(min(mbu_raw, 1.0), 4) if mbu_raw is not None else None,
                       "mbu_raw": round(mbu_raw, 4) if mbu_raw is not None else None,
                       "flops": round(item["flops"], 1) if item["flops"] else None})
    groups.sort(key=lambda item: item["dur_us"], reverse=True)
    rows.sort(key=lambda item: item["dur_us"], reverse=True)
    formula_rows = sum(1 for item in rows if item["formula_available"])
    return {"available": True,
            "chip": {"name": SETTINGS.chip.name, "hbm_tbps": SETTINGS.chip.hbm_bandwidth / 1e12,
                     "cube_bf16_tflops": SETTINGS.chip.cube_fp16_flops / 1e12},
            "total_dur_us": round(total_us, 1), "groups": groups, "rows": rows,
            "kernel_count": len(rows),
            "formula_coverage_pct": round(formula_rows / len(rows) * 100.0, 1),
            "note": "MFU 分母按每个算子@dtype 对应的 CUBE 峰值；融合算子仅统计可由 shape 重建的矩阵工作，稀疏离散访存的 MBU 保持不可用。"}
