"""Dtype-aware matrix compute metrics for MatMul and matrix-heavy fused kernels."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import re

from ..config import SETTINGS, dtype_bytes
from ..parser.names import MC2_FUSED_MARKERS, is_mc2_fused_type, op_display_name
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
    "causal_conv1d", "_attn_", "attention", "mega_moe",
)

# MC2 fused MoE kernels run a grouped GEMM and an all-to-all inside ONE recorded
# duration, so the window is compute+communication, not pure compute.  Their
# per-operator MFU is therefore not defined from an in-window phase split, and
# their "MBU" would divide shape bytes by a duration that also contains the a2a.
# Level-1 captures additionally record NO shapes for these kernels at all, so even
# the GEMM operands are missing.  What the capture DOES contain is the same layer's
# plain expert grouped GEMM, which exposes
#   * the per-expert weight matrices [K,N] (3-D [G,K,N]), and
#   * the routed-row count M (first dim of its activation operand),
# so the fused window's GEMM work is reconstructible from function + those anchors.
# The result is published as an ESTIMATED equivalent MFU of the fused window (the
# denominator still contains the internal a2a), never as FORMULA.  The marker list
# itself lives in parser.names (single source).
_FUSED_MC2_MARKERS = MC2_FUSED_MARKERS


def _has_fused_mc2(kd) -> bool:
    if kd is None or kd.empty or "Name" not in kd.columns:
        return False
    names = kd["Name"].astype(str).str.lower()
    return any(names.str.contains(marker, regex=False).any() for marker in _FUSED_MC2_MARKERS)


def _expert_gemm_anchors(kd) -> Dict[str, Any]:
    """Anchor the fused MC2 GEMM contract from sibling expert grouped GEMMs.

    Returns ``{"rows", "contracts", "dtype", "source_ops"}`` or ``{}`` when the
    capture has no usable anchor.  ``contracts`` are per-expert weight matrices
    ``(K, N)``; a fused MC2 window does one of them per call.
    """
    if kd is None or kd.empty or "Type" not in kd.columns:
        return {}
    mask = kd["Type"].astype(str).str.lower().str.contains("groupedmatmul")
    if not bool(mask.any()):
        return {}
    # The fused MC2 kernels also carry "groupedmatmul" in their symbol/type but are
    # NOT an anchor: they are the thing being estimated (no shapes, window unmixed).
    mask &= ~kd["Type"].astype(str).map(is_mc2_fused_type)
    if not bool(mask.any()):
        return {}
    columns = [c for c in ("Type", "Input Shapes", "Output Shapes", "Input Data Types") if c in kd.columns]
    contracts = set()
    activation_rows: List[int] = []
    dtype: Optional[str] = None
    source_ops = set()
    for record in kd.loc[mask, columns].to_dict(orient="records"):
        shapes_in = parse_shapes(record.get("Input Shapes"))
        shapes_out = parse_shapes(record.get("Output Shapes"))
        source_ops.add(str(record.get("Type")))
        # Expert weights appear as a 3-D [G, K, N] operand: either the input of a
        # forward grouped GEMM or the output of its weight-gradient counterpart.
        weight = next((s for s in shapes_out if len(s) == 3), None) \
            or next((s for s in shapes_in if len(s) == 3), None)
        if weight:
            _, k, n = weight
            if k and n:
                contracts.add((int(k), int(n)))
        if shapes_in and shapes_in[0]:
            activation_rows.append(int(shapes_in[0][0]))
        if dtype is None:
            for value in parse_dtypes(record.get("Input Data Types")):
                candidate = _canonical_dtype(value)
                if candidate in {"BF16", "FP16", "FP32", "FP8", "FP4"}:
                    dtype = candidate
                    break
    if not contracts or not activation_rows:
        return {}
    return {"rows": max(activation_rows), "contracts": sorted(contracts),
            "dtype": dtype or "UNKNOWN", "source_ops": sorted(source_ops)}


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


def _mega_matrix_proxy(record, dtype, duration, efficiency):
    """Only MAC work belongs to this matrix view; the denominator stays fused."""
    def number(value):
        try:
            value = float(value)
            return value if math.isfinite(value) and value >= 0 else None
        except (TypeError, ValueError):
            return None

    active = number(record.get("aic_mac_time(us)"))
    if active is None:
        ratio = number(record.get("aic_mac_ratio"))
        active = ratio * duration if ratio is not None else None
    calibration = efficiency.get("mfu_estimate") or {}
    chip = efficiency.get("chip") or {}
    scale = number(calibration.get("counter_scale"))
    anchor = number(calibration.get("counter_reference_proxy_flops"))
    peak_scale = ((chip.get("effective_peak_tflops") or 0) / chip["cube_bf16_tflops"]
                  if chip.get("cube_bf16_tflops") else 1.0)
    peak = SETTINGS.chip.peak_cube_flops(dtype) * peak_scale
    valid = (active is not None and scale and anchor and peak > 0
             and dtype in {"BF16", "FP16", "FP32", "FP8", "FP4"}
             and not efficiency.get("peak_inconsistent", True))
    return active, active * peak * 1e-6 * scale if valid else None, peak


def _estimate_fused_mc2(mc2_groups: List[Dict[str, Any]], anchor: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct the MC2 fused window's GEMM work from function + anchors.

    Every MC2 layer call does two expert grouped GEMMs (up: [M,H]x[H,2F], down:
    [M,F]x[F,H]) around an all-to-all.  The capture records neither shape for the
    fused kernel, but the sibling expert GEMM gives the per-expert weights
    ``contracts`` and the routed-row count ``M``.  Work per call is therefore
    ``2·M·Σ(K·N)`` and the whole fused stage is ``pairs`` calls.
    """
    if not mc2_groups or not anchor.get("contracts") or not anchor.get("rows"):
        return {}
    contracts = [(int(k), int(n)) for k, n in anchor["contracts"] if k > 0 and n > 0]
    if not contracts:
        return {}
    rows_per_call = int(anchor["rows"])
    pairs = max(int(item["count"]) for item in mc2_groups)
    total_dur_us = sum(float(item["dur_us"]) for item in mc2_groups)
    if pairs <= 0 or total_dur_us <= 0:
        return {}
    per_pair_flops = 2.0 * rows_per_call * sum(k * n for k, n in contracts)
    total_flops = per_pair_flops * pairs
    dtype = anchor.get("dtype") or "BF16"
    peak = SETTINGS.chip.peak_cube_flops(dtype)
    equivalent_mfu = (total_flops / (peak * total_dur_us * 1e-6)) if peak else None
    if equivalent_mfu is None:
        return {}
    return {
        "rows": rows_per_call,
        "contracts": contracts,
        "pairs": pairs,
        "total_flops": total_flops,
        "total_dur_us": total_dur_us,
        "dtype": dtype,
        "equivalent_mfu": equivalent_mfu,
        "above_peak": equivalent_mfu > 1.02,
        "source_ops": anchor.get("source_ops") or [],
        "basis": (
            "功能约定 + 同采集锚点：MC2 每次调用 = 1 个两投影专家 GEMM 组"
            f"（权重 {', '.join(f'[{k}x{n}]' for k, n in contracts)}）× 激活行数 "
            f"{rows_per_call:,}；共 {pairs} 次调用。GEMM 工作量按各算子占融合窗口的时长份额"
            "分摊（窗口内计算/通信不可拆），dtype 取自同层专家 GEMM。分母含内部 a2a，"
            "故为「融合窗口等效 MFU」= 纯计算 MFU 的下界。"
            + ("（该锚点推断出的吞吐超过物理峰值 → 锚点不可信，MFU 保持不可用。）"
               if equivalent_mfu > 1.02 else "")
        ),
    }


def _apply_fused_mc2(mc2_groups: List[Dict[str, Any]], rows: List[Dict[str, Any]],
                     estimate: Dict[str, Any]) -> None:
    """Attach the fused-window estimate to the MC2 groups and their kernels.

    Fail closed like the rest of the MFU layer: an anchor that implies more work
    than the device can physically execute means the anchor is wrong (wrong dtype,
    wrong routed-row count, a 3-D operand mistaken for the expert weight), so the
    MFU stays unavailable and only the reconstruction note is published.
    """
    if estimate.get("above_peak"):
        for item in mc2_groups:
            if item["dtype"] == "UNKNOWN":
                item["dtype"] = estimate["dtype"]
            item["fused_estimate"] = {"flops": None, "mfu": None, "basis": estimate["basis"]}
        return
    total_flops = estimate["total_flops"]
    total_dur_us = estimate["total_dur_us"]
    peak = SETTINGS.chip.peak_cube_flops(estimate["dtype"])
    for item in mc2_groups:
        if item["dtype"] == "UNKNOWN":
            item["dtype"] = estimate["dtype"]
        share = float(item["dur_us"]) / total_dur_us
        flops = total_flops * share
        item["fused_estimate"] = {
            "flops": flops,
            "mfu": (flops / (peak * item["dur_us"] * 1e-6)) if peak and item["dur_us"] else None,
            "basis": estimate["basis"],
        }
    fused_rows = [row for row in rows if row.get("fused_window")]
    fused_dur = sum(float(row["dur_us"]) for row in fused_rows) or total_dur_us
    for row in fused_rows:
        if row.get("flops") is not None:
            continue
        share = float(row["dur_us"]) / fused_dur
        flops = total_flops * share
        row["flops"] = round(flops, 1)
        row["mfu"] = (round(flops / (peak * row["dur_us"] * 1e-6), 4)
                      if peak and row["dur_us"] else None)
        row["evidence_level"] = "ESTIMATED"


def compute_matrix_power(prof, efficiency=None) -> Dict[str, Any]:
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
    matrix_columns.extend(c for c in ("Accelerator Core", "aic_mac_time(us)", "aic_mac_ratio") if c in kd.columns)
    mc2_anchor = _expert_gemm_anchors(kd) if _has_fused_mc2(kd) else {}
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
        if str(record.get("Accelerator Core", "")).upper() == "AI_CPU":
            continue
        fused_moe = "mega_moe" in f"{op_type} {name}".lower()
        fused_window = fused_moe or any(m in f"{op_type} {name}".lower() for m in _FUSED_MC2_MARKERS)
        shapes_in = parse_shapes(record.get("Input Shapes"))
        shapes_out = parse_shapes(record.get("Output Shapes")) if has_out_shapes else []
        input_dtypes = parse_dtypes(record.get("Input Data Types"))
        output_dtypes = parse_dtypes(record.get("Output Data Types")) if has_out_dtypes else []
        dtype = _matrix_dtype(shapes_in, input_dtypes, output_dtypes)
        dtype_basis = "profiler_dtype"
        if fused_moe and dtype == "UNKNOWN":
            for symbol in (op_type, name):
                # A simple Itanium symbol stores the function-name length before
                # the identifier; following ABI argument codes are not its name.
                mangled = re.match(r"_Z(\d+)(.*)", symbol)
                identifier = mangled[2][:int(mangled[1])] if mangled else symbol
                match = re.search(r"(?<![a-z0-9])(bf16|fp16|fp32|fp8|fp4)(?![a-z0-9])", identifier.lower())
                if match:
                    dtype, dtype_basis = match[1].upper(), "kernel_symbol"
                    break
        if (fused_window and not fused_moe and dtype == "UNKNOWN"
                and mc2_anchor.get("dtype") not in (None, "UNKNOWN")):
            # Level-1 MC2 rows carry no dtype either; the same layer's expert GEMM
            # records it, and the fused GEMM runs in that dtype.
            dtype, dtype_basis = mc2_anchor["dtype"], "sibling_expert_gemm_dtype"
        flops = _flops(op_type, shapes_in, shapes_out) if not fused_window else None
        formula_available = flops is not None
        bytes_moved = _bytes(shapes_in, input_dtypes, shapes_out, output_dtypes, op_type) if not fused_window else 0.0
        peak = SETTINGS.chip.peak_cube_flops(dtype)
        mac_active = None
        if fused_moe:
            mac_active, flops, peak = _mega_matrix_proxy(record, dtype, duration, efficiency or {})
        evidence = "FORMULA" if formula_available else "PROXY" if flops is not None else "UNAVAILABLE"
        duration_s = duration * 1e-6
        mfu = flops / (peak * duration_s) if flops is not None and peak and duration_s else None
        mbu_raw = bytes_moved / duration_s / SETTINGS.chip.hbm_bandwidth if bytes_moved and duration_s else None
        mbu = min(mbu_raw, 1.0) if mbu_raw is not None else None
        label = f"{op_display_name(op_type)}@{dtype}"
        group = grouped.setdefault(label, {"label": label, "operator": op_type, "dtype": dtype,
                                           "count": 0, "dur_us": 0.0, "flops": 0.0,
                                           "bytes": 0.0, "peak_time": 0.0, "modeled_count": 0,
                                           "fused_window": fused_window, "proxy_count": 0})
        group["count"] += 1
        group["dur_us"] += duration
        group["flops"] += flops or 0.0
        group["bytes"] += bytes_moved
        if flops is not None:
            group["peak_time"] += peak * duration_s
            group["modeled_count"] += 1
        group["proxy_count"] += evidence == "PROXY"
        rows.append({"operator": op_type, "name": name, "dtype": dtype, "label": label,
                     "shape": _shape_text(shapes_in, shapes_out), "dur_us": round(duration, 3),
                     "flops": round(flops, 1) if flops is not None else None,
                     "mfu": round(mfu, 4) if mfu is not None else None,
                     "mbu": round(mbu, 4) if mbu is not None else None,
                     "mbu_raw": round(mbu_raw, 4) if mbu_raw is not None else None,
                     "formula_available": formula_available, "evidence_level": evidence,
                     "fused_window": fused_window, "mac_active_us": mac_active,
                     "dtype_basis": dtype_basis})
    if not rows:
        return {"available": False, "reason": "no matrix or matrix-heavy fused kernels"}
    total_us = sum(item["dur_us"] for item in rows)
    mc2_groups = [item for item in grouped.values()
                  if item["fused_window"] and not item["modeled_count"]]
    mc2_estimate = _estimate_fused_mc2(mc2_groups, mc2_anchor)
    if mc2_estimate:
        # Re-label the fused MC2 groups with their (estimated) dtype, then publish
        # the reconstructed GEMM work split by each group's share of the fused
        # window.  The denominator is still the fused window (a2a included), so the
        # value is an equivalent MFU, explicitly marked ESTIMATED.
        _apply_fused_mc2(mc2_groups, rows, mc2_estimate)
        for item in list(grouped.values()):
            if item["fused_window"] and not item["modeled_count"]:
                item["label"] = f"{op_display_name(item['operator'])}@{item['dtype']}"
    groups: List[Dict[str, Any]] = []
    for item in grouped.values():
        duration_s = item["dur_us"] * 1e-6
        mfu = item["flops"] / item["peak_time"] if item["peak_time"] else None
        complete = item["modeled_count"] == item["count"]
        if item["fused_window"] and not complete:
            mfu = None
        mbu_raw = item["bytes"] / duration_s / SETTINGS.chip.hbm_bandwidth if item["bytes"] and duration_s else None
        estimate = item.get("fused_estimate") or {}
        groups.append({"label": item["label"], "operator": item["operator"], "dtype": item["dtype"],
                       "count": item["count"], "dur_us": round(item["dur_us"], 1),
                       "dur_pct": round(item["dur_us"] / total_us * 100.0, 3) if total_us else 0.0,
                       "mfu": (round(estimate["mfu"], 6) if estimate.get("mfu") is not None
                               else (round(mfu, 4) if mfu is not None else None)),
                       "mbu": round(min(mbu_raw, 1.0), 4) if mbu_raw is not None else None,
                       "mbu_raw": round(mbu_raw, 4) if mbu_raw is not None else None,
                       "flops": (round(estimate["flops"], 1) if estimate.get("flops") is not None
                                 else (round(item["flops"], 1)
                                       if item["modeled_count"] and (complete or not item["fused_window"])
                                       else None)),
                       "evidence_level": ("ESTIMATED" if estimate.get("mfu") is not None else
                                          "PROXY" if complete and item["proxy_count"] else
                                          "FORMULA" if complete else "UNAVAILABLE"),
                       "fused_window": item["fused_window"],
                       "estimate_basis": estimate.get("basis")})
    groups.sort(key=lambda item: item["dur_us"], reverse=True)
    rows.sort(key=lambda item: item["dur_us"], reverse=True)
    formula_rows = sum(1 for item in rows if item["formula_available"])
    note = ("MFU 分母按每个算子@dtype 对应的 CUBE 峰值。MegaMoE 的 PROXY 仅含同采集校准的 "
            "MAC/CUBE 工作量，不含 Vector；分母为完整计算/通信融合窗口，不是纯计算 MFU。"
            "MC2 融合算子（AlltoAllvGroupedMatMul / GroupedMatMulAlltoAllv）在 Level-1 采集里"
            "不记录 shape，改用功能约定的矩阵乘（专家权重 [K,N]）与同采集专家 GroupedMatmul 的"
            "激活行数 M 重建 GEMM 工作量，按各算子占融合窗口的时长份额分摊，给出 ESTIMATED 的"
            "「融合窗口等效 MFU」——分母仍含内部 a2a，故是纯计算 MFU 的下界；MBU 仍不可用。"
            "缺少校准、计数器或 dtype 时保持不可用。其它融合算子统计可由 shape 重建的矩阵工作。")
    if mc2_estimate:
        note += ("（本次 MC2 估算：每 call 行数 {rows:,}，单次权重 {contracts}，配对 {pairs} 次，"
                 "重建总 FLOPs {flops:.3e}，融合窗口 {dur:.3f}s → 等效 MFU {mfu:.1%}）").format(
            rows=mc2_estimate["rows"], contracts="、".join(f"{k}x{n}" for k, n in mc2_estimate["contracts"]),
            pairs=mc2_estimate["pairs"], flops=mc2_estimate["total_flops"],
            dur=mc2_estimate["total_dur_us"] * 1e-6, mfu=mc2_estimate["equivalent_mfu"])
    return {"available": True,
            "chip": {"name": SETTINGS.chip.name, "hbm_tbps": SETTINGS.chip.hbm_bandwidth / 1e12,
                     "cube_bf16_tflops": SETTINGS.chip.cube_fp16_flops / 1e12},
            "total_dur_us": round(total_us, 1), "groups": groups, "rows": rows,
            "kernel_count": len(rows),
            "formula_coverage_pct": round(formula_rows / len(rows) * 100.0, 1),
            "fused_mc2_estimate": mc2_estimate or None,
            "note": note}
