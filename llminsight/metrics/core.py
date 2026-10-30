"""Core table/number metrics derived from the CSVs + communication.json."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import SETTINGS
from ..parser.names import is_mc2_fused_type
from ..parser.profile import num


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _i(x) -> Optional[int]:
    """Int from a possibly-blank cell. float(NaN) is a valid float (so _f lets it
    through) but int(NaN) raises — a missing Step / Device_id should degrade to
    None rather than blow up overview()."""
    v = _f(x)
    return int(v) if pd.notna(v) else None


def _pct(x: float, total: float) -> float:
    return round(100.0 * x / total, 2) if total else 0.0


# --------------------------------------------------------------------------- #
def _overview_msprof(ms: Dict[str, Any]) -> Dict[str, Any]:
    """Overview for a msprof lightweight capture: no wall-clock step_trace, so we
    surface SUMMED compute vs communication time as an accumulated-time split.
    Labeled basis=accumulated so the UI doesn't read it as a step wall-clock."""
    compute = _f(ms.get("compute_us"))
    comm = _f(ms.get("comm_us"))
    total = compute + comm
    return {
        "available": True,
        "basis": "accumulated",
        "note": ("msprof 轻量采集：无 wall-clock step 分解，下为算子累加耗时占比"
                 "（计算/通信相互重叠，非 step 墙钟）。"),
        "us": {"computing": compute, "communication": comm,
               "comm_not_overlapped": comm, "overlapped": 0.0, "free": 0.0,
               "stage": total, "bubble": 0.0, "preparing": 0.0},
        "composition": [
            {"name": "Computing", "us": compute, "pct": _pct(compute, total)},
            {"name": "Communication", "us": comm, "pct": _pct(comm, total)},
        ],
        "ratios": {
            "effective_compute_pct": _pct(compute, total),
            "comm_pct": _pct(comm, total),
            "comm_not_overlapped_pct": _pct(comm, total),
            "free_pct": 0.0,
            "step_time_s": round(total / 1e6, 4),
        },
    }


def overview(prof) -> Dict[str, Any]:
    st = prof.step_trace
    if st.empty:
        ms = (getattr(prof, "meta", {}) or {}).get("msprof")
        if ms and (ms.get("compute_us") or ms.get("comm_us")):
            return _overview_msprof(ms)
        return {"available": False}
    # torch_npu may export several profiled steps into one directory.  Every
    # other table (kernel_details / op_statistic / api_statistic / communication)
    # covers that whole capture window, so using only st.iloc[0] here would pair
    # a multi-step numerator with a one-step denominator and inflate MFU/MAC.
    # Keep one device (the loader is single-rank today), but SUM every tagged
    # step for the internal window.  avg_us is exposed separately for the UI's
    # human-friendly "平均 Step" display.
    rows = st
    device_id = None
    if "Device_id" in st.columns:
        devices = pd.to_numeric(st["Device_id"], errors="coerce").dropna()
        if not devices.empty:
            device_id = int(devices.iloc[0])
            rows = st[pd.to_numeric(st["Device_id"], errors="coerce") == device_id]

    steps: List[int] = []
    if "Step" in rows.columns:
        for value in pd.to_numeric(rows["Step"], errors="coerce").dropna():
            step = int(value)
            if step not in steps:
                steps.append(step)
    step_count = len(steps) if steps else 1

    def _sum_col(name: str) -> float:
        if name not in rows.columns:
            return 0.0
        return float(pd.to_numeric(rows[name], errors="coerce").fillna(0.0).sum())

    computing = _sum_col("Computing")
    comm_no = _sum_col("Communication(Not Overlapped)")
    overlapped = _sum_col("Overlapped")
    communication = _sum_col("Communication")
    free = _sum_col("Free")
    stage = _sum_col("Stage") or (computing + comm_no + free)
    bubble = _sum_col("Bubble")
    preparing = _sum_col("Preparing")
    avg = {
        "computing": computing / step_count,
        "comm_not_overlapped": comm_no / step_count,
        "overlapped": overlapped / step_count,
        "communication": communication / step_count,
        "free": free / step_count,
        "stage": stage / step_count,
        "bubble": bubble / step_count,
        "preparing": preparing / step_count,
    }
    step_label: Any = steps[0] if len(steps) == 1 else (
        f"{steps[0]}–{steps[-1]}" if steps else None
    )

    return {
        "available": True,
        "basis": "step_window",
        "step": step_label,
        "steps": steps,
        "step_count": step_count,
        "window_aggregated": step_count > 1,
        "device_id": device_id,
        "us": {
            "computing": computing,
            "comm_not_overlapped": comm_no,
            "overlapped": overlapped,
            "communication": communication,
            "free": free,
            "stage": stage,
            "bubble": bubble,
            "preparing": preparing,
        },
        "avg_us": avg,
        "composition": [
            {"name": "Computing", "us": computing, "pct": _pct(computing, stage)},
            {"name": "Communication (Not Overlapped)", "us": comm_no, "pct": _pct(comm_no, stage)},
            {"name": "Free", "us": free, "pct": _pct(free, stage)},
        ],
        "ratios": {
            "effective_compute_pct": _pct(computing, stage),
            "comm_not_overlapped_pct": _pct(comm_no, stage),
            "free_pct": _pct(free, stage),
            "overlap_rate_pct": _pct(overlapped, communication),
            "step_time_s": round(avg["stage"] / 1e6, 4),
            "window_time_s": round(stage / 1e6, 4),
        },
        "note": (f"采集窗口包含 {step_count} 个 step；内部指标使用窗口总量，"
                 "Step 时间展示窗口均值。" if step_count > 1 else "单 step 采集窗口。"),
    }


# --------------------------------------------------------------------------- #
_MC2_WRAPPER_SUFFIX = "Mc2AicpuKernel"


def _mc2_wrapper_duplicate_mask(df: "pd.DataFrame") -> "pd.Series":
    """Flag AI_CPU ``*Mc2AicpuKernel`` rows that merely mirror a real kernel.

    An MC2 fused op is recorded at two layers: the AI_CPU launcher
    (``<op>Mc2AicpuKernel``) and the MIX_AIC kernel that actually executes, both
    stamped with the same wall clock.  Summing both makes the operator table
    exceed the step time and deflates every share, so the launcher row is dropped
    — but only when its base name is also present on a non-AI_CPU core, so a
    capture without the paired kernel keeps its data.
    """
    types = df["OP Type"].astype(str)
    cores = (df["Core Type"].astype(str).str.upper()
             if "Core Type" in df.columns else pd.Series([""] * len(df), index=df.index))
    is_wrapper = types.str.endswith(_MC2_WRAPPER_SUFFIX) & (cores == "AI_CPU")
    if not bool(is_wrapper.any()):
        return is_wrapper
    real_types = set(types[~cores.isin(["AI_CPU"])])
    bases = types.str.slice(0, -len(_MC2_WRAPPER_SUFFIX))
    return is_wrapper & bases.isin(real_types)


def hotspots(prof) -> Dict[str, Any]:
    op = prof.op_statistic
    if op.empty:
        return {"available": False}
    df = op.copy()
    df["Total Time(us)"] = num(df["Total Time(us)"])
    df["Count"] = num(df["Count"])
    duplicate = _mc2_wrapper_duplicate_mask(df)
    dedup_us = float(_f(df.loc[duplicate, "Total Time(us)"].sum())) if bool(duplicate.any()) else 0.0
    dedup_types = sorted(set(df.loc[duplicate, "OP Type"].astype(str))) if dedup_us else []
    reported_total = df["Total Time(us)"].sum()
    if dedup_us:
        df = df[~duplicate].copy()
    total = df["Total Time(us)"].sum()

    ops = [
        {
            "type": str(row["OP Type"]),
            "core": str(row.get("Core Type", "")),
            "count": int(row["Count"]),
            "total_us": round(_f(row["Total Time(us)"]), 1),
            "avg_us": round(_f(row.get("Avg Time(us)")), 2),
            "max_us": round(_f(row.get("Max Time(us)")), 2),
            # share of the de-duplicated operator total (NOT of op_statistic's raw
            # Σ, which double counts the MC2 launcher layer and exceeds the step)
            "ratio": round(100.0 * _f(row["Total Time(us)"]) / total, 3) if total else 0.0,
        }
        for _, row in df.iterrows()
    ]
    ops.sort(key=lambda x: x["total_us"], reverse=True)

    by_core: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        c = str(row.get("Core Type", "")) or "UNKNOWN"
        b = by_core.setdefault(c, {"core": c, "total_us": 0.0, "count": 0})
        b["total_us"] += _f(row["Total Time(us)"])
        b["count"] += int(row["Count"])
    core_rows = sorted(by_core.values(), key=lambda x: x["total_us"], reverse=True)
    for b in core_rows:
        b["total_us"] = round(b["total_us"], 1)
        b["pct"] = _pct(b["total_us"], total)

    return {
        "available": True,
        "total_us": round(total, 1),
        "ops": ops,
        "top": ops[:15],
        "by_core": core_rows,
        "dedup": {
            "excluded_us": round(dedup_us, 1),
            "excluded_types": dedup_types,
            "reported_total_us": round(_f(reported_total), 1),
            "note": ("op_statistic 把 MC2 融合算子记了两层（AI_CPU 的 *Mc2AicpuKernel 启动壳 + "
                     "真正执行的 MIX_AIC 内核，同一段墙钟）。此处剔除启动壳，占比与合计按去重后的"
                     "算子时间计算；reported_total_us 为原始 Σ（含重复层）。"),
        },
    }


# --------------------------------------------------------------------------- #
# Collective-communication helpers shared by the communication section.
#   * an HCCL op carries its parallel-group id in the name (see
#     parser.profile.comm_meta_from_name) so the per-communicator split is a
#     plain group-by;
#   * FSDP roles (dense weights vs MoE expert weights) are recovered from the
#     host-side `FSDP::all_gather (<module>)` / `FSDP::post_backward_reduce
#     (<module>)` annotations that torch_npu records in operator_details.csv.
#     Each annotation pairs 1:1 with one HCCL collective, so counting the
#     annotations per module family and matching them against a communicator's
#     op counts identifies the group without hard-coding communicator ids.
_COMM_ROLE_LABELS = {
    "dense_fsdp": "dense-FSDP",
    "expert_fsdp": "expert-FSDP",
    "ep_a2a": "EP / a2a",
    "activation": "CP / KV 激活",
    "barrier": "纯 barrier",
    "unknown": "未知",
}


def _op_transit_mb(c: Dict[str, Any]) -> float:
    return sum((l or {}).get("transit_mb", 0) or 0 for l in (c.get("links") or {}).values())


def _bw(size_mb: float, transit_ms: float):
    """Effective bus bandwidth in GB/s (MB/ms); None when nothing was moved.

    Payloads below 0.01 MB are control/heartbeat bytes (the same threshold the
    zero-payload diagnosis uses), so they must not surface as "0.0 GB/s" — a
    barrier has no bandwidth at all, it is not an infinitely slow link.
    """
    if not size_mb or size_mb <= 0.01 or not transit_ms or transit_ms <= 0:
        return None
    return round(size_mb / transit_ms, 1)


def _fsdp_unit_counts(prof) -> Dict[str, int]:
    """Count FSDP2 unit hooks per module family from operator_details.csv.

    Dense units are `FSDP::all_gather (model.layers.N)`; MoE expert units are
    `... (model.layers.N.mlp.experts)`. `all_gather_copy_out` is the host copy
    that follows each gather and must not be counted twice.
    """
    od = getattr(prof, "operator_details", None)
    if od is None or getattr(od, "empty", True) or "Name" not in getattr(od, "columns", []):
        return {}
    names = od["Name"].astype(str)
    is_ag = names.str.startswith("FSDP::all_gather") & ~names.str.startswith(
        "FSDP::all_gather_copy_out")
    is_rs = names.str.startswith("FSDP::post_backward_reduce")
    if not (is_ag.any() or is_rs.any()):
        return {}
    expert = names.str.contains(".mlp.experts", regex=False)
    return {
        "dense_ag_events": int((is_ag & ~expert).sum()),
        "expert_ag_events": int((is_ag & expert).sum()),
        "dense_rs_events": int((is_rs & ~expert).sum()),
        "expert_rs_events": int((is_rs & expert).sum()),
    }


def _classify_group_role(stats: Dict[str, Any], units: Dict[str, int],
                         link_bytes_recorded: bool = True):
    """(role, basis) for one HCCL communicator group.

    ``link_bytes_recorded=False`` means the capture carries no HCCL link bytes at
    all (see ``communication()``), so "this communicator moves no payload" is not
    an observation — it is the absence of one.  In that case no role conclusion is
    published: calling a 16/64-rank gradient communicator a "pure barrier" sends
    the reader to the wrong bottleneck.
    """
    types = stats.get("types") or {}
    ag = int(types.get("allGather", 0))
    rs = int(types.get("reduceScatter", 0))
    if any(t.startswith("alltoall") for t in types):
        return "ep_a2a", "该通信域含 alltoallv（含 MC2 融合 AllToAllv）算子"
    expert_ag = units.get("expert_ag_events", 0)
    expert_rs = units.get("expert_rs_events", 0)
    dense_ag = units.get("dense_ag_events", 0)
    dense_rs = units.get("dense_rs_events", 0)
    expert_match = bool(expert_ag and expert_rs
                        and abs(ag - expert_ag) <= 2 and abs(rs - expert_rs) <= 2)
    dense_match = bool(dense_ag and dense_rs
                       and abs(ag - dense_ag) <= 2 and abs(rs - dense_rs) <= 2)
    if expert_match and dense_match:
        # Both unit families have the same AG/RS counts (tiny models / proxies):
        # the degenerate expert-FSDP case is the one that moves no bytes, so use
        # payload as the tie-breaker instead of guessing.
        if (stats.get("size_mb") or 0.0) <= 0.01:
            return ("expert_fsdp",
                    f"AG/RS 次数 ({ag}/{rs}) 同时匹配两族单元，但 0 载荷 → 判定 expert-FSDP"
                    f"（单元数 {expert_ag}/{expert_rs}）")
        return ("dense_fsdp",
                f"AG/RS 次数 ({ag}/{rs}) 同时匹配两族单元，但有载荷 → 判定 dense-FSDP"
                f"（单元数 {dense_ag}/{dense_rs}）")
    if expert_match:
        return ("expert_fsdp",
                f"AG/RS 次数 ({ag}/{rs}) 与 FSDP::all_gather(...mlp.experts) / "
                f"post_backward_reduce(...mlp.experts) 单元数 ({expert_ag}/{expert_rs}) 匹配")
    if dense_match:
        return ("dense_fsdp",
                f"AG/RS 次数 ({ag}/{rs}) 与 FSDP 非 expert 单元数 ({dense_ag}/{dense_rs}) 匹配")
    if not link_bytes_recorded:
        return ("unknown",
                "链路字节（Transit Size/Time）整份采集未记录：无法区分「零字节 barrier」与"
                "「未采集带宽」，本通信域不下角色结论")
    if stats.get("zero_payload_count") and stats["zero_payload_count"] == stats.get("count"):
        return "barrier", "该通信域全部算子 0 载荷（纯同步）"
    if ag or rs:
        return "activation", "有 AG/RS 但次数不匹配 FSDP 单元数（激活/KV 类通信）"
    return "unknown", "无法从算子名/单元计数判定角色"


# --------------------------------------------------------------------------- #
def communication(prof) -> Dict[str, Any]:
    comms = [c for c in prof.communication if c["type"] != "Total"]
    reported_total = next((c for c in prof.communication if c["type"] == "Total"), None)
    if not comms:
        return {"available": False}

    units = _fsdp_unit_counts(prof)

    by_type: Dict[str, Dict[str, Any]] = {}
    by_group: Dict[str, Dict[str, Any]] = {}
    for c in comms:
        t = by_type.setdefault(
            c["type"],
            {"type": c["type"], "count": 0, "elapse_ms": 0.0, "wait_ms": 0.0,
             "sync_ms": 0.0, "transit_ms": 0.0, "transit_mb": 0.0, "_wait_ratio_sum": 0.0},
        )
        t["count"] += 1
        t["elapse_ms"] += c["elapse_ms"]
        t["wait_ms"] += c["wait_ms"]
        t["sync_ms"] += c["sync_ms"]
        t["transit_ms"] += c["transit_ms"]
        t["transit_mb"] += _op_transit_mb(c)
        t["_wait_ratio_sum"] += min(c.get("wait_ratio", 0) or 0, 1.0)

        gid = c.get("group")
        if gid:
            g = by_group.setdefault(
                gid,
                {"group": gid, "count": 0, "types": {}, "flavors": {},
                 "elapse_ms": 0.0, "wait_ms": 0.0, "sync_ms": 0.0, "transit_ms": 0.0,
                 "size_mb": 0.0, "zero_payload_count": 0, "zero_payload_ms": 0.0,
                 "_wait_ratio_sum": 0.0},
            )
            g["count"] += 1
            g["types"][c["type"]] = g["types"].get(c["type"], 0) + 1
            fl = c.get("flavor")
            if fl and fl != "plain":
                g["flavors"][fl] = g["flavors"].get(fl, 0) + 1
            g["elapse_ms"] += c["elapse_ms"]
            g["wait_ms"] += c["wait_ms"]
            g["sync_ms"] += c["sync_ms"]
            g["transit_ms"] += c["transit_ms"]
            g["size_mb"] += _op_transit_mb(c)
            g["_wait_ratio_sum"] += min(c.get("wait_ratio", 0) or 0, 1.0)
            if _op_transit_mb(c) <= 0.01 and (c.get("elapse_ms") or 0.0) >= 0.05:
                g["zero_payload_count"] += 1
                g["zero_payload_ms"] += c["elapse_ms"]

    type_rows = sorted(by_type.values(), key=lambda x: x["elapse_ms"], reverse=True)
    for t in type_rows:
        # mean per-op wait ratio (each op's wait/elapse); avoids the >100%
        # artifact you get from summing wait_ms / summing elapse_ms separately.
        t["wait_pct"] = round(100.0 * t["_wait_ratio_sum"] / t["count"], 1) if t["count"] else 0.0
        t.pop("_wait_ratio_sum", None)
        for k in ("elapse_ms", "wait_ms", "sync_ms", "transit_ms", "transit_mb"):
            t[k] = round(t[k], 3)
        # effective bus bandwidth = moved bytes / transit time (MB/ms ≡ GB/s). None
        # when there is no real transit (single-card: all wait, transit≈0). This is
        # the per-category 有效带宽 the user wants surfaced alongside each type.
        t["bandwidth_gbps"] = (round(t["transit_mb"] / t["transit_ms"], 1)
                               if t["transit_ms"] > 0 else None)
        t["bandwidth_with_wait_gbps"] = (round(t["transit_mb"] / t["elapse_ms"], 1)
                                         if t["transit_mb"] > 0 and t["elapse_ms"] > 0
                                         else None)
        # MC2 fused all-to-all (dispatch/combine) is folded into `alltoallv`; flag
        # it so the UI can tell "fused a2a" from a plain hcom_alltoallv.
        t["fused"] = any(c.get("fused") for c in comms if c["type"] == t["type"])
        flavors: Dict[str, int] = {}
        for c in comms:
            if c["type"] == t["type"] and c.get("flavor") not in (None, "plain"):
                flavors[c["flavor"]] = flavors.get(c["flavor"], 0) + 1
        if flavors:
            t["flavors"] = flavors

    total_elapse = sum(c["elapse_ms"] for c in comms)
    total_wait = sum(c["wait_ms"] for c in comms)
    mean_wait_ratio = sum(min(c.get("wait_ratio", 0) or 0, 1.0) for c in comms) / len(comms)
    total_transit_mb = sum(
        sum(l.get("transit_mb", 0) for l in c["links"].values()) for c in comms
    )
    total_transit_ms = sum(c["transit_ms"] for c in comms)
    # Were link bytes recorded at all?  Level-1 captures emit communication.json
    # with every "Transit Size(MB)"/"Transit Time(ms)" at 0 — the HCCL bandwidth
    # detail is simply not collected.  That absence must never be read as "these
    # collectives moved no bytes": it would turn every communicator into a
    # "pure barrier" and every op into a zero-payload diagnosis.
    link_bytes_recorded = bool(total_transit_mb > 0.01 or total_transit_ms > 0)
    overall_bandwidth_gbps = (round(total_transit_mb / total_transit_ms, 1)
                              if total_transit_ms > 0 else None)
    overall_bandwidth_with_wait_gbps = (round(total_transit_mb / total_elapse, 1)
                                        if total_transit_mb > 0 and total_elapse > 0
                                        else None)

    top = sorted(comms, key=lambda c: c["elapse_ms"], reverse=True)[:15]
    top_out = [
        {
            "name": c["name"].split("@")[0],
            "type": c["type"],
            "elapse_ms": round(c["elapse_ms"], 3),
            "wait_ms": round(c["wait_ms"], 3),
            "wait_ratio": round(c["wait_ratio"], 3),
            "transit_mb": round(_op_transit_mb(c), 2),
            # per-op 有效带宽 = transit bytes / transit time (None when transit≈0)
            "bandwidth_gbps": (round(_op_transit_mb(c) / c["transit_ms"], 1)
                               if c["transit_ms"] > 0 else None),
            "bandwidth_with_wait_gbps": (round(_op_transit_mb(c) / c["elapse_ms"], 1)
                                         if _op_transit_mb(c) > 0 and c["elapse_ms"] > 0
                                         else None),
        }
        for c in top
    ]

    # ---- per-communicator split (dense-FSDP / expert-FSDP / EP-a2a / CP-KV) ----- #
    group_rows = []
    role_by_group: Dict[str, str] = {}
    for g in by_group.values():
        for key in ("elapse_ms", "wait_ms", "sync_ms", "transit_ms", "size_mb",
                    "zero_payload_ms"):
            g[key] = round(g[key], 3)
        g["size_gb"] = round(g["size_mb"] / 1000.0, 3)
        g["bandwidth_gbps"] = _bw(g["size_mb"], g["transit_ms"])
        g["bandwidth_with_wait_gbps"] = (round(g["size_mb"] / g["elapse_ms"], 1)
                                         if g["size_mb"] > 0 and g["elapse_ms"] > 0
                                         else None)
        g["wait_pct"] = (round(100.0 * g["_wait_ratio_sum"] / g["count"], 1)
                         if g["count"] else 0.0)
        g.pop("_wait_ratio_sum", None)
        g["role"], g["role_basis"] = _classify_group_role(
            g, units, link_bytes_recorded=link_bytes_recorded)
        g["role_label"] = _COMM_ROLE_LABELS.get(g["role"], g["role"])
        # Per-group zero-payload counts are only meaningful when transit bytes were
        # recorded; otherwise report "unknown" rather than a fabricated 0.
        g["zero_payload_available"] = link_bytes_recorded
        if not link_bytes_recorded:
            g["zero_payload_count"] = None
            g["zero_payload_ms"] = None
        role_by_group[g["group"]] = g["role"]
        group_rows.append(g)
    group_rows.sort(key=lambda x: x["elapse_ms"], reverse=True)

    def _split(rows, want_type):
        sub = [c for c in rows if c["type"] == want_type]
        size_mb = sum(_op_transit_mb(c) for c in sub)
        transit_ms = sum(c["transit_ms"] for c in sub)
        elapse_ms = sum(c["elapse_ms"] for c in sub)
        per_op = sorted(_op_transit_mb(c) / c["transit_ms"]
                        for c in sub if c["transit_ms"] > 0)
        return {
            "count": len(sub),
            "size_gb": round(size_mb / 1000.0, 3),
            "elapse_ms": round(elapse_ms, 3),
            "wait_ms": round(sum(c["wait_ms"] for c in sub), 3),
            "transit_ms": round(transit_ms, 3),
            "bandwidth_gbps": _bw(size_mb, transit_ms),
            "bandwidth_with_wait_gbps": (round(size_mb / elapse_ms, 1)
                                         if size_mb > 0 and elapse_ms > 0 else None),
            "per_op_p50_gbps": (round(per_op[len(per_op) // 2], 1) if per_op else None),
        }

    # ---- a2a (MC2 fused AllToAllV) ------------------------------------------- #
    a2a_ops = [c for c in comms if (c.get("type") or "").startswith("alltoallv")]
    a2a: Dict[str, Any] = {"available": bool(a2a_ops)}
    if not a2a_ops:
        # "no alltoallv in communication.json" is NOT the same as "this run has no
        # all-to-all": with MC2 fused MoE the a2a runs inside the expert GEMM kernel
        # and never reaches communication.json.  Surface the fused window instead of
        # letting the reader conclude EP is off.
        fused_us = _fused_mc2_us(prof)
        if fused_us > 0:
            a2a["fused_window_us"] = round(fused_us, 1)
            a2a["note"] = (
                f"communication.json 中没有独立 alltoallv：本采集的 all-to-all 在 MC2 融合 "
                f"kernel 内部（融合窗口 {fused_us:,.0f}us，见「结构归因 → MoE fused "
                "dispatch/combine」），因此不能用本段判断 EP 是否开启。")
    if a2a_ops:
        size_mb = sum(_op_transit_mb(c) for c in a2a_ops)
        transit_ms = sum(c["transit_ms"] for c in a2a_ops)
        elapse_ms = sum(c["elapse_ms"] for c in a2a_ops)
        wait_ms = sum(c["wait_ms"] for c in a2a_ops)
        per_op = sorted(_op_transit_mb(c) / c["transit_ms"]
                        for c in a2a_ops if c["transit_ms"] > 0)

        def _flavor_block(sub):
            smb = sum(_op_transit_mb(c) for c in sub)
            tms = sum(c["transit_ms"] for c in sub)
            return {"count": len(sub), "size_gb": round(smb / 1000.0, 3),
                    "bandwidth_gbps": _bw(smb, tms),
                    "wait_ms": round(sum(c["wait_ms"] for c in sub), 3)}

        a2a.update({
            "count": len(a2a_ops),
            "fused": any(c.get("fused") for c in a2a_ops),
            "size_gb": round(size_mb / 1000.0, 3),
            "per_op_mb": round(size_mb / len(a2a_ops), 3),
            "elapse_ms": round(elapse_ms, 3),
            "wait_ms": round(wait_ms, 3),
            "transit_ms": round(transit_ms, 3),
            "bandwidth_gbps": _bw(size_mb, transit_ms),
            "bandwidth_with_wait_gbps": (round(size_mb / elapse_ms, 1)
                                         if elapse_ms > 0 else None),
            "per_op_bw_p50": round(per_op[len(per_op) // 2], 1) if per_op else None,
            "per_op_bw_p90": (round(per_op[min(len(per_op) - 1,
                                              int(0.9 * (len(per_op) - 1)))], 1)
                              if per_op else None),
            "per_op_bw_max": round(per_op[-1], 1) if per_op else None,
            "wait_share_pct": (round(100.0 * wait_ms / total_wait, 1)
                               if total_wait else None),
            "dispatch": _flavor_block([c for c in a2a_ops
                                       if c.get("flavor") == "mc2_dispatch"]),
            "combine": _flavor_block([c for c in a2a_ops
                                      if c.get("flavor") == "mc2_combine"]),
            "groups": sorted({c["group"] for c in a2a_ops if c.get("group")}),
        })

    # ---- dense-FSDP vs expert-FSDP AG/RS -------------------------------------- #
    dense_ops = [c for c in comms if role_by_group.get(c.get("group")) == "dense_fsdp"]
    expert_ops = [c for c in comms if role_by_group.get(c.get("group")) == "expert_fsdp"]
    fsdp: Dict[str, Any] = {"available": bool(dense_ops or expert_ops), "units": units}
    if fsdp["available"]:
        fsdp["dense"] = {
            "groups": sorted({c["group"] for c in dense_ops if c.get("group")}),
            "ag": _split(dense_ops, "allGather"),
            "rs": _split(dense_ops, "reduceScatter"),
        }
        fsdp["expert"] = {
            "groups": sorted({c["group"] for c in expert_ops if c.get("group")}),
            "ag": _split(expert_ops, "allGather"),
            "rs": _split(expert_ops, "reduceScatter"),
        }
        fsdp["note"] = (
            "角色由 operator_details 的 FSDP::all_gather / post_backward_reduce 模块名"
            "（model.layers.N 为 dense 单元，model.layers.N.mlp.experts 为 expert 单元）"
            "与各通信域 AG/RS 次数匹配得出；expert 侧 0 字节说明 efsdp 没有真正 unshard 参数，"
            "其耗时全部是同步等待。")

    # ---- wait attribution (等最慢 rank) --------------------------------------- #
    role_wait: Dict[str, float] = {}
    for c in comms:
        role = role_by_group.get(c.get("group"), "unknown")
        role_wait[role] = role_wait.get(role, 0.0) + c["wait_ms"]
    wait_attribution = {
        "total_wait_ms": round(total_wait, 2),
        "top": [
            {
                "name": c["name"].split("@")[0],
                "type": c["type"],
                "group": c.get("group"),
                "wait_ms": round(c["wait_ms"], 3),
                "share_pct": (round(100.0 * c["wait_ms"] / total_wait, 2)
                              if total_wait else None),
            }
            for c in sorted(comms, key=lambda x: x["wait_ms"], reverse=True)[:10]
        ],
        "by_role": [
            {"role": role, "role_label": _COMM_ROLE_LABELS.get(role, role),
             "wait_ms": round(v, 2),
             "share_pct": (round(100.0 * v / total_wait, 2) if total_wait else None)}
            for role, v in sorted(role_wait.items(), key=lambda kv: -kv[1])
            if v > 0
        ],
    }

    # Zero-payload collectives: ops that move (almost) no bytes but still take
    # wall-clock.  They are barriers / phase-alignment placeholders inside a
    # parallel group (e.g. a 2-rank edp group doing a 0-byte allReduce per layer),
    # so their time is pure synchronization, not bandwidth.  Surfaced separately
    # because it is invisible in the bandwidth table (transit≈0 -> bandwidth N/A).
    #
    # This is only a diagnosis when the capture actually recorded link bytes for
    # SOME op.  When not a single op carries a transit size, "0 bytes" is the
    # profiler's default for "not collected", and reporting 全部 collective 零载荷
    # would fabricate the finding (a Level-1 capture can read 0 for every record
    # while the sibling MindStudio DB still holds the real transfers).
    if not link_bytes_recorded:
        zero_transit: Dict[str, Any] = {
            "available": False,
            "evidence_level": "UNAVAILABLE",
            "count": 0,
            "reason": ("本次采集没有记录集合通信的链路字节（communication.json 的 "
                       "Transit Size(MB) / Transit Time(ms) 对全部算子均为 0，通常是 profiling "
                       "level 不足 / 未开互联采集）：无法区分「零字节 barrier」与「未采集带宽」，"
                       "因此不输出零载荷诊断，也不给出可回收结论。如需该诊断，请用更高的 "
                       "profiler level 复采。"),
        }
    else:
        zero_transit_ops = [
            c for c in comms
            if _op_transit_mb(c) <= 0.01 and (c.get("elapse_ms") or 0.0) >= 0.05
        ]
        zero_transit_elapse = round(sum(c["elapse_ms"] for c in zero_transit_ops), 3)
        zero_by_type: Dict[str, Dict[str, Any]] = {}
        zero_by_group: Dict[str, Dict[str, Any]] = {}
        for c in zero_transit_ops:
            row = zero_by_type.setdefault(c["type"], {"count": 0, "elapse_ms": 0.0,
                                                      "sync_ms": 0.0, "wait_ms": 0.0})
            row["count"] += 1
            row["elapse_ms"] += c["elapse_ms"]
            row["sync_ms"] += c.get("sync_ms") or 0.0
            row["wait_ms"] += c.get("wait_ms") or 0.0
            if c.get("group"):
                gr = zero_by_group.setdefault(c["group"], {"group": c["group"], "count": 0,
                                                           "elapse_ms": 0.0, "wait_ms": 0.0})
                gr["count"] += 1
                gr["elapse_ms"] += c["elapse_ms"]
                gr["wait_ms"] += c.get("wait_ms") or 0.0
        for row in zero_by_type.values():
            for key in ("elapse_ms", "sync_ms", "wait_ms"):
                row[key] = round(row[key], 3)
        zero_group_rows = []
        for gr in zero_by_group.values():
            gr["elapse_ms"] = round(gr["elapse_ms"], 3)
            gr["wait_ms"] = round(gr["wait_ms"], 3)
            gr["role"] = role_by_group.get(gr["group"], "unknown")
            gr["role_label"] = _COMM_ROLE_LABELS.get(gr["role"], gr["role"])
            zero_group_rows.append(gr)
        zero_transit = {
            "available": True,
            "evidence_level": "OBSERVED",
            "count": len(zero_transit_ops),
            "elapse_ms": zero_transit_elapse,
            # Share of the SUMMED per-op elapse, which is a legitimate relative
            # measure.  It is NOT a share of the step: parallel communicators
            # overlap, so Σ(elapse) can exceed the device wall clock several times
            # over.  Step-level exposure lives in step_trace_time.csv only.
            "pct_of_comm": (round(100.0 * zero_transit_elapse / total_elapse, 2)
                            if total_elapse else None),
            "mean_elapse_ms": (round(zero_transit_elapse / len(zero_transit_ops), 3)
                               if zero_transit_ops else None),
            "by_type": zero_by_type,
            "groups": sorted(zero_by_group.keys()),
            "by_group": sorted(zero_group_rows, key=lambda x: -x["elapse_ms"]),
            "note": ("Transit Size≈0 的集合通信：耗时几乎全是 wait/synchronization，"
                     "属于并行组内的 barrier / 相位对齐占位，不是链路带宽问题。"
                     "优先核查并行组划分（谁和谁同步）与能否合并/消除这些 0 字节 collective。"
                     "elapse_ms 是逐 op 累计值（多并行组区间重叠），不是 step 墙钟。"),
        }

    # msprof: device-level effective-transfer vs cross-rank-wait split (None on
    # torch_npu, whose communication.json already carries Transit/Wait per op).
    breakdown = ((getattr(prof, "meta", {}) or {}).get("msprof") or {}).get("comm_breakdown")

    if total_transit_ms > 0:
        note = ("平均带宽(含等待) = Σ Transit Size ÷ Σ Elapse；"
                "有效带宽(去等待) = Σ Transit Size ÷ Σ Transit Time（MB/ms≡GB/s）。"
                "Wait/Synchronization 为卡间等待，不计入去等待带宽分母。")
    else:
        note = ("本采集 Transit≈0：communication.json 无可用传输字节，平均/去等待带宽均 N/A；"
                "零载荷诊断与通信域角色一并停用（0 字节是「未采集」而不是「不搬数据」）。"
                "若存在 MindStudio DB，则用 device 子任务分解等待与有效传输时间，但不能反推真实链路带宽。")

    return {
        "available": True,
        "breakdown": breakdown,
        "count": len(comms),
        "total_elapse_ms": round(total_elapse, 2),
        "total_wait_ms": round(total_wait, 2),
        "overall_wait_pct": round(mean_wait_ratio * 100, 1),
        "total_transit_mb": round(total_transit_mb, 3),
        "total_transit_ms": round(total_transit_ms, 3),
        # False when no op carries transit bytes: bandwidth / zero-payload / group
        # role are then data-absent, not measured. Consumers must degrade, not
        # interpret 0 bytes as a physical fact.
        "transit_collected": link_bytes_recorded,
        # overall 有效带宽 (None when transit≈0); per-category in by_type[].bandwidth_gbps
        "overall_bandwidth_gbps": overall_bandwidth_gbps,
        "overall_bandwidth_with_wait_gbps": overall_bandwidth_with_wait_gbps,
        "by_type": type_rows,
        "by_group": group_rows,
        "a2a": a2a,
        "fsdp": fsdp,
        "wait_attribution": wait_attribution,
        "top": top_out,
        "zero_transit": zero_transit,
        "note": note,
    }


# --------------------------------------------------------------------------- #
def _api_sum(api: pd.DataFrame, pattern: str) -> Dict[str, float]:
    if api.empty or "API Name" not in api.columns:
        return {"time_us": 0.0, "count": 0, "max_us": 0.0}
    mask = api["API Name"].astype(str).str.contains(pattern, case=False, regex=True, na=False)
    sub = api[mask]
    return {
        "time_us": round(_f(num(sub["Time(us)"]).sum()), 1),
        "count": int(num(sub["Count"]).sum()) if "Count" in sub else 0,
        "max_us": round(_f(num(sub["Max(us)"]).max()), 1) if "Max(us)" in sub and not sub.empty else 0.0,
    }


def _op_time(op: pd.DataFrame, names: List[str]) -> float:
    if op.empty:
        return 0.0
    mask = op["OP Type"].astype(str).isin(names)
    return _f(num(op[mask]["Total Time(us)"]).sum())


def _fused_mc2_us(prof) -> float:
    """Wall-clock of the MC2 fused (all-to-all + grouped GEMM) kernels.

    These windows never appear in ``communication.json``, so the communication
    section (and the EP guesses in parser.derive) must read them from
    ``op_statistic``.  AI_CPU launcher rows are a second record of the same wall
    clock and are excluded.
    """
    op = getattr(prof, "op_statistic", None)
    if op is None or op.empty or "OP Type" not in op.columns:
        return 0.0
    df = op.copy()
    df["Total Time(us)"] = num(df["Total Time(us)"])
    mask = df["OP Type"].astype(str).map(is_mc2_fused_type)
    if "Core Type" in df.columns:
        mask &= df["Core Type"].astype(str).str.upper() != "AI_CPU"
    return float(df.loc[mask, "Total Time(us)"].sum())


def hidden_overhead(prof, ov: Dict[str, Any], capture: Dict[str, Any] = None,
                    recompute_view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Hidden-overhead ledger.

    ``recompute_view`` is the already-quantified ``theoretical.recompute`` payload.
    The ledger must mirror it instead of re-deriving a second, differently-defined
    number (the model projection ``computing × r_re`` and the measured replay
    duration disagreed by 32x for the same "重计算开销" label).
    """
    api = prof.api_statistic
    op = prof.op_statistic
    kd = prof.kernel_details
    stage = ov["us"]["stage"] if ov.get("available") else 0.0
    free = ov["us"]["free"] if ov.get("available") else 0.0
    comm_no = ov["us"]["comm_not_overlapped"] if ov.get("available") else 0.0

    # --- dispatch (下发) ---
    aicpu_dispatch = _op_time(op, ["HcclLaunchAicpuKernel"])
    host_launch = _api_sum(api, r"launch|LaunchKernel")
    # --- wait / sync ---
    sync_stream = _api_sum(api, r"[Ss]ynchronize")
    notify_wait = _api_sum(api, r"Notify.?Wait|StreamWait|EventWait|Wait")
    comm_wait_ms = sum(c["wait_ms"] for c in prof.communication if c["type"] != "Total")
    kernel_wait_us = _f(num(kd["Wait Time(us)"]).sum()) if (not kd.empty and "Wait Time(us)" in kd.columns) else 0.0
    # --- format / init / memory mgmt ---
    cast_init = _op_time(op, ["Cast", "ZerosLike", "TensorMove", "ConcatD", "Slice"])
    contiguous = _api_sum(api, r"[Cc]ontiguous|Copy|Memcpy")
    # --- dynamic shape ---
    dyn = _api_sum(api, r"MaskedSelect|NonZero|Unique|MaskedScatter")

    # Capture/config state is RECONSTRUCTED FROM PROFILING (parser.derive), never a
    # launch script. The host_sync suggestion and the recompute bucket below follow
    # these derived facts so they stay consistent across datasets and with
    # rules/engine.py + summarizer.py — no hardcoded blocking / full-recompute.
    cap = capture or {}
    cap_state = cap.get("capture", {}) or {}
    blocking = bool((cap_state.get("blocking", {}) or {}).get("value")) if cap_state else \
        ((cap.get("env") or {}).get("ASCEND_LAUNCH_BLOCKING") == "1")
    recompute = (cap_state.get("recompute", {}) or {}).get("value")  # full|selective|off|None
    # Heaviest aclnn API this capture actually spent host time in (parser.derive).
    # The suggestion text used to hardcode "aclnnMaskedSelect", which is simply not
    # the API in every capture (e.g. aclnnNonzeroV2 dominates here).
    sync_root = (cap_state.get("host_sync_stall", {}) or {}).get("root_api") if cap_state else None

    buckets = [
        {
            # NOT a separate launch overhead: HcclLaunchAicpuKernel is the AI_CPU
            # operator that *executes* the collectives (AICPU-unfold), so its time
            # is the same wall-clock as step_trace "Communication" (here ~100%
            # Wait). Shown as an operator-view lens but marked non-additive so it
            # is never summed on top of the 未掩盖通信 bucket (double-count).
            "key": "aicpu_dispatch", "domain": "device", "additive": False,
            "label": "AICPU 集合通信执行 (HcclLaunchAicpuKernel · 同段通信不计入合计)",
            "us": round(aicpu_dispatch, 1),
            "detail": f"AI_CPU 驱动集合通信合计 {aicpu_dispatch:,.0f}us（占 step {_pct(aicpu_dispatch, stage)}%）。"
                      # 拓扑相关结论必须由数据决定：单卡采集里该算子几乎全是 Wait、
                      # Transit≈0；多卡采集里它是 AICPU 展开执行集合通信的算子视角。
                      # 早期版本把「单卡」写死，在多卡采集上直接是错误陈述。
                      + ("单卡下几乎全为 Wait、Transit≈0 → " if _single_card(prof)
                         else "多卡采集下这是 AICPU 展开执行集合通信的算子视角（耗时含等待与实际传输）→ ")
                      + "这是「未掩盖通信」的算子视角，与 Communication 为同一段时间，"
                        "已从 Device 合计中剔除以免与未掩盖通信重复计入。",
            "source": "op_statistic (AI_CPU)",
            "suggestion": "本质是通信而非下发延迟：优先掩盖到计算下（moe-fb-overlap / 异步通信），再缩短通信本身（HCCL 算法/buffsize、EP 规模）。",
        },
        {
            "key": "comm_not_overlapped", "domain": "device",
            "label": "通信未掩盖 (Not Overlapped)",
            "us": round(comm_no, 1),
            "detail": f"未掩盖通信 {comm_no:,.0f}us 直接计入 step；集合通信 Wait 合计 {comm_wait_ms*1000.0:,.0f}us "
                      "（Σ 逐 op 累计口径：多个并行域的区间互相重叠，不等于 step 墙钟）；"
                      f"kernel Wait {kernel_wait_us:,.0f}us",
            "source": "step_trace + communication.json + kernel_details",
            "suggestion": "扩大计算-通信重叠（moe-fb-overlap / 异步通信），减少同步点。",
        },
        {
            "key": "free", "domain": "device",
            "label": "空泡 / Free",
            "us": round(free, 1),
            "detail": f"Free {free:,.0f}us（占 step {_pct(free, stage)}%）",
            "source": "step_trace_time.csv",
            "suggestion": "定位空泡来源（同步等待 vs 下发不及时），与等待/下发联合优化。",
        },
        {
            # format_init (Cast / ZerosLike / TensorMove / Concat / Slice) is device
            # compute that ALREADY lives inside Computing.  It is shown as a lens for
            # "how much of the compute is format/init work", never summed into the
            # hidden-overhead total.
            "key": "format_init", "domain": "device", "additive": False,
            "label": "格式转换 / 内存初始化 (Cast/ZerosLike/TensorMove · 已含在 Computing 内)",
            "us": round(cast_init, 1),
            "detail": f"device 侧 Cast/ZerosLike/TensorMove/Concat/Slice {cast_init:,.0f}us"
                      "（这部分时间已包含在 Computing 内，仅作构成视角，不并入 Device 合计，"
                      "否则会把正常计算重复算成隐性开销）",
            "source": "op_statistic",
            "suggestion": "算子融合 / 内存复用 / 减少不必要的 dtype 转换与拷贝。",
        },
        {
            "key": "host_launch", "domain": "host",
            "label": "Host Kernel Launch 下发",
            "us": round(host_launch["time_us"], 1),
            "detail": f"host 侧 launch 类 API 累计 {host_launch['time_us']:,.0f}us（{host_launch['count']} 次）",
            "source": "api_statistic",
            "suggestion": "减少小算子数量 / 开启下发队列（TASK_QUEUE_ENABLE）/ 图模式降低 host 下发压力。",
        },
        {
            "key": "host_sync", "domain": "host",
            "label": "Host 同步阻塞 (aclrtSynchronize*)",
            "us": round(sync_stream["time_us"], 1),
            "detail": f"aclrtSynchronize* 累计 {sync_stream['time_us']:,.0f}us，单次 max {sync_stream['max_us']:,.0f}us",
            "source": "api_statistic",
            "suggestion": (
                "ASCEND_LAUNCH_BLOCKING=1 放大了同步开销（采集干扰项）；正式训练应关闭。" if blocking else
                "本次未检出 blocking（profiling 反推）：该同步主要来自动态 shape 的 D2H 同步"
                + (f"（最重 aclnn 调用为 {sync_root}）" if sync_root else "")
                + " → 固定 expert capacity / 对路由结果 padding，合并或减少同步点。"
            ),
        },
        {
            "key": "dynamic_shape", "domain": "host",
            "label": "动态 shape 抖动 (MaskedSelect/NonZero)",
            "us": round(dyn["time_us"], 1),
            "detail": f"host 累计 {dyn['time_us']:,.0f}us，单次 max {dyn['max_us']:,.0f}us（方差大）",
            "source": "api_statistic",
            "suggestion": "MoE 路由/掩码导致 host 重编译/同步 → 固定 capacity / padding。",
        },
    ]
    # 重计算桶：仅当 profiling 反推重计算开启（full/selective）时呈现；反推为 off/未知时
    # 本次配置无此开销，不臆造（与「配置只来自 profiling、不依赖启动脚本」一致）。
    # 数值必须来自 theoretical.recompute（单一真源）：本函数早于 theoretical 计算时曾
    # 自行用 computing × r_re 再算一次，两处对同一个「重计算开销」相差 32 倍。
    _rc_view = recompute_view or None
    _rc_gran = (_rc_view or {}).get("granularity") or recompute
    if _rc_gran in ("full", "selective"):
        _rc_kind = "Full" if _rc_gran == "full" else "Selective"
        _rc_us = (_rc_view or {}).get("overhead_us")
        _fp_us = (_rc_view or {}).get("footprint_us")
        if _rc_us is not None:
            _rc_detail = (
                "profiling 反推重计算={g}（前向/反向调用数 + 可识别重放算子）→ 反向重跑前向。"
                "实测可识别重放算子耗时 ≈ {us:,.0f}us（step 的 {pct}%）。"
                "该口径只计入能直接配对到的重放算子，因此是保守下界；"
            ).format(g=_rc_gran, us=_rc_us, pct=_pct(_rc_us, stage))
            if _fp_us:
                _rc_detail += (
                    "若全部前向算子都被重放，足迹上界约 {fp:,.0f}us（step 的 {fp_pct}%，"
                    "computing × r_re 投影）。"
                ).format(fp=_fp_us, fp_pct=_pct(_fp_us, stage))
            _rc_detail += "重放与反向计算并发，可回收收益见「理论上界 / What-if」重计算项。"
        else:
            # 没有共享视图时（独立调用本函数）退回模型投影，并显式标注它是投影。
            _computing = ov["us"]["computing"] if ov.get("available") else 0.0
            _rc_ov = _recompute_overhead(_computing, cap_state.get("recompute"))
            _rc_us = _rc_ov["us"] if _rc_ov else None
            _rc_detail = (
                "profiling 反推重计算={g}（FA 前向/反向次数比）→ 反向重跑前向。"
                "估算重算足迹 ≈ {us:,.0f}us（step 的 {pct}%，band {lo:,.0f}–{hi:,.0f}us）"
                "= computing × {sh:.0%} 的投影，假设前向算子全部重放。"
            ).format(g=_rc_gran, us=_rc_ov["us"] if _rc_ov else 0.0,
                     pct=_pct(_rc_ov["us"], stage) if _rc_ov else 0.0,
                     lo=_rc_ov["us_lo"] if _rc_ov else 0.0,
                     hi=_rc_ov["us_hi"] if _rc_ov else 0.0,
                     sh=_rc_ov["flops_share"] if _rc_ov else 0.0) if _rc_ov else \
                f"profiling 反推重计算={_rc_gran}→ 反向重跑前向；缺前向/反向计数，未能量化耗时。"
        buckets.append({
            "key": "recompute", "domain": "config", "additive": False,
            "label": f"重计算开销 ({_rc_kind} Recompute)",
            "us": _rc_us,
            "detail": _rc_detail,
            "source": "profiling 反推 (parser.derive) + 可识别重放算子",
            "suggestion": "评估「减少/关闭重计算」做显存↔耗时平衡；量化收益见 What-if 重计算项。",
        })
    device_total = sum(b["us"] for b in buckets
                       if b["domain"] == "device" and b.get("additive", True)
                       and isinstance(b["us"], (int, float)))
    host_total = sum(b["us"] for b in buckets if b["domain"] == "host" and isinstance(b["us"], (int, float)))
    return {
        "available": True,
        "buckets": buckets,
        "device_total_us": round(device_total, 1),
        "host_total_us": round(host_total, 1),
        "stage_us": stage,
        "step_count": int(ov.get("step_count") or 1),
        "average_stage_us": round(
            stage / max(int(ov.get("step_count") or 1), 1), 1
        ),
        "note": (
            "host 与 device 时间不可直接相加（部分并发/被 blocking 放大）。device 桶与 step 同口径可比；"
            "host 桶反映下发/同步压力。AICPU 集合通信执行与「未掩盖通信」为同一段时间，仅作算子视角展示、"
            "不并入 Device 合计（避免重复计入）。用于「相对量级与归因」。"
        ),
    }


# --------------------------------------------------------------------------- #
# MC2 fused dispatch/combine GEMM labels, matched case-insensitively as SUBSTRINGS.
# A capture that records op attributes publishes the profiler label
# ("AlltoAllvGroupedMatMul"), but a Level-1 capture (no op attrs) publishes the raw
# kernel symbol as OP Type instead, e.g.
#   _ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel31Ascend950AllToAllVGroupedMatmul…
# Exact equality then silently dropped the whole fused window into
# Elementwise/Other and left moe_focus.fused_dispatch_combine_us = 0.  Keep the
# markers narrow (parser.names owns the single list).
def _module_type_mask(types: "pd.Series", name: str) -> "pd.Series":
    """Rows whose OP Type is *name*, tolerating mangled/fused kernel symbols.

    Exact equality for every module rule; the MC2 fused GEMMs additionally match
    by marker so qualified C++ symbols (Level-1 captures) are attributed too.
    """
    mask = types == name
    if is_mc2_fused_type(name):
        mask = mask | types.str.lower().str.contains(name.lower(), regex=False)
    return mask


_MODULE_RULES = [
    # MC2 fused dispatch/combine GEMMs.  They carry both the expert GEMM and the
    # all-to-all inside one window, so they get their own module instead of
    # falling through to the Elementwise/Other residual (which used to swallow the
    # whole fused window and made "MoE-Dispatch (alltoallv) = 0" look plausible).
    ("MoE fused dispatch/combine (GEMM+comm)",
     ["AlltoAllvGroupedMatMul", "GroupedMatMulAlltoAllv"]),
    ("MoE-Experts", ["GroupedMatmul", "GroupedMatmulAdd", "SwiGlu", "SwiGluGrad",
                      "ScatterAdd", "InplaceIndexAdd", "ScatterElementsV2",
                      "GatherElements"]),
    ("MoE-Router", ["TopKV2", "Sigmoid", "SigmoidGrad", "ArgMaxWithValue", "Sort",
                     "Cumsum", "ReduceSum", "LpNormV2"]),
    ("Attention", ["FlashAttentionScore", "FlashAttentionScoreGrad",
                        "SparseFlashAttention", "SparseFlashAttentionGrad",
                        "SparseFlashMla", "SparseFlashMlaGrad",
                        "RotaryPositionEmbedding", "RotaryPositionEmbeddingGrad"]),
    ("Norm", ["RmsNorm", "RmsNormGrad"]),
    ("Optimizer", ["ApplyAdamWV2", "ApplyAdamW"]),
    ("Embedding/Loss", ["GatherV2", "EmbeddingDenseGradV2", "Exp", "Log"]),
    ("GEMM/Projections (shared)", ["MatMulV3", "GemmV3", "MatMul", "BatchMatMul",
                                    "BatchMatMulV2", "BatchMatMulV3"]),
]


def attribution(prof) -> Dict[str, Any]:
    op = prof.op_statistic
    if op.empty:
        return {"available": False}
    df = op.copy()
    kd = prof.kernel_details
    optimizer_us = 0.0
    if "Workload Phase" in kd.columns:
        device = kd[~kd["Type"].astype(str).str.lower().str.startswith(("hccl", "hcom"))].copy()
        if "Accelerator Core" in device:
            device = device[device["Accelerator Core"].astype(str).str.upper() != "AI_CPU"]
        device["Duration(us)"] = num(device["Duration(us)"])
        optimizer_us = float(device.loc[device["Workload Phase"] == "optimizer", "Duration(us)"].sum())
        model = device[device["Workload Phase"] != "optimizer"]
        df = model.groupby("Type", as_index=False)["Duration(us)"].sum().rename(
            columns={"Type": "OP Type", "Duration(us)": "Total Time(us)"})
    df["Total Time(us)"] = num(df["Total Time(us)"])
    # Exclude communication-dispatch ops (AI_CPU / HCCL) — they are wall-clock
    # comm, not model compute, and would otherwise dump 676k us into Elementwise.
    core_col = df["Core Type"].astype(str).str.upper() if "Core Type" in df.columns else None
    type_col = df["OP Type"].astype(str)
    keep = ~type_col.str.lower().str.startswith(("hccl", "hcom"))
    if core_col is not None:
        keep &= core_col != "AI_CPU"
    df = df[keep].copy()
    assigned = {name: 0.0 for name, _ in _MODULE_RULES}
    assigned["Optimizer"] = optimizer_us
    elementwise = 0.0
    seen = set()
    op_types = df["OP Type"].astype(str)
    for module, types in _MODULE_RULES:
        for t in types:
            mask = _module_type_mask(op_types, t)
            assigned[module] += _f(df[mask]["Total Time(us)"].sum())
            seen.update(df[mask].index.tolist())
    for idx, row in df.iterrows():
        if idx not in seen:
            elementwise += _f(row["Total Time(us)"])

    # --- device compute ring (consistent device-time basis) ---
    # Fused expert work keeps its entire window, including internal communication.
    fused_mask = df["OP Type"].astype(str).str.lower().str.contains("mega_moe", regex=False)
    fused_us = float(df.loc[fused_mask, "Total Time(us)"].sum())
    elementwise -= fused_us
    modules = [{"module": m, "us": round(v, 1)} for m, v in assigned.items() if v > 0]
    if fused_us:
        modules.append({"module": "MoE fused compute/communication", "us": round(fused_us, 1)})
    modules.append({"module": "Elementwise/Other", "us": round(elementwise, 1)})
    modules.sort(key=lambda x: x["us"], reverse=True)
    total = sum(m["us"] for m in modules)
    for m in modules:
        m["pct"] = _pct(m["us"], total)

    # --- communication shown separately (wall-clock, mostly wait — NOT summed
    #     into the compute ring to avoid mixing bases) ---
    comm_moe = sum(c["elapse_ms"] for c in prof.communication if c["type"] == "alltoallv") * 1000.0
    comm_other = sum(c["elapse_ms"] for c in prof.communication
                     if c["type"] not in ("alltoallv", "Total")) * 1000.0
    comm_total = comm_moe + comm_other
    comm_breakdown = [
        {"module": "MoE-Dispatch (alltoallv)", "us": round(comm_moe, 1),
         "pct": _pct(comm_moe, comm_total)},
        {"module": "TP/SP/DP (allGather/reduceScatter/allReduce)", "us": round(comm_other, 1),
         "pct": _pct(comm_other, comm_total)},
    ]

    # MoE专属：专家计算 vs 分发通信
    moe_compute = assigned.get("MoE-Experts", 0.0)
    mc2_fused = assigned.get("MoE fused dispatch/combine (GEMM+comm)", 0.0)
    moe_focus = {
        "expert_compute_us": round(moe_compute, 1),
        "dispatch_comm_us": round(comm_moe, 1),
        "fused_expert_window_us": round(fused_us, 1),
        # MC2 dispatch/combine windows: expert GEMM and the a2a share one duration,
        # so they cannot be split from this capture and are NOT added to
        # dispatch_comm_us (which only counts standalone alltoallv ops).
        "fused_dispatch_combine_us": round(mc2_fused, 1),
    }

    return {
        "available": True,
        "modules": modules,
        "total_us": round(total, 1),
        "comm_breakdown": comm_breakdown,
        "comm_total_us": round(comm_total, 1),
        "moe_focus": moe_focus,
        "note": (
            "按算子命名启发式归因，基于 device 计算时间；GEMM/Projections 为 Attention 投影 / Router / "
            "LM-Head 共用未细分。MoE fused 为包含内部通信的完整窗口，不能解释为纯计算。"
            "MC2 融合算子（AlltoAllvGroupedMatMul / GroupedMatMulAlltoAllv）的 all-to-all 在算子内部，"
            "不会出现在 communication.json 里，因此其 dispatch/combine 时间单列为"
            "「MoE fused dispatch/combine」窗口，既不进 Elementwise，也不计入外部通信。"
            "外部通信单独列出，不与融合窗口相加。"
        ),
    }


# --------------------------------------------------------------------------- #
def memory(prof) -> Dict[str, Any]:
    """Back-compat shim → metrics.memory.compute_memory. The real logic moved to
    memory.py once the 3 memory-level files became ingestible; build.py calls
    compute_memory directly, this keeps any external `core.memory` import working."""
    from .memory import compute_memory
    return compute_memory(prof)


# --------------------------------------------------------------------------- #
# Realistic optimization ceilings for the What-if "现实地板" analysis. These are
# state-of-art engineering priors (achievable, not physical limits); they are
# applied to the *currently loaded* profile's measured slices, so every floor
# below adapts to whatever profiling is loaded — never hardcoded to one sample.
REALISTIC_COMM_OVERLAP = 0.85          # achievable compute–comm overlap (good MoE schedule)
REALISTIC_COMM_OVERLAP_BAND = (0.80, 0.90)
REALISTIC_FREE_RESIDUAL = 0.03         # residual Free as a fraction of step (blocking-off + graph)
REALISTIC_FREE_RESIDUAL_BAND = (0.02, 0.05)


def _single_card(prof) -> bool:
    """Infer training topology without confusing a single-rank shard with one card.

    communication_matrix wraps each
    step as {step: {p2p:{}, collective:{}}}, so emptiness must be tested on the inner
    groups, not the (always-present) outer wrapper."""
    cm = getattr(prof, "communication_matrix", None) or {}
    if not isinstance(cm, dict):
        return not bool(cm)
    for step_v in cm.values():
        if isinstance(step_v, dict):
            if step_v.get("p2p") or step_v.get("collective"):
                return False
        elif step_v:
            return False
    # HCCL collectives prove distributed execution even when this export only
    # contains rank 0 and therefore has an empty peer matrix.
    if any(c.get("type") != "Total" for c in (getattr(prof, "communication", None) or [])):
        return False
    return True


def _whatif_realistic(stage, computing, comm_no, free, comm_total, overlapped,
                      op_reclaim, step_mfu, blocking, single_card, flags, recompute=None):
    """Per-lever 「能否减到 0？不能则能减到多少」 analysis, derived from the LOADED
    profile's measured slices (never the static sample). Floors come from realistic
    ceilings applied to measured values; 失真 caveats are conditioned on this capture's
    blocking / single-card state. Returns a render-ready dict consumed identically by
    the web What-if panel and the shareable report."""
    flags = flags or {}

    def _mfu_at(new_step):
        return (round(min(step_mfu * stage / new_step, 1.0), 4)
                if (step_mfu and new_step and new_step > 0) else None)

    def _exposed_floor(overlap):  # exposed comm left if overlap reaches `overlap`
        return max(0.0, comm_total * (1.0 - overlap))

    levers = []

    # ---- ① 未掩盖通信: overlap up to a realistic ceiling, never to 0 ----
    f_mid = min(comm_no, _exposed_floor(REALISTIC_COMM_OVERLAP))
    f_lo = min(comm_no, _exposed_floor(REALISTIC_COMM_OVERLAP_BAND[0]))   # 0.80 → higher floor
    f_hi = min(comm_no, _exposed_floor(REALISTIC_COMM_OVERLAP_BAND[1]))   # 0.90 → lower floor
    rec_mid, rec_lo, rec_hi = comm_no - f_mid, comm_no - f_lo, comm_no - f_hi
    comm_caveats = []
    if single_card:
        comm_caveats.append("单卡采集（communication_matrix 为空）：集合通信几乎全为等待、"
                            "Transit≈0，多卡真实暴露需以多卡重采为准。")
    if blocking:
        comm_caveats.append("ASCEND_LAUNCH_BLOCKING=1 阻止异步重叠，当前暴露被放大；"
                            "关 blocking 重采后真实暴露更低。")
    overlap_now = _pct(overlapped, comm_total)
    levers.append({
        "id": "comm_overlap", "title": "未掩盖通信", "can_reach_zero": False,
        "scenario": "未掩盖通信 {m:.1f}%→{f:.1f}%".format(
            m=_pct(comm_no, stage), f=_pct(f_mid, stage)),
        "measured_us": round(comm_no, 1), "measured_pct": _pct(comm_no, stage),
        "floor_us": round(f_mid, 1), "floor_pct": _pct(f_mid, stage),
        "recoverable_us": round(rec_mid, 1), "recoverable_pct": _pct(rec_mid, stage),
        "recoverable_lo_us": round(rec_lo, 1), "recoverable_hi_us": round(rec_hi, 1),
        "new_step_us": round(stage - rec_mid, 1), "new_mfu": _mfu_at(stage - rec_mid),
        "floor_basis": "把计算-通信重叠率从 {now}% 提到 ~{tgt:.0f}%（业界可达 {lo:.0f}–{hi:.0f}%）→ "
                       "暴露 = 通信总量 × (1−overlap)，不能到 0。".format(
                           now=overlap_now, tgt=REALISTIC_COMM_OVERLAP * 100,
                           lo=REALISTIC_COMM_OVERLAP_BAND[0] * 100,
                           hi=REALISTIC_COMM_OVERLAP_BAND[1] * 100),
        "methods": [
            "计算-通信重叠：--moe-fb-overlap / --moe-permutation-async-comm，把 dispatch/combine "
            "压到相邻 microbatch 的专家计算之下",
            "缩短通信本身：HCCL 算法/buffsize 调优、合并小通信、EP 规模权衡（EP↓+TP↑）",
            "减少同步点：异步集合通信 + event 依赖替代 stream 同步",
        ],
        "reasons": [
            "MoE 串行依赖链 Router→dispatch→Expert→combine 的首尾无独立计算可掩盖，必落关键路径",
            "可重叠量受并发独立计算与硬件并发上限约束（本例计算总量{rel}通信总量，"
            "瓶颈在依赖与调度而非算力预算）".format(rel=("＞" if computing > comm_total else "≤")),
        ],
        "caveats": comm_caveats,
    })

    # ---- ② 空泡 Free: collapse most of it, but a residual remains ----
    g_mid = min(free, stage * REALISTIC_FREE_RESIDUAL)
    g_lo = min(free, stage * REALISTIC_FREE_RESIDUAL_BAND[1])   # 0.05 → higher floor
    g_hi = min(free, stage * REALISTIC_FREE_RESIDUAL_BAND[0])   # 0.02 → lower floor
    rec2_mid, rec2_lo, rec2_hi = free - g_mid, free - g_lo, free - g_hi
    # floor_basis / methods follow the PROFILING-DERIVED blocking: only blame & propose
    # closing ASCEND_LAUNCH_BLOCKING when it was actually detected. When derive says
    # async (the case on both samples), the cheapest-cut method and the "关 blocking"
    # floor prefix would be a fabricated config assertion — drop them.
    free_caveats = []
    if blocking:
        free_caveats.append("当前 Free 主因是 ASCEND_LAUNCH_BLOCKING=1 的逐算子同步（采集失真）；"
                            "关掉后大部分气泡即塌缩。")
        _free_floor_pre = "关 blocking + 图模式 + 固定 capacity 后"
        _free_methods = [
            "关 ASCEND_LAUNCH_BLOCKING（最大且最廉价的一刀）",
            "图模式 / ACL Graph 下沉、TASK_QUEUE_ENABLE 下发队列降 host 下发压力",
            "固定 MoE capacity / padding，消除动态 shape 的 host 往返",
            "减少 D2H；评估 --swap-optimizer 换入换出代价",
        ]
    else:
        free_caveats.append("采集未开 blocking（profiling 反推）：Free 多为真实下发/同步气泡，按下述方法逐项压缩。")
        _free_floor_pre = "图模式 + 固定 capacity 后"
        _free_methods = [
            "图模式 / ACL Graph 下沉、TASK_QUEUE_ENABLE 下发队列降 host 下发压力",
            "固定 MoE capacity / padding，消除动态 shape 的 host 往返",
            "减少 D2H；评估 --swap-optimizer 换入换出代价",
        ]
    levers.append({
        "id": "free_zero", "title": "空泡 Free", "can_reach_zero": False,
        "scenario": "空泡 Free {m:.1f}%→{f:.1f}%".format(
            m=_pct(free, stage), f=_pct(g_mid, stage)),
        "measured_us": round(free, 1), "measured_pct": _pct(free, stage),
        "floor_us": round(g_mid, 1), "floor_pct": _pct(g_mid, stage),
        "recoverable_us": round(rec2_mid, 1), "recoverable_pct": _pct(rec2_mid, stage),
        "recoverable_lo_us": round(rec2_lo, 1), "recoverable_hi_us": round(rec2_hi, 1),
        "new_step_us": round(stage - rec2_mid, 1), "new_mfu": _mfu_at(stage - rec2_mid),
        "floor_basis": "{pre}，Free 通常落到 step 的 "
                       "~{tgt:.0f}%（{lo:.0f}–{hi:.0f}%），无法归零。".format(
                           pre=_free_floor_pre,
                           tgt=REALISTIC_FREE_RESIDUAL * 100,
                           lo=REALISTIC_FREE_RESIDUAL_BAND[0] * 100,
                           hi=REALISTIC_FREE_RESIDUAL_BAND[1] * 100),
        "methods": _free_methods,
        "reasons": [
            "数据依赖的 host 往返（MoE 路由需在 host 读 token 计数）无法完全消除",
            "流水的填充/排空首尾各有一段空泡",
            "跨流 / event 的真实同步点仍需保留",
        ],
        "caveats": free_caveats,
    })

    # ---- ③ 算子余量: NOT a '→0' lever — compute is useful work ----
    if op_reclaim > 0:
        levers.append({
            "id": "op_ceiling", "title": "算子余量", "can_reach_zero": False,
            "scenario": "算子极致优化（达 MFU 天花板）",
            "measured_us": round(op_reclaim, 1), "measured_pct": _pct(op_reclaim, stage),
            "floor_us": round(computing - op_reclaim, 1),
            "floor_pct": _pct(computing - op_reclaim, stage),
            "recoverable_us": round(op_reclaim, 1), "recoverable_pct": _pct(op_reclaim, stage),
            "recoverable_lo_us": round(op_reclaim, 1), "recoverable_hi_us": round(op_reclaim, 1),
            "new_step_us": round(stage - op_reclaim, 1), "new_mfu": _mfu_at(stage - op_reclaim),
            "floor_basis": "本项非「→0」：计算是有用功，可回收即各算子达 MFU 天花板后的 "
                           "ceiling-relative 余量（已是现实值）。",
            "methods": [
                "将 matmul/FA/FAG 推到各自现实 MFU 天花板（已达标算子不再投入）",
                "访存类（Cast/ZerosLike/TensorMove）靠算子融合 / 内存复用 / 去无谓 dtype 转换",
            ],
            "reasons": [
            "计算地板仅针对已建模部分；融合通信及未知工作量不能按纯计算回收",
                "matmul 已接近天花板，余量主要在 FA/FAG 与访存类尾部",
            ],
            "caveats": [],
        })

    # ---- ④ 重计算: turning recompute off removes the recomputed forward ENTIRELY.
    # Disjoint from 算子余量 — the caller already carved the recomputed kernels' headroom
    # out of op_reclaim (× (1−r_re)) — so it stacks into the combined like the others.
    # Floor can reach 0 when memory allows full-off → recoverable = the whole recompute time.
    if recompute and (recompute.get("us") or 0) > 0:
        rc_us = min(float(recompute["us"]), computing)
        rc_lo = min(float(recompute["us_lo"] if recompute.get("us_lo") is not None else rc_us), computing)
        rc_hi = min(float(recompute["us_hi"] if recompute.get("us_hi") is not None else rc_us), computing)
        levers.append({
            "id": "recompute_off", "title": "重计算", "can_reach_zero": True,
            "scenario": "关闭/减少重计算（反向不重跑前向）",
            "measured_us": round(rc_us, 1), "measured_pct": _pct(rc_us, stage),
            "floor_us": 0.0, "floor_pct": 0.0,
            "recoverable_us": round(rc_us, 1), "recoverable_pct": _pct(rc_us, stage),
            "recoverable_lo_us": round(rc_lo, 1), "recoverable_hi_us": round(rc_hi, 1),
            "new_step_us": round(stage - rc_us, 1), "new_mfu": _mfu_at(stage - rc_us),
            "evidence_level": "PROXY",
            "floor_basis": "重算候选的累计耗时上界；尚未扣除与通信或其它算子的重叠，关键路径收益可能为零。",
            "methods": [
                "显存有余量时减少/关闭重计算层（--recompute-num-layers↓ 或关 full）",
                "选择性重计算（只重算激活大、计算省的算子）做显存↔吞吐平衡",
                "配合 memory 采集确认显存 headroom 再调",
            ],
            "reasons": [
                "重算是为省激活显存而多做的前向，非模型必需功 → 显存够则可全回收",
                "回收上限 = 候选算子反向重跑前向的估计时间（HFU 与 MFU 之差的时间体现）",
            ],
            "caveats": ["关闭重计算抬高激活显存峰值，需先确认显存 headroom（见显存板块），否则 OOM。"],
        })

    # ---- combined realistic floor (sum of disjoint recoverables) ----
    rec_total = sum(l["recoverable_us"] for l in levers)
    rec_total_lo = sum(l["recoverable_lo_us"] for l in levers)   # conservative
    rec_total_hi = sum(l["recoverable_hi_us"] for l in levers)   # optimistic
    combined = {
        "recoverable_us": round(rec_total, 1), "recoverable_pct": _pct(rec_total, stage),
        "new_step_us": round(stage - rec_total, 1), "new_mfu": _mfu_at(stage - rec_total),
        "new_step_lo_us": round(stage - rec_total_lo, 1),   # slower / less optimized
        "new_step_hi_us": round(stage - rec_total_hi, 1),   # faster / more optimized
        "new_mfu_lo": _mfu_at(stage - rec_total_lo),
        "new_mfu_hi": _mfu_at(stage - rec_total_hi),
        "basis": "各项现实地板之和；与物理上界（全部 →0，不可达）的差，即不可消除部分。",
    }
    return {
        "levers": levers,
        "combined": combined,
        "blocking": bool(blocking),
        "single_card": single_card,
        "note": "现实地板 = 业界可达优化上限（重叠 80–90% / Free 残留 2–5% / 算子达 MFU 天花板）"
                "作用于当前加载 profiling 的实测值；与物理上界（全部 →0，不可达）的差即不可消除部分；"
                "失真提示按本次采集的 blocking 与单卡状态自动判定。",
    }


# --------------------------------------------------------------------------- #
# Standard transformer FLOPs split: forward F, backward ≈ 2F (input-grad + weight-
# grad). Let m=fwd/grad−1 be repeated forward passes per model forward (full→1,
# off→0).  With R=bwd/fwd, recomputed FLOPs / executed FLOPs is exactly
# r_re=m/(1+R+m).  A fully modeled SFA fwd/grad pair supplies R; otherwise retain
# the conservative training fallback band R∈[2,2.5].
_RECOMPUTE_BWD_FWD = 2.0
_RECOMPUTE_BWD_FWD_BAND = (2.0, 2.5)


def _recompute_overhead(
    computing: float,
    recompute_fact: Optional[Dict[str, Any]],
    bwd_fwd_flop_ratio: Optional[float] = None,
    recompute_forward_multiplier: Optional[float] = None,
    component_flops: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Time the backward spends RE-running the forward (activation recompute), in μs,
    estimated from modeled component FLOPs when they fully close to the executed
    useful-FLOP total. None when recompute is off/unknown."""
    rf = recompute_fact or {}
    gran = rf.get("value")
    if gran not in ("full", "selective") or not computing or computing <= 0:
        return None
    ratio = rf.get("fwd_grad_ratio")
    multiplier = recompute_forward_multiplier
    if multiplier is None:
        multiplier = rf.get("recompute_forward_multiplier")
    if multiplier is None and ratio and ratio > 0:
        multiplier = max(float(ratio) - 1.0, 0.0)
    if multiplier is None:
        multiplier = 1.0 if gran == "full" else 0.5
    multiplier = max(float(multiplier), 0.0)
    if multiplier <= 0:
        return None
    rho = multiplier / (1.0 + multiplier)

    def _share(R: float) -> float:
        return multiplier / (1.0 + R + multiplier)

    exact_ratio = (
        float(bwd_fwd_flop_ratio)
        if bwd_fwd_flop_ratio is not None and float(bwd_fwd_flop_ratio) > 0
        else None
    )
    components = component_flops or {}
    mm_flops = float(components.get("matmul_flops") or 0.0)
    attn_flops = float(components.get("attention_flops") or 0.0)
    model_flops = float(components.get("model_flops") or 0.0)
    attn_recompute_flops = float(components.get("attention_recompute_flops") or 0.0)
    mm_recompute_explicit = components.get("matmul_recompute_flops")
    mm_recompute_lo_explicit = components.get("matmul_recompute_flops_lo")
    mm_recompute_hi_explicit = components.get("matmul_recompute_flops_hi")
    component_total = mm_flops + attn_flops
    components_close = bool(
        model_flops > 0
        and abs(component_total - model_flops) <= max(model_flops, 1.0) * 1e-9
    )
    component_valid = bool(
        exact_ratio is not None
        and components_close
        and 0 <= attn_recompute_flops <= attn_flops
    )
    if component_valid:
        # Linear/GEMM backward is two matmuls per forward (R=2).  Preserve the
        # established [2,2.5] uncertainty band for non-attention matmuls, while
        # using the exact modeled SFA recompute FLOPs instead of applying SFA's
        # larger backward ratio to every compute kernel.
        explicit_mm_valid = bool(
            mm_recompute_explicit is not None
            and 0 <= float(mm_recompute_explicit) <= mm_flops
        )
        if explicit_mm_valid:
            mm_recompute = float(mm_recompute_explicit)
            mm_recompute_lo = float(
                mm_recompute_lo_explicit
                if mm_recompute_lo_explicit is not None else mm_recompute
            )
            mm_recompute_hi = float(
                mm_recompute_hi_explicit
                if mm_recompute_hi_explicit is not None else mm_recompute
            )
            if not (0 <= mm_recompute_lo <= mm_recompute_hi <= mm_flops):
                explicit_mm_valid = False
        if not explicit_mm_valid:
            mm_recompute = mm_flops * _share(_RECOMPUTE_BWD_FWD)
            mm_recompute_lo = mm_flops * _share(_RECOMPUTE_BWD_FWD_BAND[1])
            mm_recompute_hi = mm_recompute
        r_re = (mm_recompute + attn_recompute_flops) / component_total
        r_lo = (mm_recompute_lo + attn_recompute_flops) / component_total
        r_hi = (mm_recompute_hi + attn_recompute_flops) / component_total
        R = exact_ratio
        ratio_source = (
            "component_weighted_matmul_phase_split_sparse_attention_exact"
            if explicit_mm_valid
            else "component_weighted_matmul_R2_sparse_attention_exact"
        )
    else:
        # A SFA-only ratio must never be projected onto unclosed model FLOPs.
        # Fall back to the standard training band until every modeled component
        # contributing to the MFU numerator has an explicit phase split.
        R = _RECOMPUTE_BWD_FWD
        r_re = _share(R)
        r_lo = _share(_RECOMPUTE_BWD_FWD_BAND[1])
        r_hi = _share(_RECOMPUTE_BWD_FWD_BAND[0])
        ratio_source = ("fallback_training_band_component_mismatch"
                        if exact_ratio is not None else "fallback_training_band")
    return {
        "us": round(computing * r_re, 1),
        "us_lo": round(computing * r_lo, 1),
        "us_hi": round(computing * r_hi, 1),
        "flops_share": round(r_re, 4),                       # also = (HFU − MFU)/HFU
        "flops_share_lo": round(r_lo, 4),
        "flops_share_hi": round(r_hi, 4),
        "rho": round(rho, 4),
        "recompute_forward_multiplier": round(multiplier, 4),
        "bwd_fwd_flop_ratio": round(R, 4),
        "flop_ratio_source": ratio_source,
        "granularity": gran,
        "basis": ("重算开销 = computing × r_re；m={m:.2f}（融合 Attention 调用数推导的"
                  "重复前向倍数）。稀疏 Attention/Indexer 使用建模 FLOPs 与精确调用计数；"
                  "matmul 优先使用可配对线性层的方向/次数相位拆分，未配对部分使用"
                  "R=2（band 2–2.5）。"
                  "SFA R={R:.4f}，source={source}。"
                  ).format(m=multiplier, R=R, source=ratio_source),
    }


# --------------------------------------------------------------------------- #
def theoretical(prof, ov: Dict[str, Any], eff: Dict[str, Any],
                capture: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not ov.get("available"):
        return {"available": False}
    u = ov["us"]
    stage = u["stage"]
    computing = u["computing"]
    comm_no = u["comm_not_overlapped"]
    free = u["free"]

    # Recompute overhead (full activation recompute re-runs the forward before backward),
    # quantified from the FA fwd/grad ratio. r_re = recomputed-FLOPs share of executed
    # FLOPs. Used to (a) strip HFU→MFU and (b) add the 重计算 optimization lever, with
    # 算子余量 carved down by (1−r_re) so the two stay disjoint in the combined.
    rc_state = ((capture or {}).get("capture", {}) or {}).get("recompute")
    attn_flops = eff.get("attention_training_flops") or {}
    matmul_flops = eff.get("matmul_training_flops") or {}
    exact_attn_flops = bool(attn_flops.get("available") and eff.get("efficiency_reliable"))
    component_flops = None
    if exact_attn_flops:
        component_flops = {
            "matmul_flops": eff.get("matmul_flops_total"),
            "attention_flops": eff.get("attention_flops_total"),
            "model_flops": eff.get("useful_flops_total"),
            "attention_recompute_flops": attn_flops.get("recompute_forward_flops"),
            "matmul_recompute_flops": matmul_flops.get("recompute_forward_flops"),
            "matmul_recompute_flops_lo": matmul_flops.get("recompute_forward_flops_lo"),
            "matmul_recompute_flops_hi": matmul_flops.get("recompute_forward_flops_hi"),
        }
    recompute_est = eff.get("recompute_estimate") or {}
    if rc_state and recompute_est.get("available") and not eff.get("efficiency_reliable"):
        r_est = float(recompute_est.get("flops_share") or 0.0)
        r_lo_est = float(recompute_est.get("flops_share_lo") or r_est)
        r_hi_est = float(recompute_est.get("flops_share_hi") or r_est)
        multiplier = float(rc_state.get("recompute_forward_multiplier") or 0.0)
        candidates_us = round(sum(c.get("duration_us", 0)
                                  for c in recompute_est.get("candidates", [])), 1)
        # Two different questions share the "recompute" label and must not be
        # conflated:
        #   footprint   = how much of Computing is replay work (model estimate
        #                 computing × r_re; same one the hidden-overhead ledger uses)
        #   reclaimable = wall time owned by the *identifiable* replay kernels
        #                 (excess forward calls); unidentified families keep theirs,
        #                 so this is a conservative lower bound on the saving.
        # The lever uses the conservative bound, but the granularity classification
        # comes from the caller counts (parser.derive): efficiency's share-based
        # heuristic measures "how much FLOPs", not full-vs-selective, so letting it
        # win made the What-if card say selective while the ledger said full.
        footprint = _recompute_overhead(
            computing,
            rc_state,
            bwd_fwd_flop_ratio=(attn_flops.get("backward_forward_flop_ratio")
                                if exact_attn_flops else None),
            recompute_forward_multiplier=(attn_flops.get("recompute_forward_multiplier")
                                          if exact_attn_flops else None),
            component_flops=component_flops,
        )
        rc_ov = {
            "us": candidates_us,
            "us_lo": 0.0,
            "us_hi": candidates_us,
            "flops_share": round(r_est, 4),
            "flops_share_lo": round(r_lo_est, 4),
            "flops_share_hi": round(r_hi_est, 4),
            "rho": round(multiplier / (1.0 + multiplier), 4) if multiplier > 0 else 0.0,
            "recompute_forward_multiplier": round(multiplier, 4),
            "bwd_fwd_flop_ratio": 2.0,
            "flop_ratio_source": "operator_candidates_plus_counter_calibration",
            "granularity": rc_state.get("value") or recompute_est.get("likely_granularity"),
            # 重算在 Computing 里的全量足迹（与隐性开销桶同口径），仅作对照，不并入收益。
            "footprint_us": footprint.get("us") if footprint else None,
            "footprint_pct": _pct(footprint["us"], stage) if footprint else None,
            "basis": ("可回收上界 = 已识别重放算子（超额 forward 调用）的累计耗时，未识别到的"
                      "模型家族不计入收益，因此是保守值；重算在 Computing 中的全量足迹另见"
                      "隐性开销「重计算」桶（computing × r_re 估算）。"),
            "candidates": recompute_est.get("candidates") or [],
            "confidence": recompute_est.get("confidence") or "medium",
        }
    else:
        rc_ov = _recompute_overhead(
            computing,
            rc_state,
            bwd_fwd_flop_ratio=(attn_flops.get("backward_forward_flop_ratio")
                                if exact_attn_flops else None),
            recompute_forward_multiplier=(attn_flops.get("recompute_forward_multiplier")
                                          if exact_attn_flops else None),
            component_flops=component_flops,
        )
    r_re = rc_ov["flops_share"] if rc_ov else 0.0
    r_re_lo = rc_ov["flops_share_lo"] if rc_ov else 0.0
    r_re_hi = rc_ov["flops_share_hi"] if rc_ov else 0.0
    recompute_us = min(rc_ov["us"], computing) if rc_ov else 0.0

    # Atomic optimization levers (not preset combos): 未掩盖通信 and Free are the two
    # disjoint, independently-removable slices of Stage. Each row reports the gain of
    # enabling *only* that lever; the UI lets the user tick any subset and the combined
    # effect is derived from these (see whatif_combined). A stable `id` keys each lever.
    whatif = [
        {
            "id": "comm_overlap",
            "scenario": "通信完全掩盖（未掩盖通信→0）",
            "new_step_us": round(stage - comm_no, 1),
            "save_us": round(comm_no, 1),
            "save_pct": _pct(comm_no, stage),
            "basis": "把 Communication(Not Overlapped) 全部与计算重叠。",
        },
        {
            "id": "free_zero",
            "scenario": "消除空泡（Free→0）",
            "new_step_us": round(stage - free, 1),
            "save_us": round(free, 1),
            "save_pct": _pct(free, stage),
            "basis": "理想下发与同步，Free 归零（上界估计）。",
        },
    ]

    # 算子极致优化: tune the modeled compute kernels (matmul / FA / FAG) up to their
    # realistic MFU ceiling. The reclaimed time comes from the Computing slice —
    # disjoint from 未掩盖通信 and Free — so this lever stacks with the other two.
    # Kernels already at/above their ceiling are excluded (no further tuning), so the
    # gain is the honest ceiling-relative headroom, not a naive "everything→100%".
    # 算子极致优化 (op_ceiling_opt) 只覆盖公式建模的 matmul/FA/FAG 行，因此它是否
    # 可用与「模型级 FLOP 公式覆盖率」无关 —— KDA/causal-conv 等自定义算子未建模时
    # efficiency_reliable=False（且 counter 校准兜底 estimate），但公式覆盖的算子
    # 本身的 ceiling 收益仍然成立，杠杆不应被整体隐藏。仅在观测峰值与所选芯片矛盾
    # （peak_inconsistent，fail-closed）时禁用。
    oco = (eff.get("op_ceiling_opt")
           if eff.get("available") and not eff.get("peak_inconsistent") else None)
    op_reclaim_full = min(float((oco or {}).get("total_reclaim_us") or 0.0), computing)
    # carve the recomputed kernels' ceiling headroom out of 算子余量 so it is disjoint
    # from the 重计算 lever (which removes those kernels wholesale). op_reclaim_full still
    # feeds the matmul compute-bound footnote below.
    op_reclaim = op_reclaim_full * (1.0 - r_re)
    if oco and op_reclaim > 0:
        cl = oco.get("ceilings") or {}
        def _ceil_pct(x):
            return int(round((x or 0) * 100))
        whatif.append({
            "id": "op_ceiling",
            "scenario": "算子极致优化（计算算子达 MFU 天花板）",
            "new_step_us": round(stage - op_reclaim, 1),
            "save_us": round(op_reclaim, 1),
            "save_pct": _pct(op_reclaim, stage),
            "basis": ("将 matmul/FA/FAG 优化到各自 MFU 天花板（matmul {m}% / FA {a}% / FAG {g}%）；"
                      "已达天花板的 {n} 个算子不再优化。".format(
                          m=_ceil_pct(cl.get("matmul")), a=_ceil_pct(cl.get("attention")),
                          g=_ceil_pct(cl.get("attention_grad")), n=oco.get("n_capped", 0))
                     + ("（已扣除重算算子余量，与「重计算」项不重叠）" if r_re > 0 else "")),
        })

    # 重计算: full/selective recompute re-runs the forward; turning it off removes that
    # whole time. Disjoint from 算子余量 (carved out above), so it joins the combined.
    if recompute_us > 0:
        whatif.append({
            "id": "recompute_off",
            "scenario": "关闭/减少重计算（反向不再重跑前向，需显存余量）",
            "new_step_us": round(stage - recompute_us, 1),
            "save_us": round(recompute_us, 1),
            "save_pct": _pct(recompute_us, stage),
            "basis": rc_ov["basis"],
        })

    # End-to-end (step) MFU + the MFU each what-if would unlock. Useful FLOPs and
    # the silicon peak are constant, so end-to-end MFU scales inversely with step
    # time: new_mfu = step_mfu × (stage / new_step). A shorter step ⇒ higher MFU,
    # which is exactly the payoff of hiding comm / removing bubbles.
    hfu = None
    mfu_estimated = False
    mfu_basis = "formula_flops"
    useful_flops_lo = None
    useful_flops_hi = None
    if eff.get("available"):
        useful_flops = eff.get("useful_flops_total")
        if not useful_flops:
            useful_flops = eff.get("useful_flops_estimated_total")
            useful_flops_lo = eff.get("useful_flops_estimated_lo")
            useful_flops_hi = eff.get("useful_flops_estimated_hi")
            mfu_estimated = bool(useful_flops)
            mfu_basis = "formula_flops_plus_same_capture_counter_calibration"
        peak_tflops = (eff.get("chip") or {}).get("effective_peak_tflops")
        if useful_flops and peak_tflops and stage > 0:
            # executed-FLOPs step rate = HFU — it counts the forward FLOPs the
            # backward RE-RAN under activation recompute (hardware utilization).
            hfu = useful_flops / (peak_tflops * 1e12 * (stage * 1e-6))
    # Strip recompute to get the textbook (model) MFU: recomputed forward FLOPs are
    # hardware work, not model work. MFU = HFU × (1 − r_re). step_mfu (the headline) is
    # now the MODEL MFU; step_hfu carries the executed-FLOPs figure. (rc_ov / r_re were
    # computed at the top so the levers above could use them.)
    step_mfu = (hfu * (1.0 - r_re)) if hfu is not None else None
    if mfu_estimated and useful_flops_lo and useful_flops_hi and peak_tflops and stage > 0:
        denom = peak_tflops * 1e12 * (stage * 1e-6)
        re_lo_flops = float(recompute_est.get("recompute_flops_lo") or 0.0)
        re_hi_flops = float(recompute_est.get("recompute_flops_hi") or 0.0)
        step_mfu_lo = max(float(useful_flops_lo) - re_hi_flops, 0.0) / denom
        step_mfu_hi = max(float(useful_flops_hi) - re_lo_flops, 0.0) / denom
    else:
        step_mfu_lo = (hfu * (1.0 - r_re_hi)) if hfu is not None else None
        step_mfu_hi = (hfu * (1.0 - r_re_lo)) if hfu is not None else None
    # Step-normalized model MAC: the numerator is accumulated MAC-active time over
    # GEMM + attention kernels; the denominator is the full training Stage, so
    # exposed communication and Free/bubbles lower it just as they lower MFU/HFU.
    model_mac_active_us = (
        eff.get("model_mac_active_us") if eff.get("available") else None
    )
    model_mac_counter_coverage_pct = eff.get("model_mac_counter_coverage_pct")
    model_mac_insufficient_coverage = bool(
        model_mac_counter_coverage_pct is None
        or float(model_mac_counter_coverage_pct) < 90.0
    )
    model_mac_ratio_raw = (
        float(model_mac_active_us) / stage
        if model_mac_active_us is not None and stage > 0 else None
    )
    model_mac_inconsistent = bool(
        model_mac_ratio_raw is not None and model_mac_ratio_raw > 1.05
    )
    model_mac_ratio = (
        None
        if model_mac_inconsistent or model_mac_insufficient_coverage
        else model_mac_ratio_raw
    )
    # Time-only levers (comm/free/op) keep useful FLOPs fixed, so post-opt MFU scales
    # as mfu × stage/new_step — the same factor for MFU or HFU.
    physical_floor_us = (step_mfu * stage) if step_mfu else 0.0
    for w in whatif:
        raw_new_step = max(stage - float(w.get("save_us") or 0.0), 0.0)
        new_step = max(raw_new_step, physical_floor_us) if step_mfu else raw_new_step
        if new_step > raw_new_step + 1.0:
            w["raw_new_step_us"] = round(raw_new_step, 1)
            w["capped_by_physical_mfu"] = True
        else:
            w["capped_by_physical_mfu"] = False
        w["new_step_us"] = round(new_step, 1)
        w["new_mfu"] = (round(min(step_mfu * stage / new_step, 1.0), 4)
                        if (step_mfu and new_step > 0) else None)

    # Combined what-if: every lever enabled at once. Since the levers are disjoint
    # slices of Stage their savings add, and the floor is pure Computing. This is the
    # true upper bound and the default for the UI's "已启用组合" row (all ticked).
    raw_combined_save_us = comm_no + free + op_reclaim + recompute_us
    raw_combined_step_us = max(stage - raw_combined_save_us, 0.0)
    combined_step_us = (max(raw_combined_step_us, physical_floor_us)
                        if step_mfu else raw_combined_step_us)
    combined_save_us = max(stage - combined_step_us, 0.0)
    combined_capped = combined_step_us > raw_combined_step_us + 1.0
    combined_parts = ["通信掩盖", "空泡"]
    if op_reclaim > 0:
        combined_parts.append("算子极致优化")
    if recompute_us > 0:
        combined_parts.append("关闭重计算")
    whatif_combined = {
        "save_us": round(combined_save_us, 1),
        "save_pct": _pct(combined_save_us, stage),
        "raw_save_us": round(raw_combined_save_us, 1),
        "raw_save_pct": _pct(raw_combined_save_us, stage),
        "new_step_us": round(combined_step_us, 1),
        "raw_new_step_us": round(raw_combined_step_us, 1),
        "physical_floor_us": round(physical_floor_us, 1) if physical_floor_us else None,
        "capped_by_physical_mfu": combined_capped,
        "new_mfu": (round(min(step_mfu * stage / combined_step_us, 1.0), 4)
                    if (step_mfu and combined_step_us > 0) else None),
        "basis": "当前可建模优化项叠加（" + " / ".join(combined_parts) + "，收益按不重叠口径相加）。",
    }

    matmul_mfu = eff.get("matmul_mfu") if eff.get("available") else None
    compute_bound_note = None
    if eff.get("peak_inconsistent"):
        chipinfo = eff.get("chip", {})
        compute_bound_note = {
            "peak_inconsistent": True,
            "peak_underestimated": False,
            "matmul_mfu_pct": None,
            "ideal_matmul_us": None,
            "headroom_us": None,
            "assumed_peak_tflops": chipinfo.get("peak_bf16_tflops"),
            "observed_peak_tflops": chipinfo.get("observed_peak_tflops"),
            "hint": "观测吞吐超过所选芯片/精度峰值；MFU 与算子 What-if 已停用，请核对芯片、精度和 shape 语义。",
        }
    if matmul_mfu:
        if matmul_mfu > 1.0:
            # achieved exceeds the assumed peak -> peak is underestimated, not a
            # real >100% utilization. Report the flag instead of a negative headroom.
            compute_bound_note = {
                "matmul_mfu_pct": round(matmul_mfu * 100, 2),
                "peak_underestimated": True,
                "ideal_matmul_us": None,
                "headroom_us": None,
                "hint": "实测算力超过假设峰值 → ChipSpec 峰值偏低，请按实际 SKU 调整。",
            }
        else:
            # Ceiling-aware headroom: the reclaimable compute time is what's left
            # after tuning matmul/FA/FAG up to their MFU ceilings (not to 100%), so
            # this number matches the 算子极致优化 lever exactly instead of implying
            # a fictitious gap that the lever would never chase.
            chipinfo = eff.get("chip", {})
            compute_bound_note = {
                "matmul_mfu_pct": round(matmul_mfu * 100, 2),
                "peak_underestimated": False,
                "ideal_matmul_us": round(computing - op_reclaim_full, 1),
                "headroom_us": round(op_reclaim_full, 1),
                "ceiling_based": True,
                "calibrated": bool(chipinfo.get("calibrated")),
                "assumed_peak_tflops": chipinfo.get("peak_bf16_tflops"),
                "observed_peak_tflops": chipinfo.get("observed_peak_tflops"),
            }

    # Realistic「能否减到 0 / 现实地板」analysis — derived from THIS profile's measured
    # slices + capture state (blocking / single-card), not the static sample.
    # blocking comes from the PROFILING-DERIVED capture fact (parser.derive), never a
    # launch script; mirror engine.py so the free_zero lever's caveats/methods/floor
    # follow the same truth as the rule cards. env is only a legacy fallback (always
    # empty under derive — it never asserts ASCEND_LAUNCH_BLOCKING).
    cap = capture or {}
    _cap_state = cap.get("capture", {}) or {}
    blocking = bool((_cap_state.get("blocking", {}) or {}).get("value")) if _cap_state else \
        ((cap.get("env") or {}).get("ASCEND_LAUNCH_BLOCKING") == "1")
    sc_fact = (_cap_state.get("single_card", {}) or {}).get("value")
    single_card = sc_fact if isinstance(sc_fact, bool) else _single_card(prof)
    realistic = _whatif_realistic(
        stage, computing, comm_no, free,
        u.get("communication", 0.0) or 0.0, u.get("overlapped", 0.0) or 0.0,
        op_reclaim, step_mfu, blocking, single_card, cap.get("flags"), recompute=rc_ov)

    # Recompute summary (for the rule card + the MFU/HFU foot). The optimization itself
    # is now a first-class lever inside whatif / realistic.levers (id=recompute_off),
    # counted in the combined; 算子余量 was carved by (1−r_re) to keep them disjoint.
    recompute = None
    if rc_ov:
        rc_save = recompute_us
        _new_step = max(stage - rc_save, 1.0)
        recompute = {
            "id": "recompute_off",
            "granularity": rc_ov["granularity"],
            "scenario": "关闭/减少重计算（反向不再重跑前向）",
            "overhead_us": rc_ov["us"],
            "overhead_us_lo": rc_ov["us_lo"],
            "overhead_us_hi": rc_ov["us_hi"],
            "overhead_pct": _pct(rc_ov["us"], stage),
            # 足迹口径（与 hidden_overhead 桶一致）：仅当该分支用「可识别重放耗时」
            # 作为收益上界时才有值，供卡片说明两者区别。
            "footprint_us": rc_ov.get("footprint_us"),
            "footprint_pct": rc_ov.get("footprint_pct"),
            "rho": rc_ov["rho"],
            "flops_share": rc_ov["flops_share"],
            "flops_share_lo": rc_ov["flops_share_lo"],
            "flops_share_hi": rc_ov["flops_share_hi"],
            "recompute_forward_multiplier": rc_ov["recompute_forward_multiplier"],
            "bwd_fwd_flop_ratio": rc_ov["bwd_fwd_flop_ratio"],
            "flop_ratio_source": rc_ov["flop_ratio_source"],
            "confidence": rc_ov.get("confidence", "medium"),
            "candidates": rc_ov.get("candidates") or [],
            "save_us": round(rc_save, 1),
            "save_pct": _pct(rc_save, stage),
            "new_step_us": round(_new_step, 1),
            "new_mfu": (round(step_mfu * stage / _new_step, 4) if step_mfu else None),
            "hfu": round(hfu, 4) if hfu else None,
            "mfu": round(step_mfu, 4) if step_mfu else None,
            "mfu_lo": round(step_mfu_lo, 4) if step_mfu_lo else None,
            "mfu_hi": round(step_mfu_hi, 4) if step_mfu_hi else None,
            "basis": rc_ov["basis"],
            "note": "已作为「重计算」项纳入 What-if 优化组合（与算子余量 disjoint：算子余量已扣除重算算子余量）；"
                    "关闭需显存有余量（见显存板块）。",
        }

    return {
        "available": True,
        "current_step_us": round(stage, 1),
        # `stage` is the complete capture window.  For a multi-step export the
        # per-step figure is a display normalization only; all numerators and
        # What-if slices remain on the same window denominator.
        "window_step_count": int(ov.get("step_count") or 1),
        "average_step_us": round(
            stage / max(int(ov.get("step_count") or 1), 1), 1
        ),
        "whatif": whatif,
        "whatif_combined": whatif_combined,
        "realistic": realistic,
        "compute_bound": compute_bound_note,
        # step_mfu = MODEL MFU (recompute stripped); step_hfu = executed-FLOPs (HFU).
        "step_mfu": round(step_mfu, 4) if step_mfu else None,
        "step_mfu_lo": round(step_mfu_lo, 4) if step_mfu_lo else None,
        "step_mfu_hi": round(step_mfu_hi, 4) if step_mfu_hi else None,
        "step_hfu": round(hfu, 4) if hfu else None,
        "step_hfu_basis": "model_execution_excluding_optimizer",
        "mfu_estimated": mfu_estimated,
        "mfu_basis": mfu_basis,
        "mfu_confidence": (
            (eff.get("mfu_estimate") or {}).get("confidence")
            if mfu_estimated else "high"
        ),
        "mfu_opinion": (
            ("MFU 为 PROXY 量级参考；先核对融合算子工作量和阶段归属"
             if mfu_estimated and (eff.get("fused_moe") or {}).get("available")
             else "端到端 MFU 偏低，重计算与未掩盖通信是主要拖累"
             if step_mfu is not None and step_mfu < 0.30
             else "端到端 MFU 中等，仍应优先处理重计算与通信暴露"
             if step_mfu is not None and step_mfu < 0.50
             else "端到端 MFU 较高，优化重点转向尾部通信与局部算子")
            if step_mfu is not None else None
        ),
        "model_mac_ratio": (
            round(model_mac_ratio, 4) if model_mac_ratio is not None else None
        ),
        "model_mac_active_us": model_mac_active_us,
        "model_mac_denominator_us": round(stage, 1),
        "model_mac_counter_coverage_pct": model_mac_counter_coverage_pct,
        "model_mac_insufficient_coverage": model_mac_insufficient_coverage,
        "model_mac_inconsistent": model_mac_inconsistent,
        "recompute": recompute,
        "note": ("端到端 MFU=模型有效FLOPs/(峰值×完整step)（不含重计算和优化器工作量）；此处 HFU 为模型执行 FLOPs 利用率，含重算、不含优化器。"
                 "稀疏 Attention/Indexer 按前后向调用计数拆分；matmul 优先按线性层方向/次数拆分，"
                 "无法配对的部分不设重算点估计，仅保留敏感性上界；"
                 "主导算子 FLOP 公式不完整时，使用同一采集内已建模算子校准 MAC/Vector 周期代理，"
                 "输出 PROXY MFU(est)，区间为假设敏感性而非统计置信区间；未建模算子的 Roofline 收益仍不臆测。"
                 "峰值口径不一致时才停用 MFU。"),
    }
