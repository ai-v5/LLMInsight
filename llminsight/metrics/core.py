"""Core table/number metrics derived from the CSVs + communication.json."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import SETTINGS
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
    r = st.iloc[0]
    computing = _f(r.get("Computing"))
    comm_no = _f(r.get("Communication(Not Overlapped)"))
    overlapped = _f(r.get("Overlapped"))
    communication = _f(r.get("Communication"))
    free = _f(r.get("Free"))
    stage = _f(r.get("Stage")) or (computing + comm_no + free)
    bubble = _f(r.get("Bubble"))
    preparing = _f(r.get("Preparing"))

    return {
        "available": True,
        "step": _i(r.get("Step")),
        "device_id": _i(r.get("Device_id")),
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
            "step_time_s": round(stage / 1e6, 4),
        },
    }


# --------------------------------------------------------------------------- #
def hotspots(prof) -> Dict[str, Any]:
    op = prof.op_statistic
    if op.empty:
        return {"available": False}
    df = op.copy()
    df["Total Time(us)"] = num(df["Total Time(us)"])
    df["Count"] = num(df["Count"])
    total = df["Total Time(us)"].sum()

    ops = [
        {
            "type": str(row["OP Type"]),
            "core": str(row.get("Core Type", "")),
            "count": int(row["Count"]),
            "total_us": round(_f(row["Total Time(us)"]), 1),
            "avg_us": round(_f(row.get("Avg Time(us)")), 2),
            "max_us": round(_f(row.get("Max Time(us)")), 2),
            "ratio": round(_f(row.get("Ratio(%)")), 3),
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
    }


# --------------------------------------------------------------------------- #
def communication(prof) -> Dict[str, Any]:
    comms = [c for c in prof.communication if c["type"] != "Total"]
    reported_total = next((c for c in prof.communication if c["type"] == "Total"), None)
    if not comms:
        return {"available": False}

    by_type: Dict[str, Dict[str, Any]] = {}
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
        t["transit_mb"] += sum(l.get("transit_mb", 0) for l in c["links"].values())
        t["_wait_ratio_sum"] += min(c.get("wait_ratio", 0) or 0, 1.0)

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

    total_elapse = sum(c["elapse_ms"] for c in comms)
    total_wait = sum(c["wait_ms"] for c in comms)
    mean_wait_ratio = sum(min(c.get("wait_ratio", 0) or 0, 1.0) for c in comms) / len(comms)
    total_transit_mb = sum(
        sum(l.get("transit_mb", 0) for l in c["links"].values()) for c in comms
    )
    total_transit_ms = sum(c["transit_ms"] for c in comms)
    overall_bandwidth_gbps = (round(total_transit_mb / total_transit_ms, 1)
                              if total_transit_ms > 0 else None)
    overall_bandwidth_with_wait_gbps = (round(total_transit_mb / total_elapse, 1)
                                        if total_transit_mb > 0 and total_elapse > 0
                                        else None)

    def _op_transit_mb(c):
        return sum(l.get("transit_mb", 0) for l in c["links"].values())

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

    # msprof: device-level effective-transfer vs cross-rank-wait split (None on
    # torch_npu, whose communication.json already carries Transit/Wait per op).
    breakdown = ((getattr(prof, "meta", {}) or {}).get("msprof") or {}).get("comm_breakdown")

    if total_transit_ms > 0:
        note = ("平均带宽(含等待) = Σ Transit Size ÷ Σ Elapse；"
                "有效带宽(去等待) = Σ Transit Size ÷ Σ Transit Time（MB/ms≡GB/s）。"
                "Wait/Synchronization 为卡间等待，不计入去等待带宽分母。")
    else:
        note = ("本采集 Transit≈0：communication.json 无可用传输字节，平均/去等待带宽均 N/A；"
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
        # overall 有效带宽 (None when transit≈0); per-category in by_type[].bandwidth_gbps
        "overall_bandwidth_gbps": overall_bandwidth_gbps,
        "overall_bandwidth_with_wait_gbps": overall_bandwidth_with_wait_gbps,
        "by_type": type_rows,
        "top": top_out,
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


def hidden_overhead(prof, ov: Dict[str, Any], capture: Dict[str, Any] = None) -> Dict[str, Any]:
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
                      "单卡下几乎全为 Wait、Transit≈0 → 这是「未掩盖通信」的算子视角，与 Communication 为同一段时间，"
                      "已从 Device 合计中剔除以免与未掩盖通信重复计入。",
            "source": "op_statistic (AI_CPU)",
            "suggestion": "本质是通信而非下发延迟：优先掩盖到计算下（moe-fb-overlap / 异步通信），再缩短通信本身（HCCL 算法/buffsize、EP 规模）。",
        },
        {
            "key": "comm_not_overlapped", "domain": "device",
            "label": "通信未掩盖 (Not Overlapped)",
            "us": round(comm_no, 1),
            "detail": f"未掩盖通信 {comm_no:,.0f}us 直接计入 step；集合通信 Wait 总量 {comm_wait_ms*1000.0:,.0f}us（其余与计算重叠）；kernel Wait {kernel_wait_us:,.0f}us",
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
            "key": "format_init", "domain": "device",
            "label": "格式转换 / 内存初始化 (Cast/ZerosLike/TensorMove)",
            "us": round(cast_init, 1),
            "detail": f"device 侧 Cast/ZerosLike/TensorMove/Concat/Slice {cast_init:,.0f}us",
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
                "（aclnnMaskedSelect 等）→ 固定 expert capacity / 对路由结果 padding，合并或减少同步点。"
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
    if recompute in ("full", "selective"):
        _rc_kind = "Full" if recompute == "full" else "Selective"
        # Quantify the recompute time (computing × r_re from the FA ratio). domain=
        # config + additive:False → a lens, NOT summed into device/host (it already
        # lives inside Computing).
        _computing = ov["us"]["computing"] if ov.get("available") else 0.0
        _rc_ov = _recompute_overhead(_computing, cap_state.get("recompute"))
        if _rc_ov:
            _rc_detail = (
                "profiling 反推重计算={g}（FA 前向/反向次数比）→ 反向重跑前向。"
                "估算重算耗时 ≈ {us:,.0f}us（step 的 {pct}%，band {lo:,.0f}–{hi:,.0f}us）"
                "= computing × {sh:.0%}；覆盖全部前向计算算子，非仅 FA。"
            ).format(g=recompute, us=_rc_ov["us"], pct=_pct(_rc_ov["us"], stage),
                     lo=_rc_ov["us_lo"], hi=_rc_ov["us_hi"], sh=_rc_ov["flops_share"])
        else:
            _rc_detail = (f"profiling 反推重计算={recompute}（FA 前向/反向次数比）→ 反向重跑前向；"
                          "缺前向/反向计数，未能量化耗时。")
        buckets.append({
            "key": "recompute", "domain": "config", "additive": False,
            "label": f"重计算开销 ({_rc_kind} Recompute)",
            "us": _rc_ov["us"] if _rc_ov else None,
            "detail": _rc_detail,
            "source": "profiling 反推 (parser.derive) + 1:2:1 估算",
            "suggestion": "评估「选择性重计算 / 减少重计算层」做显存↔耗时平衡；量化收益见 What-if 重计算项。",
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
        "note": (
            "host 与 device 时间不可直接相加（部分并发/被 blocking 放大）。device 桶与 step 同口径可比；"
            "host 桶反映下发/同步压力。AICPU 集合通信执行与「未掩盖通信」为同一段时间，仅作算子视角展示、"
            "不并入 Device 合计（避免重复计入）。用于「相对量级与归因」。"
        ),
    }


# --------------------------------------------------------------------------- #
_MODULE_RULES = [
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
    elementwise = 0.0
    seen = set()
    for module, types in _MODULE_RULES:
        for t in types:
            mask = df["OP Type"].astype(str) == t
            assigned[module] += _f(df[mask]["Total Time(us)"].sum())
            seen.update(df[mask].index.tolist())
    for idx, row in df.iterrows():
        if idx not in seen:
            elementwise += _f(row["Total Time(us)"])

    # --- device compute ring (consistent device-time basis) ---
    modules = [{"module": m, "us": round(v, 1)} for m, v in assigned.items() if v > 0]
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
    moe_focus = {
        "expert_compute_us": round(moe_compute, 1),
        "dispatch_comm_us": round(comm_moe, 1),
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
            "LM-Head 共用未细分。通信为 wall-clock（多为等待），单独列出不并入计算环。"
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
                "计算是有用功，地板 = 有效 FLOPs / 峰值算力，永远 >0",
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
        rc_lo = min(float(recompute.get("us_lo") or rc_us), computing)   # conservative (smaller)
        rc_hi = min(float(recompute.get("us_hi") or rc_us), computing)   # optimistic (larger)
        levers.append({
            "id": "recompute_off", "title": "重计算", "can_reach_zero": True,
            "scenario": "关闭/减少重计算（反向不重跑前向）",
            "measured_us": round(rc_us, 1), "measured_pct": _pct(rc_us, stage),
            "floor_us": 0.0, "floor_pct": 0.0,
            "recoverable_us": round(rc_us, 1), "recoverable_pct": _pct(rc_us, stage),
            "recoverable_lo_us": round(rc_lo, 1), "recoverable_hi_us": round(rc_hi, 1),
            "new_step_us": round(stage - rc_us, 1), "new_mfu": _mfu_at(stage - rc_us),
            "floor_basis": "重算 = 反向重跑前向的额外计算（≈computing×{:.0%}，覆盖全部前向算子）；"
                           "显存允许时可完全关闭 → 全部回收。".format(recompute.get("flops_share") or 0),
            "methods": [
                "显存有余量时减少/关闭重计算层（--recompute-num-layers↓ 或关 full）",
                "选择性重计算（只重算激活大、计算省的算子）做显存↔吞吐平衡",
                "配合 memory 采集确认显存 headroom 再调",
            ],
            "reasons": [
                "重算是为省激活显存而多做的前向，非模型必需功 → 显存够则可全回收",
                "回收上限 = 反向重跑前向的实测时间（HFU 与 MFU 之差的时间体现）",
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
# grad). Full activation recompute reruns the forward before the backward (+F). The
# measured FlashAttention fwd/grad ratio gives ρ=(fwd-grad)/fwd — HOW MUCH forward is
# recomputed (full→0.5, off→0, selective between). Recomputed FLOPs as a fraction of
# EXECUTED FLOPs: r_re = 2ρ/(1+R+2ρ), R=bwd/fwd. R=2 (textbook) with full ρ=0.5 → 1/4,
# i.e. the classic 1:2:1 fwd:bwd:recompute split. Band over R∈[2,2.5].
_RECOMPUTE_BWD_FWD = 2.0
_RECOMPUTE_BWD_FWD_BAND = (2.0, 2.5)


def _recompute_overhead(computing: float,
                        recompute_fact: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Time the backward spends RE-running the forward (activation recompute), in μs,
    estimated from the measured FA fwd/grad ratio applied to the WHOLE compute slice
    (every recomputed forward op, not just FA). None when recompute is off/unknown."""
    rf = recompute_fact or {}
    gran = rf.get("value")
    if gran not in ("full", "selective") or not computing or computing <= 0:
        return None
    fa_fwd = rf.get("fa_fwd")
    fa_grad = rf.get("fa_grad")
    ratio = rf.get("fwd_grad_ratio")
    if fa_fwd and fa_grad is not None and fa_fwd > 0:
        rho = max(0.0, (fa_fwd - fa_grad) / fa_fwd)          # measured recompute share
    elif ratio and ratio > 0:
        rho = max(0.0, (ratio - 1.0) / ratio)                # (fwd-grad)/fwd from the ratio
    else:
        rho = 0.5 if gran == "full" else 0.25                # fallback when raw counts absent
    if rho <= 0:
        return None

    def _share(R: float) -> float:                           # recomputed FLOPs / executed FLOPs
        return (2.0 * rho) / (1.0 + R + 2.0 * rho)
    r_re = _share(_RECOMPUTE_BWD_FWD)
    r_lo = _share(_RECOMPUTE_BWD_FWD_BAND[1])                 # larger R → smaller share (conservative)
    r_hi = _share(_RECOMPUTE_BWD_FWD_BAND[0])                 # smaller R → larger share (optimistic)
    return {
        "us": round(computing * r_re, 1),
        "us_lo": round(computing * r_lo, 1),
        "us_hi": round(computing * r_hi, 1),
        "flops_share": round(r_re, 4),                       # also = (HFU − MFU)/HFU
        "rho": round(rho, 4),
        "granularity": gran,
        "basis": ("重算开销 = computing × r_re；r_re = 2ρ/(1+R+2ρ)，"
                  "ρ={rho:.2f}（FA 实测前向重算占比），R=反向/前向 FLOPs≈2（band 2–2.5）。"
                  "full(ρ≈0.5) → r_re≈¼（前向:反向:重算≈1:2:1）；覆盖全部前向计算算子，非仅 FA。"
                  ).format(rho=rho),
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
    rc_ov = _recompute_overhead(computing, rc_state)
    r_re = rc_ov["flops_share"] if rc_ov else 0.0
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
    oco = (eff.get("op_ceiling_opt")
           if eff.get("available") and eff.get("efficiency_reliable") else None)
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
    if eff.get("available"):
        useful_flops = eff.get("useful_flops_total")
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
            "rho": rc_ov["rho"],
            "flops_share": rc_ov["flops_share"],
            "save_us": round(rc_save, 1),
            "save_pct": _pct(rc_save, stage),
            "new_step_us": round(_new_step, 1),
            "new_mfu": (round(step_mfu * stage / _new_step, 4) if step_mfu else None),
            "hfu": round(hfu, 4) if hfu else None,
            "mfu": round(step_mfu, 4) if step_mfu else None,
            "basis": rc_ov["basis"],
            "note": "已作为「重计算」项纳入 What-if 优化组合（与算子余量 disjoint：算子余量已扣除重算算子余量）；"
                    "关闭需显存有余量（见显存板块）。",
        }

    return {
        "available": True,
        "current_step_us": round(stage, 1),
        "whatif": whatif,
        "whatif_combined": whatif_combined,
        "realistic": realistic,
        "compute_bound": compute_bound_note,
        # step_mfu = MODEL MFU (recompute stripped); step_hfu = executed-FLOPs (HFU).
        "step_mfu": round(step_mfu, 4) if step_mfu else None,
        "step_hfu": round(hfu, 4) if hfu else None,
        "recompute": recompute,
        "note": ("端到端 MFU=模型理论FLOPs/(峰值×step)（不含重计算）；HFU 含重计算重复执行的 FLOPs。"
                 "当主导算子 FLOP 模型不完整或峰值口径不一致时，MFU/HFU 与算子收益保持 unavailable；"
                 "通信、空泡和重计算仍按各自证据单独估算。"),
    }
