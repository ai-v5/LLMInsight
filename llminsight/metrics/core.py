"""Core table/number metrics derived from the CSVs + communication.json."""
from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd

from ..config import SETTINGS
from ..parser.profile import num


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _pct(x: float, total: float) -> float:
    return round(100.0 * x / total, 2) if total else 0.0


# --------------------------------------------------------------------------- #
def overview(prof) -> Dict[str, Any]:
    st = prof.step_trace
    if st.empty:
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
        "step": int(_f(r.get("Step"))),
        "device_id": int(_f(r.get("Device_id"))),
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

    total_elapse = sum(c["elapse_ms"] for c in comms)
    total_wait = sum(c["wait_ms"] for c in comms)
    mean_wait_ratio = sum(min(c.get("wait_ratio", 0) or 0, 1.0) for c in comms) / len(comms)
    total_transit_mb = sum(
        sum(l.get("transit_mb", 0) for l in c["links"].values()) for c in comms
    )

    top = sorted(comms, key=lambda c: c["elapse_ms"], reverse=True)[:15]
    top_out = [
        {
            "name": c["name"].split("@")[0],
            "type": c["type"],
            "elapse_ms": round(c["elapse_ms"], 3),
            "wait_ms": round(c["wait_ms"], 3),
            "wait_ratio": round(c["wait_ratio"], 3),
        }
        for c in top
    ]

    return {
        "available": True,
        "count": len(comms),
        "total_elapse_ms": round(total_elapse, 2),
        "total_wait_ms": round(total_wait, 2),
        "overall_wait_pct": round(mean_wait_ratio * 100, 1),
        "total_transit_mb": round(total_transit_mb, 3),
        "by_type": type_rows,
        "top": top_out,
        "note": (
            "单卡采集：集合通信几乎全部为 Wait/Synchronization，Transit≈0、带宽≈0 —— "
            "通信时间以「等待对端」为主，需结合多卡数据才能看真实链路带宽。"
        ),
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


def hidden_overhead(prof, ov: Dict[str, Any]) -> Dict[str, Any]:
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

    buckets = [
        {
            "key": "aicpu_dispatch", "domain": "device",
            "label": "AICPU 通信下发 (HcclLaunchAicpuKernel)",
            "us": round(aicpu_dispatch, 1),
            "detail": f"device 侧 AI_CPU 通信下发合计 {aicpu_dispatch:,.0f}us，占 step {_pct(aicpu_dispatch, stage)}%",
            "source": "op_statistic",
            "suggestion": "EP64 alltoall 启动密集 → 调 HCCL buffsize/通信算法、减少 dispatch 次数、合并下发。",
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
            "suggestion": "ASCEND_LAUNCH_BLOCKING=1 放大了同步开销（采集干扰项）；正式训练应关闭。",
        },
        {
            "key": "dynamic_shape", "domain": "host",
            "label": "动态 shape 抖动 (MaskedSelect/NonZero)",
            "us": round(dyn["time_us"], 1),
            "detail": f"host 累计 {dyn['time_us']:,.0f}us，单次 max {dyn['max_us']:,.0f}us（方差大）",
            "source": "api_statistic",
            "suggestion": "MoE 路由/掩码导致 host 重编译/同步 → 固定 capacity / padding。",
        },
        {
            "key": "recompute", "domain": "config",
            "label": "重计算开销 (Full Recompute)",
            "us": None,
            "detail": "训练脚本启用 --recompute-granularity full（uniform, 1 层）→ 反向重跑前向。本次采集未单独标注，估算见理论分析。",
            "source": "训练脚本配置",
            "suggestion": "评估「选择性重计算 / 减少重计算层」做显存↔耗时平衡。",
        },
    ]
    device_total = sum(b["us"] for b in buckets if b["domain"] == "device" and isinstance(b["us"], (int, float)))
    host_total = sum(b["us"] for b in buckets if b["domain"] == "host" and isinstance(b["us"], (int, float)))
    return {
        "available": True,
        "buckets": buckets,
        "device_total_us": round(device_total, 1),
        "host_total_us": round(host_total, 1),
        "stage_us": stage,
        "note": (
            "host 与 device 时间不可直接相加（部分并发/被 blocking 放大）。device 桶与 step 同口径可比；"
            "host 桶反映下发/同步压力。用于「相对量级与归因」。"
        ),
    }


# --------------------------------------------------------------------------- #
_MODULE_RULES = [
    ("MoE-Experts", ["GroupedMatmul", "SwiGlu", "SwiGluGrad", "ScatterAdd",
                      "InplaceIndexAdd", "ScatterElementsV2", "GatherElements"]),
    ("MoE-Router", ["TopKV2", "Sigmoid", "SigmoidGrad", "ArgMaxWithValue", "Sort",
                     "Cumsum", "ReduceSum", "LpNormV2"]),
    ("Attention-MLA", ["FlashAttentionScore", "FlashAttentionScoreGrad",
                        "RotaryPositionEmbedding", "RotaryPositionEmbeddingGrad"]),
    ("Norm", ["RmsNorm", "RmsNormGrad"]),
    ("Optimizer", ["ApplyAdamWV2", "ApplyAdamW"]),
    ("Embedding/Loss", ["GatherV2", "EmbeddingDenseGradV2", "Exp", "Log"]),
    ("GEMM/Projections (shared)", ["MatMulV3", "GemmV3", "MatMul", "BatchMatMul"]),
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
            "按算子命名启发式归因，基于 device 计算时间；GEMM/Projections 为 MLA 投影 / Router / "
            "LM-Head 共用未细分。通信为 wall-clock（多为等待），单独列出不并入计算环。"
        ),
    }


# --------------------------------------------------------------------------- #
def memory(prof) -> Dict[str, Any]:
    m = SETTINGS.model
    init_ops = _op_time(prof.op_statistic, ["ZerosLike", "TensorMove", "Fill", "OnesLike"])
    return {
        "available": False,
        "hbm_timeline_available": False,
        "reason": "本次采集仅含 AI Core Freq counter，无 memory-level 采集（memory_record.csv / npu_module_mem.csv 缺失）→ 无法绘制真实 HBM 峰值时间线。",
        "config_tradeoffs": [
            {
                "feature": "--recompute-granularity full (uniform, 1 层)",
                "effect": "省激活显存，代价是反向重跑前向 → 增加计算耗时。",
                "advice": "显存不紧张时改选择性重计算 / 减少重计算层，换取吞吐。",
            },
            {
                "feature": "--swap-optimizer",
                "effect": "优化器状态在 HBM↔Host 间换入换出，省 HBM、代价是 H2D/D2H 拷贝与同步。",
                "advice": "若 PCIe/同步成为瓶颈，评估关闭 swap 或仅 swap 部分状态。",
            },
        ],
        "init_overhead_us": round(init_ops, 1),
        "ai_core_freq_mhz": 1650,
        "note": "显存峰值/构成分解待接入 memory-level 采集；当前给出配置驱动的内存-时间权衡顾问。",
    }


# --------------------------------------------------------------------------- #
def theoretical(prof, ov: Dict[str, Any], eff: Dict[str, Any]) -> Dict[str, Any]:
    if not ov.get("available"):
        return {"available": False}
    u = ov["us"]
    stage = u["stage"]
    computing = u["computing"]
    comm_no = u["comm_not_overlapped"]
    free = u["free"]

    whatif = [
        {
            "scenario": "通信完全掩盖（未掩盖通信→0）",
            "new_step_us": round(computing + free, 1),
            "save_us": round(comm_no, 1),
            "save_pct": _pct(comm_no, stage),
            "basis": "把 Communication(Not Overlapped) 全部与计算重叠。",
        },
        {
            "scenario": "消除空泡（Free→0）",
            "new_step_us": round(computing + comm_no, 1),
            "save_us": round(free, 1),
            "save_pct": _pct(free, stage),
            "basis": "理想下发与同步，Free 归零（上界估计）。",
        },
        {
            "scenario": "通信掩盖 + 空泡减半",
            "new_step_us": round(computing + free * 0.5, 1),
            "save_us": round(comm_no + free * 0.5, 1),
            "save_pct": _pct(comm_no + free * 0.5, stage),
            "basis": "组合优化的综合估计。",
        },
    ]

    matmul_mfu = eff.get("matmul_mfu") if eff.get("available") else None
    compute_bound_note = None
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
            ideal_compute_us = computing * matmul_mfu
            chipinfo = eff.get("chip", {})
            compute_bound_note = {
                "matmul_mfu_pct": round(matmul_mfu * 100, 2),
                "peak_underestimated": False,
                "ideal_matmul_us": round(ideal_compute_us, 1),
                "headroom_us": round(computing - ideal_compute_us, 1),
                "calibrated": bool(chipinfo.get("calibrated")),
                "assumed_peak_tflops": chipinfo.get("peak_bf16_tflops"),
                "observed_peak_tflops": chipinfo.get("observed_peak_tflops"),
            }

    # End-to-end (step) MFU: total executed matmul+attention FLOPs over the FULL
    # step wall-clock against the (calibrated) silicon peak. Unlike matmul_mfu —
    # which divides by matmul *kernel* time and reads ~cube quality — this divides
    # by the whole step, so 未掩盖通信 + 空泡 directly drag it down. It is the number
    # one would quote as "训练 MFU". Same single-step basis as compute_bound above.
    step_mfu = None
    if eff.get("available"):
        useful_flops = eff.get("useful_flops_total")
        peak_tflops = (eff.get("chip") or {}).get("effective_peak_tflops")
        if useful_flops and peak_tflops and stage > 0:
            step_mfu = useful_flops / (peak_tflops * 1e12 * (stage * 1e-6))

    return {
        "available": True,
        "current_step_us": round(stage, 1),
        "whatif": whatif,
        "compute_bound": compute_bound_note,
        "step_mfu": round(step_mfu, 4) if step_mfu else None,
        "note": "What-if 为基于 step 时间构成的上界估算，用于优化排序，非精确预测。芯片峰值为假设值。",
    }
