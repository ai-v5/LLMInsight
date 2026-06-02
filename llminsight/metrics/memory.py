"""Memory-level metrics (H8): HBM peak timeline, fragmentation decomposition,
per-module & per-operator allocation — built from the 3 optional memory files
(memory_record.csv / npu_module_mem.csv / operator_memory.csv).

These files only exist when the profiler ran at memory level. When they are
absent the section degrades gracefully to available:False, preserving the
config-driven recompute/swap advisor so the single-card OLD sample (and the
verify.py regression baselines) stay byte-for-byte unchanged.

Units: Ascend reports "MB" as MiB (PyTorch / torch_npu convention), so capacity
headroom converts via 1 MB = 2**20 bytes against the chip's decimal-GB capacity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import SETTINGS
from ..parser.profile import num
from .core import _op_time

_MB = 1024.0 * 1024.0  # bytes per reported "MB" (== MiB)

# Driver modules that are structurally always-zero noise in npu_module_mem.csv —
# we surface only modules that actually reserve HBM.
_MODULE_EPS_MB = 1.0


# --------------------------------------------------------------------------- #
def _tradeoffs() -> List[Dict[str, str]]:
    """Config-driven recompute/swap advisor (shared by both code paths so the
    unavailable fallback is unchanged from the historic core.memory output)."""
    return [
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
    ]


def _unavailable(prof) -> Dict[str, Any]:
    init_ops = _op_time(prof.op_statistic, ["ZerosLike", "TensorMove", "Fill", "OnesLike"])
    return {
        "available": False,
        "hbm_timeline_available": False,
        "reason": "本次采集无 memory-level 数据（memory_record.csv / npu_module_mem.csv 缺失）→ 无法绘制真实 HBM 峰值时间线。",
        "config_tradeoffs": _tradeoffs(),
        "init_overhead_us": round(init_ops, 1),
        "ai_core_freq_mhz": 1650,
        "note": "显存峰值/构成分解待接入 memory-level 采集；当前给出配置驱动的内存-时间权衡顾问。",
    }


# --------------------------------------------------------------------------- #
def _downsample(ts: pd.Series, val: pd.Series, t0: float, span_us: float,
                bins: int) -> List[List[float]]:
    """Peak-preserving downsample to <=bins points. x = seconds from t0, y = MB.
    Bins by equal-width time buckets and keeps the MAX in each so plateaus/peaks
    survive; returns monotonic-x [[t_s, mb], ...]."""
    df = pd.DataFrame({"t": ts, "v": val}).dropna()
    if df.empty:
        return []
    if span_us <= 0 or len(df) <= bins:
        df = df.sort_values("t")
        return [[round((t - t0) / 1e6, 4), round(v, 1)] for t, v in zip(df["t"], df["v"])]
    width = span_us / bins
    b = ((df["t"] - t0) / width).astype(int).clip(0, bins - 1)
    g = df.assign(_b=b).groupby("_b").agg(t=("t", "mean"), v=("v", "max"))
    g = g.sort_index()
    return [[round((t - t0) / 1e6, 4), round(v, 1)] for t, v in zip(g["t"], g["v"])]


def _alloc_component(comps: set) -> Optional[str]:
    """The framework caching-allocator component (live + pool). Prefer PTA+GE."""
    for c in ("PTA+GE", "PTA"):
        if c in comps:
            return c
    return None


def _modules(prof) -> List[Dict[str, Any]]:
    nm = prof.npu_module_mem
    if nm is None or nm.empty or "Component" not in nm.columns \
            or "Total Reserved(MB)" not in nm.columns:
        return []
    df = nm.copy()
    df["_r"] = num(df["Total Reserved(MB)"])
    by = df.groupby("Component")["_r"].max().dropna()
    by = by[by >= _MODULE_EPS_MB].sort_values(ascending=False)
    out = [{"module": str(k), "mb": round(float(v), 1)} for k, v in by.items()]
    total = sum(o["mb"] for o in out) or 1.0
    for o in out:
        o["pct"] = round(100.0 * o["mb"] / total, 1)
    return out


def _operator_memory(prof) -> Dict[str, Any]:
    om = prof.operator_memory
    if om is None or om.empty or "Name" not in om.columns or "Size(KB)" not in om.columns:
        return {"top_allocators": [], "longest_lived": []}
    df = om.copy()
    df["_kb"] = num(df["Size(KB)"])
    life = None
    if "Active Duration(us)" in df.columns:
        life = num(df["Active Duration(us)"])
    if "Duration(us)" in df.columns:
        d = num(df["Duration(us)"])
        life = d if life is None else life.fillna(d)
    df["_life"] = life if life is not None else 0.0

    g = df.groupby("Name")["_kb"].agg(["sum", "count", "max"]).sort_values(
        "sum", ascending=False).head(8)
    top_allocators = [
        {"name": str(name), "total_mb": round(float(r["sum"]) / 1024.0, 1),
         "count": int(r["count"]), "max_mb": round(float(r["max"]) / 1024.0, 1)}
        for name, r in g.iterrows()
    ]

    # Longest-lived tensors that actually hold memory (>=1 MB) — these pin HBM.
    big = df[df["_kb"] >= 1024.0].sort_values("_life", ascending=False).head(5)
    longest_lived = [
        {"name": str(r["Name"]), "life_s": round(float(r["_life"]) / 1e6, 3),
         "mb": round(float(r["_kb"]) / 1024.0, 1)}
        for _, r in big.iterrows()
    ]
    return {"top_allocators": top_allocators, "longest_lived": longest_lived}


# --------------------------------------------------------------------------- #
def compute_memory(prof) -> Dict[str, Any]:
    mr = getattr(prof, "memory_record", None)
    if mr is None or mr.empty or "Total Reserved(MB)" not in mr.columns \
            or "Timestamp(us)" not in mr.columns:
        return _unavailable(prof)

    df = mr.copy()
    df["_ts"] = num(df["Timestamp(us)"])
    df["_res"] = num(df["Total Reserved(MB)"])
    df["_alloc"] = num(df["Total Allocated(MB)"]) if "Total Allocated(MB)" in df.columns \
        else pd.Series(dtype=float)
    df = df.dropna(subset=["_ts"])
    if df.empty:
        return _unavailable(prof)

    comps = set(df["Component"].astype(str)) if "Component" in df.columns else set()
    t0 = float(df["_ts"].min())
    t1 = float(df["_ts"].max())
    span_us = t1 - t0

    peak_reserved = float(df["_res"].max()) if df["_res"].notna().any() else 0.0
    peak_allocated = float(df["_alloc"].max()) if df["_alloc"].notna().any() else 0.0

    # Allocator pool (PTA / PTA+GE) reserved — torch caching-allocator footprint.
    alloc_comp = _alloc_component(comps)
    pool_reserved = 0.0
    alloc_series = df
    if alloc_comp and "Component" in df.columns:
        sub = df[df["Component"].astype(str) == alloc_comp]
        if not sub.empty:
            alloc_series = sub
            if sub["_res"].notna().any():
                pool_reserved = float(sub["_res"].max())
    if pool_reserved <= 0:
        pool_reserved = peak_allocated  # degenerate: no separate pool reading

    # Process-reserved (whole-HBM footprint) — APP component if present, else global.
    proc_series = df
    if "Component" in df.columns and "APP" in comps:
        app = df[df["Component"].astype(str) == "APP"]
        if not app.empty and app["_res"].notna().any():
            proc_series = app
            peak_reserved = max(peak_reserved, float(app["_res"].max()))

    # Decomposition (peak-basis approximation — each term at its own peak).
    alloc_slack = max(pool_reserved - peak_allocated, 0.0)          # allocator cache slack
    nontensor = max(peak_reserved - pool_reserved, 0.0)            # comm/workspace/runtime
    fragmentation = max(peak_reserved - peak_allocated, 0.0)       # total reserved-but-not-live

    # Capacity headroom (MiB vs decimal-GB chip capacity).
    cap_bytes = float(SETTINGS.chip.hbm_capacity_bytes or 0.0)
    cap_mib = cap_bytes / _MB if cap_bytes > 0 else 0.0
    peak_reserved_bytes = peak_reserved * _MB
    util_pct = round(100.0 * peak_reserved_bytes / cap_bytes, 1) if cap_bytes > 0 else None
    headroom_mb = round(cap_mib - peak_reserved, 1) if cap_mib > 0 else None
    near_oom = bool(util_pct is not None and util_pct >= 90.0)

    modules = _modules(prof)
    hccl_mb = next((o["mb"] for o in modules if o["module"].upper() == "HCCL"), 0.0)
    op_mem = _operator_memory(prof)

    bins = int(getattr(SETTINGS, "timeline_bins", 600) or 600)
    timeline = {
        "reserved": _downsample(proc_series["_ts"], proc_series["_res"], t0, span_us, bins),
        "allocated": _downsample(alloc_series["_ts"], alloc_series["_alloc"], t0, span_us, bins),
        "capacity_mb": round(cap_mib, 1) if cap_mib > 0 else None,
    }

    init_ops = _op_time(prof.op_statistic, ["ZerosLike", "TensorMove", "Fill", "OnesLike"])

    return {
        "available": True,
        "hbm_timeline_available": True,
        "summary": {
            "peak_reserved_mb": round(peak_reserved, 1),
            "peak_reserved_gib": round(peak_reserved / 1024.0, 2),
            "peak_allocated_mb": round(peak_allocated, 1),
            "peak_allocated_gib": round(peak_allocated / 1024.0, 2),
            "pool_reserved_mb": round(pool_reserved, 1),
            "alloc_slack_mb": round(alloc_slack, 1),
            "nontensor_reserved_mb": round(nontensor, 1),
            "fragmentation_mb": round(fragmentation, 1),
            "fragmentation_pct": round(100.0 * fragmentation / peak_reserved, 1) if peak_reserved else None,
            "hccl_reserved_mb": round(float(hccl_mb), 1),
            "capacity_gb": round(cap_bytes / 1e9, 1) if cap_bytes > 0 else None,
            "capacity_mib": round(cap_mib, 1) if cap_mib > 0 else None,
            "util_pct": util_pct,
            "headroom_mb": headroom_mb,
            "headroom_gib": round(headroom_mb / 1024.0, 2) if headroom_mb is not None else None,
            "near_oom": near_oom,
            "span_s": round(span_us / 1e6, 3),
            "samples": int(len(df)),
        },
        "timeline": timeline,
        "modules": modules,
        "top_allocators": op_mem["top_allocators"],
        "longest_lived": op_mem["longest_lived"],
        "config_tradeoffs": _tradeoffs(),
        "init_overhead_us": round(init_ops, 1),
        "ai_core_freq_mhz": 1650,
        "note": (
            "已保留=进程 HBM 占用（计入容量），已分配=活跃张量；二者之差为「保留未占用」"
            "（分配器缓存碎片 + 通信/workspace/运行时保留）。峰值口径为各项各自峰值的近似。"
        ),
    }
