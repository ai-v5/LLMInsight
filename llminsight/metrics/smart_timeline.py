"""Smart timeline: profiler-style operator Gantt + utilization lanes.

Replicates the LLMperf timeline paradigm in ECharts: device/communication trace
slices are laid out as horizontal bars across **stream lanes** (Cube / Flash-
Attention / Vector / MIX / Communication / other), and three time-aligned
utilization lanes (compute / HBM bandwidth / communication) are stacked below.
显存容量 / 主机内存 are surfaced as "待采集" placeholders (need memory_record.csv).

Two layers, mirroring the chip-switch design elsewhere:
  * chip-independent **geometry** — one streaming pass over trace_view.json,
    cached on disk by the trace signature (so chip switches never re-scan the
    104MB trace). Stores per-bin Σflops / Σbytes / comm-coverage and the
    down-sampled slice list (full kernel name + stream + start/dur).
  * cheap chip-dependent **overlay** — turns Σflops/Σbytes into utilization %
    against the (calibrated) peak / HBM bandwidth, and attaches a representative
    MFU/MBU/dtype per slice by joining the trace name to efficiency's
    `kernel_index`. Recomputed on every chip switch; the geometry cache is reused.

Trace slices carry only a name (no shapes), so MFU/MBU/FLOPs are best-effort
per-name averages from kernel_index — surfaced as reference values, "—" when a
slice has no model (communication, AI_CPU dispatch).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..cache import cached_json, file_signature
from ..config import SETTINGS
from ..parser.trace import iter_events, event_ts_us
from .efficiency import MATMUL_TYPES, ATTENTION_TYPES

# stream key -> (label, color). Colors borrowed from LLMperf's lane palette.
STREAMS = [
    ("cube",       "Cube / AI_CORE",  "#1565C0"),
    ("flash_attn", "FlashAttention",  "#6A1B9A"),
    ("vector",     "Vector / AI_VECTOR", "#2E7D32"),
    ("mix",        "MIX",             "#00838F"),
    ("comm",       "Communication",   "#C62828"),
    ("other",      "其它 / 下发",      "#757575"),
]
DEVICE_PROC = "Ascend Hardware"
COMM_PROC = "Communication"

_GEOM_VERSION = "v1"  # bump when geometry/stream logic changes (cache invalidation)


def _stream_from_meta(typ: Optional[str], core: Optional[str]) -> str:
    """Lane for a device slice that matched kernel_index (has Type / Core)."""
    t = typ or ""
    c = (core or "").upper()
    if t in ATTENTION_TYPES:
        return "flash_attn"
    if t in MATMUL_TYPES or "CUBE" in c or "AI_CORE" in c or "AICORE" in c:
        return "cube"
    if "VECTOR" in c or "AIV" in c:
        return "vector"
    if "MIX" in c:
        return "mix"
    if t.lower().startswith(("hccl", "hcom")):
        return "comm"
    return "other"


def _stream_from_name(name: str) -> str:
    """Heuristic lane for a device slice with no kernel_index match (rare —
    mostly AI_CPU dispatch ops, which efficiency.py excludes from the index)."""
    n = (name or "").lower()
    if "flashattention" in n or "fusedinferattention" in n or \
            "promptflashattention" in n or "increflashattention" in n:
        return "flash_attn"
    if "matmul" in n or "gemm" in n or "batchmatmul" in n or "bmm" in n:
        return "cube"
    if n.startswith(("hcom", "hccl")) or "allreduce" in n or "allgather" in n or \
            "reducescatter" in n or "alltoall" in n or "broadcast" in n:
        return "comm"
    return "other"


def _build_geometry(prof, kindex: Dict[str, Any]) -> Dict[str, Any]:
    """Chip-independent geometry: one streaming pass over the trace.

    `kindex` supplies Type/Core/dtype/FLOPs/bytes per kernel name. Only the
    chip-independent fields are used here, so the cached result is valid for any
    selected chip.
    """
    path = prof.trace_path
    if not path:
        return {"available": False, "reason": "trace_view.json missing"}

    bins = SETTINGS.timeline_bins
    max_slices = SETTINGS.timeline_max_slices

    # pass 1: process_name map + global time span
    pid_name: Dict[Any, str] = {}
    t0 = float("inf")
    t1 = 0.0
    for ev in iter_events(path):
        ph = ev.get("ph")
        if ph == "M":
            if ev.get("name") == "process_name":
                args = ev.get("args", {}) or {}
                pid_name[ev.get("pid")] = args.get("name")
        elif ph == "X":
            ts = event_ts_us(ev)
            if ts:
                t1 = max(t1, ts + float(ev.get("dur") or 0.0))
                t0 = min(t0, ts)
    if t0 == float("inf") or t1 <= t0:
        return {"available": False, "reason": "no timed events"}

    span_us = t1 - t0
    bin_us = span_us / bins
    name_to_pid = {v: k for k, v in pid_name.items()}
    dev_pid = name_to_pid.get(DEVICE_PROC)
    comm_pid = name_to_pid.get(COMM_PROC)

    flops_sum = [0.0] * bins
    bytes_sum = [0.0] * bins
    comm_occ = [0.0] * bins

    def spread(arr: List[float], s: float, e: float, total: float) -> None:
        """Distribute `total` across the time bins [s,e] proportionally."""
        s = max(s, t0)
        e = min(e, t1)
        if e <= s:
            return
        dur = e - s
        b0 = max(0, min(int((s - t0) / bin_us), bins - 1))
        b1 = max(0, min(int((e - t0) / bin_us), bins - 1))
        if b0 == b1:
            arr[b0] += total
            return
        arr[b0] += total * (((t0 + (b0 + 1) * bin_us) - s) / dur)
        mid = total * (bin_us / dur)
        for b in range(b0 + 1, b1):
            arr[b] += mid
        arr[b1] += total * ((e - (t0 + b1 * bin_us)) / dur)

    # pass 2: classify device / communication slices
    slices: List[Dict[str, Any]] = []
    total_dev = 0
    matched_dev = 0
    modeled_dev_us = 0.0
    total_dev_us = 0.0

    for ev in iter_events(path):
        if ev.get("ph") != "X":
            continue
        pid = ev.get("pid")
        is_dev = pid == dev_pid
        is_comm = pid == comm_pid
        if not (is_dev or is_comm):
            continue
        ts = event_ts_us(ev)
        dur = float(ev.get("dur") or 0.0)
        if not ts or dur <= 0:
            continue
        name = str(ev.get("name") or "")
        ki = kindex.get(name)

        if is_comm:
            stream = "comm"
            typ = ki["type"] if ki else None
            core = ki["core"] if ki else None
            dtype = ki["dtype"] if ki else None
            spread(comm_occ, ts, ts + dur, dur)
        else:
            total_dev += 1
            total_dev_us += dur
            if ki:
                matched_dev += 1
                typ, core, dtype = ki["type"], ki["core"], ki["dtype"]
                stream = _stream_from_meta(typ, core)
                if ki.get("flops") is not None:
                    modeled_dev_us += dur
                    spread(flops_sum, ts, ts + dur, ki["flops"])
                if ki.get("bytes") is not None:
                    spread(bytes_sum, ts, ts + dur, ki["bytes"])
            else:
                typ = core = dtype = None
                stream = _stream_from_name(name)

        slices.append({
            "name": name,           # FULL name (kept for the chip-overlay join)
            "stream": stream,
            "type": typ,
            "core": core,
            "dtype": dtype,
            "start_ms": (ts - t0) / 1e3,
            "dur_ms": dur / 1e3,
        })

    if not slices:
        return {"available": False,
                "reason": "no device / communication slices in trace"}

    # down-sample: keep the longest bars (sub-bin slices are visually invisible),
    # then restore chronological order for the Gantt.
    total_slices = len(slices)
    if total_slices > max_slices:
        slices.sort(key=lambda s: s["dur_ms"], reverse=True)
        slices = slices[:max_slices]
    slices.sort(key=lambda s: s["start_ms"])
    for s in slices:
        s["start_ms"] = round(s["start_ms"], 4)
        s["dur_ms"] = round(s["dur_ms"], 4)

    return {
        "available": True,
        "t0_us": t0,
        "span_us": round(span_us, 1),
        "span_s": round(span_us / 1e6, 4),
        "bins": bins,
        "bin_us": round(bin_us, 3),
        "slices": slices,
        "flops_sum": [round(v, 1) for v in flops_sum],
        "bytes_sum": [round(v, 1) for v in bytes_sum],
        "comm_occ": [round(v, 3) for v in comm_occ],
        "total_slices": total_slices,
        "shown_slices": len(slices),
        "matched_dev": matched_dev,
        "total_dev": total_dev,
        "modeled_dev_us": round(modeled_dev_us, 1),
        "total_dev_us": round(total_dev_us, 1),
    }


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1.0 else x)


def _apply_chip(geom: Dict[str, Any], eff: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap chip-dependent overlay: utilization % + per-slice MFU/MBU."""
    if not geom.get("available"):
        return dict(geom)

    chip = (eff or {}).get("chip", {}) or {}
    effective_peak = (chip.get("effective_peak_tflops") or 0.0) * 1e12
    hbm_bw = (chip.get("hbm_tbps") or 0.0) * 1e12
    kindex = (eff or {}).get("kernel_index", {}) or {}

    bins = geom["bins"]
    bin_us = geom["bin_us"]
    bin_s = bin_us * 1e-6
    flops_sum = geom["flops_sum"]
    bytes_sum = geom["bytes_sum"]
    comm_occ = geom["comm_occ"]

    if effective_peak > 0 and bin_s > 0:
        compute_series = [round(_clamp01(flops_sum[b] / (bin_s * effective_peak)), 4)
                          for b in range(bins)]
    else:
        compute_series = [0.0] * bins
    if hbm_bw > 0 and bin_s > 0:
        hbm_series = [round(_clamp01(bytes_sum[b] / (bin_s * hbm_bw)), 4)
                      for b in range(bins)]
    else:
        hbm_series = [0.0] * bins
    comm_series = ([round(_clamp01(comm_occ[b] / bin_us), 4) for b in range(bins)]
                   if bin_us > 0 else [0.0] * bins)

    # per-slice representative MFU/MBU (by name) + truncated display name
    out_slices: List[Dict[str, Any]] = []
    for s in geom["slices"]:
        ki = kindex.get(s["name"])
        nm = s["name"]
        out_slices.append({
            "name": nm if len(nm) <= 90 else nm[:88] + "…",
            "stream": s["stream"],
            "type": s["type"],
            "core": s["core"],
            "dtype": s["dtype"],
            "start_ms": s["start_ms"],
            "dur_ms": s["dur_ms"],
            "mfu": ki.get("mfu") if ki else None,
            "mbu": ki.get("mbu") if ki else None,
        })

    # "已建模占比" = matmul/attention compute time / wall-clock span — the same
    # story the compute-utilization lane tells (NOT / Σ device-slice time, which is
    # inflated many-fold by parallel device streams). Clamp for the rare case where
    # attention + GEMM overlap on distinct cores pushes modeled time past wall-clock.
    modeled_pct = (round(min(geom["modeled_dev_us"] / geom["span_us"] * 100, 100.0), 1)
                   if geom.get("span_us") else None)

    utilization = [
        {"key": "compute", "label": "算力 (Cube)", "available": True, "unit": "%",
         "color": "#3fb6e0", "series": compute_series,
         "note": "Σ建模FLOPs /（桶时长 × 有效峰值）"},
        {"key": "hbm_bw", "label": "显存带宽 (HBM)", "available": True, "unit": "%",
         "color": "#d29922", "series": hbm_series,
         "note": "Σ读写字节 /（桶时长 × HBM 带宽）"},
        {"key": "comm", "label": "通信 (HCCL)", "available": True, "unit": "%",
         "color": "#f85149", "series": comm_series,
         "note": "Communication 泳道每桶时间覆盖率"},
        {"key": "hbm_cap", "label": "显存容量 (HBM)", "available": False,
         "reason": "待 memory_record.csv 采集（与「显存洞察」页口径一致）"},
        {"key": "host_mem", "label": "主机内存", "available": False,
         "reason": "待 memory_record.csv 采集"},
    ]

    return {
        "available": True,
        "t0_us": geom["t0_us"],
        "span_us": geom["span_us"],
        "span_s": geom["span_s"],
        "bins": bins,
        "bin_us": bin_us,
        "lane_count": len(STREAMS),
        "streams": [{"key": k, "label": l, "color": c} for (k, l, c) in STREAMS],
        "slices": out_slices,
        "utilization": utilization,
        "modeled_pct": modeled_pct,
        "total_slices": geom["total_slices"],
        "shown_slices": geom["shown_slices"],
        "note": (
            "算子按 stream 泳道铺成 Gantt；悬停显示 名称/类型/Core/start/dur，"
            "命中按名 join 的 kernel_index 时追加 MFU/MBU/dtype（按名平均，参考值）。"
            "利用率：算力 = Σ建模FLOPs /（桶 × 有效峰值），显存带宽 = Σ字节 /（桶 × HBM 带宽），"
            "通信 = 每桶时间覆盖率。显存容量 / 主机内存待 memory_record.csv 采集。"
            f"切片共 {geom['total_slices']} 个，按时长下采样保留最长 {geom['shown_slices']} 个。"
        ),
    }


def compute_smart_timeline(prof, eff: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point: geometry (cached by trace signature) + chip overlay."""
    if not getattr(prof, "trace_path", None):
        return {"available": False, "reason": "trace_view.json missing"}
    if not eff or not eff.get("available"):
        return {"available": False,
                "reason": "kernel_details.csv missing（无法按名 join 算子）"}
    kindex = eff.get("kernel_index", {}) or {}
    sig = file_signature(prof.trace_path)
    key = (f"smarttl:{_GEOM_VERSION}:{sig}:"
           f"{SETTINGS.timeline_bins}:{SETTINGS.timeline_max_slices}")
    geom = cached_json(key, lambda: _build_geometry(prof, kindex))
    return _apply_chip(geom, eff)
