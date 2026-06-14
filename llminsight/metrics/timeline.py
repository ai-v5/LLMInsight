"""Trace-derived timeline: lane occupancy bins + overlap segments + top slices.

Computed by streaming trace_view.json. Result is small (JSON ~100KB) and cached
on disk keyed by the trace file signature so server restarts are instant.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..cache import cached_json, file_signature
from ..config import SETTINGS
from ..parser.trace import iter_events, event_ts_us

# pid process_name -> our lane label
LANE_LABELS = {
    "Ascend Hardware": "Device (Ascend Hardware)",
    "Communication": "Communication",
    "CANN": "Host Runtime (CANN)",
    "Python": "Framework (Python)",
}
OVERLAP_LANE = "Overlap Analysis"

# The top-slice table should surface real compute kernels, not host/device sync
# barriers. DAVID_EVENT_WAIT / NOTIFY_WAIT / *_RECORD / stream-sync ops dominate raw
# duration but carry no optimization signal — drop them so matmul/FlashAttention/comm
# kernels (previously crowded out of the top 60) rise to the top.
_SYNC_RE = re.compile(r"WAIT|NOTIFY|EVENT_RECORD|BARRIER|STREAM", re.I)


def _build(prof) -> Dict[str, Any]:
    path = prof.trace_path
    if not path:
        return {"available": False, "reason": "trace_view.json missing"}

    bins = SETTINGS.timeline_bins

    # pass 1: process_name map + global time span
    pid_name: Dict[Any, str] = {}
    tid_name: Dict[Any, str] = {}
    t0 = float("inf")
    t1 = 0.0
    for ev in iter_events(path):
        ph = ev.get("ph")
        if ph == "M":
            args = ev.get("args", {}) or {}
            if ev.get("name") == "process_name":
                pid_name[ev.get("pid")] = args.get("name")
            elif ev.get("name") == "thread_name":
                tid_name[(ev.get("pid"), ev.get("tid"))] = args.get("name")
        elif ph == "X":
            ts = event_ts_us(ev)
            dur = float(ev.get("dur") or 0.0)
            if ts:
                t0 = min(t0, ts)
                t1 = max(t1, ts + dur)
    if t0 == float("inf") or t1 <= t0:
        return {"available": False, "reason": "no timed events"}

    span_us = t1 - t0
    bin_us = span_us / bins

    name_to_pid = {v: k for k, v in pid_name.items()}
    lane_pids = {label: name_to_pid.get(proc) for proc, label in LANE_LABELS.items()}
    overlap_pid = name_to_pid.get(OVERLAP_LANE)

    lane_occ: Dict[str, List[float]] = {lbl: [0.0] * bins for lbl in LANE_LABELS.values()}
    pid_to_label = {pid: lbl for lbl, pid in lane_pids.items() if pid is not None}

    overlap_tracks = ("Computing", "Communication", "Communication(Not Overlapped)", "Free")
    overlap_occ: Dict[str, List[float]] = {t: [0.0] * bins for t in overlap_tracks}
    top_slices: List[Dict[str, Any]] = []
    freq: List[Dict[str, Any]] = []

    def add_occ(arr: List[float], s: float, e: float):
        s = max(s, t0)
        e = min(e, t1)
        if e <= s:
            return
        b0 = int((s - t0) / bin_us)
        b1 = int((e - t0) / bin_us)
        b0 = max(0, min(b0, bins - 1))
        b1 = max(0, min(b1, bins - 1))
        if b0 == b1:
            arr[b0] += e - s
            return
        # first partial
        arr[b0] += (t0 + (b0 + 1) * bin_us) - s
        for b in range(b0 + 1, b1):
            arr[b] += bin_us
        arr[b1] += e - (t0 + b1 * bin_us)

    # pass 2: aggregate
    for ev in iter_events(path):
        ph = ev.get("ph")
        if ph == "X":
            pid = ev.get("pid")
            label = pid_to_label.get(pid)
            ts = event_ts_us(ev)
            dur = float(ev.get("dur") or 0.0)
            if label and ts and dur > 0:
                add_occ(lane_occ[label], ts, ts + dur)
                if label.startswith("Device") and dur > 1500:
                    nm = str(ev.get("name"))
                    if not _SYNC_RE.search(nm):
                        top_slices.append(
                            {"name": nm[:60], "lane": "Device",
                             "start_us": round(ts - t0, 1), "dur_us": round(dur, 1)}
                        )
            if pid == overlap_pid:
                track = tid_name.get((pid, ev.get("tid")), "")
                if track in overlap_tracks and ts and dur > 0:
                    add_occ(overlap_occ[track], ts, ts + dur)
        elif ph == "C":
            nm = ev.get("name") or ""
            if "Freq" in nm and "Die 0" in nm:
                args = ev.get("args", {}) or {}
                t_us = event_ts_us(ev) - t0
                # counter samples often predate the first ph="X" kernel (negative
                # t_us) and were stretching the value-axis to a symmetric [-4,4]s.
                # Keep only samples inside the kernel span so the axis can pin to it.
                if 0 <= t_us <= span_us:
                    freq.append({"t_us": round(t_us, 1), "mhz": args.get("MHz", 0)})

    # normalize occupancy to [0,1] (per-bin busy fraction; multi-stream overlap clamps)
    for lbl in lane_occ:
        lane_occ[lbl] = [round(min(v / bin_us, 1.0), 3) for v in lane_occ[lbl]]
    # The four Overlap tracks are NOT mutually exclusive: "Communication" (total) runs
    # concurrently with "Computing" when comm is hidden, so all four sum to ~1.3-2.0.
    # The three that DO partition the step are Computing + Communication(Not Overlapped)
    # + Free (sum ~1) — that triple is what the frontend stacks as a true 100% band.
    for trk in overlap_occ:
        overlap_occ[trk] = [round(min(v / bin_us, 1.0), 3) for v in overlap_occ[trk]]

    def _avg_pct(arr: List[float]) -> float:
        return round(100.0 * sum(arr) / len(arr), 1) if arr else 0.0

    computing_pct = _avg_pct(overlap_occ["Computing"])
    not_overlapped_pct = _avg_pct(overlap_occ["Communication(Not Overlapped)"])
    free_pct = _avg_pct(overlap_occ["Free"])

    top_slices.sort(key=lambda s: s["dur_us"], reverse=True)
    top_slices = top_slices[:60]
    if len(freq) > 240:
        step = len(freq) // 240
        freq = freq[::step]

    return {
        "available": True,
        "t0_us": t0,
        "span_us": round(span_us, 1),
        "span_s": round(span_us / 1e6, 4),
        "bins": bins,
        "bin_us": round(bin_us, 1),
        "lanes": [{"label": lbl, "occupancy": lane_occ[lbl]} for lbl in lane_occ],
        "overlap_bins": [{"track": t, "occupancy": overlap_occ[t]} for t in overlap_tracks],
        "computing_pct": computing_pct,
        "not_overlapped_pct": not_overlapped_pct,
        "free_pct": free_pct,
        "top_slices": top_slices,
        "ai_core_freq": freq,
        "note": "热力图：绿=满载（主机流常驻属正常），暖色/红=占用骤降的空泡（可优化）。Overlap 时间条把每个时间桶按 有效计算(Computing)/未掩盖通信/Free 三段拆分（三者合计=步长 100%）——红=未掩盖通信、琥珀=Free 空泡，越多越值得优化；已掩盖通信隐含在 Computing 墙钟内，单独见上方 Communication 泳道。三图共享同一时间轴可纵向对照。切片表已过滤同步等待项（WAIT/NOTIFY/…），仅保留真实计算 kernel。",
    }


def compute_timeline(prof) -> Dict[str, Any]:
    # msprof captures have no trace_view.json; build the timeline from the SQLite
    # slice/counter tables instead (same output shape).
    if (getattr(prof, "meta", {}) or {}).get("format") == "msprof":
        from .msprof_timeline import compute_msprof_timeline
        return compute_msprof_timeline(prof)
    if not prof.trace_path:
        return {"available": False, "reason": "trace_view.json missing"}
    sig = file_signature(prof.trace_path)
    # v3 schema: overlap_bins + computing/not-overlapped/free pcts (the 3 step-
    # partitioning tracks), dropped overlap_segments, freq filtered to span, sync
    # kernels filtered from top_slices. Bump the key on any shape change so a cache
    # written under an older schema is never served.
    key = f"timeline:v3:{sig}:{SETTINGS.timeline_bins}"
    return cached_json(key, lambda: _build(prof))
