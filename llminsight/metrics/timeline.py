"""Trace-derived timeline: lane occupancy bins + overlap segments + top slices.

Computed by streaming trace_view.json. Result is small (JSON ~100KB) and cached
on disk keyed by the trace file signature so server restarts are instant.
"""
from __future__ import annotations

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
    overlap_segments: List[Dict[str, Any]] = []
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
                    top_slices.append(
                        {"name": str(ev.get("name"))[:60], "lane": "Device",
                         "start_us": round(ts - t0, 1), "dur_us": round(dur, 1)}
                    )
            if pid == overlap_pid:
                track = tid_name.get((pid, ev.get("tid")), "")
                if track in overlap_tracks and dur > 0:
                    overlap_segments.append(
                        {"track": track, "start_us": round(ts - t0, 1),
                         "dur_us": round(dur, 1)}
                    )
        elif ph == "C":
            nm = ev.get("name") or ""
            if "Freq" in nm and "Die 0" in nm:
                args = ev.get("args", {}) or {}
                freq.append({"t_us": round(event_ts_us(ev) - t0, 1),
                             "mhz": args.get("MHz", 0)})

    # normalize occupancy to [0,1]
    for lbl in lane_occ:
        lane_occ[lbl] = [round(min(v / bin_us, 1.0), 3) for v in lane_occ[lbl]]

    overlap_segments.sort(key=lambda s: s["dur_us"], reverse=True)
    overlap_segments = overlap_segments[:1500]
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
        "overlap_segments": overlap_segments,
        "top_slices": top_slices,
        "ai_core_freq": freq,
        "note": "占用率为每个时间桶内事件覆盖比例（多流叠加已截断到 1.0）。Overlap 段来自 profiler 的 Overlap Analysis 泳道。",
    }


def compute_timeline(prof) -> Dict[str, Any]:
    if not prof.trace_path:
        return {"available": False, "reason": "trace_view.json missing"}
    sig = file_signature(prof.trace_path)
    key = f"timeline:{sig}:{SETTINGS.timeline_bins}"
    return cached_json(key, lambda: _build(prof))
