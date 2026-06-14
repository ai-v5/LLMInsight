"""Timeline for msprof captures, built from the SQLite slice/counter tables.

The 1.28GB msprof_*.json carries the same trace, but the DB is far cheaper to
query. Emits the SAME shape as metrics.timeline.compute_timeline so the frontend
renders it unchanged: lane occupancy bins + the 3-track overlap band + AI Core
frequency + top slices. slice.timestamp/duration are in NANOSECONDS here (the
torch_npu trace path is µs) — converted to µs on the way out.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, List

from ..cache import cached_json, file_signature
from ..config import SETTINGS

_LANES = {  # process_name -> lane label (mirrors timeline.LANE_LABELS)
    "Ascend Hardware": "Device (Ascend Hardware)",
    "Communication": "Communication",
    "CANN": "Host Runtime (CANN)",
}
_OVERLAP = ("Computing", "Communication", "Communication(Not Overlapped)", "Free")
_SYNC_RE = re.compile(r"WAIT|NOTIFY|EVENT_RECORD|BARRIER|STREAM", re.I)


def _dec(b):
    if isinstance(b, bytes):
        for e in ("utf-8", "gbk"):
            try:
                return b.decode(e)
            except Exception:
                pass
        return b.decode("utf-8", "replace")
    return b


def compute_msprof_timeline(prof) -> Dict[str, Any]:
    db = ((prof.meta or {}).get("msprof") or {}).get("db_path")
    if not db:
        return {"available": False, "reason": "msprof db missing"}
    sig = file_signature(db)
    return cached_json(f"msprof_timeline:v3:{sig}:{SETTINGS.timeline_bins}",
                       lambda: _build(db))


def _build(db: str) -> Dict[str, Any]:
    bins = SETTINGS.timeline_bins
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.text_factory = _dec
    cur = con.cursor()
    try:
        row = cur.execute(
            "SELECT MIN(timestamp), MAX(timestamp+duration) FROM slice WHERE duration>0").fetchone()
        if not row or row[0] is None:
            return {"available": False, "reason": "no timed slices"}
        t0, t1 = float(row[0]), float(row[1])
        span = t1 - t0
        if span <= 0:
            return {"available": False, "reason": "empty span"}
        bin_w = span / bins

        # --- lanes: per-process per-bin busy fraction (SQL start-bin aggregate;
        # long comm slices concentrate at their start bin -> occupancy is a coarse
        # overview, the overlap band below is the exact step decomposition).
        lane_occ: Dict[str, List[float]] = {lbl: [0.0] * bins for lbl in _LANES.values()}
        for proc, lbl in _LANES.items():
            for b, s in cur.execute(
                "SELECT CAST((s.timestamp-?)/? AS INT) b, SUM(s.duration) FROM slice s "
                "JOIN thread t ON s.track_id=t.track_id JOIN process p ON t.pid=p.pid "
                "WHERE p.process_name=? AND s.duration>0 GROUP BY b", (t0, bin_w, proc)):
                if b is not None and 0 <= b < bins:
                    lane_occ[lbl][int(b)] = round(min(float(s or 0) / bin_w, 1.0), 3)

        # --- overlap band: exact cross-bin spread over the Overlap Analysis tracks
        # (Computing / Communication / Communication(Not Overlapped) / Free).
        pid_name = {pid: nm for pid, nm in cur.execute("SELECT pid, process_name FROM process")}
        ov_track: Dict[int, str] = {}
        for tid, pid, tname in cur.execute("SELECT track_id, pid, thread_name FROM thread"):
            if "Overlap" in (pid_name.get(pid, "")) and tname in _OVERLAP:
                ov_track[tid] = tname
        ov_occ: Dict[str, List[float]] = {t: [0.0] * bins for t in _OVERLAP}

        def add(arr, s, e):
            s = max(s, t0); e = min(e, t1)
            if e <= s:
                return
            b0 = max(0, min(int((s - t0) / bin_w), bins - 1))
            b1 = max(0, min(int((e - t0) / bin_w), bins - 1))
            if b0 == b1:
                arr[b0] += e - s; return
            arr[b0] += (t0 + (b0 + 1) * bin_w) - s
            for b in range(b0 + 1, b1):
                arr[b] += bin_w
            arr[b1] += e - (t0 + b1 * bin_w)

        if ov_track:
            qmarks = ",".join("?" * len(ov_track))
            for ts, dur, tid in cur.execute(
                f"SELECT timestamp, duration, track_id FROM slice "
                f"WHERE duration>0 AND track_id IN ({qmarks})", tuple(ov_track)):
                add(ov_occ[ov_track[tid]], float(ts), float(ts) + float(dur))
        for t in ov_occ:
            ov_occ[t] = [round(min(v / bin_w, 1.0), 3) for v in ov_occ[t]]

        # --- AI Core frequency (counter table) ---
        freq: List[Dict[str, Any]] = []
        for ts, args in cur.execute(
            "SELECT timestamp, args FROM counter WHERE name LIKE '%Freq%Die 0%' ORDER BY timestamp"):
            try:
                mhz = json.loads(args).get("MHz", 0)
            except Exception:
                mhz = 0
            tu = (float(ts) - t0) / 1e3                      # ns -> us
            if 0 <= tu <= span / 1e3:
                freq.append({"t_us": round(tu, 1), "mhz": mhz})
        if len(freq) > 240:
            freq = freq[::len(freq) // 240]

        # --- top real slices per lane: Device compute + Communication, non-sync.
        # Per-lane caps so big multi-second comm slices don't crowd out the (smaller)
        # compute kernels. CANN host-API rows (AscendCL@/Node@) are excluded. ---
        top: List[Dict[str, Any]] = []
        THRESH_NS = 1.5e6   # 1500 us, mirrors the torch_npu timeline's Device cutoff
        for proc, lane in (("Ascend Hardware", "Device"), ("Communication", "Comm")):
            cnt = 0
            for nm, ts, dur in cur.execute(
                "SELECT s.name, s.timestamp, s.duration FROM slice s "
                "JOIN thread t ON s.track_id=t.track_id JOIN process p ON t.pid=p.pid "
                "WHERE p.process_name=? AND s.duration>? ORDER BY s.duration DESC LIMIT 200",
                (proc, THRESH_NS)):
                if _SYNC_RE.search(str(nm)):
                    continue
                top.append({"name": str(nm)[:60], "lane": lane,
                            "start_us": round((float(ts) - t0) / 1e3, 1),
                            "dur_us": round(float(dur) / 1e3, 1)})
                cnt += 1
                if cnt >= 30:
                    break
        top.sort(key=lambda s: s["dur_us"], reverse=True)
    finally:
        con.close()

    def avg_pct(a):
        return round(100.0 * sum(a) / len(a), 1) if a else 0.0

    return {
        "available": True,
        "t0_us": round(t0 / 1e3, 1),
        "span_us": round(span / 1e3, 1),
        "span_s": round(span / 1e9, 4),
        "bins": bins,
        "bin_us": round(bin_w / 1e3, 1),
        "lanes": [{"label": l, "occupancy": lane_occ[l]} for l in lane_occ],
        "overlap_bins": [{"track": t, "occupancy": ov_occ[t]} for t in _OVERLAP],
        "computing_pct": avg_pct(ov_occ["Computing"]),
        "not_overlapped_pct": avg_pct(ov_occ["Communication(Not Overlapped)"]),
        "free_pct": avg_pct(ov_occ["Free"]),
        "top_slices": top,
        "ai_core_freq": freq,
        "source": "msprof",
        "note": ("源自 msprof SQLite（slice/counter）。Overlap 时间条为精确跨桶分摊"
                 "（计算/未掩盖通信/Free 三段=步长 100%）；上方泳道占用为按起始桶的"
                 "粗聚合概览（长通信切片会偏向其起始桶）。"),
    }
