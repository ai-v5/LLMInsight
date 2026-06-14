"""Smart timeline: profiler-style operator Gantt + utilization lanes.

Replicates the LLMperf timeline paradigm in ECharts as ONE merged multi-lane
chart on a single time axis: four utilization lanes (Cube / Vector / HBM / 通信,
each 0–100%) stacked on top, and device/communication trace slices laid out as
horizontal Gantt bars across **stream lanes** (Cube / FlashAttention / Vector /
MIX / Communication / other) below. Notify_Wait synchronization stalls are
excluded from every lane. 显存容量 / 主机内存 are surfaced as "待采集"
placeholders (need memory_record.csv).

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

_GEOM_VERSION = "v3"  # bump when geometry/stream logic changes (cache invalidation)


def _is_notify_wait(name: str) -> bool:
    """Synchronization stalls (HCCL Notify_Wait / NOTIFY_WAIT_SQE) — pure idle
    waiting, excluded from every lane so they don't inflate comm / occupancy."""
    return "notify_wait" in (name or "").lower()


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

    bins = SETTINGS.smart_timeline_bins
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
    vec_occ = [0.0] * bins  # vector ops have no FLOP model → time occupancy

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
        if _is_notify_wait(name):
            continue  # synchronization idle — not a real lane occupant
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
                # attribute work ∝ this slice's actual duration at the name's average
                # throughput (FLOP/μs). Using a fixed per-call average flops here would
                # dilute long instances (util = mfu × avg_dur/this_dur); throughput ×
                # dur instead keeps a fully-covered bin's util ≈ the name's MFU.
                fpu = ki.get("flops_per_us")
                if fpu is not None:
                    modeled_dev_us += dur
                    spread(flops_sum, ts, ts + dur, fpu * dur)
                bpu = ki.get("bytes_per_us")
                if bpu is not None:
                    spread(bytes_sum, ts, ts + dur, bpu * dur)
            else:
                typ = core = dtype = None
                stream = _stream_from_name(name)
            if stream == "vector":
                spread(vec_occ, ts, ts + dur, dur)

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
        "vec_occ": [round(v, 3) for v in vec_occ],
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
    vec_occ = geom.get("vec_occ", [0.0] * bins)

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
    vector_series = ([round(_clamp01(vec_occ[b] / bin_us), 4) for b in range(bins)]
                     if bin_us > 0 else [0.0] * bins)

    # un-clamped absolute per-bin values for the hover tooltip (reveal >100% overshoot
    # that the clamped utilization % hides — e.g. overlapping streams pushing past peak)
    cube_abs = ([round(flops_sum[b] / bin_s / 1e12, 2) for b in range(bins)]
                if bin_s > 0 else [0.0] * bins)        # achieved TFLOP/s
    hbm_abs = ([round(bytes_sum[b] / bin_s / 1e9, 1) for b in range(bins)]
               if bin_s > 0 else [0.0] * bins)         # achieved GB/s
    vec_abs = [round(vec_occ[b], 1) for b in range(bins)]    # busy μs within bin
    comm_abs = [round(comm_occ[b], 1) for b in range(bins)]  # busy μs within bin
    peak_tflops = round(effective_peak / 1e12, 1)
    peak_gbps = round(hbm_bw / 1e9, 1)
    bin_us_r = round(bin_us, 1)

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
        {"key": "cube", "label": "Cube 利用率", "available": True, "unit": "%",
         "color": "#4f9fe0", "series": compute_series,
         "abs": cube_abs, "abs_unit": "TFLOP/s", "peak": peak_tflops,
         "peak_unit": "TFLOP/s", "kind": "rate",
         "note": "Σ建模FLOPs /（桶时长 × Cube 有效峰值）"},
        {"key": "vector", "label": "Vector 利用率", "available": True, "unit": "%",
         "color": "#4caf50", "series": vector_series,
         "abs": vec_abs, "abs_unit": "μs", "peak": bin_us_r,
         "peak_unit": "μs", "kind": "occupancy",
         "note": "Vector 泳道每桶时间占用率（向量算子无 FLOP 模型，按占用计）"},
        {"key": "hbm_bw", "label": "HBM 利用率", "available": True, "unit": "%",
         "color": "#e3b341", "series": hbm_series,
         "abs": hbm_abs, "abs_unit": "GB/s", "peak": peak_gbps,
         "peak_unit": "GB/s", "kind": "rate",
         "note": "Σ读写字节 /（桶时长 × HBM 带宽）"},
        {"key": "comm", "label": "通信 利用率", "available": True, "unit": "%",
         "color": "#f0655c", "series": comm_series,
         "abs": comm_abs, "abs_unit": "μs", "peak": bin_us_r,
         "peak_unit": "μs", "kind": "occupancy",
         "note": "Communication 泳道每桶时间占用率（已剔除 Notify_Wait 同步等待）"},
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
            "单张大图、统一时间轴：上方 Cube / Vector / HBM / 通信 四条利用率泳道，"
            "下方算子按 stream 泳道铺成 Gantt。悬停算子显示 名称/类型/Core/start/dur，"
            "命中按名 join 的 kernel_index 时追加 MFU/MBU/dtype（按名平均，参考值）。"
            "利用率口径：Cube = Σ建模FLOPs /（桶 × Cube 有效峰值），"
            "Vector = 向量泳道每桶时间占用率（无 FLOP 模型），"
            "HBM = Σ字节 /（桶 × HBM 带宽），通信 = Communication 泳道每桶时间占用率。"
            "Notify_Wait 同步等待已全程剔除，不进入任何泳道。"
            "显存容量 / 主机内存待 memory_record.csv 采集。"
            f"切片共 {geom['total_slices']} 个，按时长下采样保留最长 {geom['shown_slices']} 个。"
        ),
    }


def compute_smart_timeline(prof, eff: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point: geometry (cached by trace signature) + chip overlay."""
    if (getattr(prof, "meta", {}) or {}).get("format") == "msprof":
        return compute_msprof_smart_timeline(prof, eff)
    # torch_npu WITHOUT shapes: Cube/HBM (FLOP/byte models) are dead and the
    # trace+kernel_index geometry can't classify the ~98% of ops missing from the
    # shape-scored index, so Cube/Vector come out empty. Fall back to the kd-occupancy
    # timeline (Accelerator Core + Duration straight from kernel_details, like msprof)
    # — it fills Cube/Vector and skips the multi-GB trace pass.
    kd = getattr(prof, "kernel_details", None)
    has_shapes = (kd is not None and not kd.empty and "Input Shapes" in kd.columns and
                  bool((~kd["Input Shapes"].astype(str).str.strip().str.upper()
                        .isin(("", "N/A", "NAN", "NONE"))).any()))
    if not has_shapes:
        return compute_msprof_smart_timeline(prof, eff)
    if not getattr(prof, "trace_path", None):
        return {"available": False, "reason": "trace_view.json missing"}
    if not eff or not eff.get("available"):
        return {"available": False,
                "reason": "kernel_details.csv missing（无法按名 join 算子）"}
    kindex = eff.get("kernel_index", {}) or {}
    sig = file_signature(prof.trace_path)
    key = (f"smarttl:{_GEOM_VERSION}:{sig}:"
           f"{SETTINGS.smart_timeline_bins}:{SETTINGS.timeline_max_slices}")
    geom = cached_json(key, lambda: _build_geometry(prof, kindex))
    return _apply_chip(geom, eff)


def _wait_xfer_from_task_slices(data_dir, t0, bin_us, bins):
    """No-db msprof: per-bin comm wait vs transfer from task_time_slice_*.csv, whose
    kernel_type column carries the device sub-tasks (NOTIFY_WAIT_SQE / UBDMA / SDMA /
    DAVID_EVENT_WAIT / ...). Start-bin aggregate (sub-tasks are short, so it's
    accurate); cached by the slice files' sizes. Returns (wait_occ, xfer_occ) in us."""
    import glob
    import os as _os
    # msprof RAW export shards as task_time_slice_*.csv; torch_npu ships a single
    # task_time.csv with the SAME columns (kernel_type / task_start(us) / task_time(us)).
    files = (sorted(glob.glob(_os.path.join(data_dir, "task_time_slice_*.csv"))) or
             sorted(glob.glob(_os.path.join(data_dir, "task_time.csv"))))
    if not files:
        return [0.0] * bins, [0.0] * bins
    sig = "|".join(str(_os.path.getsize(f)) for f in files)

    def _build():
        import pandas as pd
        wait = [0.0] * bins
        xfer = [0.0] * bins
        for f in files:
            try:
                tdf = pd.read_csv(f, low_memory=False,
                                  usecols=["kernel_type", "task_start(us)", "task_time(us)"])
            except Exception:
                continue
            kt = tdf["kernel_type"].astype(str)
            ts = pd.to_numeric(tdf["task_start(us)"], errors="coerce")
            tt = pd.to_numeric(tdf["task_time(us)"], errors="coerce").fillna(0.0)
            bi = ((ts - t0) / bin_us).fillna(-1).astype("int64")
            wmask = kt.str.contains("WAIT|NOTIFY", case=False, na=False, regex=True)
            xmask = kt.str.contains("DMA|MEMCPY", case=False, na=False, regex=True)
            for mask, occ in ((wmask, wait), (xmask, xfer)):
                grp = tt[mask].groupby(bi[mask]).sum()
                for b, v in grp.items():
                    if 0 <= int(b) < bins:
                        occ[int(b)] += float(v)
        return {"wait": wait, "xfer": xfer}

    res = cached_json(f"msprof_ts_wx:{sig}:{bins}:{int(t0)}:{int(round(bin_us))}", _build)
    return res["wait"], res["xfer"]


def compute_msprof_smart_timeline(prof, eff: Dict[str, Any]) -> Dict[str, Any]:
    """Smart timeline for msprof captures, built from kernel_details (there is no
    trace_view.json). The operator Gantt and the Cube/Vector/通信 lanes are time
    OCCUPANCY (no FLOP/byte model without shapes), so they read as "busy fraction"
    rather than MFU/MBU; the HBM-rate lane degrades to unavailable. Per-slice
    MFU/MBU are joined from kernel_index when a shape-recording capture has them."""
    import pandas as pd
    kd = getattr(prof, "kernel_details", None)
    if kd is None or kd.empty:
        return {"available": False, "reason": "no kernels"}
    bins = SETTINGS.smart_timeline_bins
    max_slices = SETTINGS.timeline_max_slices
    start = pd.to_numeric(kd["Start Time(us)"], errors="coerce")
    dur = pd.to_numeric(kd["Duration(us)"], errors="coerce")
    names = kd["Name"].astype(str)
    types = kd["Type"].astype(str)
    cores = kd["Accelerator Core"].astype(str)
    mask = start.notna() & (dur > 0)
    if not mask.any():
        return {"available": False, "reason": "no timed kernels"}
    t0 = float(start[mask].min())
    t1 = float((start + dur)[mask].max())
    span = t1 - t0
    if span <= 0:
        return {"available": False, "reason": "empty span"}
    bin_us = span / bins

    cube_occ = [0.0] * bins
    vec_occ = [0.0] * bins
    comm_occ = [0.0] * bins
    p2p_occ = [0.0] * bins   # point-to-point (PP send/recv) AICPU blocking — a
    # distinct kind of cross-rank wait that never expands into device sub-tasks.

    def spread(arr, s, e):
        s = max(s, t0); e = min(e, t1)
        if e <= s:
            return
        b0 = max(0, min(int((s - t0) / bin_us), bins - 1))
        b1 = max(0, min(int((e - t0) / bin_us), bins - 1))
        if b0 == b1:
            arr[b0] += e - s; return
        arr[b0] += (t0 + (b0 + 1) * bin_us) - s
        for b in range(b0 + 1, b1):
            arr[b] += bin_us
        arr[b1] += e - (t0 + b1 * bin_us)

    slices: List[Dict[str, Any]] = []
    for nm, ty, co, s, d in zip(names[mask], types[mask], cores[mask],
                                start[mask], dur[mask]):
        if _is_notify_wait(nm):
            continue
        stream = _stream_from_meta(ty, co)
        s = float(s); e = s + float(d)
        cu = co.upper()
        nl = nm.lower()
        if nl.startswith("hcom_send") or nl.startswith("hcom_receive"):
            spread(p2p_occ, s, e)        # P2P/PP blocking (AICPU; not on device cube)
        if "VECTOR" in cu or "AIV" in cu:
            spread(vec_occ, s, e)
        elif "COMMUNICATION" in cu or stream == "comm":
            spread(comm_occ, s, e)
        else:
            spread(cube_occ, s, e)
        slices.append({"name": nm, "stream": stream, "type": ty, "core": co,
                       "dtype": None, "start_ms": (s - t0) / 1e3, "dur_ms": float(d) / 1e3})
    if not slices:
        return {"available": False, "reason": "no lane slices"}

    total = len(slices)
    if total > max_slices:
        # per-stream quota: multi-second comm slices would otherwise crowd out every
        # (shorter) compute kernel under a global longest-first cut, leaving an
        # all-comm Gantt. Keep the longest few per lane so each stream is visible.
        from collections import defaultdict
        by_stream: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in slices:
            by_stream[s["stream"]].append(s)
        quota = max(1, max_slices // max(len(by_stream), 1))
        kept: List[Dict[str, Any]] = []
        for grp in by_stream.values():
            grp.sort(key=lambda x: x["dur_ms"], reverse=True)
            kept.extend(grp[:quota])
        slices = kept
    slices.sort(key=lambda x: x["start_ms"])

    # comm wait vs transfer per-bin, from the device HCCL sub-tasks in the db slice
    # table (NOTIFY/EVENT_WAIT = cross-rank waiting, UBDMA/SDMA/MEMCPY = moving data).
    # SQL start-bin aggregate (sub-tasks are short, so bin attribution is accurate);
    # durations sum across cores/queues -> occupancy can exceed 1 and is clamped.
    wait_occ = [0.0] * bins
    xfer_occ = [0.0] * bins
    db = ((getattr(prof, "meta", {}) or {}).get("msprof") or {}).get("db_path")
    if db and bin_us > 0:
        import sqlite3
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "replace") if isinstance(b, bytes) else b
        try:
            for bi, cat, dns in con.execute(
                "SELECT CAST((s.timestamp/1e3 - ?)/? AS INT) b, "
                "CASE WHEN s.name LIKE '%WAIT%' OR s.name LIKE '%NOTIFY%' THEN 'w' "
                "     WHEN s.name LIKE '%DMA%' OR s.name LIKE '%MEMCPY%' THEN 'x' "
                "     ELSE 'o' END c, SUM(s.duration) "
                "FROM slice s JOIN thread t ON s.track_id=t.track_id "
                "JOIN process p ON t.pid=p.pid "
                "WHERE p.process_name='Ascend Hardware' AND s.name NOT LIKE 'aclnn%' "
                "GROUP BY b, c", (t0, bin_us)):
                if bi is None or not (0 <= int(bi) < bins):
                    continue
                us = float(dns or 0) / 1e3                 # ns -> us
                if cat == "w":
                    wait_occ[int(bi)] += us
                elif cat == "x":
                    xfer_occ[int(bi)] += us
        except Exception:
            wait_occ = [0.0] * bins
            xfer_occ = [0.0] * bins
        finally:
            con.close()
    elif bin_us > 0:
        # no db (msprof raw export under PROF_xxx): rebuild wait/transfer from
        # task_time_slice_*.csv, whose kernel_type carries the same device sub-tasks
        # (NOTIFY_WAIT_SQE / UBDMA / SDMA / ...). Cached by the slice files' sizes.
        wait_occ, xfer_occ = _wait_xfer_from_task_slices(
            (getattr(prof, "meta", {}) or {}).get("data_dir", ""), t0, bin_us, bins)

    kindex = (eff or {}).get("kernel_index", {}) or {}
    out_slices: List[Dict[str, Any]] = []
    for s in slices:
        ki = kindex.get(s["name"])
        nm = s["name"]
        out_slices.append({
            "name": nm if len(nm) <= 90 else nm[:88] + "…",
            "stream": s["stream"], "type": s["type"], "core": s["core"], "dtype": None,
            "start_ms": round(s["start_ms"], 4), "dur_ms": round(s["dur_ms"], 4),
            "mfu": ki.get("mfu") if ki else None, "mbu": ki.get("mbu") if ki else None,
        })

    def occ_series(arr):
        return [round(_clamp01(v / bin_us), 4) for v in arr] if bin_us > 0 else [0.0] * bins
    bin_us_r = round(bin_us, 1)

    # comm composition per bin: wait vs transfer as a FRACTION of comm activity, so
    # red+green=1 when comm is active and both=0 in compute gaps. Clamped occupancy
    # hid the difference (both lanes saturated to 100% during comm); the fraction
    # shows whether each moment is dominated by waiting or by actual transfer.
    comm_tot = [wait_occ[b] + xfer_occ[b] for b in range(bins)]
    wait_frac = [round(wait_occ[b] / comm_tot[b], 4) if comm_tot[b] > 0 else 0.0
                 for b in range(bins)]
    xfer_frac = [round(xfer_occ[b] / comm_tot[b], 4) if comm_tot[b] > 0 else 0.0
                 for b in range(bins)]

    utilization = [
        {"key": "cube", "label": "Cube 占用率", "available": True, "unit": "%",
         "color": "#4f9fe0", "series": occ_series(cube_occ),
         "abs": [round(v, 1) for v in cube_occ], "abs_unit": "μs", "peak": bin_us_r,
         "peak_unit": "μs", "kind": "occupancy",
         "note": "AI_CORE 泳道每桶时间占用率（无 shape 采集 → 按占用计，非 FLOP MFU）"},
        {"key": "vector", "label": "Vector 占用率", "available": True, "unit": "%",
         "color": "#4caf50", "series": occ_series(vec_occ),
         "abs": [round(v, 1) for v in vec_occ], "abs_unit": "μs", "peak": bin_us_r,
         "peak_unit": "μs", "kind": "occupancy",
         "note": "AI_VECTOR_CORE 泳道每桶时间占用率"},
        {"key": "comm_wait", "label": "集合通信-卡间等待占比", "available": True, "unit": "%",
         "color": "#f0655c", "series": wait_frac,
         "abs": [round(v, 1) for v in wait_occ], "abs_unit": "μs", "peak": bin_us_r,
         "peak_unit": "μs", "kind": "occupancy",
         "note": "集合通信(allReduce/alltoall…)的 device 同步等待占比(NOTIFY/EVENT_WAIT)；红+绿=该桶集合通信构成、计算间隙为 0；悬停 abs 为绝对耗时(μs，跨核累加)"},
        {"key": "comm_xfer", "label": "集合通信-有效传输占比", "available": True, "unit": "%",
         "color": "#5ee0b8", "series": xfer_frac,
         "abs": [round(v, 1) for v in xfer_occ], "abs_unit": "μs", "peak": bin_us_r,
         "peak_unit": "μs", "kind": "occupancy",
         "note": "集合通信的 device 搬数据占比(UBDMA/SDMA/MEMCPY)；悬停 abs 为绝对耗时(μs)"},
        {"key": "comm_p2p", "label": "P2P/PP 通信阻塞", "available": True, "unit": "%",
         "color": "#d29922", "series": occ_series(p2p_occ),
         "abs": [round(v, 1) for v in p2p_occ], "abs_unit": "μs", "peak": bin_us_r,
         "peak_unit": "μs", "kind": "occupancy",
         "note": "点对点 hcom_send/receive 的 AICPU 阻塞占用——PP 流水的卡间等待，不经 device cube 展开(故不在上面两条集合通信里)；占用率(含跨核累加)"},
        {"key": "hbm_bw", "label": "HBM 利用率", "available": False,
         "reason": "无 shape 采集 → 无字节模型，HBM 利用率不可用"},
        {"key": "hbm_cap", "label": "显存容量 (HBM)", "available": False,
         "reason": "待 memory 采集"},
        {"key": "host_mem", "label": "主机内存", "available": False,
         "reason": "待 memory 采集"},
    ]

    _is_msprof = ((getattr(prof, "meta", {}) or {}).get("format") == "msprof")
    _cap = "msprof " if _is_msprof else "torch_npu 半采集 "
    return {
        "available": True, "t0_us": t0, "span_us": round(span, 1),
        "span_s": round(span / 1e6, 4), "bins": bins, "bin_us": bin_us,
        "lane_count": len(STREAMS),
        "streams": [{"key": k, "label": l, "color": c} for (k, l, c) in STREAMS],
        "slices": out_slices, "utilization": utilization, "modeled_pct": None,
        "total_slices": total, "shown_slices": len(out_slices),
        "source": "msprof" if _is_msprof else "torch_npu",
        "note": (f"{_cap}智能时间线：算子按 stream 泳道铺成 Gantt；上方 Cube / Vector / 通信 "
                 "为每桶时间占用率泳道（无 shape 采集，按占用计、非 FLOP MFU，HBM 利用率不可用）。"
                 "集合通信 device 子任务（等待 NOTIFY/EVENT_WAIT vs 传输 UBDMA/SDMA）取自 "
                 "task_time(_slice).csv。Notify_Wait 同步等待已从算子泳道剔除。"
                 f"切片共 {total} 个，按时长下采样保留最长 {len(out_slices)} 个。"),
    }
