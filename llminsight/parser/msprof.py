"""Load a MindStudio **msprof-export** profiling dir into the same ProfileData
model used by the torch_npu ASCEND_PROFILER_OUTPUT loader.

Two profiling layouts exist in the wild:

  * torch_npu  -> ASCEND_PROFILER_OUTPUT/  (kernel_details.csv, trace_view.json, …)
                  handled by parser.profile.load_profile
  * msprof     -> mindstudio_profiler_output/  (mindstudio_insight_data.db +
                  op_summary_*.csv + task_time_slice_*.csv + msprof_*.json)
                  handled here

The msprof SQLite db (`mindstudio_insight_data.db`) is the richest source: table
`kernel_detail` holds one row per launched task (name, accelerator_core, duration).
We map it onto the kernel_details schema the metrics layer already understands.

CAVEAT — lightweight captures: when the profiler ran WITHOUT op-attr/shape
recording, `op_type` / `input_shapes` / `input_data_types` are all "N/A". MFU/MBU
and the Roofline need shapes, so they degrade to unavailable; op timing, the
compute/comm split, and the timeline remain fully usable. `meta["has_shapes"]`
tells the UI which case it is.
"""
from __future__ import annotations

import glob
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

import pandas as pd

from .profile import ProfileData
from ..cache import cached_json


# --------------------------------------------------------------------------- #
def is_msprof_dir(data_dir: str) -> bool:
    """A dir is msprof-export if it carries the insight DB or an op_summary_*.csv
    (and crucially NOT a kernel_details.csv, which marks the torch_npu layout)."""
    try:
        if os.path.isfile(os.path.join(data_dir, "kernel_details.csv")):
            return False
        if os.path.isfile(os.path.join(data_dir, "mindstudio_insight_data.db")):
            return True
        return bool(glob.glob(os.path.join(data_dir, "op_summary_*.csv")))
    except OSError:
        return False


def _first(data_dir: str, pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(data_dir, pattern)))
    return hits[-1] if hits else None


# --- op-type / core recovery from the kernel name --------------------------- #
# msprof Op Names look like `aclnn<Api>_<Impl>_<OpType>` (last seg is the op
# type), or `hcom_<verb>_AicpuKernel_...` / `HcclLaunchAicpuKernel` for comms.
_HCOM_RE = re.compile(r"^(hcom_[A-Za-z]+)", re.IGNORECASE)
_VECTOR_HINTS = (
    "RmsNorm", "SwiGlu", "Swiglu", "Cast", "ZerosLike", "Transpose", "Slice",
    "Gather", "Scatter", "RotaryPosition", "Sigmoid", "Softmax", "ReduceSum",
    "ReduceMean", "RealDiv", "Mul", "Add", "Sub", "Exp", "Sin", "Cos", "Neg",
    "Square", "Fill", "BroadcastTo", "Concat", "TensorMove", "LayerNorm",
    "ApplyAdam", "Sort", "TopK", "Range", "MaskedSelect", "MaskedFill", "NonZero",
)
_CUBE_HINTS = ("MatMul", "Matmul", "Gemm", "FlashAttention", "Conv", "BatchMatMul")


def op_type_from_name(name: str) -> str:
    """Recover an op TYPE from a msprof kernel name (op_type column is N/A)."""
    n = str(name or "").strip()
    if not n:
        return "Unknown"
    if "Hccl" in n:
        return "HcclLaunch"
    m = _HCOM_RE.match(n)
    if m:
        return m.group(1).lower()                       # hcom_send / hcom_receive
    if n.startswith("aclnn"):
        seg = n.split("_")[-1]                           # aclnn..._..._<OpType>
        if seg and not seg.isdigit():
            return seg
    parts = [p for p in n.split("_") if p and not p.isdigit()]
    return parts[-1] if parts else n


def core_from_name(name: str, db_core: str, op_type: str) -> str:
    """Recover an Accelerator-Core class. db gives only N/A / COMMUNICATION, so we
    refine compute kernels into AI_CORE (cube) vs AI_VECTOR_CORE from the name."""
    if (db_core or "").upper() == "COMMUNICATION":
        return "COMMUNICATION"
    if "Hccl" in name or _HCOM_RE.match(name or ""):
        return "COMMUNICATION"
    if any(h in name for h in _CUBE_HINTS):
        return "AI_CORE"
    if any(h in name for h in _VECTOR_HINTS):
        return "AI_VECTOR_CORE"
    return "AI_CORE"


# --------------------------------------------------------------------------- #
def _decode(b):
    """db text may be UTF-8 or GBK (Windows paths) — never crash on decode."""
    if isinstance(b, bytes):
        for enc in ("utf-8", "gbk"):
            try:
                return b.decode(enc)
            except Exception:
                pass
        return b.decode("utf-8", "replace")
    return b


def _load_kernel_detail_db(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = _decode
    try:
        df = pd.read_sql_query(
            "SELECT name, accelerator_core, duration, start_time, task_id, step_id, "
            "deviceId, input_shapes, input_data_types, output_shapes, output_data_types "
            "FROM kernel_detail", con)
    finally:
        con.close()
    return df


def _kernel_details_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map the msprof kernel rows onto the kernel_details.csv schema the metrics
    layer consumes. Shapes/dtypes stay N/A (not captured)."""
    names = df["name"].astype(str)
    op_types = names.map(op_type_from_name)
    cores = [core_from_name(n, c, t)
             for n, c, t in zip(names, df["accelerator_core"].astype(str), op_types)]
    n = len(df)

    def _col(name):
        # Use the db's shape/dtype columns verbatim: "N/A" on a lightweight capture,
        # but REAL on a shape-recording one -> MFU/MBU then work with no extra code.
        return df[name].astype(str) if name in df.columns else ["N/A"] * n

    return pd.DataFrame({
        "Name": names,
        "Type": op_types,
        "Accelerator Core": cores,
        "Duration(us)": pd.to_numeric(df["duration"], errors="coerce").fillna(0.0),
        "Input Shapes": _col("input_shapes"),
        "Input Data Types": _col("input_data_types"),
        "Output Shapes": _col("output_shapes"),
        "Output Data Types": _col("output_data_types"),
        "Task ID": df["task_id"],
        # kernel_detail.start_time is NANOSECONDS while duration is microseconds
        # (mixed units in this table); normalize start to us so start+duration and
        # any span math are consistent.
        "Start Time(us)": pd.to_numeric(df["start_time"], errors="coerce") / 1e3,
    })


def _op_statistic_from_kd(kd: pd.DataFrame) -> pd.DataFrame:
    """Aggregate kernel rows into the op_statistic.csv schema (hotspots view)."""
    g = kd.groupby(["Type", "Accelerator Core"], dropna=False)
    rows = []
    total = float(kd["Duration(us)"].sum()) or 1.0
    for (otype, core), sub in g:
        tot = float(sub["Duration(us)"].sum())
        cnt = int(len(sub))
        rows.append({
            "Device_id": 0,
            "OP Type": otype,
            "Core Type": core,
            "Count": cnt,
            "Total Time(us)": round(tot, 3),
            "Min Time(us)": round(float(sub["Duration(us)"].min()), 3),
            "Avg Time(us)": round(tot / cnt, 3) if cnt else 0.0,
            "Max Time(us)": round(float(sub["Duration(us)"].max()), 3),
            "Ratio(%)": round(100.0 * tot / total, 3),
        })
    rows.sort(key=lambda r: r["Total Time(us)"], reverse=True)
    return pd.DataFrame(rows)


def _communication_from_kd(kd: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build a normalized communication list from COMMUNICATION kernels. The
    msprof lightweight capture has no per-link transit/bandwidth, so those are 0;
    elapse time comes from the kernel duration (μs -> ms)."""
    comm = kd[kd["Accelerator Core"] == "COMMUNICATION"]
    out: List[Dict[str, Any]] = []
    for name, sub in comm.groupby(comm["Name"].astype(str).map(op_type_from_name)):
        elapse_ms = float(sub["Duration(us)"].sum()) / 1e3
        out.append({
            "step": "step", "kind": "collective", "name": name, "type": name,
            "elapse_ms": elapse_ms, "transit_ms": 0.0, "wait_ms": 0.0,
            "sync_ms": 0.0, "idle_ms": 0.0, "wait_ratio": 0.0, "sync_ratio": 0.0,
            "start_us": 0.0, "links": {},
        })
    return out


# --------------------------------------------------------------------------- #
def _overlap_breakdown(db_path: str) -> Dict[str, float]:
    """Read the Overlap Analysis layer -> wall-clock Computing / Communication /
    Communication(Not Overlapped) / Free totals (us). slice.duration is ns."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = _decode
    try:
        rows = con.execute(
            "SELECT t.thread_name, SUM(s.duration) FROM slice s "
            "JOIN thread t ON s.track_id=t.track_id "
            "JOIN process p ON t.pid=p.pid "
            "WHERE p.process_name LIKE '%Overlap%' GROUP BY t.thread_name").fetchall()
    except Exception:
        return {}
    finally:
        con.close()
    return {str(name): float(dur or 0) / 1e3 for name, dur in rows}   # ns -> us


def _step_trace_from_overlap(ov: Dict[str, float], device_id) -> pd.DataFrame:
    """One-row step_trace from the Overlap layer so the standard overview()
    (wall-clock Computing/Comm/Free decomposition) works unchanged."""
    comp = ov.get("Computing", 0.0)
    comm = ov.get("Communication", 0.0)
    cno = ov.get("Communication(Not Overlapped)", 0.0)
    free = ov.get("Free", 0.0)
    stage = comp + cno + free
    return pd.DataFrame([{
        "Step": 0, "Device_id": device_id or 0,
        "Computing": comp, "Communication": comm,
        "Communication(Not Overlapped)": cno,
        "Overlapped": max(comm - cno, 0.0), "Free": free, "Stage": stage,
    }])


def _comm_task_cat(name: str) -> str:
    """Classify a device HCCL/sync sub-task by name."""
    u = (name or "").upper()
    if "WAIT" in u or "NOTIFY" in u:
        return "wait"           # cross-rank waiting for the peer / event sync
    if "DMA" in u or "MEMCPY" in u:
        return "transfer"       # actual data movement (UBDMA / SDMA / MEMCPY)
    if "REDUCE" in u:
        return "reduce"         # in-flight reduction
    if "HCCL" in u or "HCOM" in u:
        return "launch"         # op dispatch (AICPU launch), neither wait nor move
    return "other"


def _comm_breakdown(db_path: str) -> Dict[str, Any]:
    """Split device communication/sync sub-tasks into effective transfer vs
    cross-rank waiting. A device HCCL op expands into NOTIFY_WAIT/EVENT_WAIT (waiting
    for the peer) and UBDMA/SDMA/MEMCPY (actually moving data) — classify by task
    name. Durations are SUMMED across cores/queues (overlap-inclusive), so this is
    the internal COMPOSITION of comm time, not a wall-clock figure."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = _decode
    try:
        rows = con.execute(
            "SELECT s.name, COUNT(*), SUM(s.duration) FROM slice s "
            "JOIN thread t ON s.track_id=t.track_id JOIN process p ON t.pid=p.pid "
            "WHERE p.process_name='Ascend Hardware' AND s.name NOT LIKE 'aclnn%' "
            "GROUP BY s.name").fetchall()
    except Exception:
        return {}
    finally:
        con.close()
    cats = {"wait": 0.0, "transfer": 0.0, "reduce": 0.0, "launch": 0.0, "other": 0.0}
    per_name: List[Dict[str, Any]] = []
    for name, cnt, dur in rows:
        us = float(dur or 0) / 1e3                       # ns -> us
        cat = _comm_task_cat(name)
        cats[cat] += us
        per_name.append({"name": str(name)[:48], "cat": cat,
                         "count": int(cnt or 0), "us": round(us, 1)})
    comm_total = cats["wait"] + cats["transfer"] + cats["reduce"]
    if comm_total <= 0:
        return {}
    top_wait = sorted((p for p in per_name if p["cat"] == "wait"),
                      key=lambda x: x["us"], reverse=True)[:8]
    top_xfer = sorted((p for p in per_name if p["cat"] in ("transfer", "reduce")),
                      key=lambda x: x["us"], reverse=True)[:8]
    return {
        "basis": "accumulated",
        "wait_us": round(cats["wait"], 1),
        "transfer_us": round(cats["transfer"] + cats["reduce"], 1),
        "launch_us": round(cats["launch"], 1),
        "comm_total_us": round(comm_total, 1),
        "wait_pct": round(100.0 * cats["wait"] / comm_total, 1),
        "transfer_pct": round(100.0 * (cats["transfer"] + cats["reduce"]) / comm_total, 1),
        "top_wait": top_wait,
        "top_transfer": top_xfer,
    }


def _comm_breakdown_from_task_slices(data_dir: str) -> Dict[str, Any]:
    """No-db msprof: device wait/transfer split from task_time_slice_*.csv — same shape
    as _comm_breakdown (db). The kernel_type column carries the sub-task class
    (NOTIFY_WAIT_SQE / UBDMA / SDMA / DAVID_EVENT_WAIT / ...); reuse _comm_task_cat on
    it. Durations are summed across cores/queues (overlap-inclusive), so this is the
    internal COMPOSITION of comm time, not a wall-clock figure. Cached by file sizes."""
    files = sorted(glob.glob(os.path.join(data_dir, "task_time_slice_*.csv")))
    if not files:
        return {}
    sig = "|".join(str(os.path.getsize(f)) for f in files)

    def _build():
        cats = {"wait": 0.0, "transfer": 0.0, "reduce": 0.0, "launch": 0.0, "other": 0.0}
        per: Dict[str, Dict[str, Any]] = {}
        for f in files:
            try:
                tdf = pd.read_csv(f, low_memory=False,
                                  usecols=["kernel_type", "task_time(us)"])
            except Exception:
                continue
            tt = pd.to_numeric(tdf["task_time(us)"], errors="coerce").fillna(0.0)
            agg = (pd.DataFrame({"kt": tdf["kernel_type"].astype(str), "us": tt})
                   .groupby("kt")["us"].agg(["sum", "count"]))
            for kt, row in agg.iterrows():
                cat = _comm_task_cat(kt)
                cats[cat] += float(row["sum"])
                if cat in ("wait", "transfer", "reduce"):
                    e = per.setdefault(kt, {"name": str(kt)[:48], "cat": cat,
                                            "count": 0, "us": 0.0})
                    e["count"] += int(row["count"])
                    e["us"] += float(row["sum"])
        comm_total = cats["wait"] + cats["transfer"] + cats["reduce"]
        if comm_total <= 0:
            return {}
        for e in per.values():
            e["us"] = round(e["us"], 1)
        top_wait = sorted((p for p in per.values() if p["cat"] == "wait"),
                          key=lambda x: x["us"], reverse=True)[:8]
        top_xfer = sorted((p for p in per.values() if p["cat"] in ("transfer", "reduce")),
                          key=lambda x: x["us"], reverse=True)[:8]
        return {
            "basis": "accumulated",
            "wait_us": round(cats["wait"], 1),
            "transfer_us": round(cats["transfer"] + cats["reduce"], 1),
            "launch_us": round(cats["launch"], 1),
            "comm_total_us": round(comm_total, 1),
            "wait_pct": round(100.0 * cats["wait"] / comm_total, 1),
            "transfer_pct": round(100.0 * (cats["transfer"] + cats["reduce"]) / comm_total, 1),
            "top_wait": top_wait,
            "top_transfer": top_xfer,
        }

    return cached_json(f"msprof_ts_cb:{sig}", _build)


def _kernel_details_from_op_summary(path: str) -> pd.DataFrame:
    """Load kernels from op_summary_*.csv (msprof RAW export, no SQLite db). Same
    output schema as _kernel_details_from_df; op type / core fall back to the name,
    and Task Start Time is already in us here (not ns like the db)."""
    df = pd.read_csv(path, low_memory=False)
    n = len(df)
    names = (df["Op Name"] if "Op Name" in df.columns else df.iloc[:, 4]).astype(str)
    rawt = df["OP Type"].astype(str) if "OP Type" in df.columns else pd.Series([""] * n)
    op_types = [t if t.strip().upper() not in ("", "N/A", "NAN", "NONE") else op_type_from_name(nm)
                for nm, t in zip(names, rawt)]
    rawc = df["Task Type"].astype(str) if "Task Type" in df.columns else pd.Series([""] * n)
    cores = [core_from_name(nm, (c if c.strip().upper() not in ("", "N/A", "NAN", "NONE") else ""), t)
             for nm, c, t in zip(names, rawc, op_types)]

    def _col(c):
        return df[c].astype(str) if c in df.columns else ["N/A"] * n

    def _num(c):
        return (pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                if c in df.columns else pd.Series([0.0] * n))

    return pd.DataFrame({
        "Name": names, "Type": op_types, "Accelerator Core": cores,
        "Duration(us)": _num("Task Duration(us)"),
        "Input Shapes": _col("Input Shapes"), "Input Data Types": _col("Input Data Types"),
        "Output Shapes": _col("Output Shapes"), "Output Data Types": _col("Output Data Types"),
        "Task ID": (df["Task ID"] if "Task ID" in df.columns else pd.Series(range(n))),
        "Start Time(us)": _num("Task Start Time(us)"),     # already us in op_summary
    })


def load_msprof_profile(data_dir: str) -> ProfileData:
    db_path = os.path.join(data_dir, "mindstudio_insight_data.db")
    has_db = os.path.isfile(db_path)
    if has_db:
        raw = _load_kernel_detail_db(db_path)
        kd = _kernel_details_from_df(raw)
        _d = raw["deviceId"].iloc[0] if not raw.empty else None
        dev = int(_d) if pd.notna(_d) else None  # blank deviceId -> None, not int(NaN)
        overlap = _overlap_breakdown(db_path)            # Overlap layer (step_trace)
        comm_breakdown = _comm_breakdown(db_path)         # device wait/transfer
    else:
        # RAW export (torch_npu PROF_xxx/mindstudio_profiler_output): no db, only
        # op_summary_*.csv. The Overlap layer + device comm sub-tasks live in the db,
        # so step_trace degrades to accumulated and comm wait/transfer is unavailable;
        # hotspots / efficiency / communication / smart-timeline still work off kd.
        op_sum = _first(data_dir, "op_summary_*.csv")
        kd = (_kernel_details_from_op_summary(op_sum) if op_sum
              else pd.DataFrame(columns=["Name", "Type", "Accelerator Core", "Duration(us)",
                                         "Input Shapes", "Input Data Types", "Output Shapes",
                                         "Output Data Types", "Task ID", "Start Time(us)"]))
        dev = 0
        overlap = {}
        comm_breakdown = _comm_breakdown_from_task_slices(data_dir)

    empty_kd = kd.empty
    has_shapes = (not empty_kd) and bool((~kd["Input Shapes"].astype(str).str.strip().str.upper()
                                          .isin(("", "N/A", "NAN", "NONE"))).any())
    op_stat = _op_statistic_from_kd(kd) if not empty_kd else pd.DataFrame()
    communication = _communication_from_kd(kd) if not empty_kd else []

    # compute / comm wall-ish split (summed durations; not overlap-aware yet)
    if not empty_kd:
        comm_us = float(kd.loc[kd["Accelerator Core"] == "COMMUNICATION", "Duration(us)"].sum())
        compute_us = float(kd.loc[kd["Accelerator Core"] != "COMMUNICATION", "Duration(us)"].sum())
        start = pd.to_numeric(kd["Start Time(us)"], errors="coerce")
        dur = pd.to_numeric(kd["Duration(us)"], errors="coerce")
        wall_us = float((start + dur).max() - start.min()) if start.notna().any() else 0.0
    else:
        comm_us = compute_us = wall_us = 0.0

    trace_path = _first(data_dir, "msprof_*.json")
    step_trace = _step_trace_from_overlap(overlap, dev) if overlap else pd.DataFrame()

    meta = {
        "data_dir": data_dir,
        "format": "msprof",
        "has_shapes": has_shapes,            # True iff the capture recorded op shapes
        "rank": 0,
        "device_id": dev,
        "step": None,
        "multi_card": comm_us > 0,
        "counts": {
            "op_types": int(op_stat.shape[0]),
            "kernels": int(kd.shape[0]),
            "comm_ops": len(communication),
        },
        "msprof": {
            "compute_us": round(compute_us, 1),
            "comm_us": round(comm_us, 1),
            "wall_us": round(wall_us, 1),
            "comm_pct": round(100.0 * comm_us / (compute_us + comm_us), 1) if (compute_us + comm_us) else 0.0,
            "db_path": db_path if has_db else None,
            "trace_path": trace_path,
            "overlap": {k: round(v, 1) for k, v in overlap.items()},
            "comm_breakdown": comm_breakdown,
        },
        "memory_level": False,
        "file_sizes": {},
    }

    empty = pd.DataFrame()
    return ProfileData(
        data_dir=data_dir,
        step_trace=step_trace,               # synthesized from the Overlap layer
        op_statistic=op_stat,
        api_statistic=empty,
        kernel_details=kd,
        operator_details=empty,
        communication=communication,
        communication_raw={},
        communication_matrix={},
        trace_path=None,                     # msprof.json handled separately (big)
        meta=meta,
    )
