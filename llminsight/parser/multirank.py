"""Multi-rank profiling root support (cluster capture layout).

A multi-node Ascend training run writes ONE directory per NPU rank, usually

    <host>_<pid>_<timestamp>_ascend_pt/
        ASCEND_PROFILER_OUTPUT/        # the 8-file torch_npu capture
        profiler_info_<id>.json
        PROF_<id>/...

A user naturally points LLMInsight at the PARENT of those rank directories.
The single-rank metric model cannot aggregate 256 full captures, and silently
loading zero files there would render an empty dashboard, so this module:

  1. recognises the root (`discover_rank_dirs`),
  2. picks a deterministic, statistically representative rank
     (median step time; overridable with ``LLMINSIGHT_RANK``),
  3. summarises every rank's ``step_trace_time.csv`` into a lightweight
     cluster view (M1 slow-rank / load-imbalance evidence).

The cluster summary deliberately reads only the small per-rank
``step_trace_time.csv`` (a few hundred bytes each).  Reading every rank's
``communication.json`` is opt-in (``LLMINSIGHT_CLUSTER_COMM=1``) because a
256-rank capture is ~130 MB of JSON.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional

# Files that identify a torch_npu ``ASCEND_PROFILER_OUTPUT`` capture.  Kept in
# sync with ``llminsight.server.app._PROFILE_MARKERS`` but defined here so the
# parser layer does not import the HTTP layer.
_PROFILE_MARKERS = (
    "kernel_details.csv", "trace_view.json", "step_trace_time.csv",
    "op_statistic.csv", "operator_details.csv", "communication.json",
)
_MSPROF_MARKERS = ("mindstudio_insight_data.db",)

_RANK_DIR_RE = re.compile(r"^(?P<host>.+?)_(?P<pid>\d+)_(?P<ts>\d+)_ascend_pt$")
_OUTPUT_SUBDIR = "ASCEND_PROFILER_OUTPUT"

# Metric columns summarised for every rank (name -> step_trace column).
_STEP_COLUMNS = {
    "computing_us": "Computing",
    "comm_not_overlapped_us": "Communication(Not Overlapped)",
    "overlapped_us": "Overlapped",
    "communication_us": "Communication",
    "free_us": "Free",
    "stage_us": "Stage",
    "bubble_us": "Bubble",
    "preparing_us": "Preparing",
}


def has_profile_files(path: str) -> bool:
    """True when *path* itself carries a recognisable profiler capture."""
    if not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    if any(m in names for m in _PROFILE_MARKERS) or any(m in names for m in _MSPROF_MARKERS):
        return True
    return any(n.startswith("op_summary_") and n.endswith(".csv") for n in names)


def rank_host(name: str) -> str:
    """Host part of ``<host>_<pid>_<ts>_ascend_pt`` (falls back to the name)."""
    m = _RANK_DIR_RE.match(name)
    if m:
        return m.group("host")
    return name[:-len("_ascend_pt")] if name.endswith("_ascend_pt") else name


def _is_rank_dir_name(name: str) -> bool:
    return name.endswith("_ascend_pt")


def rank_profile_dir(rank_dir: str) -> Optional[str]:
    """Directory to hand to the single-rank loader for one rank directory."""
    out = os.path.join(rank_dir, _OUTPUT_SUBDIR)
    if has_profile_files(out):
        return out
    if has_profile_files(rank_dir):
        return rank_dir
    return None


def discover_rank_dirs(root: str) -> List[str]:
    """Rank directories directly below *root* ([] when *root* is single-rank)."""
    if not root or not os.path.isdir(root):
        return []
    if has_profile_files(root):
        return []                      # already a leaf capture
    try:
        with os.scandir(root) as it:
            entries = sorted(it, key=lambda e: e.name)
    except OSError:
        return []
    found: List[str] = []
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        name = entry.name
        if _is_rank_dir_name(name):
            if rank_profile_dir(entry.path):
                found.append(entry.path)
        elif name != _OUTPUT_SUBDIR:
            # tolerate roots whose rank dirs lost the `_ascend_pt` suffix but
            # still hold an ASCEND_PROFILER_OUTPUT capture
            if rank_profile_dir(entry.path):
                found.append(entry.path)
    return found


def looks_like_multirank_root(path: str) -> bool:
    """Cheap check used by the directory picker / load validation."""
    if not path or not os.path.isdir(path):
        return False
    if has_profile_files(path):
        return False
    try:
        with os.scandir(path) as entries:
            for i, entry in enumerate(entries):
                if i >= 512:
                    break
                try:
                    if entry.is_dir() and (_is_rank_dir_name(entry.name)
                                           or os.path.isdir(os.path.join(entry.path, _OUTPUT_SUBDIR))):
                        return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


# --------------------------------------------------------------------------- #
# per-rank step_trace summary
def _num(value: Any) -> Optional[float]:
    try:
        f = float(str(value).strip().strip('"').strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read_step_row(profile_dir: str) -> Dict[str, Any]:
    """Parse the first step of a rank's step_trace_time.csv into plain floats."""
    import pandas as pd

    path = os.path.join(profile_dir, "step_trace_time.csv")
    row: Dict[str, Any] = {}
    if not os.path.isfile(path):
        return row
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python")
        except Exception:
            return row
    if df is None or df.empty:
        return row
    first = df.iloc[0]
    for key, column in _STEP_COLUMNS.items():
        if column in df.columns:
            val = _num(first.get(column))
            if val is not None:
                row[key] = val
    for key, column in (("device_id", "Device_id"), ("step", "Step")):
        if column in df.columns:
            val = _num(first.get(column))
            if val is not None:
                row[key] = int(val) if val.is_integer() else val
    if "step_count" in df.columns:
        row["step_count"] = len(df)
    else:
        row["step_count"] = len(df)
    stage = row.get("stage_us")
    if stage and stage > 0:
        comm_visible = row.get("comm_not_overlapped_us") or 0.0
        computing = row.get("computing_us") or 0.0
        comm_total = row.get("communication_us")
        row["comm_total_pct"] = round(100.0 * comm_total / stage, 3) if comm_total else None
        row["effective_compute_pct"] = round(100.0 * computing / stage, 3)
        row["comm_not_overlapped_pct"] = round(100.0 * comm_visible / stage, 3)
        overlap = row.get("overlapped_us") or 0.0
        if comm_total and comm_total > 0:
            row["overlap_rate_pct"] = round(100.0 * overlap / comm_total, 3)
        else:
            row["overlap_rate_pct"] = None
    return row


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {}
    import statistics

    ordered = sorted(values)
    n = len(ordered)
    mean = float(statistics.fmean(ordered))
    median = float(statistics.median(ordered))
    std = float(statistics.pstdev(ordered)) if n > 1 else 0.0
    mad = float(statistics.median([abs(v - median) for v in ordered]))
    def _q(frac: float) -> float:
        idx = min(n - 1, max(0, int(round(frac * (n - 1)))))
        return ordered[idx]
    return {
        "n": n,
        "mean": round(mean, 3),
        "median": round(median, 3),
        "std": round(std, 3),
        "cv_pct": round(100.0 * std / mean, 4) if mean else None,
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "p05": round(_q(0.05), 3),
        "p95": round(_q(0.95), 3),
        "mad": round(mad, 3),
        "max_over_median": round(ordered[-1] / median, 4) if median else None,
        "min_over_median": round(ordered[0] / median, 4) if median else None,
    }


def _detect_outliers(ranks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stage-time outliers vs the cluster median (robust MAD threshold)."""
    stages = [r["stage_us"] for r in ranks if r.get("stage_us")]
    if len(stages) < 4:
        return {"rule": "too few ranks", "ranks": []}
    import statistics

    median = statistics.median(stages)
    mad = statistics.median([abs(v - median) for v in stages])
    sigma = 1.4826 * mad
    threshold = median + max(3.0 * sigma, 0.02 * median)
    flagged = [r for r in ranks if r.get("stage_us") and r["stage_us"] > threshold]
    return {
        "rule": ("stage_us > median + max(3 x 1.4826 x MAD, 2% x median); "
                 f"median={median:.1f}us mad={mad:.1f}us threshold={threshold:.1f}us"),
        "threshold_us": round(threshold, 1),
        "ranks": [
            {"rank_dir": r["rank_dir"], "host": r["host"], "device_id": r.get("device_id"),
             "stage_us": r["stage_us"], "vs_median_pct": round(100.0 * (r["stage_us"] / median - 1.0), 3)}
            for r in sorted(flagged, key=lambda r: r["stage_us"], reverse=True)
        ],
    }


def _by_host(ranks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in ranks:
        groups.setdefault(r.get("host") or "?", []).append(r)
    out = []
    for host, members in groups.items():
        stages = [m["stage_us"] for m in members if m.get("stage_us")]
        computings = [m["computing_us"] for m in members if m.get("computing_us")]
        out.append({
            "host": host,
            "n_ranks": len(members),
            "stage_mean_us": round(sum(stages) / len(stages), 1) if stages else None,
            "stage_max_us": round(max(stages), 1) if stages else None,
            "stage_spread_us": round(max(stages) - min(stages), 1) if len(stages) > 1 else 0.0,
            "computing_mean_us": round(sum(computings) / len(computings), 1) if computings else None,
        })
    return sorted(out, key=lambda h: (h["stage_mean_us"] or 0.0), reverse=True)


def _read_comm_totals(profile_dir: str) -> Optional[Dict[str, Any]]:
    """Rank-level communication totals (opt-in; each file is ~0.5 MB)."""
    import json

    path = os.path.join(profile_dir, "communication.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    by_type: Dict[str, Dict[str, float]] = {}
    for step_name, groups in (raw or {}).items():
        if not isinstance(groups, dict):
            continue
        for kind in ("collective", "p2p"):
            for name, info in (groups.get(kind) or {}).items():
                m = re.search(r"hcom_([A-Za-z0-9]+?)_", name or "")
                ctype = m.group(1) if m else ("Total" if str(name).lower().startswith("total") else "unknown")
                tinfo = info.get("Communication Time Info", {}) or {}
                if ctype == "Total":
                    continue
                slot = by_type.setdefault(ctype, {"elapse_ms": 0.0, "wait_ms": 0.0, "count": 0})
                slot["elapse_ms"] += float(tinfo.get("Elapse Time(ms)", 0) or 0)
                slot["wait_ms"] += float(tinfo.get("Wait Time(ms)", 0) or 0)
                slot["count"] += 1
    for slot in by_type.values():
        for key in ("elapse_ms", "wait_ms"):
            slot[key] = round(slot[key], 3)
    total_elapse = sum(v["elapse_ms"] for v in by_type.values())
    total_wait = sum(v["wait_ms"] for v in by_type.values())
    return {
        "by_type": by_type,
        "total_elapse_ms": round(total_elapse, 3),
        "total_wait_ms": round(total_wait, 3),
        "wait_pct": round(100.0 * total_wait / total_elapse, 3) if total_elapse else None,
    }


def pick_representative(ranks: List[Dict[str, Any]], requested: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Median-step rank (``median_step_time``) or an explicit rank request.

    ``LLMINSIGHT_RANK`` accepts a 0-based index into the sorted rank list or a
    substring of the rank directory name (e.g. ``CPU1_1276060``).
    """
    if not ranks:
        return None
    ordered = sorted(ranks, key=lambda r: r["rank_dir"])
    if requested:
        req = str(requested).strip()
        if req.isdigit():
            idx = int(req)
            if 0 <= idx < len(ordered):
                return ordered[idx]
        for r in ordered:
            if req in r["rank_dir"]:
                return r
    with_stage = [r for r in ranks if r.get("stage_us")]
    if not with_stage:
        return ordered[0]
    import statistics

    median = statistics.median([r["stage_us"] for r in with_stage])
    return min(with_stage, key=lambda r: (abs(r["stage_us"] - median), r["rank_dir"]))


def summarize_cluster(root: str, rank_dirs: List[str],
                      requested: Optional[str] = None,
                      deep_comm: bool = False) -> Dict[str, Any]:
    ranks: List[Dict[str, Any]] = []
    for idx, rank_dir in enumerate(sorted(rank_dirs)):
        profile_dir = rank_profile_dir(rank_dir)
        if not profile_dir:
            continue
        name = os.path.basename(rank_dir)
        row = read_step_row(profile_dir)
        if not row:
            continue
        entry = {
            "index": idx,
            "rank_dir": name,
            "profile_dir": profile_dir,
            "host": rank_host(name),
            **{k: row.get(k) for k in
               ("device_id", "step", "step_count", *_STEP_COLUMNS.keys())},
            "comm_total_pct": row.get("comm_total_pct"),
            "effective_compute_pct": row.get("effective_compute_pct"),
            "comm_not_overlapped_pct": row.get("comm_not_overlapped_pct"),
            "overlap_rate_pct": row.get("overlap_rate_pct"),
        }
        if deep_comm:
            comm = _read_comm_totals(profile_dir)
            if comm:
                entry["comm_elapse_ms"] = comm["total_elapse_ms"]
                entry["comm_wait_ms"] = comm["total_wait_ms"]
                entry["comm_wait_pct"] = comm["wait_pct"]
                entry["comm_by_type"] = comm["by_type"]
        ranks.append(entry)
    if not ranks:
        return {"available": False, "reason": "no readable step_trace_time.csv under rank dirs"}

    rep = pick_representative(ranks, requested)
    stats = {key: _stats([r[key] for r in ranks if r.get(key) is not None])
             for key in ("stage_us", "computing_us", "comm_not_overlapped_us",
                         "overlapped_us", "communication_us", "free_us",
                         "comm_total_pct", "effective_compute_pct", "overlap_rate_pct")}
    stage_ranked = sorted([r for r in ranks if r.get("stage_us")],
                          key=lambda r: r["stage_us"], reverse=True)
    outliers = _detect_outliers(ranks)
    by_host = _by_host(ranks)
    devices = sorted({r["device_id"] for r in ranks if r.get("device_id") is not None})
    steps = sorted({r["step"] for r in ranks if r.get("step") is not None})
    payload: Dict[str, Any] = {
        "available": True,
        "requested_dir": root,
        "n_ranks": len(ranks),
        "layout": "torch_npu_ascend_pt_per_rank",
        "device_ids": devices,
        "step_ids": steps,
        "single_step_capture": len(steps) <= 1,
        "representative": {
            "rank_dir": rep["rank_dir"],
            "profile_dir": rep["profile_dir"],
            "host": rep["host"],
            "device_id": rep.get("device_id"),
            "step": rep.get("step"),
            "stage_us": rep.get("stage_us"),
            "selection": ("requested:%s" % requested) if requested else "median_step_time",
        },
        "ranks": ranks,
        "stats": stats,
        "slowest": stage_ranked[:5],
        "fastest": list(reversed(stage_ranked[-5:])),
        "outliers": outliers,
        "by_host": by_host,
        "deep_comm": bool(deep_comm),
        "note": ("Cluster summary from each rank's step_trace_time.csv; the "
                 "single-rank sections use the representative rank. A capture "
                 "with one step window is a snapshot, not a steady-state average."),
    }
    if deep_comm:
        elapses = [r["comm_elapse_ms"] for r in ranks if r.get("comm_elapse_ms") is not None]
        payload["comm_stats"] = _stats(elapses)
        agg: Dict[str, Dict[str, float]] = {}
        for r in ranks:
            for ctype, slot in (r.get("comm_by_type") or {}).items():
                slot_out = agg.setdefault(ctype, {"elapse_ms": 0.0, "wait_ms": 0.0, "count": 0.0})
                slot_out["elapse_ms"] += slot["elapse_ms"]
                slot_out["wait_ms"] += slot["wait_ms"]
                slot_out["count"] += slot["count"]
        for ctype, slot in agg.items():
            for key in ("elapse_ms", "wait_ms"):
                slot[key] = round(slot[key], 1)
            slot["wait_pct"] = round(100.0 * slot["wait_ms"] / slot["elapse_ms"], 2) if slot["elapse_ms"] else None
            slot["rank_mean_elapse_ms"] = round(slot["elapse_ms"] / len(ranks), 1)
        payload["comm_by_type_rank_mean"] = agg
    return payload


def resolve_multirank(data_dir: str, requested: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve *data_dir* to a representative single rank + cluster summary.

    Returns ``None`` when *data_dir* is a normal single-rank capture (so callers
    keep the existing behaviour untouched).
    """
    root = os.path.abspath(data_dir)
    rank_dirs = discover_rank_dirs(root)
    if not rank_dirs:
        return None
    requested = requested if requested is not None else os.environ.get("LLMINSIGHT_RANK")
    deep_comm = str(os.environ.get("LLMINSIGHT_CLUSTER_COMM", "")).strip() not in ("", "0", "false", "False")
    cluster = summarize_cluster(root, rank_dirs, requested=requested, deep_comm=deep_comm)
    if not cluster.get("available"):
        return None
    return {
        "kind": "multirank",
        "root": root,
        "rank_dirs": rank_dirs,
        "representative": cluster["representative"]["profile_dir"],
        "cluster": cluster,
    }
