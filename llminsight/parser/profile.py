"""Load the 8 Ascend profiler-output files into one ProfileData model.

Indexed conceptually by (rank, step). This sample is single-card (rank 0) /
single-step (step5); the rank dimension is reserved for future multi-card data.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .trace import iter_events

_COMM_TYPE_RE = re.compile(r"hcom_([A-Za-z0-9]+?)_", re.IGNORECASE)


# --------------------------------------------------------------------------- #
def _read_csv(path: str, **kw) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", **kw)
        except Exception:
            return pd.DataFrame()


def num(series: pd.Series) -> pd.Series:
    """Coerce a possibly-dirty (trailing tab/quote) column to float."""
    if series.dtype.kind in "if":
        return series
    return pd.to_numeric(
        series.astype(str).str.strip().str.strip('"').str.strip(), errors="coerce"
    )


def comm_type_from_name(name: str) -> str:
    m = _COMM_TYPE_RE.search(name or "")
    if m:
        return m.group(1)
    if name and name.lower().startswith("total"):
        return "Total"
    return "unknown"


# --------------------------------------------------------------------------- #
@dataclass
class ProfileData:
    data_dir: str
    step_trace: pd.DataFrame
    op_statistic: pd.DataFrame
    api_statistic: pd.DataFrame
    kernel_details: pd.DataFrame
    operator_details: pd.DataFrame
    communication: List[Dict[str, Any]]          # normalized collective ops
    communication_raw: Dict[str, Any]
    communication_matrix: Dict[str, Any]
    trace_path: Optional[str]
    meta: Dict[str, Any] = field(default_factory=dict)

    # trace is never held in memory; stream on demand
    def iter_trace_events(self) -> Iterator[Dict[str, Any]]:
        if self.trace_path and os.path.exists(self.trace_path):
            yield from iter_events(self.trace_path)


# --------------------------------------------------------------------------- #
def _normalize_communication(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for step_name, groups in (raw or {}).items():
        if not isinstance(groups, dict):
            continue
        for group_kind in ("collective", "p2p"):
            entries = groups.get(group_kind, {}) or {}
            for op_name, info in entries.items():
                tinfo = info.get("Communication Time Info", {}) or {}
                binfo = info.get("Communication Bandwidth Info", {}) or {}
                links = {}
                for link, ld in binfo.items():
                    if isinstance(ld, dict):
                        links[link] = {
                            "transit_mb": ld.get("Transit Size(MB)", 0) or 0,
                            "transit_ms": ld.get("Transit Time(ms)", 0) or 0,
                            "bandwidth_gbps": ld.get("Bandwidth(GB/s)", 0) or 0,
                        }
                out.append(
                    {
                        "step": step_name,
                        "kind": group_kind,
                        "name": op_name,
                        "type": comm_type_from_name(op_name),
                        "elapse_ms": tinfo.get("Elapse Time(ms)", 0) or 0,
                        "transit_ms": tinfo.get("Transit Time(ms)", 0) or 0,
                        "wait_ms": tinfo.get("Wait Time(ms)", 0) or 0,
                        "sync_ms": tinfo.get("Synchronization Time(ms)", 0) or 0,
                        "idle_ms": tinfo.get("Idle Time(ms)", 0) or 0,
                        "wait_ratio": tinfo.get("Wait Time Ratio", 0) or 0,
                        "sync_ratio": tinfo.get("Synchronization Time Ratio", 0) or 0,
                        "start_us": tinfo.get("Start Timestamp(us)", 0) or 0,
                        "links": links,
                    }
                )
    return out


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------- #
def load_profile(data_dir: str) -> ProfileData:
    p = lambda f: os.path.join(data_dir, f)

    step_trace = _read_csv(p("step_trace_time.csv"))
    op_statistic = _read_csv(p("op_statistic.csv"))
    api_statistic = _read_csv(p("api_statistic.csv"))
    kernel_details = _read_csv(p("kernel_details.csv"), low_memory=False)
    operator_details = _read_csv(p("operator_details.csv"), low_memory=False)

    comm_raw = _load_json(p("communication.json"))
    comm_matrix = _load_json(p("communication_matrix.json"))
    communication = _normalize_communication(comm_raw)

    trace_path = p("trace_view.json")
    if not os.path.exists(trace_path):
        trace_path = None

    device_id = None
    step_id = None
    if not step_trace.empty:
        if "Device_id" in step_trace.columns:
            device_id = int(step_trace["Device_id"].iloc[0])
        if "Step" in step_trace.columns:
            step_id = int(step_trace["Step"].iloc[0])

    def fsize(f: str) -> int:
        try:
            return os.path.getsize(p(f))
        except OSError:
            return 0

    meta = {
        "data_dir": data_dir,
        "rank": 0,                      # reserved; single-card sample
        "device_id": device_id,
        "step": step_id,
        "multi_card": bool(comm_matrix.get("step5", {}).get("collective")),
        "counts": {
            "op_types": int(op_statistic.shape[0]),
            "api": int(api_statistic.shape[0]),
            "kernels": int(kernel_details.shape[0]),
            "operators": int(operator_details.shape[0]),
            "comm_ops": len(communication),
        },
        "file_sizes": {
            f: fsize(f)
            for f in (
                "step_trace_time.csv",
                "op_statistic.csv",
                "api_statistic.csv",
                "kernel_details.csv",
                "operator_details.csv",
                "communication.json",
                "communication_matrix.json",
                "trace_view.json",
            )
        },
    }

    return ProfileData(
        data_dir=data_dir,
        step_trace=step_trace,
        op_statistic=op_statistic,
        api_statistic=api_statistic,
        kernel_details=kernel_details,
        operator_details=operator_details,
        communication=communication,
        communication_raw=comm_raw,
        communication_matrix=comm_matrix,
        trace_path=trace_path,
        meta=meta,
    )
