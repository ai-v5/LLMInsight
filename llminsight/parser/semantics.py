"""Recover framework evidence omitted from device CSVs, without changing raw data."""
from pathlib import Path
from contextlib import closing
import sqlite3

import pandas as pd


def annotate_workload(data_dir, kernels):
    evidence = {"optimizer_window_count": 0, "optimizer_kernel_count": 0,
                "mega_moe_input_shapes": [], "source": None}
    required = {"Device_id", "Stream ID", "Task ID", "Start Time(us)"}
    if not required.issubset(kernels.columns):
        return evidence
    for path in sorted(Path(data_dir).resolve().glob("ascend_pytorch_profiler_*.db")):
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as con:
                windows = con.execute("""
                    SELECT CAST(p.startNs AS INTEGER), CAST(p.endNs AS INTEGER), p.globalTid
                    FROM PYTORCH_API p JOIN STRING_IDS s ON p.name=s.id
                    WHERE s.value LIKE 'Optimizer.step%'
                """).fetchall()
                # Worker-thread launches need not share the annotation thread.
                # This is a process-local optimizer window inference, not call ancestry.
                launches = con.execute("""
                    SELECT t.deviceId,t.streamId,t.taskId,t.startNs,a.startNs,a.globalTid
                    FROM TASK t JOIN CANN_API a ON t.connectionId=a.connectionId
                """).fetchall() if windows else []
                shapes = con.execute("""
                    SELECT DISTINCT sh.value FROM PYTORCH_API p
                    JOIN STRING_IDS s ON p.name=s.id
                    JOIN STRING_IDS sh ON p.inputShapes=sh.id
                    WHERE LOWER(s.value) LIKE '%mega%moe%'
                      AND LOWER(s.value) NOT LIKE '%backward%'
                """).fetchall()
        except (sqlite3.Error, OSError):
            continue
        optimizer_keys = {}
        for device, stream, task, start, launch, tid in launches:
            if any(lo <= launch < hi and (int(tid) >> 32) == (int(owner) >> 32)
                   for lo, hi, owner in windows):
                optimizer_keys.setdefault((device, stream, task), []).append(start / 1000.0)
        cols = [pd.to_numeric(kernels[c], errors="coerce") for c in
                ("Device_id", "Stream ID", "Task ID", "Start Time(us)")]
        mask = [any(abs(ts - value) <= 1.0 for value in optimizer_keys.get((d, s, t), []))
                if pd.notna(ts) else False
                for d, s, t, ts in zip(*cols)]
        if "Workload Phase" not in kernels:
            kernels["Workload Phase"] = "model"
        kernels.loc[mask, "Workload Phase"] = "optimizer"
        evidence["optimizer_window_count"] += len(windows)
        evidence["optimizer_kernel_count"] += sum(mask)
        evidence["mega_moe_input_shapes"].extend(x[0] for x in shapes if x[0])
        evidence["source"] = "framework_db_optimizer_launch_windows"
    evidence["mega_moe_input_shapes"] = sorted(set(evidence["mega_moe_input_shapes"]))
    return evidence
