"""Parser layer: raw Ascend profiler output -> unified ProfileData model.

Three on-disk layouts are auto-detected and dispatched:
  * multi-rank training capture root  -> parser.multirank (representative rank +
    cluster summary; each rank's leaf capture is loaded by one of the loaders
    below)
  * torch_npu  ASCEND_PROFILER_OUTPUT  -> parser.profile  (kernel_details.csv …)
  * MindStudio msprof export           -> parser.msprof   (mindstudio_insight_data.db …)
"""
from .profile import ProfileData, load_profile as _load_ascend
from .msprof import is_msprof_dir, load_msprof_profile
from .multirank import discover_rank_dirs, looks_like_multirank_root, resolve_multirank

__all__ = ["ProfileData", "load_profile", "is_msprof_dir",
           "discover_rank_dirs", "looks_like_multirank_root", "resolve_multirank"]


def _load_single(data_dir: str) -> ProfileData:
    """Auto-detect the single-rank capture layout and dispatch to its loader."""
    if is_msprof_dir(data_dir):
        return load_msprof_profile(data_dir)
    return _load_ascend(data_dir)


def _empty_capture_warning(prof: ProfileData) -> ProfileData:
    counts = (prof.meta or {}).get("counts") or {}
    has_counts = any(int(v or 0) > 0 for v in counts.values())
    has_steps = getattr(prof, "step_trace", None) is not None and not prof.step_trace.empty
    has_comm = bool(getattr(prof, "communication", None))
    if has_counts or has_steps or has_comm:
        return prof
    prof.meta.setdefault("warnings", []).append(
        "在该目录下没有识别到 profiling 数据文件（step_trace_time.csv / "
        "kernel_details.csv / trace_view.json 等）；如果这是一个多 rank 采集根目录，"
        "请确认它下面有 *_ascend_pt/ASCEND_PROFILER_OUTPUT 子目录。")
    return prof


def load_profile(data_dir: str) -> ProfileData:
    """Load a capture directory.

    A multi-rank training root (one ``*_ascend_pt`` directory per NPU) is turned
    into its representative rank (median step time, or ``LLMINSIGHT_RANK``) plus
    a ``prof.cluster`` summary, so the single-rank metric pipeline can run on
    real cluster captures without the caller picking a rank by hand.
    """
    resolved = resolve_multirank(data_dir)
    if resolved is not None:
        prof = _load_single(resolved["representative"])
        cluster = resolved["cluster"]
        prof.cluster = cluster
        prof.meta["multirank"] = {
            "requested_dir": resolved["root"],
            "n_ranks": cluster.get("n_ranks"),
            "representative": cluster.get("representative"),
            "rank_dirs": [r["rank_dir"] for r in cluster.get("ranks", [])],
        }
        prof.meta["data_dir"] = resolved["root"]
        return _empty_capture_warning(prof)
    return _empty_capture_warning(_load_single(data_dir))
