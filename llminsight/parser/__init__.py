"""Parser layer: raw Ascend profiler output -> unified ProfileData model.

Two on-disk layouts are auto-detected and dispatched:
  * torch_npu  ASCEND_PROFILER_OUTPUT  -> parser.profile  (kernel_details.csv …)
  * MindStudio msprof export           -> parser.msprof   (mindstudio_insight_data.db …)
"""
from .profile import ProfileData, load_profile as _load_ascend
from .msprof import is_msprof_dir, load_msprof_profile

__all__ = ["ProfileData", "load_profile", "is_msprof_dir"]


def load_profile(data_dir: str) -> ProfileData:
    """Auto-detect the profiling layout and dispatch to the matching loader."""
    if is_msprof_dir(data_dir):
        return load_msprof_profile(data_dir)
    return _load_ascend(data_dir)
