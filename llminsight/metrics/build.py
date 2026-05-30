"""Orchestrator: ProfileData -> full metrics dict (all sections)."""
from __future__ import annotations

from typing import Any, Dict

from ..config import SETTINGS
from . import core
from .efficiency import compute_efficiency
from .timeline import compute_timeline
from .smart_timeline import compute_smart_timeline


def compute_all(prof, capture: Dict[str, Any] = None) -> Dict[str, Any]:
    # `capture` (launch-script env/flags) feeds the What-if 现实地板 caveats. The
    # server passes the one it already parsed; standalone callers (report CLI) leave
    # it None and we read it here so the analysis still reflects the loaded run.
    if capture is None:
        from ..rules.engine import read_capture_config
        capture = read_capture_config()
    ov = core.overview(prof)
    eff = compute_efficiency(prof)
    # smart_timeline joins trace slices to eff["kernel_index"]; drop that heavy
    # index from the public efficiency payload once the timeline has consumed it.
    smart_tl = compute_smart_timeline(prof, eff)
    eff.pop("kernel_index", None)
    return {
        "meta": {**prof.meta, "settings": SETTINGS.to_dict()},
        "overview": ov,
        "hotspots": core.hotspots(prof),
        "efficiency": eff,
        "communication": core.communication(prof),
        "hidden_overhead": core.hidden_overhead(prof, ov),
        "attribution": core.attribution(prof),
        "memory": core.memory(prof),
        "theoretical": core.theoretical(prof, ov, eff, capture),
        "timeline": compute_timeline(prof),
        "smart_timeline": smart_tl,
    }
