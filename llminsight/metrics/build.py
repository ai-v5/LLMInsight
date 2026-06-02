"""Orchestrator: ProfileData -> full metrics dict (all sections)."""
from __future__ import annotations

from typing import Any, Dict

from ..config import SETTINGS
from . import core
from .efficiency import compute_efficiency
from .timeline import compute_timeline
from .smart_timeline import compute_smart_timeline


def compute_all(prof, capture: Dict[str, Any] = None) -> Dict[str, Any]:
    # `capture` carries the model + training/capture config DERIVED FROM THE
    # PROFILING ITSELF (never a launch script — see parser.derive). It feeds the
    # What-if 现实地板 caveats and the rule cards. The server passes the one it
    # already derived; standalone callers (report CLI) leave it None and we
    # derive it here so the analysis always reflects the loaded run.
    if capture is None:
        from ..parser.derive import derive_config
        capture = derive_config(prof)
    ov = core.overview(prof)
    eff = compute_efficiency(prof)
    # smart_timeline joins trace slices to eff["kernel_index"]; drop that heavy
    # index from the public efficiency payload once the timeline has consumed it.
    smart_tl = compute_smart_timeline(prof, eff)
    eff.pop("kernel_index", None)
    return {
        # `config` exposes the profiling-derived model/capture/guesses (KB-level,
        # secret-free) so the report header and UI show what the DATA says, not a
        # hardcoded ModelConfig.
        "meta": {**prof.meta, "settings": SETTINGS.to_dict(), "config": capture},
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
