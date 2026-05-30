"""Orchestrator: ProfileData -> full metrics dict (all sections)."""
from __future__ import annotations

from typing import Any, Dict

from ..config import SETTINGS
from . import core
from .efficiency import compute_efficiency
from .timeline import compute_timeline


def compute_all(prof) -> Dict[str, Any]:
    ov = core.overview(prof)
    eff = compute_efficiency(prof)
    return {
        "meta": {**prof.meta, "settings": SETTINGS.to_dict()},
        "overview": ov,
        "hotspots": core.hotspots(prof),
        "efficiency": eff,
        "communication": core.communication(prof),
        "hidden_overhead": core.hidden_overhead(prof, ov),
        "attribution": core.attribution(prof),
        "memory": core.memory(prof),
        "theoretical": core.theoretical(prof, ov, eff),
        "timeline": compute_timeline(prof),
    }
