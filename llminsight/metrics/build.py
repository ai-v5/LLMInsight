"""Orchestrator: ProfileData -> full metrics dict (all sections)."""
from __future__ import annotations

from typing import Any, Dict

from ..config import SETTINGS
from . import core
from .efficiency import compute_efficiency
from .memory import compute_memory
from .matrix_power import compute_matrix_power
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
    matrix_power = compute_matrix_power(prof, eff)
    # smart_timeline joins trace slices to eff["kernel_index"]; drop that heavy
    # index from the public efficiency payload once the timeline has consumed it.
    smart_tl = compute_smart_timeline(prof, eff)
    eff.pop("kernel_index", None)
    # theoretical BEFORE hidden_overhead: the recompute view is quantified once,
    # there, and the ledger bucket mirrors it.  Computing the ledger first forced a
    # second, differently-defined estimate (model projection vs measured replay
    # duration) and the two sections published a 32x-disagreeing "recompute cost".
    theoretical = core.theoretical(prof, ov, eff, capture)
    return {
        # `config` exposes the profiling-derived model/capture/guesses (KB-level,
        # secret-free) so the report header and UI show what the DATA says, not a
        # hardcoded ModelConfig.
        "meta": {**prof.meta, "settings": SETTINGS.to_dict(), "config": capture},
        # Per-rank summary for multi-rank capture roots (empty -> unavailable);
        # every other section is the representative rank's single-rank analysis.
        "cluster": getattr(prof, "cluster", None) or {"available": False},
        "overview": ov,
        "hotspots": core.hotspots(prof),
        "efficiency": eff,
        "matrix_power": matrix_power,
        "communication": core.communication(prof),
        "hidden_overhead": core.hidden_overhead(
            # `or {}` (not None) is the explicit "quantified, found nothing" signal:
            # the ledger must not fall back to its own projection there.
            prof, ov, capture, recompute_view=theoretical.get("recompute") or {}),
        "attribution": core.attribution(prof),
        "memory": compute_memory(prof),
        "theoretical": theoretical,
        "timeline": compute_timeline(prof),
        "smart_timeline": smart_tl,
    }
