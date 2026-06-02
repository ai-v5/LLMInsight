"""Insight layer: rule-engine cards + (optional, disabled-by-default) LLM narrative.

`generate_insights` always returns the structured rule cards (the fact base) so
the UI works with zero LLM configured. If an LLM provider is explicitly enabled
AND a key is present, it additionally fills `narrative` with a natural-language
diagnosis built ONLY from the privacy-safe summary. Any call failure degrades
gracefully to cards-only with the error surfaced.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .provider import get_provider, load_config, LLMConfig, Provider
from .summarizer import build_summary, build_messages

__all__ = ["generate_insights", "get_provider", "load_config",
           "LLMConfig", "Provider", "build_summary", "build_messages"]


def generate_insights(m: Dict[str, Any],
                      cards: Optional[List[Dict[str, Any]]] = None,
                      capture: Optional[Dict[str, Any]] = None,
                      call_llm: bool = True) -> Dict[str, Any]:
    """Build the insight payload (rule cards + privacy-safe summary).

    The actual LLM network call happens only when `call_llm` is True AND a
    provider is enabled+keyed. The GET /api/insights endpoint passes
    call_llm=False so merely opening the panel never spends a request; the
    explicit POST /api/llm (the "运行 LLM 分析" button) is the only trigger.
    """
    from ..rules import run_rules

    if capture is None:
        # Use the profiling-derived config from compute_all (never a script).
        capture = (m.get("meta", {}) or {}).get("config") or {"found": False, "env": {}, "flags": {}}
    if cards is None:
        cards = run_rules(m, capture)

    summary = build_summary(m, cards, capture)
    provider = get_provider()
    status = provider.status()

    result: Dict[str, Any] = {
        "llm": status,            # provider/model/enabled/available/reason (no key)
        "cards": cards,           # always present — renders without any LLM
        "summary": summary,       # exactly what WOULD be sent (transparency)
        "narrative": None,
        "error": None,
    }

    if call_llm and provider.available:
        try:
            system, user = build_messages(summary)
            result["narrative"] = provider.complete(system, user)
        except Exception as exc:  # network/auth/parse — degrade to cards-only
            result["error"] = f"{type(exc).__name__}: {exc}"

    return result
