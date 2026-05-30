"""Rule engine: metrics -> structured insight cards (现象/根因/建议/预计收益/置信度)."""
from .engine import run_rules, read_capture_config

__all__ = ["run_rules", "read_capture_config"]
