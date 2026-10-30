"""Operator-name helpers shared by the parser, metrics and presentation layers.

Two problems this module solves, both caused by the same thing: a profiler export
does not guarantee a stable, human-sized operator label.

* **Fused MC2 kernels.**  A capture that records op attributes publishes the
  profiler label ("AlltoAllvGroupedMatMul" / "GroupedMatMulAlltoAllv"), but a
  Level-1 capture (no op attrs) publishes the raw C++ kernel symbol instead, e.g.
  ``_ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel31Ascend950AllToAllVGroupedMatmul…EPh``.
  Exact-name matching silently loses those rows (they used to fall into the
  Elementwise/Other residual while the fused dispatch/combine window read 0), so
  every consumer matches them through :func:`is_mc2_fused_type`.
* **Display length.**  Those symbols are 400-600 characters long and wreck table
  layouts and card titles, so :func:`op_display_name` shortens them for display
  while the raw name stays the data key.
"""
from __future__ import annotations

# Case-insensitive markers that identify the MC2 fused AllToAllv + GroupedMatMul
# kernels in either naming form.  Keep narrow: they must never match the plain
# expert GEMM ("GroupedMatmul") or an ordinary MatMul.
MC2_FUSED_MARKERS = ("alltoallvgroupedmatmul", "groupedmatmulalltoallv")

# Canonical display labels per marker (order matters: first match wins).
MC2_FUSED_LABELS = (
    ("alltoallvgroupedmatmul", "AlltoAllvGroupedMatMul"),
    ("groupedmatmulalltoallv", "GroupedMatMulAlltoAllV"),
)

DEFAULT_DISPLAY_LIMIT = 44


def is_mc2_fused_type(name: str) -> bool:
    """True when *name* is an MC2 fused AllToAllv + GroupedMatMul kernel."""
    value = (name or "").lower()
    return any(marker in value for marker in MC2_FUSED_MARKERS)


def op_display_name(name: str, limit: int = DEFAULT_DISPLAY_LIMIT) -> str:
    """Human-sized label for an OP Type / kernel name.

    Known fused families get their canonical label (plus a hint that the record was
    a raw kernel symbol); everything else keeps its name, shortened with an ellipsis
    when it exceeds *limit* characters.  Display only — never use as a data key.
    """
    raw = str(name or "")
    if len(raw) <= limit:
        return raw
    low = raw.lower()
    for marker, label in MC2_FUSED_LABELS:
        if marker in low:
            return f"{label}（融合内核符号）"
    return raw[: limit - 1] + "…"
