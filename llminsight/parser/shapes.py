"""Helpers for parsing the quirky shape / dtype strings in kernel_details.csv.

Shapes look like   \"\"\"230602752\"\"\"   (single 1-D tensor) or
                   \"16384,7168;7168,2048\"   (two operands, ';'-separated).
After CSV decoding stray quote characters can remain, so we strip aggressively.
"""
from __future__ import annotations

from math import prod
from typing import List

from ..config import dtype_bytes

_NULLISH = {"", "N/A", "NAN", "NONE", "NULL"}


def parse_shapes(raw) -> List[List[int]]:
    """'16384,7168;7168,2048' -> [[16384,7168],[7168,2048]]."""
    if raw is None:
        return []
    s = str(raw).strip().strip('"').strip()
    if s.upper() in _NULLISH:
        return []
    tensors: List[List[int]] = []
    for part in s.split(";"):
        part = part.strip().strip('"').strip()
        if not part or part.upper() in _NULLISH:
            continue
        dims: List[int] = []
        for d in part.split(","):
            d = d.strip().strip('"').strip()
            if not d:
                continue
            try:
                dims.append(int(float(d)))
            except ValueError:
                dims = []
                break
        if dims:
            tensors.append(dims)
    return tensors


def parse_dtypes(raw) -> List[str]:
    """'BF16;BF16' or 'FLOAT' -> ['BF16','BF16'] / ['FLOAT']."""
    if raw is None:
        return []
    s = str(raw).strip().strip('"').strip()
    if s.upper() in _NULLISH:
        return []
    sep = ";" if ";" in s else ","
    return [t.strip().strip('"').strip() for t in s.split(sep) if t.strip()]


def numel(dims: List[int]) -> int:
    return int(prod(dims)) if dims else 0


def tensors_bytes(shapes: List[List[int]], dtypes: List[str]) -> int:
    """Total bytes across a list of tensors, pairing each shape with its dtype."""
    total = 0
    for i, dims in enumerate(shapes):
        dt = dtypes[i] if i < len(dtypes) else (dtypes[-1] if dtypes else "")
        total += numel(dims) * dtype_bytes(dt)
    return total
