"""Streaming reader for the (large) Chrome/Perfetto trace_view.json.

The file is a flat JSON array of event objects. We avoid building the whole
list at once by walking it with json.JSONDecoder.raw_decode, yielding one event
dict at a time. Peak memory ~= file size (held as one text buffer) + one event.
"""
from __future__ import annotations

import json
from typing import Iterator, Dict, Any


def iter_events(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    dec = json.JSONDecoder()
    n = len(text)
    i = text.find("[")
    if i < 0:
        return
    i += 1
    while i < n:
        c = text[i]
        if c in " \t\r\n,":
            i += 1
            continue
        if c == "]":
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            break
        yield obj
        i = end


def event_ts_us(ev: Dict[str, Any]) -> float:
    """Trace timestamps are strings of microseconds; coerce safely."""
    ts = ev.get("ts")
    if ts is None:
        return 0.0
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0
