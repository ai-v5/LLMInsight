"""Tiny on-disk JSON cache so we don't reparse the 104MB trace every restart.

Keyed by (logical key + source file size/mtime). Lives in .llminsight_cache/.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, Optional

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".llminsight_cache"
)


def _key_path(key: str) -> str:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"{key.split(':')[0]}_{h}.json")


def file_signature(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return "missing"


def cached_json(key: str, builder: Callable[[], object]) -> object:
    """Return cached JSON for `key`, else build, store, and return it."""
    path = _key_path(key)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    value = builder()
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(value, fh)
    except OSError:
        pass
    return value
