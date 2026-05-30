#!/usr/bin/env python3
"""Build a self-contained LLMInsight wheel.

In the dev layout the runtime assets live at the repo ROOT (``web/`` and
``configs/``), OUTSIDE the importable ``llminsight`` package. A plain
``pip wheel .`` would therefore ship code without its UI or chip specs and the
installed app would not start. This script stages copies of those dirs INTO the
package (``llminsight/web``, ``llminsight/configs``) just long enough to build
the wheel, then removes them again so the source tree stays clean.

The server resolves either location at runtime (in-package first, repo-root
fallback), so both the wheel and a dev checkout work unchanged.

Usage:  python scripts/build_wheel.py
Output: dist/llminsight-<version>-py3-none-any.whl
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "llminsight"
DIST = ROOT / "dist"
# (source at repo root, staged copy inside the package)
STAGED = [(ROOT / "web", PKG / "web"), (ROOT / "configs", PKG / "configs")]


def _stage() -> None:
    for src, dst in STAGED:
        if not src.is_dir():
            raise SystemExit(f"missing source dir: {src}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  staged {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def _unstage() -> None:
    for _, dst in STAGED:
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
    for leftover in [ROOT / "build", *ROOT.glob("*.egg-info"), PKG / "llminsight.egg-info"]:
        if leftover.exists():
            shutil.rmtree(leftover, ignore_errors=True)


def main() -> int:
    if DIST.exists():
        shutil.rmtree(DIST)
    print("staging runtime assets into the package ...")
    _stage()
    try:
        try:
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(DIST)],
                cwd=ROOT, check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Fall back to pip's wheel builder if `build` is unavailable.
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
                 "--no-build-isolation", "-w", str(DIST)],
                cwd=ROOT, check=True,
            )
    finally:
        _unstage()

    wheels = sorted(DIST.glob("llminsight-*.whl"))
    if not wheels:
        print("ERROR: no wheel produced", file=sys.stderr)
        return 1
    print(f"\nbuilt: {wheels[-1].relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
