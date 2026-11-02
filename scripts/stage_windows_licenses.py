"""Collect the actual bundled Python distributions' notices for the installer."""
from __future__ import annotations

import importlib.metadata
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "build" / "windows-dist" / "LLMInsight" / "licenses"
DISTRIBUTIONS = (
    "numpy", "pandas", "PyYAML", "python-dateutil", "pytz", "tzdata", "pyinstaller",
)


def main() -> None:
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        raise SystemExit(f"Python license missing: {python_license}")
    (DEST / "python").mkdir(parents=True, exist_ok=True)
    shutil.copy2(python_license, DEST / "python" / "LICENSE.txt")

    for name in DISTRIBUTIONS:
        dist = importlib.metadata.distribution(name)
        files = dist.files or ()
        metadata = [p for p in files if p.name == "METADATA" and ".dist-info/" in p.as_posix()]
        if len(metadata) != 1:
            raise SystemExit(f"Distribution metadata missing or ambiguous: {name}")
        info_dir = metadata[0].parts[0]
        notices = [p for p in files if p.parts[0] == info_dir and (
            p.name == "METADATA" or "licenses" in p.parts or
            p.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
        )]
        if len(notices) < 2:
            raise SystemExit(f"Distribution license missing: {name}")
        for notice in notices:
            target = DEST / "python-packages" / name / Path(*notice.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dist.locate_file(notice), target)
        print(f"Included {name} {dist.version} metadata and notices")


if __name__ == "__main__":
    main()
