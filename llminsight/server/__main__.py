"""`python -m llminsight.server` — launch the LLMInsight web app."""
from __future__ import annotations

import argparse

from .app import serve


def main() -> None:
    ap = argparse.ArgumentParser(prog="llminsight.server",
                                 description="LLMInsight — Ascend NPU profiling insight web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
