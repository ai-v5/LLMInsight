"""`python -m llminsight.server` — launch the LLMInsight web app."""
from __future__ import annotations

import argparse

from .app import LOOPBACK_HOST, serve


def main() -> None:
    ap = argparse.ArgumentParser(prog="llminsight.server",
                                 description="LLMInsight — Ascend NPU profiling insight web app")
    ap.add_argument("--host", default=LOOPBACK_HOST)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()
    if args.host != LOOPBACK_HOST:
        ap.error(f"--host 仅支持 {LOOPBACK_HOST}，不支持远程监听")
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
