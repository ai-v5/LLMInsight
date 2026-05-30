"""LLMInsight backend — pure-stdlib http.server (no FastAPI dependency, so the
app is trivially distributable).

Lifecycle: the profile is parsed and all metrics computed ONCE at startup, then
held in memory; metric sections are also disk-cached (see cache.py) so restarts
are fast. Each `/api/*` route returns a pre-computed JSON section. Static files
under `web/` are served for everything else (SPA-style, index.html default).

The LLM call is never made at startup. `/api/insights` returns rule cards +
the privacy-safe summary (narrative=null unless a provider is enabled). Only an
explicit `POST /api/llm` triggers a network call, and only if enabled+keyed.
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict
from urllib.parse import urlparse

from ..config import SETTINGS, CHIP_PRESETS, set_chip
from ..parser import load_profile
from ..metrics import compute_all
from ..metrics import core as metrics_core
from ..metrics.efficiency import compute_efficiency
from ..rules import run_rules, read_capture_config
from ..insight import generate_insights, get_provider

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "web")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".map": "application/json",
}


class AppState:
    """Loads the profile + computes everything once; routes read from here."""

    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.metrics: Dict[str, Any] = {}
        self.cards: list = []
        self.capture: Dict[str, Any] = {}
        self.load_seconds = 0.0
        # The parsed profile is retained (the 104MB trace is streamed, never held,
        # so this is cheap) to allow chip-switch recomputation without a reload.
        self.prof: Any = None
        self.lock = threading.Lock()

    def build(self) -> None:
        t0 = time.time()
        try:
            self.prof = load_profile(SETTINGS.data_dir)
            self.metrics = compute_all(self.prof)
            self.capture = read_capture_config()
            self.cards = run_rules(self.metrics, self.capture)
            self.ready = True
        except Exception as exc:  # surface load failures to the UI
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.load_seconds = round(time.time() - t0, 2)

    def switch_chip(self, key: str) -> Dict[str, Any]:
        """Re-point the chip preset and recompute only the chip-dependent
        sections (efficiency + theoretical) plus the rule cards, in place. The
        step decomposition, hotspots, communication, timeline, etc. are
        chip-independent and are left untouched. Serialized by a lock so two
        concurrent switches can't interleave."""
        with self.lock:
            if key not in CHIP_PRESETS:
                return {"ok": False, "error": f"unknown chip '{key}'"}
            if not self.ready or self.prof is None:
                return {"ok": False, "error": "profile not loaded yet"}
            set_chip(key)
            eff = compute_efficiency(self.prof)
            self.metrics["efficiency"] = eff
            self.metrics["theoretical"] = metrics_core.theoretical(
                self.prof, self.metrics.get("overview", {}), eff)
            self.metrics["meta"] = {**self.metrics.get("meta", {}),
                                    "settings": SETTINGS.to_dict()}
            self.cards = run_rules(self.metrics, self.capture)
            return {
                "ok": True,
                "chip_key": key,
                "chip": eff.get("chip"),
                "efficiency": eff,
                "theoretical": self.metrics["theoretical"],
                "cards_count": len(self.cards),
            }


STATE = AppState()


# --------------------------------------------------------------------------- #
# Route handlers — each returns a JSON-serializable object.
# --------------------------------------------------------------------------- #
def _meta() -> Dict[str, Any]:
    return {
        "ready": STATE.ready,
        "error": STATE.error,
        "load_seconds": STATE.load_seconds,
        "meta": STATE.metrics.get("meta"),
        "llm": get_provider().status(),
        "data_dir": SETTINGS.data_dir,
        "sections": ["overview", "hotspots", "efficiency", "communication",
                     "hidden_overhead", "attribution", "memory", "theoretical",
                     "timeline", "insights"],
    }


def _section(name: str) -> Callable[[], Any]:
    return lambda: STATE.metrics.get(name, {"available": False})


def _insights() -> Dict[str, Any]:
    # Cards + summary only — never spend an LLM request just to open the panel.
    # The narrative is fetched on demand via POST /api/llm.
    return generate_insights(STATE.metrics, STATE.cards, STATE.capture, call_llm=False)


def _llm_run() -> Dict[str, Any]:
    """Explicit LLM trigger. Returns narrative or a clear disabled/err reason."""
    res = generate_insights(STATE.metrics, STATE.cards, STATE.capture, call_llm=True)
    return {"llm": res["llm"], "narrative": res["narrative"],
            "error": res["error"], "summary": res["summary"]}


ROUTES: Dict[str, Callable[[], Any]] = {
    "/api/meta": _meta,
    "/api/overview": _section("overview"),
    "/api/hotspots": _section("hotspots"),
    "/api/efficiency": _section("efficiency"),
    "/api/communication": _section("communication"),
    "/api/hidden_overhead": _section("hidden_overhead"),
    "/api/attribution": _section("attribution"),
    "/api/memory": _section("memory"),
    "/api/theoretical": _section("theoretical"),
    "/api/timeline": _section("timeline"),
    "/api/insights": _insights,
    "/api/llm": _llm_run,
    "/api/all": lambda: STATE.metrics,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "LLMInsight/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter console
        return

    # -- response helpers ---------------------------------------------------
    def _send(self, body: bytes, ctype: str, status: int = 200):
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
        headers = [("Content-Type", ctype)]
        if accepts_gzip and len(body) > 1400:
            body = gzip.compress(body, 5)
            headers.append(("Content-Encoding", "gzip"))
        headers.append(("Content-Length", str(len(body))))
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj: Any, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._handle_api(path)
        else:
            self._serve_static(path)

    do_HEAD = do_GET

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/llm":
            self._handle_api(path)
        elif path == "/api/chip":
            self._handle_chip()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_chip(self):
        """POST /api/chip {"chip": "950DT"} — switch the reference chip and
        return the freshly recomputed efficiency/theoretical sections."""
        if not STATE.ready:
            self._send_json({"ok": False, "error": STATE.error or "loading"}, 503)
            return
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n > 0 else b""
            body = json.loads(raw.decode("utf-8")) if raw else {}
            key = str(body.get("chip", "")).strip()
        except Exception as exc:
            self._send_json({"ok": False, "error": f"bad request: {exc}"}, 400)
            return
        try:
            res = STATE.switch_chip(key)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
            return
        self._send_json(res, 200 if res.get("ok") else 400)

    def _handle_api(self, path: str):
        fn = ROUTES.get(path)
        if fn is None:
            self._send_json({"error": f"unknown endpoint {path}"}, 404)
            return
        if not STATE.ready and path != "/api/meta":
            self._send_json({"available": False, "ready": False,
                             "error": STATE.error or "loading"}, 503)
            return
        try:
            self._send_json(fn())
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _serve_static(self, path: str):
        rel = path.lstrip("/") or "index.html"
        full = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not full.startswith(os.path.normpath(WEB_DIR)):
            self._send_json({"error": "forbidden"}, 403)
            return
        if not os.path.isfile(full):
            full = os.path.join(WEB_DIR, "index.html")  # SPA fallback
            if not os.path.isfile(full):
                self._send_json({"error": "web/ not built"}, 404)
                return
        ext = os.path.splitext(full)[1].lower()
        ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(full, "rb") as fh:
            self._send(fh.read(), ctype)


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    print(f"[LLMInsight] loading profile from: {SETTINGS.data_dir}")
    try:
        STATE.build()
    except Exception:
        print(f"[LLMInsight] FAILED to load: {STATE.error}")
        raise
    print(f"[LLMInsight] ready in {STATE.load_seconds}s — "
          f"{len(STATE.cards)} insight cards, "
          f"LLM={'on' if get_provider().available else 'off (default)'}")
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"[LLMInsight] serving at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: _try_open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[LLMInsight] shutting down")
        httpd.shutdown()


def _try_open(url: str):
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
