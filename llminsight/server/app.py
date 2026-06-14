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
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from ..config import SETTINGS, set_chip, save_last_dir, load_last_dir
from ..parser import load_profile
from ..metrics import compute_all
from ..metrics import core as metrics_core
from ..metrics.efficiency import compute_efficiency
from ..metrics.smart_timeline import compute_smart_timeline
from ..rules import run_rules
from ..parser.derive import derive_config
from ..insight import generate_insights, get_provider
from ..report import build_report_html

def _resolve_web_dir() -> str:
    """Locate the static web/ dir. In an installed wheel it ships INSIDE the
    package (llminsight/web); in a dev checkout it sits at the repo root
    (<repo>/web). LLMINSIGHT_WEB_DIR overrides both."""
    env = os.environ.get("LLMINSIGHT_WEB_DIR")
    if env:
        return env
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # <...>/llminsight
    for cand in (os.path.join(pkg, "web"),                       # installed wheel
                 os.path.join(os.path.dirname(pkg), "web")):     # dev checkout (<repo>/web)
        if os.path.isdir(cand):
            return cand
    return os.path.join(os.path.dirname(pkg), "web")


WEB_DIR = _resolve_web_dir()

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".map": "application/json",
}

# A directory "looks like" an Ascend profiler output if it carries any of these.
# Used both to gate POST /api/load and to flag candidates in the GET /api/browse
# directory listing (so the UI can highlight loadable folders).
_PROFILE_MARKERS = (
    "kernel_details.csv", "trace_view.json", "step_trace_time.csv",
    "op_statistic.csv", "operator_details.csv", "communication.json",
)


def _looks_like_profile_dir(path: str) -> bool:
    try:
        if any(os.path.isfile(os.path.join(path, f)) for f in _PROFILE_MARKERS):
            return True
        # also accept the MindStudio msprof-export layout (insight DB / op_summary)
        from ..parser.msprof import is_msprof_dir
        return is_msprof_dir(path)
    except OSError:
        return False


def _list_drives() -> List[str]:
    """Windows drive roots (C:\\, D:\\ …) so the picker can jump across volumes.
    Empty on POSIX, where everything hangs off '/'."""
    if os.name != "nt":
        return []
    import string
    return [f"{c}:\\" for c in string.ascii_uppercase if os.path.isdir(f"{c}:\\")]


def _browse_dir(raw: str) -> Dict[str, Any]:
    """List the SUBDIRECTORIES of a server-side path (files are irrelevant to a
    directory chooser). Marks each child that looks like a profiler output. This
    is a localhost dev tool, so browsing the local filesystem is intentional and
    unsandboxed (unlike the static file server). Errors are returned, not raised."""
    if not raw:
        parent = os.path.dirname(SETTINGS.data_dir)
        raw = parent if os.path.isdir(parent) else os.path.expanduser("~")
    path = os.path.normpath(os.path.expanduser(str(raw).strip()))
    if not os.path.isdir(path):
        return {"ok": False, "error": f"目录不存在：{path}"}
    try:
        names = []
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir():
                        names.append(e.name)
                except OSError:
                    continue
    except PermissionError:
        return {"ok": False, "error": f"无权限读取：{path}"}
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    names.sort(key=str.lower)
    dirs = [{"name": n, "path": os.path.join(path, n),
             "is_profile": _looks_like_profile_dir(os.path.join(path, n))}
            for n in names]
    parent = os.path.dirname(path)
    if parent == path:  # filesystem / drive root has no parent
        parent = None
    return {
        "ok": True,
        "path": path,
        "parent": parent,
        "sep": os.sep,
        "is_profile_dir": _looks_like_profile_dir(path),
        "drives": _list_drives(),
        "dirs": dirs,
    }


class AppState:
    """Holds the parsed profile + computed metrics for the *currently loaded*
    directory. Loading is lazy: the app starts idle and the user picks a
    profiling directory in the UI (POST /api/load), which (re)builds in place."""

    def __init__(self) -> None:
        # status: idle (awaiting selection) | loading | ready | error
        self.status = "idle"
        self.ready = False
        self.error: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.cards: list = []
        self.capture: Dict[str, Any] = {}
        self.load_seconds = 0.0
        self.loaded_dir: Optional[str] = None  # dir whose data is currently held
        # The parsed profile is retained (the 104MB trace is streamed, never held,
        # so this is cheap) to allow chip-switch recomputation without a reload.
        self.prof: Any = None
        self.lock = threading.Lock()
        self._load_lock = threading.Lock()  # serializes load triggers

    def build(self, data_dir: str) -> None:
        """(Re)load a profiling directory and recompute everything. Blocking —
        callers wanting a responsive socket should go through start_load()."""
        t0 = time.time()
        SETTINGS.data_dir = data_dir
        # Drop any prior data up front so a failed/partial load can never serve
        # stale numbers from the previous directory.
        self.status = "loading"
        self.ready = False
        self.error = None
        self.metrics = {}
        self.cards = []
        self.prof = None
        try:
            self.prof = load_profile(data_dir)
            # Derive model + capture config FROM THE PROFILING (not a launch
            # script) BEFORE compute_all so the What-if 现实地板 caveats and rule
            # cards reflect THIS run's actual config.
            self.capture = derive_config(self.prof)
            self.metrics = compute_all(self.prof, self.capture)
            self.cards = run_rules(self.metrics, self.capture)
            self.loaded_dir = data_dir
            save_last_dir(data_dir)        # remember for the next server start
            self.ready = True
            self.status = "ready"
        except Exception as exc:  # surface load failures to the UI
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            raise
        finally:
            self.load_seconds = round(time.time() - t0, 2)

    def start_load(self, data_dir: str) -> Dict[str, Any]:
        """Validate + kick off a background (re)build. Returns immediately so the
        HTTP socket stays responsive; the frontend polls /api/meta for progress."""
        data_dir = os.path.normpath(os.path.expanduser(str(data_dir or "").strip()))
        if not data_dir or not os.path.isdir(data_dir):
            return {"ok": False, "error": f"目录不存在：{data_dir}"}
        if not _looks_like_profile_dir(data_dir):
            return {"ok": False, "error": ("该目录下找不到 profiling 文件（需含 "
                    "kernel_details.csv / trace_view.json / step_trace_time.csv，或 "
                    "msprof 的 mindstudio_insight_data.db / op_summary_*.csv 之一）")}
        with self._load_lock:
            if self.status == "loading":
                return {"ok": False, "error": "正在加载中，请稍候"}
            self.status = "loading"
            self.ready = False
            self.error = None
        threading.Thread(target=self._load_worker, args=(data_dir,), daemon=True).start()
        return {"ok": True, "loading": True, "dir": data_dir}

    def _load_worker(self, data_dir: str) -> None:
        try:
            self.build(data_dir)
        except Exception:
            pass  # status/error already recorded by build()

    def reset(self) -> Dict[str, Any]:
        """Drop loaded data and return to the idle (awaiting-selection) state."""
        with self._load_lock:
            if self.status == "loading":
                return {"ok": False, "error": "正在加载中，无法重置"}
            self.status = "idle"
            self.ready = False
            self.error = None
            self.metrics = {}
            self.cards = []
            self.prof = None
            self.loaded_dir = None
            self.load_seconds = 0.0
        return {"ok": True}

    def switch_chip(self, key: str) -> Dict[str, Any]:
        """Re-point the chip preset and recompute only the chip-dependent
        sections (efficiency + theoretical) plus the rule cards, in place. The
        step decomposition, hotspots, communication, timeline, etc. are
        chip-independent and are left untouched. Serialized by a lock so two
        concurrent switches can't interleave."""
        with self.lock:
            if not self.ready or self.prof is None:
                return {"ok": False, "error": "profile not loaded yet"}
            if not set_chip(key):  # loads configs/chips/<key>.yaml (alias-aware)
                return {"ok": False, "error": f"unknown chip '{key}'"}
            eff = compute_efficiency(self.prof)
            # MFU/MBU/算力/带宽 on the smart-timeline scale with the chip; recompute
            # its cheap overlay (the geometry layer hits the trace-signature cache,
            # so no 104MB re-scan) before dropping the heavy index from eff.
            self.metrics["smart_timeline"] = compute_smart_timeline(self.prof, eff)
            eff.pop("kernel_index", None)
            self.metrics["efficiency"] = eff
            self.metrics["theoretical"] = metrics_core.theoretical(
                self.prof, self.metrics.get("overview", {}), eff, self.capture)
            self.metrics["meta"] = {**self.metrics.get("meta", {}),
                                    "settings": SETTINGS.to_dict()}
            self.cards = run_rules(self.metrics, self.capture)
            return {
                "ok": True,
                "chip_key": SETTINGS.chip_key,  # canonical YAML stem (alias-resolved)
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
        "status": STATE.status,            # idle | loading | ready | error
        "error": STATE.error,
        "load_seconds": STATE.load_seconds,
        "meta": STATE.metrics.get("meta"),
        "llm": get_provider().status(),
        "data_dir": STATE.loaded_dir,       # currently-loaded dir (null when idle)
        # the last loaded dir (persisted) seeds the picker; falls back to the sample
        "suggested_dir": load_last_dir() or SETTINGS.data_dir,
        "sections": ["overview", "hotspots", "efficiency", "communication",
                     "hidden_overhead", "attribution", "memory", "theoretical",
                     "timeline", "smart_timeline", "insights"],
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
    "/api/smart_timeline": _section("smart_timeline"),
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
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/browse":          # works pre-load (no ready guard)
            self._handle_browse(parsed.query)
        elif path.startswith("/api/"):
            self._handle_api(path)
        elif path == "/report.html":
            self._serve_report()
        else:
            self._serve_static(path)

    do_HEAD = do_GET

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/llm":
            self._handle_api(path)
        elif path == "/api/chip":
            self._handle_chip()
        elif path == "/api/load":
            self._handle_load()
        elif path == "/api/reset":
            self._send_json(STATE.reset())
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_browse(self, query: str):
        """GET /api/browse?path=<abs> — list server-side subdirectories for the
        directory picker. Defaults to the suggested sample's parent when empty."""
        raw = (parse_qs(query or "").get("path", [""])[0] or "").strip()
        res = _browse_dir(raw)
        self._send_json(res, 200 if res.get("ok") else 400)

    def _handle_load(self):
        """POST /api/load {"dir": "<abs>"} — validate + start a background
        (re)load of a profiling directory. The frontend then polls /api/meta."""
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n > 0 else b""
            body = json.loads(raw.decode("utf-8")) if raw else {}
            data_dir = str(body.get("dir", "")).strip()
        except Exception as exc:
            self._send_json({"ok": False, "error": f"bad request: {exc}"}, 400)
            return
        if not data_dir:
            self._send_json({"ok": False, "error": "missing 'dir'"}, 400)
            return
        res = STATE.start_load(data_dir)
        self._send_json(res, 200 if res.get("ok") else 400)

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

    def _serve_report(self):
        """GET /report.html — render the self-contained shareable report on the
        fly from the in-memory metrics + cards (same aggregated numbers the UI
        shows; never the raw trace). 503 until the profile is loaded."""
        if not STATE.ready:
            self._send_json({"error": STATE.error or "loading"}, 503)
            return
        try:
            html = build_report_html(STATE.metrics, STATE.cards)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return
        self._send(html.encode("utf-8"), "text/html; charset=utf-8")

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


def _autoload_enabled() -> bool:
    return os.environ.get("LLMINSIGHT_AUTOLOAD", "").strip().lower() in (
        "1", "true", "yes", "on")


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    # Lazy by default: start idle and let the user choose a profiling directory in
    # the UI (POST /api/load). The socket therefore opens immediately. Set
    # LLMINSIGHT_AUTOLOAD=1 to eagerly (re)load the default / LLMINSIGHT_DATA_DIR
    # sample at startup in the background (the old one-shot behavior).
    if _autoload_enabled():
        target = load_last_dir() or SETTINGS.data_dir
        print(f"[LLMInsight] autoload (LLMINSIGHT_AUTOLOAD): {target}")
        STATE.start_load(target)
    else:
        print("[LLMInsight] idle — open the app and choose a profiling directory")
    print(f"[LLMInsight] LLM={'on' if get_provider().available else 'off (default)'}")
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
