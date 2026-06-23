#!/usr/bin/env bash
# restart_insight.sh — stop any running LLMInsight server, then start a fresh one.
#
# Usage:
#   ./restart_insight.sh                          # restart on default host/port
#   PORT=9000 ./restart_insight.sh                # override port
#   HOST=0.0.0.0 PORT=9000 ./restart_insight.sh   # bind all interfaces
#   LLMINSIGHT_LLM_ENABLED=0 ./restart_insight.sh # start with the LLM off
#
# It finds the running instance in three ways, in order: the .insight.pid file,
# a process match on "llminsight.server", and finally whatever is listening on
# $PORT. Each is asked to exit gracefully (SIGTERM) and force-killed (SIGKILL)
# only if it refuses. A fresh instance is then launched in the background
# (nohup), its PID recorded to .insight.pid, output to insight.log, and
# readiness confirmed by polling /api/meta.
set -euo pipefail

# --- locate repo root (the directory holding this script) ------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --- config (all overridable from the environment) -------------------------
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
PIDFILE="$REPO_DIR/.insight.pid"
LOGFILE="$REPO_DIR/insight.log"

# The LLM is enabled by default (the bundled key in secret/api_key.txt
# authorizes it); override with LLMINSIGHT_LLM_ENABLED=0 for a pure
# rule-engine run. Only the KB-level metric summary is ever sent upstream.
export LLMINSIGHT_LLM_ENABLED="${LLMINSIGHT_LLM_ENABLED:-1}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING="utf-8"

# --- pick a python interpreter ---------------------------------------------
# Prefer the project-local virtualenv (.venv) so the server runs with the repo's
# pinned deps without the caller having to "activate" first; fall back to any
# python3/python on PATH. Handles both the POSIX (.venv/bin) and Windows
# (.venv/Scripts) venv layouts so the same repo works under bash on either OS.
if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON="$REPO_DIR/.venv/bin/python"
elif [[ -x "$REPO_DIR/.venv/Scripts/python.exe" ]]; then
    PYTHON="$REPO_DIR/.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "[restart_insight] ERROR: no .venv and no python3/python on PATH. Create the venv:  python3 -m venv .venv  then  .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

# --- helper: ask a PID to stop (SIGTERM, wait ~5s, then SIGKILL) ------------
stop_pid() {
    local pid="$1"
    [[ -n "$pid" ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0      # already gone
    echo "[restart_insight] stopping PID $pid ..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.1
    done
    echo "[restart_insight] PID $pid ignored SIGTERM, sending SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
}

# --- stop existing instance(s) ---------------------------------------------
# 1) the PID we recorded last time
if [[ -f "$PIDFILE" ]]; then
    stop_pid "$(cat "$PIDFILE" 2>/dev/null || true)"
    rm -f "$PIDFILE"
fi

# 2) any process whose command line mentions our module
if command -v pgrep >/dev/null 2>&1; then
    while read -r p; do
        stop_pid "$p"
    done < <(pgrep -f "llminsight.server" 2>/dev/null || true)
fi

# 3) whatever still holds the port (lsof, then fuser as a fallback)
if command -v lsof >/dev/null 2>&1; then
    while read -r p; do
        stop_pid "$p"
    done < <(lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null || true)
elif command -v fuser >/dev/null 2>&1; then
    fuser -k "$PORT/tcp" 2>/dev/null || true
fi

# --- start a fresh instance ------------------------------------------------
echo "[restart_insight] starting LLMInsight on http://$HOST:$PORT (LLM=$LLMINSIGHT_LLM_ENABLED) ..."
nohup "$PYTHON" -m llminsight.server --host "$HOST" --port "$PORT" --no-browser \
    >"$LOGFILE" 2>&1 &
new_pid=$!
echo "$new_pid" > "$PIDFILE"
echo "[restart_insight] launched PID $new_pid (log: $LOGFILE)"

# --- wait for readiness ----------------------------------------------------
# Poll /api/meta until it reports ready:true (or the process dies / we time out).
# The socket only opens AFTER the profile is parsed, so a cold cache (104MB
# trace) can take a while on first launch — hence the generous ~120s budget.
if command -v curl >/dev/null 2>&1; then
    for _ in $(seq 1 240); do
        if ! kill -0 "$new_pid" 2>/dev/null; then
            echo "[restart_insight] ERROR: server exited early — see $LOGFILE" >&2
            exit 1
        fi
        if curl -fsS "http://$HOST:$PORT/api/meta" 2>/dev/null \
             | grep -Eq '"ready":[[:space:]]*true'; then
            echo "[restart_insight] ready — serving at http://$HOST:$PORT/"
            exit 0
        fi
        sleep 0.5
    done
    echo "[restart_insight] WARNING: not ready after timeout — check $LOGFILE" >&2
else
    echo "[restart_insight] (curl not found — skipping readiness check; see $LOGFILE)"
fi
