"""Install to an isolated directory, exercise the bundled app, then uninstall."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
import winreg
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
INSTALL = BUILD / f"installer-smoke-{os.getpid()}"
STATE = BUILD / f"installer-smoke-state-{os.getpid()}"
INSTALLER = ROOT / "dist" / "LLMInsight-Setup-x64.exe"
FIXTURE = ROOT / "tests" / "fixtures" / "profiling_recipe"
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{B8A24B86-83CB-4B5A-AE5F-E5D7B3D504BC}_is1"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check_no_existing_install() -> None:
    if INSTALL.exists():
        raise RuntimeError(f"Refusing to overwrite an existing path: {INSTALL}")
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
            try:
                with winreg.OpenKey(hive, UNINSTALL_KEY, 0, winreg.KEY_READ | view):
                    raise RuntimeError("Existing LLMInsight installation found; refusing to overwrite it")
            except FileNotFoundError:
                pass


def request(url: str, payload: dict | None = None) -> bytes:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {
        "Content-Type": "application/json",
        "X-LLMInsight-Request": "1",
        "Origin": url.rsplit("/api/", 1)[0],
    }
    with OPENER.open(urllib.request.Request(url, data=body, headers=headers), timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP status: {response.status}")
        return response.read()


def main() -> None:
    check_no_existing_install()
    if not INSTALLER.is_file() or not FIXTURE.is_dir():
        raise RuntimeError("Installer or synthetic profiling fixture is missing")
    STATE.mkdir(parents=True, exist_ok=True)
    temp = BUILD / "installer-smoke-tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLMINSIGHT_")}
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    env["LLMINSIGHT_STATE_DIR"] = str(STATE)
    env["TEMP"] = env["TMP"] = str(temp)
    process = None
    installed = False
    try:
        subprocess.run([str(INSTALLER), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LANG=chinesesimp",
                        "/NOICONS", f"/DIR={INSTALL}"], check=True, timeout=180, env=env)
        installed = True
        exe = INSTALL / "LLMInsight.exe"
        for required in (exe, INSTALL / "LICENSE", INSTALL / "THIRD_PARTY_NOTICES.md",
                         INSTALL / "licenses" / "python" / "LICENSE.txt",
                         INSTALL / "licenses" / "python-packages" / "numpy" / "METADATA"):
            if not required.is_file():
                raise RuntimeError(f"Installed file missing: {required}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        with (BUILD / "installer-smoke-server.log").open("wb") as log:
            process = subprocess.Popen([str(exe), "--port", str(port), "--no-browser"],
                                       cwd=INSTALL, env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Installed server exited: {process.returncode}")
                try:
                    meta = json.loads(request(base + "/api/meta"))
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Installed server did not start")
                    time.sleep(0.5)
            if meta["status"] != "idle" or b"LLMInsight" not in request(base + "/"):
                raise RuntimeError("Installed application did not start in idle state")
            for path in ("/css/app.css", "/js/app.js", "/vendor/echarts.min.js"):
                if not request(base + path):
                    raise RuntimeError(f"Empty installed web asset: {path}")
            result = json.loads(request(base + "/api/load", {"dir": str(FIXTURE)}))
            if not result.get("ok"):
                raise RuntimeError(f"Synthetic profile rejected: {result}")
            deadline = time.monotonic() + 60
            while True:
                meta = json.loads(request(base + "/api/meta"))
                if meta["status"] == "ready":
                    break
                if meta["status"] == "error" or time.monotonic() >= deadline:
                    raise RuntimeError(f"Synthetic profile did not load: {meta.get('error')}")
                time.sleep(0.5)
            json.loads(request(base + "/api/overview"))
            chip = json.loads(request(base + "/api/chip", {"chip": "950DT"}))
            if not chip.get("ok"):
                raise RuntimeError(f"Bundled chip configuration failed: {chip}")
            print("PASS: installed app started without Python on PATH, served assets, loaded synthetic data and switched chip")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        uninstaller = INSTALL / "unins000.exe"
        if installed and uninstaller.is_file():
            uninstall_log = BUILD / f"installer-smoke-uninstall-{os.getpid()}.log"
            subprocess.run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                            f"/LOG={uninstall_log}"],
                           check=True, timeout=180, env=env)
            deadline = time.monotonic() + 10
            while True:
                leftovers = [path for path in INSTALL.rglob("*") if path.is_file()]
                if not leftovers or time.monotonic() >= deadline:
                    break
                time.sleep(0.25)
            if leftovers:
                raise RuntimeError(f"Uninstaller left {len(leftovers)} file(s) in {INSTALL}; see {uninstall_log}")
            print("PASS: isolated test installation removed its files")


if __name__ == "__main__":
    main()
