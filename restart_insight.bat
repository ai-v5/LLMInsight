@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM restart_insight.bat - stop any running LLMInsight server, then start a fresh one (Windows).
REM Windows counterpart of restart_insight.sh.
REM
REM Usage (run from a terminal, cmd.exe or PowerShell - do NOT double-click, or the
REM server dies when the console closes):
REM   restart_insight.bat                                  : restart on default host/port
REM   set "PORT=9000" & restart_insight.bat                : override port
REM   set "HOST=0.0.0.0" & set "PORT=9000" & restart_insight.bat   : bind all interfaces
REM   set "LLMINSIGHT_LLM_ENABLED=0" & restart_insight.bat : start with the LLM off
REM
REM It finds the running instance three ways, in order: the .insight.pid file, a
REM process whose command line mentions "llminsight.server", and finally whatever
REM listens on %PORT%. A fresh instance is launched in the background (start /B),
REM its PID recorded to .insight.pid, output to insight.log, and readiness
REM confirmed by polling /api/meta.

REM --- locate repo root (the directory holding this script) ------------------
set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
cd /d "%REPO_DIR%" || ( echo [restart_insight] ERROR: cannot cd to %REPO_DIR% 1>&2 & exit /b 1 )

REM --- config (all overridable from the environment) -------------------------
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8765"
set "PIDFILE=%REPO_DIR%\.insight.pid"
set "LOGFILE=%REPO_DIR%\insight.log"

REM Keep outbound LLM calls disabled by default. Set LLMINSIGHT_LLM_ENABLED=1
REM explicitly when the operator has approved sending the KB-level metric
REM summary to the configured provider.
if not defined LLMINSIGHT_LLM_ENABLED set "LLMINSIGHT_LLM_ENABLED=0"
set "PYTHONIOENCODING=utf-8"
if defined PYTHONPATH ( set "PYTHONPATH=%REPO_DIR%;%PYTHONPATH%" ) else ( set "PYTHONPATH=%REPO_DIR%" )

REM 0.0.0.0 isn't directly pollable; check readiness over loopback instead.
set "CHECKHOST=%HOST%"
if "%HOST%"=="0.0.0.0" set "CHECKHOST=127.0.0.1"

REM --- pick a python interpreter ---------------------------------------------
REM Prefer the project-local virtualenv (.venv) so the server runs with the repo's
REM pinned deps without the caller having to "activate" first; fall back to any
REM python/py/python3 on PATH. When uv/python aren't on PATH, that .venv (created
REM with:  python -m venv .venv  then  .venv\Scripts\python -m pip install -r requirements.txt )
REM is what makes restart work.
set "PYTHON="
if exist "%REPO_DIR%\.venv\Scripts\python.exe" set PYTHON="%REPO_DIR%\.venv\Scripts\python.exe"
if not defined PYTHON for %%C in (python py python3) do if not defined PYTHON ( where %%C >nul 2>&1 && set "PYTHON=%%C" )
if not defined PYTHON ( echo [restart_insight] ERROR: no .venv and no python/py/python3 on PATH. Create the venv:  python -m venv .venv  then  .venv\Scripts\python -m pip install -r requirements.txt 1>&2 & exit /b 1 )
if "%PYTHON%"=="py" set "PYTHON=py -3"

REM --- stop existing instance(s) ---------------------------------------------
echo [restart_insight] stopping any existing LLMInsight server ...

REM 1) the PID we recorded last time
if exist "%PIDFILE%" (
  set "OLDPID="
  set /p OLDPID=<"%PIDFILE%"
  if defined OLDPID call :stop_pid !OLDPID!
  del /q "%PIDFILE%" 2>nul
)

REM 2) any python process whose command line mentions our module
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%%' AND CommandLine LIKE '%%llminsight.server%%'\").ProcessId"`) do call :stop_pid %%P

REM 3) whatever still holds the port
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "(Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue).OwningProcess"`) do call :stop_pid %%P

REM --- start a fresh instance ------------------------------------------------
echo [restart_insight] starting LLMInsight on http://%HOST%:%PORT% (LLM=%LLMINSIGHT_LLM_ENABLED%) ...
start "LLMInsight" /B %PYTHON% -m llminsight.server --host %HOST% --port %PORT% --no-browser > "%LOGFILE%" 2>&1

REM discover the freshly-launched PID (the only llminsight.server now running is ours)
set "NEWPID="
for /l %%N in (1,1,20) do if not defined NEWPID (
  for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$p=@((Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%%' AND CommandLine LIKE '%%llminsight.server%%'\").ProcessId); if ($p) { $p[0] }"`) do set "NEWPID=%%I"
  if not defined NEWPID ping -n 2 127.0.0.1 >nul
)
if defined NEWPID (
  > "%PIDFILE%" echo !NEWPID!
  echo [restart_insight] launched PID !NEWPID! ^(log: %LOGFILE%^)
) else (
  echo [restart_insight] launched ^(PID not captured; log: %LOGFILE%^)
)

REM --- wait until the server is up -------------------------------------------
REM The app is lazy-by-default: the socket opens immediately and the server sits
REM idle until a profiling directory is chosen in the browser (or autoloaded via
REM LLMINSIGHT_AUTOLOAD). So we wait for /api/meta to RESPOND (any status), not
REM for "ready":true - the data is loaded later, from the UI.
echo [restart_insight] waiting for the server to come up ...
set "READY="
set "SERVERDEAD="
for /l %%N in (1,1,60) do if not defined READY if not defined SERVERDEAD call :poll_once
if defined SERVERDEAD (
  echo [restart_insight] ERROR: server exited early - see %LOGFILE% 1>&2
  type "%LOGFILE%" 2>nul
  exit /b 1
)
if defined READY (
  echo [restart_insight] up - open http://%HOST%:%PORT%/ and choose a profiling directory
  exit /b 0
)
echo [restart_insight] WARNING: server did not come up after timeout - check %LOGFILE% 1>&2
exit /b 1

REM ===========================================================================
REM helpers
REM ===========================================================================

:stop_pid
REM ask a PID to stop (graceful, then force-kill the tree if it ignores us)
set "_P=%~1"
if not defined _P goto :eof
tasklist /FI "PID eq %_P%" /NH /FO CSV 2>nul | find "%_P%" >nul
if errorlevel 1 goto :eof
echo [restart_insight] stopping PID %_P% ...
taskkill /PID %_P% >nul 2>&1
ping -n 2 127.0.0.1 >nul
tasklist /FI "PID eq %_P%" /NH /FO CSV 2>nul | find "%_P%" >nul
if not errorlevel 1 (
  echo [restart_insight] PID %_P% ignored stop, force-killing
  taskkill /F /T /PID %_P% >nul 2>&1
)
goto :eof

:poll_once
REM one readiness probe: bail if the server died, else check /api/meta
if defined NEWPID (
  tasklist /FI "PID eq %NEWPID%" /NH /FO CSV 2>nul | find "%NEWPID%" >nul
  if errorlevel 1 ( set "SERVERDEAD=1" & goto :eof )
)
for /f "usebackq delims=" %%R in (`powershell -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri 'http://%CHECKHOST%:%PORT%/api/meta'; if ($r.Content -match '\"status\"') { 'ready' } } catch { }"`) do set "READY=%%R"
if not defined READY ping -n 2 127.0.0.1 >nul
goto :eof
