param(
    [string]$Python = "python",
    [string]$InnoCompiler = "ISCC.exe"
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$version = & $Python -c "from llminsight import __version__; print(__version__)"
if ($LASTEXITCODE -ne 0 -or $version -notmatch '^\d+\.\d+\.\d+$') { throw 'Invalid application version' }
New-Item -ItemType Directory -Force -Path 'build', 'dist' | Out-Null
$env:PYINSTALLER_CONFIG_DIR = Join-Path $root 'build/pyinstaller-cache'

& $Python -m PyInstaller --clean --noconfirm --log-level WARN --onedir --console --name LLMInsight `
    --paths $root --distpath 'build/windows-dist' --workpath 'build/pyinstaller' `
    --specpath 'build' --add-data "$(Join-Path $root 'web');llminsight/web" `
    --add-data "$(Join-Path $root 'configs');llminsight/configs" 'scripts/windows_launcher.py'
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }

& $Python 'scripts/stage_windows_licenses.py'
if ($LASTEXITCODE -ne 0) { throw 'Bundled license staging failed' }

& $InnoCompiler --quiet "--define=AppVersion=$version" 'scripts/windows-installer.iss'
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup compilation failed' }
$installer = Get-Item -LiteralPath 'dist/LLMInsight-Setup-x64.exe'
Write-Output "Installer: $($installer.FullName)"
Write-Output "SHA256: $((Get-FileHash -LiteralPath $installer.FullName -Algorithm SHA256).Hash.ToLowerInvariant())"
