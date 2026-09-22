$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BuildRequirements = Join-Path $PSScriptRoot "requirements\build.txt"

& $Python -m pip install -r $BuildRequirements
if ($LASTEXITCODE -ne 0) { throw "Build requirements installation failed" }
& $Python -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw "Project installation failed" }
& $Python -m PyInstaller --clean --noconfirm packaging\lattice_studio.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
