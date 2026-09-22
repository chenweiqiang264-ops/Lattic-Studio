param(
    [switch]$RebuildOnedir
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$DistRoot = Join-Path $ProjectRoot "dist\LatticeStudio"
$ExePath = Join-Path $DistRoot "LatticeStudio.exe"

if ($RebuildOnedir -or -not (Test-Path -LiteralPath $ExePath)) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_onedir.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "onedir build failed with exit code $LASTEXITCODE"
    }
}

$IsccCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
)
$Iscc = $IsccCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (-not $Iscc) {
    throw "Inno Setup compiler (ISCC.exe) was not found. Install Inno Setup 6 first."
}

& $Iscc (Join-Path $PSScriptRoot "lattice_studio.iss")
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup build failed with exit code $LASTEXITCODE"
}

Write-Host "Installer created under dist\installer"
