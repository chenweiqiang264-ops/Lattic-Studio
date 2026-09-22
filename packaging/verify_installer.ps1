param(
    [string]$InstallerPath = "",
    [ValidateSet("All", "Install", "Uninstall")][string]$Phase = "All",
    [string]$TestRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $InstallerPath) {
    $InstallerPath = Join-Path $ProjectRoot "dist\installer\LatticeStudio-Setup-0.1.1-x64.exe"
}
if (-not $TestRoot) {
    $TestRoot = Join-Path $ProjectRoot ("build\installer-acceptance-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}
$TestRoot = [IO.Path]::GetFullPath($TestRoot)
$appRoot = Join-Path $TestRoot "app"
$exe = Join-Path $appRoot "LatticeStudio.exe"
$registration = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{B9F42F2E-1EA4-4D2B-92C8-4F7F1A9F7D0A}_is1'

function Get-TestProcesses {
    @(Get-CimInstance Win32_Process -Filter "Name='LatticeStudio.exe'" |
        Where-Object { $_.ExecutablePath -eq $exe })
}

function Invoke-CheckedProcess {
    param([string]$File, [string[]]$Arguments, [int]$TimeoutSeconds = 180)
    $process = Start-Process -FilePath $File -ArgumentList $Arguments -WindowStyle Hidden -PassThru
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        foreach ($item in (Get-TestProcesses)) { Stop-Process -Id $item.ProcessId -Force }
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
        throw "Timed out: $File. Evidence retained at $TestRoot"
    }
    $process.Refresh()
    if ($process.ExitCode -ne 0) { throw "$File exited with $($process.ExitCode); see $TestRoot" }
}

if ($Phase -ne "Uninstall") {
    if (Test-Path -LiteralPath $appRoot) { throw "Use a new TestRoot; existing files will not be overwritten: $appRoot" }
    if (Test-Path $registration) { throw "An installation is already registered; avoid overwriting it during acceptance." }
    New-Item -ItemType Directory -Path $TestRoot -Force | Out-Null
    $InstallerPath = (Resolve-Path -LiteralPath $InstallerPath).Path
    Invoke-CheckedProcess $InstallerPath @(
        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/TASKS=desktopicon',
        ('/DIR="' + $appRoot + '"'), ('/LOG="' + (Join-Path $TestRoot 'install.log') + '"')
    )
    if (-not (Test-Path -LiteralPath $exe)) { throw "Installed executable is missing" }
    $reportPath = Join-Path $TestRoot 'runtime.json'
    $savedEnvironment = @{}
    $names = @('PATH', 'PYTHONHOME', 'PYTHONPATH', 'CUDA_HOME', 'CONDA_PREFIX', 'TPMS_DISABLE_NUMBA_CUDA', 'TPMS_ENABLE_NUMBA_CUDA')
    $names += @(Get-ChildItem Env: | Where-Object Name -like 'CUDA_PATH*' | ForEach-Object Name)
    try {
        foreach ($name in ($names | Select-Object -Unique)) {
            $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
        $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
        Invoke-CheckedProcess $exe @('--verify-runtime', ('"' + $reportPath + '"'))
        $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
        if (-not $report.ok) { throw "Installed runtime check failed: $($report.error)" }
        $env:TPMS_DISABLE_NUMBA_CUDA = '1'
        $cpuReport = Join-Path $TestRoot 'runtime-cpu.json'
        Invoke-CheckedProcess $exe @('--verify-runtime', ('"' + $cpuReport + '"'))
        $cpu = Get-Content -Raw -LiteralPath $cpuReport | ConvertFrom-Json
        if (-not $cpu.ok -or $cpu.checks.tpms.using_gpu) { throw "CPU fallback check failed" }
    }
    finally {
        foreach ($name in $savedEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
        }
    }
    if (@(Get-TestProcesses).Count -ne 0) { throw "Application left running descendants" }
    Write-Host "Runtime and CPU fallback checks passed. Evidence: $TestRoot"
}

if ($Phase -ne "Install") {
    if (@(Get-TestProcesses).Count -ne 0) { throw "Close the test application normally before uninstall validation." }
    $uninstaller = Join-Path $appRoot 'unins000.exe'
    if (-not (Test-Path -LiteralPath $uninstaller)) { throw "No test uninstaller at $uninstaller" }
    Invoke-CheckedProcess $uninstaller @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
        ('/LOG="' + (Join-Path $TestRoot 'uninstall.log') + '"'))
    # Observe only. Never delete residual files to make an uninstall pass.
    for ($attempt = 0; $attempt -lt 20 -and (Test-Path -LiteralPath $appRoot); $attempt++) {
        Start-Sleep -Milliseconds 500
    }
    $shortcuts = @(
        (Join-Path ([Environment]::GetFolderPath('Programs')) 'Lattice Studio\Lattice Studio.lnk'),
        (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Lattice Studio.lnk')
    )
    $remainingShortcuts = @($shortcuts | Where-Object { Test-Path -LiteralPath $_ })
    $result = [ordered]@{
        ExecutableRemoved = -not (Test-Path -LiteralPath $exe)
        InstallDirectoryRemoved = -not (Test-Path -LiteralPath $appRoot)
        RegistrationRemoved = -not (Test-Path $registration)
        ShortcutsRemoved = $remainingShortcuts.Count -eq 0
        ProcessesRemaining = @(Get-TestProcesses).Count
        TestScriptDeletedInstallFiles = $false
    }
    $result | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $TestRoot 'uninstall-result.json')
    $result | Format-List
    if (-not $result.ExecutableRemoved -or -not $result.InstallDirectoryRemoved -or
        -not $result.RegistrationRemoved -or -not $result.ShortcutsRemoved -or $result.ProcessesRemaining) {
        throw "Uninstall left remnants. Evidence retained at $TestRoot"
    }
}
