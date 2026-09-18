# ForgeX one-command setup (Windows PowerShell).
#
#   git clone https://github.com/abishek1123/forgex-kla-ps01.git
#   cd forgex-kla-ps01
#   .\setup.ps1
#
# Creates .venv, installs dependencies, then runs the laptop check.
# Safe to re-run: it reuses an existing .venv instead of rebuilding it.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host ""
Write-Host "  ForgeX setup" -ForegroundColor Cyan
Write-Host "  ------------------------------------------------------------"

# --- 1. find a python -------------------------------------------------------
$py = $null
foreach ($cand in @("py -3.13", "py -3", "python", "python3")) {
    $parts = $cand.Split(" ")
    $exe = Get-Command $parts[0] -ErrorAction SilentlyContinue
    if ($exe) {
        try {
            $v = & $parts[0] $parts[1..($parts.Length-1)] --version 2>&1
            if ($v -match "Python 3\.") { $py = $cand; break }
        } catch { }
    }
}
if (-not $py) {
    Write-Host "  FAIL: no Python 3 found on PATH." -ForegroundColor Red
    Write-Host "  Install Python 3.13 from python.org and TICK 'Add to PATH'."
    exit 1
}
Write-Host "  python      : $py"

# --- 2. virtualenv ----------------------------------------------------------
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    Write-Host "  venv        : reusing .venv"
} else {
    Write-Host "  venv        : creating .venv ..."
    $parts = $py.Split(" ")
    & $parts[0] $parts[1..($parts.Length-1)] -m venv .venv
    if (-not (Test-Path $venvPy)) {
        Write-Host "  FAIL: could not create .venv" -ForegroundColor Red
        exit 1
    }
}

# --- 3. pick the requirements file ------------------------------------------
$hasGpu = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$req = if ($hasGpu) { "requirements.txt" } else { "requirements-cpu.txt" }
if (-not (Test-Path (Join-Path $root $req))) { $req = "requirements.txt" }
Write-Host "  gpu         : $(if ($hasGpu) { 'nvidia-smi found' } else { 'none -- installing CPU build' })"
Write-Host "  installing  : $req  (this takes a few minutes)"

& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r $req --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Pinned install failed -- most likely your Python is not 3.13," -ForegroundColor Yellow
    Write-Host "  so the exact torch wheel does not exist for it." -ForegroundColor Yellow
    Write-Host "  Falling back to an unpinned install, enough to run inference." -ForegroundColor Yellow
    Write-Host ""
    & $venvPy -m pip install torch numpy --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAIL: could not install torch." -ForegroundColor Red
        Write-Host "  Tell Abishek. Do not keep retrying."
        exit 1
    }
    Write-Host "  NOTE: this laptop is INFERENCE-ONLY (no training deps)." -ForegroundColor Yellow
}

# --- 4. output dirs ---------------------------------------------------------
foreach ($d in @("outputs", "runs")) {
    if (-not (Test-Path (Join-Path $root $d))) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# --- 5. verify --------------------------------------------------------------
Write-Host ""
& $venvPy (Join-Path $root "tools\doctor.py")
$code = $LASTEXITCODE

Write-Host "  From now on, use this python for everything:"
Write-Host "      .\.venv\Scripts\python.exe <script>" -ForegroundColor Cyan
Write-Host ""
exit $code
