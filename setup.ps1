param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonExe = $null
$pythonArgs = @()
$candidates = @(
    @{Name='py'; Args=@('-3.12')},
    @{Name='py'; Args=@('-3')},
    @{Name='python'; Args=@()},
    @{Name='python3'; Args=@()}
)
foreach ($candidate in $candidates) {
    $command = Get-Command $candidate.Name -ErrorAction SilentlyContinue
    if ($command) {
        $candidateArgs = $candidate.Args
        & $command.Source @candidateArgs -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] < (3,14)))'
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = $command.Source
            $pythonArgs = $candidateArgs
            break
        }
    }
}
if (-not $pythonExe) { throw 'Install Python 3.12 and add it to PATH.' }
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    & $pythonExe @pythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$requirements = if ($Cpu) { 'requirements-cpu.txt' } else { 'requirements-inference.txt' }
& $venvPy -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; no unpinned fallback was installed.' }
& $venvPy tools/doctor.py
exit $LASTEXITCODE
