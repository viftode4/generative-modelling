param(
  [string]$VenvPath = ".venv",
  [string]$Python = "",
  [switch]$SkipPackageInstall
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
  param([string]$Requested)

  if ($Requested) { return $Requested }

  $uvPython = (uv python find 3.10 2>$null)
  if ($LASTEXITCODE -eq 0 -and $uvPython) { return $uvPython.Trim() }

  $py310 = (py -3.10 -c "import sys; print(sys.executable)" 2>$null)
  if ($LASTEXITCODE -eq 0 -and $py310) { return $py310.Trim() }

  $python = (Get-Command python -ErrorAction SilentlyContinue)
  if ($python) { return $python.Source }

  throw "Could not find Python. Install Python 3.10 or pass -Python <path>."
}

$pythonExe = Resolve-Python -Requested $Python
Write-Host "Using Python: $pythonExe"
& $pythonExe -c "import sys; print('Python version:', sys.version); raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 10)"
if ($LASTEXITCODE -ne 0) {
  throw "Zigzag upstream expects Python 3.10. Pass a Python 3.10 executable with -Python <path>."
}

if (-not (Test-Path $VenvPath)) {
  Write-Host "Creating virtual environment at $VenvPath"
  & $pythonExe -m venv $VenvPath
}

$venvPython = Join-Path $VenvPath "Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Virtual environment Python not found at $venvPython"
}

& $venvPython -m pip install --upgrade pip

if (-not $SkipPackageInstall) {
  Write-Host "Installing Zigzag pinned requirements from requirements-zigzag.txt"
  & $venvPython -m pip install -r requirements-zigzag.txt
} else {
  Write-Host "Skipping package installation as requested."
}

Write-Host "Environment ready. Activate with:"
Write-Host "  .\\$VenvPath\\Scripts\\Activate.ps1"
