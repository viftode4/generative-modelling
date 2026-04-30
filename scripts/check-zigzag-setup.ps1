param(
  [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Continue"

Write-Host "== Git repository =="
git status --short --branch

Write-Host "`n== Submodule =="
if (Test-Path "external/zigzag-diffusion-sampling/.git") {
  git submodule status external/zigzag-diffusion-sampling
  git -C external/zigzag-diffusion-sampling rev-parse HEAD
} else {
  Write-Warning "Submodule is missing. Run: git submodule update --init --recursive"
}

Write-Host "`n== Python =="
$venvPython = Join-Path $VenvPath "Scripts/python.exe"
if (Test-Path $venvPython) {
  & $venvPython --version
  $pythonCheck = @"
import importlib.util
mods = ['torch', 'diffusers', 'transformers', 'einops', 'tqdm', 'PIL']
for mod in mods:
    print(f'{mod}:', 'ok' if importlib.util.find_spec(mod) else 'missing')
try:
    import torch
    print('torch version:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('cuda device:', torch.cuda.get_device_name(0))
except Exception as exc:
    print('torch check error:', exc)
"@
  $pythonCheck | & $venvPython -
} else {
  Write-Warning "No venv found at $VenvPath. Run ./scripts/setup-zigzag-env.ps1 first."
  python --version 2>$null
}

Write-Host "`n== Output directories =="
foreach ($dir in @('outputs', 'runs', 'results')) {
  if (Test-Path $dir) { Write-Host "$dir exists" } else { Write-Host "$dir not present (ok; ignored when created)" }
}
