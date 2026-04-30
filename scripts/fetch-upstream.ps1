param(
  [string]$Destination = "external/zigzag-diffusion-sampling"
)

$ErrorActionPreference = "Stop"
$repo = "https://github.com/xie-lab-ml/zigzag-diffusion-sampling.git"

if (Test-Path $Destination) {
  Write-Host "Updating existing upstream clone at $Destination"
  git -C $Destination fetch --all --prune
  git -C $Destination status --short --branch
  git -C $Destination rev-parse HEAD
} else {
  New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
  Write-Host "Cloning upstream reference implementation into $Destination"
  git clone $repo $Destination
  git -C $Destination rev-parse HEAD
}

Write-Host "Note: external/ is ignored by this repository. Record the printed upstream commit hash in experiment logs."
