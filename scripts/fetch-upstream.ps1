param(
  [string]$Destination = "external/zigzag-diffusion-sampling"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".gitmodules")) {
  throw "No .gitmodules file found. The upstream reference should be configured as a submodule."
}

Write-Host "Initializing/updating upstream Zigzag submodule at $Destination"
git submodule update --init --recursive -- $Destination

git -C $Destination fetch --all --prune
git submodule status -- $Destination
$commit = git -C $Destination rev-parse HEAD
Write-Host "Upstream commit: $commit"
Write-Host "Record this commit hash in experiment logs."
