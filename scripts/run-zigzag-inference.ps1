param(
  [string]$VenvPath = ".venv",
  [ValidateSet("cuda", "cpu", "mps")]
  [string]$Device = "cuda",
  [string]$SaveDir = "../../outputs/zigzag-run",
  [int]$Seed = 42,
  [int]$InferStep = 50,
  [int]$LambdaStep = 49,
  [int]$ImageSize = 1024,
  [double]$Gamma1 = 5.5,
  [double]$Gamma2 = 0,
  [int]$TMax = 1
)

$ErrorActionPreference = "Stop"

$upstream = "external/zigzag-diffusion-sampling"
if (-not (Test-Path $upstream)) {
  throw "Missing upstream submodule. Run: git submodule update --init --recursive"
}

$venvPython = Join-Path $VenvPath "Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Missing virtual environment at $VenvPath. Run: ./scripts/setup-zigzag-env.ps1"
}

New-Item -ItemType Directory -Force -Path (Join-Path $upstream $SaveDir) | Out-Null

Push-Location $upstream
try {
  & (Resolve-Path "../../$venvPython") ./infer.py `
    --device $Device `
    --save_dir $SaveDir `
    --seed $Seed `
    --infer_step $InferStep `
    --lambda_step $LambdaStep `
    --image_size $ImageSize `
    --gamma_1 $Gamma1 `
    --gamma_2 $Gamma2 `
    --T_max $TMax
} finally {
  Pop-Location
}
