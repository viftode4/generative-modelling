# Baseline Experiment Config

Purpose: compare upstream standard sampling against Z-Sampling using the default prompt list from `infer.py`.

Upstream commit: `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`

Suggested command:

```powershell
./scripts/run-zigzag-inference.ps1 `
  -Device cuda `
  -SaveDir ../../outputs/zigzag-baseline-seed42 `
  -Seed 42 `
  -InferStep 50 `
  -LambdaStep 49 `
  -ImageSize 1024 `
  -Gamma1 5.5 `
  -Gamma2 0 `
  -TMax 1
```

Expected outputs:

- One standard sampling image per prompt.
- One Z-Sampling image per prompt.

Before running:

- Confirm CUDA availability with `./scripts/check-zigzag-setup.ps1`.
- Confirm Hugging Face model access for Stable Diffusion XL.
