# External Code

This directory contains external reference implementations.

Tracked submodules:

- `zigzag-diffusion-sampling/` -> https://github.com/xie-lab-ml/zigzag-diffusion-sampling

Initialize or update submodules with:

```powershell
git submodule update --init --recursive
# or
./scripts/fetch-upstream.ps1
```

Record the submodule commit hash in any experiment log that depends on it.
