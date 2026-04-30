# Upstream Reference Code Notes

## Location

Tracked Git submodule:

```text
external/zigzag-diffusion-sampling/
```

## Source

- Repository: https://github.com/xie-lab-ml/zigzag-diffusion-sampling
- Pinned commit: `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`
- License shown by GitHub repository page: Apache-2.0

## Current top-level files to inspect

- `README.md` - install/use instructions and method summary
- `requirements.txt` - dependency pins
- `infer.py` - primary inference entry point
- `utils/` - helper code
- `res/` - sample/generated assets in upstream repository

## Reproduction caution

The upstream README indicates CUDA PyTorch and `diffusers == 0.22.3`. Before running inference, create an isolated environment and confirm GPU availability, model download requirements, and output paths.

Use `knowledgebase/setup.md` for the current repo-local setup workflow.
