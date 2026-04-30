# Research Log

Chronological research notes. Prefer short entries with sources and next actions.

## 2026-04-30

- Initialized project knowledgebase for Group 14.
- Recorded primary paper and code links.
- Source URLs:
  - https://openreview.net/forum?id=MKvQH1ekeY
  - https://openreview.net/pdf?id=MKvQH1ekeY
  - https://github.com/xie-lab-ml/zigzag-diffusion-sampling

Next actions:

- Read PDF and extract equations/algorithm.
- Inspect upstream `infer.py` after cloning reference code locally.
- Decide whether the course deliverable needs reproduction, presentation, or both.

## 2026-04-30 - Upstream code fetched

- Cloned the upstream reference implementation into ignored local path `external/zigzag-diffusion-sampling/`.
- Upstream commit recorded at clone time: `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`.
- This local clone is intentionally not tracked by this repository; use it for inspection/reproduction and record the commit hash in experiment logs.

## 2026-04-30 - Repo setup formalized

- Converted upstream reference access from an ignored local clone into a tracked Git submodule.
- Added Python 3.10 environment setup helpers and inference/check scripts.
- Created a local `.venv` with Python 3.10.18 using `./scripts/setup-zigzag-env.ps1 -SkipPackageInstall`.
- Package installation was intentionally skipped during setup verification to avoid large PyTorch/CUDA downloads until a GPU-ready environment is confirmed.
