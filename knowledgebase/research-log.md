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

## 2026-04-30 - Course PDFs added

- Added official project description and project rubric PDFs from Downloads into `docs/course/`.
- Extracted text versions into `knowledgebase/course/` for search and AI-assisted planning.
- Summarized assessment requirements in `knowledgebase/course/requirements-summary.md`.
- Initialized planning, experiments, report, and presentation folders based on the rubric.

## 2026-05-14 - Vlad's local env set up (M4 Max, MPS)

- Hardware: Apple M4 Max (arm64, no CUDA). Using MPS (Metal) as the inference device.
- Python: created `.venv` with CPython 3.10.20 via `uv venv .venv --python 3.10`.
- Submodule `external/zigzag-diffusion-sampling` initialized at upstream commit `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`.
- Installed packages (relaxed from upstream pins to get working MPS support):
  - `torch==2.12.0` (upstream pins `2.0.1`; that version's MPS support is too weak for SDXL inversion).
  - `diffusers==0.37.1` (upstream pins `0.22.3`).
  - `transformers==5.8.1` (upstream pins `4.41.0`).
  - Also: `accelerate`, `safetensors`, `einops`, `torchsde`, `tqdm`, `huggingface_hub`.
- Verified `torch.backends.mps.is_available()` is True and a small MPS matmul runs.
- Added a minimal model loader at `src/zigzag_project/models.py` (`load_dreamshaper_xl_turbo`) and a smoke script at `scripts/smoke_dreamshaper.py`.
- Editable-installed the project: `uv pip install -e .` so `import zigzag_project` works without `PYTHONPATH`.
- Open risk: dependency divergence from upstream pins. Any parity-check vs. upstream needs to account for this. Group should decide whether final results runs use a CUDA box with strict pins or accept MPS divergence.
