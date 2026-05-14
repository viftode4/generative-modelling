# DreamShaper Z-Sampling Probe

## Metadata

- **Date:** 2026-05-14
- **Owner:** Vlad / AI assistant
- **Goal:** Verify that the repo-local SDXL Z-Sampling utility can execute a minimal DreamShaper-xl-v2-turbo generation on Apple Silicon MPS.
- **Source code commit:** Working tree based on `7fea0cc` (uncommitted local changes: `src/zigzag_project/zsampling.py`, `scripts/smoke_dreamshaper.py`, tests/docs updates).
- **Upstream commit:** `external/zigzag-diffusion-sampling` at `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`.
- **Environment:** `.venv` CPython 3.10; local package editable-installed; `torch==2.12.0`, `diffusers==0.37.1` per `knowledgebase/research-log.md` entry dated 2026-05-14.
- **Hardware:** Apple M4 Max, MPS device.
- **Seed(s):** `42`

## Command(s)

```bash
.venv/bin/python scripts/smoke_dreamshaper.py \
  --sampler zigzag \
  --steps 2 \
  --lambda-step 1 \
  --size 512 \
  --seed 42 \
  --out outputs/smoke/dreamshaper_zigzag_probe.png
```

Verification commands:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

## Inputs

- Prompt default from `scripts/smoke_dreamshaper.py`: `a small yellow dog sitting on grass`
- Model: `Lykon/dreamshaper-xl-v2-turbo` via `src/zigzag_project/models.py`
- Z-Sampling config: `gamma_1=2.0`, `gamma_2=0.0`, `infer_step=2`, `lambda_step=1`, `t_max=1`
- Implementation provenance: adapted from upstream `utils/pipeline_stable_diffusion_xl.py::z_sampling_call` at commit `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`; kept in repo-local utility rather than modifying the submodule.

## Outputs

- Generated image: `outputs/smoke/dreamshaper_zigzag_probe.png` (gitignored; not committed)
- Runtime evidence from command output:
  - Model loaded on `mps` in `9.8s`
  - Z-Sampling generation completed in `3.8s`
- Test evidence:
  - Ruff: `All checks passed!`
  - Pytest: `15 passed in 6.52s`

## Observations

- The minimal Z-Sampling path executed end-to-end with the installed modern `diffusers==0.37.1` stack.
- Diffusers emitted expected compatibility warnings about ignored scheduler/config fields and deprecated `upcast_vae`; these did not block generation.
- The probe was intentionally tiny (`2` steps, `512x512`) and only validates runtime wiring, not visual quality or parity with the upstream SDXL-base implementation.

## Interpretation

The W2 implementation lane has a first working repo-local Z-Sampling function for SDXL-like pipelines. This is not yet a parity result: DreamShaper turbo, two steps, and MPS-divergent dependencies are a runtime smoke check only.

## Follow-up

- Run `--sampler both` at a report-relevant step count to create a standard-vs-Z pair from shared latents.
- Add a parity-oriented check against upstream SDXL-base on a CUDA environment with upstream-compatible settings.
- Decide the shared metric and prompt subset before scaling runs.
