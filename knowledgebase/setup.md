# Zigzag Diffusion Sampling Setup

This repository tracks the upstream reference implementation as a Git submodule at:

```text
external/zigzag-diffusion-sampling/
```

The submodule points to:

- Repository: https://github.com/xie-lab-ml/zigzag-diffusion-sampling
- Current pinned commit: `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`

## Clone this repo with the submodule

```powershell
git clone --recurse-submodules <this-repo-url>
```

If the repo was already cloned without submodules:

```powershell
git submodule update --init --recursive
```

## Python environment

The upstream README specifies Python 3.10.14, CUDA PyTorch, and pinned diffusion dependencies. This repo includes helper scripts to make setup repeatable.

### Apple Silicon / MPS variant (Vlad's lane)

Upstream pins (`torch==2.0.1`) do not run SDXL inversion on MPS. Use the
verified-working pin set instead:

```bash
.venv/bin/pip install -r requirements-vlad.txt
.venv/bin/pip install -e .[dev,scoring]
```

Then patch the OpenCLIP BPE vocab missing from `hpsv2==1.2.0` (see
`scripts/score_hpsv2_pairs.py` docstring) before running scoring. Full
end-to-end pipeline:

```bash
scripts/repro-vlad.sh
```

Create a local virtual environment without installing packages yet:

```powershell
./scripts/setup-zigzag-env.ps1 -SkipPackageInstall
```

Create a local virtual environment and install upstream pinned requirements:

```powershell
./scripts/setup-zigzag-env.ps1
```

> Note: installing the full requirements can download large PyTorch/CUDA/model dependencies. Use a machine with appropriate GPU/CUDA support for inference.

## Check setup

```powershell
./scripts/check-zigzag-setup.ps1
```

This checks submodule presence, Python version, import availability, and CUDA visibility when packages are installed.

## Run upstream inference

After environment setup and model access are ready:

```powershell
./scripts/run-zigzag-inference.ps1 -Device cuda -SaveDir ../../outputs/zigzag-smoke
```

Outputs should go under ignored directories such as `outputs/`, `runs/`, or `results/`.

## Record experiments

For every run, copy `knowledgebase/templates/experiment-log-template.md` into `knowledgebase/logs/` and record:

- repo commit
- upstream submodule commit
- command
- environment
- hardware
- seed
- output path
- observations
