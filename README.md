# Zigzag Diffusion Sampling Reproduction

This repository contains our reproduction and evaluation of the paper:

**Zigzag Diffusion Sampling**
Bai et al., ICLR 2025
https://openreview.net/forum?id=MKvQH1ekeY

The project was completed as part of the TU Delft Generative Modelling course.

## Goal

The objective of this project is to reproduce and evaluate several diffusion sampling methods and compare their behaviour across multiple models and benchmark datasets.

Methods investigated include:

* Standard DDIM sampling
* Resampling (RePaint-style sampling)
* Align Your Steps (AYS)
* Zigzag Diffusion Sampling (Z-Sampling)
* AYS + Z-Sampling combinations

Experiments were conducted primarily using:

* DreamShaper XL Turbo
* Stable Diffusion XL
* Stable Diffusion 2.1
* Hunyuan-DiT

Evaluation datasets include:

* Pick-a-Pic prompts
* DrawBench prompts
* GenEval

---

## Repository Structure

```text
data/
    Prompt datasets and metadata.

experiments/
    Experiment configurations, prompt lists, and result indexes.

kaggle/
    Kaggle-specific execution files.

notebooks/
    Development and exploratory notebooks.

outputs/
    Generated images and evaluation outputs.

scripts/
    Command-line scripts used to run experiments and evaluations.

src/
    Main project implementation.

tests/
    Unit and integration tests.
```

---

## Main Source Modules

### Sampling

```text
src/zigzag_project/zsampling.py
src/zigzag_project/samplers/zigzag.py
```

Implementation of Zigzag Diffusion Sampling.

### Resampling

```text
src/zigzag_project/resampling_utils.py
src/zigzag_project/samplers/resampling.py
```

RePaint-style resampling schedules and utilities.

### AYS

```text
src/zigzag_project/ays.py
```

Implementation of the Align Your Steps timestep schedule.

### Models

```text
src/zigzag_project/models.py
src/zigzag_project/model_utils.py
```

Model loading and scheduler configuration.

### Evaluation

```text
src/zigzag_project/metrics.py
src/zigzag_project/geneval_io.py
src/zigzag_project/dataset_utils.py
```

Dataset loading and evaluation support.

---

## Setup

Clone the repository and initialize submodules:

```bash
git submodule update --init --recursive
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Examples:

```bash
python scripts/run_eugen_pickapic.py
```

```bash
python scripts/run_exp2_sdxl_geneval.py
```

Evaluation:

```bash
python scripts/score_hpsv2_pairs.py
```

Some scripts assume access to Hugging Face models and may require authentication.

---

## Attribution

### Zigzag Diffusion Sampling

Official implementation:

https://github.com/xie-lab-ml/zigzag-diffusion-sampling

The repository includes the official implementation as a Git submodule:

```text
external/zigzag-diffusion-sampling
```

The following files were implemented using the official repository as a reference:

```text
src/zigzag_project/zsampling.py
src/zigzag_project/samplers/zigzag.py
```

---

### RePaint

Original paper:

RePaint: Inpainting using Denoising Diffusion Probabilistic Models

The resampling schedule implemented in:

```text
src/zigzag_project/resampling_utils.py
src/zigzag_project/samplers/resampling.py
```

follows the methodology described in the RePaint paper.

---

### Align Your Steps

Original paper:

Align Your Steps: Optimizing Sampling Schedules in Diffusion Models

The AYS schedule implemented in:

```text
src/zigzag_project/ays.py
```

uses the published SDXL 10-step schedule.

---

### External Libraries

This project relies heavily on:

* Hugging Face Diffusers
* Transformers
* Accelerate
* HPSv2
* ImageReward
* CLIP
* OpenCLIP