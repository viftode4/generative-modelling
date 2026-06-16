# Zigzag Diffusion Sampling: Paper Notes

## Source metadata

- **Title:** Zigzag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection
- **OpenReview:** https://openreview.net/forum?id=MKvQH1ekeY
- **PDF:** https://openreview.net/pdf?id=MKvQH1ekeY
- **Local PDF:** `docs/papers/zigzag-diffusion-sampling.pdf` (downloaded from OpenReview on 2026-06-03)
- **Reference code:** https://github.com/xie-lab-ml/zigzag-diffusion-sampling
- **Local reference submodule:** `external/zigzag-diffusion-sampling/` at commit `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`
- **Venue status in local PDF:** Published as a conference paper at ICLR 2025.
- **Authors in local PDF:** Lichen Bai, Shitong Shao, Zikai Zhou, Zipeng Qi, Zhiqiang Xu, Haoyi Xiong, Zeke Xie.

## Status of this note

Validated against the official OpenReview PDF on **2026-06-03**. The earlier version of this note was directionally correct but incomplete: it captured the high-level idea, but not enough experiment settings, benchmark distinctions, or caveats to safely support report claims.

## TL;DR from the paper

Zigzag Diffusion Sampling (Z-Sampling) is a **training-free sampling-time method**. It alternates strong-guidance denoising and weak-guidance inversion so that the **guidance gap** between the two operations injects prompt-related semantic information into the latent across the sampling path. The paper evaluates this as a plug-and-play improvement over standard sampling across several diffusion backbones, datasets, and metrics.

## Core method

Paper Algorithm 1 defines Z-Sampling with:

- denoising process at timestep `t`: `Φ^t`
- inversion process at timestep `t`: `Ψ^t`
- prompt: `c`
- denoising guidance: `γ1`
- inversion guidance: `γ2`
- total inference steps: `T`
- zigzag optimization steps: `λ`

The loop starts from Gaussian latent `x_T`. At each timestep it first denoises with `γ1`; if the step is inside the first `λ` zigzag steps, it then inverts with `γ2` and denoises again with `γ1`. In paper terms, `λ = 0` reduces to standard sampling, and larger `λ` applies the zigzag operation to more early denoising steps.

Key implication for implementation: the method needs a meaningful deterministic inversion step. The paper’s discussion and appendix caution that stochastic samplers such as Euler ancestral can make inversion inaccurate and reduce Z-Sampling’s effectiveness.

## Guidance gap

The paper defines the guidance gap as:

```text
δγ = γ1 − γ2
```

The experiments vary `γ2` while fixing `γ1` and report that benefits shrink as the gap approaches zero; if `γ2 > γ1`, the gain can become negative. This is important for our implementation because `γ2=0` is not arbitrary — it is the weak-inversion side of the paper’s mechanism.

## Experiment settings from the paper

### Datasets

The main paper uses:

- Pick-a-Pic
- DrawBench
- GenEval

Appendix A.1 describes GenEval as an object-focused compositional benchmark with 553 prompts covering object co-occurrence, position, count, and color.

### Metrics

The paper reports multiple preference/image-quality metrics:

- HPS v2
- PickScore
- ImageReward (IR)
- AES

For GenEval tables, the paper reports compositional pass-rate categories such as single object, two object, counting, colors, position, color attribution, and overall.

### Backbone/hyperparameter settings

From Section 4.1:

| Model family | Paper denoising steps | Paper `γ1` |
| --- | ---: | ---: |
| SD2.1 | 50 | 5.5 |
| SDXL | 50 | 5.5 |
| Hunyuan-DiT | 50 | 6.0 |
| DreamShaper-xl-v2-turbo | 4 | 3.5 |

Unless otherwise specified, the paper uses:

- `λ = T − 1`
- `γ2 = 0`

Image size is **not pinned in the extracted Section 4.1 text**, but the upstream `infer.py` defaults `image_size=1024` for SDXL (`external/zigzag-diffusion-sampling/infer.py:16`). The SDXL Experiment #8 run uses **1024** accordingly. Vlad's earlier 768px DreamShaper run was a local choice and should be described as such.

### Efficiency comparison (Experiment #8 — Figure 10, p.10)

The efficiency study is a **time-matched comparison**, not a single fixed-step timing:

- Standard noise-prediction ops = `T_s`; Z-Sampling ops = `T_z + 2·λ`.
- The paper sets **`T_z = T_s / 2`** and **`λ = T_z / 2 = T_s / 4`**, so Z ops equal `T_s` (matched compute).
- It sweeps several time budgets and plots **mean HPS v2 (Pick-a-Pic, SDXL) vs average time/image** for Standard and Z-Sampling.
- Headline claim: *"even with 36% less computational time, Z-Sampling can surpass the best performance of standard sampling."*
- This `λ = T_s/4` is specific to the efficiency experiment and differs from the default `λ = T − 1`.

## Main quantitative claims relevant to our project

- The paper’s headline qualitative examples cover style, position, color, counting, text rendering, and object co-occurrence.
- The abstract says DreamShaper with Z-Sampling reaches an HPSv2 win rate up to 94% over original results.
- Figure 6 reports DreamShaper-xl-v2-turbo win rates over standard sampling on Pick-a-Pic and DrawBench; HPS v2 is 88% on Pick-a-Pic and 94% on DrawBench.
- Table 1 reports Z-Sampling improvements over standard sampling and resampling for SD2.1, SDXL, DreamShaper-xl-v2-turbo, and Hunyuan-DiT on Pick-a-Pic and DrawBench.
- Table 2 reports GenEval results for **SDXL**, not DreamShaper. Z-Sampling improves overall GenEval from 52.52% standard to 57.04%.
- Appendix Table 7 reports DreamShaper-xl-v2-turbo on MS-COCO 2014 with 4 denoising steps.

## Paper limitations and caveats

The paper explicitly flags:

1. Z-Sampling relies on deterministic inversion; extending it to SDE-style/stochastic diffusion frameworks is future work.
2. The study is limited to text-to-image diffusion models.
3. Zigzag steps add compute; the paper argues the cost is acceptable and includes a time-matched comparison, but the method is not free.
4. Distilling the Z-Sampling path into the model is suggested as future work.

## Our current Vlad experiment vs. paper

### Matches paper-backed settings

- Uses Z-Sampling as training-free inference-time modification.
- Uses strong denoising guidance and weak inversion guidance: `γ1=3.5`, `γ2=0`.
- Uses DreamShaper-xl-v2-turbo with `T=4`.
- Uses `λ=T−1=3`, matching the paper’s default “throughout the path” setting.
- Uses deterministic DDIM/DDIMInverse schedulers, consistent with the paper’s deterministic-inversion requirement and the upstream reference implementation.

### Deliberate deviations / report caveats

- Our Vlad cell is **DreamShaper-xl-v2-turbo + GenEval prompts**, but the paper’s main GenEval table is **SDXL**. Treat our setup as a group-assigned cross-cell stress test, not a direct reproduction of Table 2.
- Our prompt set is a balanced 18-prompt subset (first 3 prompts from each of 6 GenEval tags), not the full 553-prompt GenEval benchmark.
- Our metrics are official GenEval detector pass/fail on the 18-prompt subset plus HPSv2.1 pair scoring; this is still much smaller than the full paper benchmark.
- Our local run uses Apple Silicon/MPS and newer package versions, not the paper/upstream CUDA stack.
- Our 768px image size needs to be described as a local choice unless another source is added.

## Implementation anchors from the reference repository

The upstream GitHub README and `infer.py` list:

- Python 3.10.14
- CUDA PyTorch
- `diffusers == 0.22.3`
- Stable Diffusion XL inference support
- Main script: `infer.py`
- Key arguments: `--gamma_1`, `--gamma_2`, `--infer_step`, `--lambda_step`, `--image_size`, `--T_max`, `--seed`, `--device`, `--save_dir`

Project-local implementation provenance:

- `src/zigzag_project/zsampling.py` ports the upstream `StableDiffusionXLPipeline.z_sampling_call` loop from `external/zigzag-diffusion-sampling/utils/pipeline_stable_diffusion_xl.py` at commit `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`.
- `src/zigzag_project/models.py` keeps DDIM and DDIMInverse paired for deterministic inversion.
