# Group Report Outline

Source update: the 2026-06-04 course clarification says the final artifact is a **group report** structured like a scientific paper, with section titles mentioning who did what. Older individual-report notes are now source material, not the final structure.

Working title:

> Reproducing Zigzag Diffusion Sampling: Inference-Time Self-Reflection for Text-to-Image Diffusion Models

## 0. Abstract

- One-paragraph summary of the paper question, our implementation scope, main results, and limitations.
- Keep quantitative claims to verified numbers only.

## 1. Introduction and motivation

- Problem: diffusion models can produce plausible images while missing prompt details such as count, color, and spatial relations.
- Paper idea: Z-Sampling tries to improve alignment at sampling time, without retraining.
- Our project question: can we reproduce the core behaviour in a small controlled course setting?
- Scope statement: not a full benchmark reproduction; a scoped implementation + paired evaluation.

## 2. Background and method

### 2.1 Standard sampling

- Briefly explain standard DDIM-style denoising.
- Keep math light; define only terms used later.

### 2.2 Resampling baseline

- Explain the planned resampling baseline at a high level.
- Mention whether final report-ready results are available; do not overclaim.

### 2.3 Z-Sampling

- Explain denoise with `gamma_1`, invert with `gamma_2`, then re-denoise.
- Define guidance gap `gamma_1 - gamma_2`.
- Mention deterministic DDIM inversion requirement.
- Cite paper notes / upstream commit.

## 3. Project plan and task division

- Use the midterm plan as the starting point:
  - Standard, Resampling, Z-Sampling.
  - Multiple model/dataset lanes.
  - Multiple metrics if feasible.
- Include a compact responsibility table.
- Then explain final reality:
  - verified result lanes are smaller than the original plan;
  - implementation/integration took substantial effort;
  - report emphasizes evidence we can defend.

## 4. Implementation

### 4.1 Shared codebase

- Repo-local implementation in `src/zigzag_project/`.
- Upstream submodule treated as external reference.
- Standard / Z / Resampling interfaces and guardrails.
- Scripts and manifests for reproducible runs.

### 4.2 Experimental controls

- Matched prompt, seed, scheduler family, and initial latent for paired comparisons.
- Fixed hyperparameters per lane.
- Generated images stay in ignored outputs; curated figures/tables live under `report/`.

### 4.3 Integration challenges

- Old CUDA/diffusers stack vs modern local/Colab environments.
- GenEval detector compatibility.
- Config mixups between sampler types after branch integration.
- Runtime cost of Z-Sampling.

## 5. Experiments and results

### 5.1 Eugen: SD2.1 + Pick-a-Pic

- 100 prompts x 3 seeds = 300 paired comparisons.
- HPSv2.1 result: Z wins 213/300 = 71.0%.
- Mean HPS delta: +0.0110, 95% CI [+0.0087, +0.0132].
- Evidence: `report/tables/eugen_results.md` and curated figures in `report/figures/eugen_sd2.1_pickapic/`.

### 5.2 Vlad: DreamShaper-xl-v2-turbo + GenEval subset

- 18 prompts x 3 seeds = 54 paired comparisons.
- GenEval detector: standard 31.5% (17/54), Z 50.0% (27/54), +18.5 points.
- HPSv2.1: Z wins 53/54, mean delta +0.0358.
- Strongest detector gains: colors, counting, two-object prompts.
- Evidence: `report/tables/vlad_results.md`, `vlad_geneval_detector.md`, `vlad_prompt_level.md`, figures in `report/figures/vlad_geneval/`.

### 5.3 Runtime / compute control

- Vlad 4-step standard: 6.16 s/image.
- Vlad 4-step Z-Sampling: 13.80 s/image.
- 9-step standard control: 18.14 s/image; Z still wins 42/54 HPS pairs, mean delta +0.0245.
- Evidence: `report/tables/vlad_compute_matched.md`.

### 5.4 Missing or weaker lanes

- Add teammate results only if they are backed by committed tables/figures/logs.
- If no verified table exists, discuss as implementation progress/challenge, not as quantitative result.

## 6. Discussion and critical appraisal

- Results support a controlled positive reproduction, not full paper parity.
- HPSv2 and GenEval detector measure different things; HPS preference can reward aesthetics/sharpness, not only correctness.
- Z-Sampling gains are not uniform across prompt categories.
- Runtime cost is real; compute-matched control narrows but does not erase the HPS margin.
- Environment differences limit direct comparison with the paper.

## 7. Conclusion

- We implemented and tested the core inference-time mechanism.
- Verified lanes show positive evidence for Z-Sampling.
- Main limitations: small scale, partial benchmark coverage, metric mismatch, runtime, environment differences.
- Future work: full GenEval / DrawBench / Pick-a-Pic coverage, more teammate lanes, more detector scoring, scheduler/ablation studies.

## Figures and tables to include

Recommended minimum:

1. One method diagram: Standard vs Resampling vs Z-Sampling.
2. One implementation pipeline diagram.
3. One compact result table with Eugen + Vlad headline metrics.
4. One qualitative image pair from Vlad or Eugen.
5. Optional runtime/control table if space permits.
