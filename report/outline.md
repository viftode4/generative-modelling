# Report Outline

Target: 3 pages, double-column, individual report.

## Working title

Reproducing and Analyzing Zigzag Diffusion Sampling for Text-to-Image Diffusion Models

## 1. Introduction / Motivation

- Why prompt-image alignment is difficult in diffusion models.
- Why Z-Sampling is interesting: sampling-time self-reflection without full retraining.
- Individual/group motivation and chosen reproduction scope.

## 2. Method and Approach

- Brief explanation of standard diffusion sampling.
- Z-Sampling: alternating denoising and inversion, guidance gap, key hyperparameters.
- Upstream codebase and exact commit used.
- Experiment design and constraints.

## 3. Experiment Setting

- Hardware and environment.
- Model: Stable Diffusion XL base via upstream implementation.
- Prompts, seeds, image size, inference steps, gamma values.
- Baseline: standard sampling from the same initial latent.

## 4. Results

- Qualitative comparison table/figure.
- Any quantitative or human-evaluation observations if feasible.
- Troubleshooting and reproducibility notes.

## 5. Interpretation and Critical Appraisal

- What worked and what did not.
- Whether observations support the paper's claims within our limited setup.
- Limitations: compute, dependency pinning, prompt selection, evaluation subjectivity.

## 6. Conclusion

- Main findings.
- Lessons about reproducibility and sampling-time improvements.
- Future work / extensions.

## Required evidence to collect

- [ ] Screenshots or generated images for standard vs. Z-Sampling.
- [ ] Exact command(s), seed(s), and environment.
- [ ] Upstream commit hash.
- [ ] Explanation of failures/blockers if full reproduction is not possible.
