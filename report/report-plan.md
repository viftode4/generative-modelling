# Group Report Plan

One group report on the ACM `sigconf` template. Quantitative claims come only from
committed tables — see [`experiments/results-index/coverage.md`](../experiments/results-index/coverage.md).

Working title: *Reproducing Zigzag Diffusion Sampling: Inference-Time Self-Reflection for Text-to-Image Diffusion Models.*

## Section map (template file → content → owner)

| File | Content | Ownership |
|------|---------|-----------|
| `s1_introduction` | Prompt-alignment problem; Z-Sampling pitch (no retraining); project question + scope statement; plan & task-division table | Shared (1 author) |
| `s2_background` | Standard sampling; resampling; Z-Sampling (denoise γ₁ → invert γ₂ → re-denoise; guidance gap; DDIM inversion). **Light math** — TA explicitly said no deep math | Shared (1 author) |
| `s3_methodology` | Shared codebase (`src/zigzag_project/`); experimental controls (matched prompt/seed/scheduler/latent); per-lane setup; integration challenges | Per-person subsections + infra owners |
| `s4_results` | Per-lane results + analysis vs. paper; discussion / critical appraisal | Per-person subsections |
| `s5_conclusion` | Bounded positive takeaway; limitations; future work | Shared (1 author) |
| `appendix` | Full prompt lists, seeds, repro commands, extra tables, individual reflections | Mixed |

**Open structural vote:** keep methodology (s3) and results (s4) separate, *or* merge into one "Method & Results per person" block. Merging reads better for a per-person split.

## Verified results to write (numbers are final)

- **Eugen — SD2.1 × Pick-a-Pic:** 300 pairs, HPSv2 Z wins 213/300 = 71.0%, mean Δ +0.0110, 95% CI [+0.0087, +0.0132]. → `tables/eugen_results.md`
- **Daniel — SDXL × DrawBench:** 600 pairs (200×3), HPSv2 Z wins 540/600, AES Z wins 420/600, PickScore ~tie. **ImageReward = nan (fix or drop).** → `notebooks/daniel-sdxl-drawbench/`
- **Vlad — DreamShaper-xl-v2-turbo × GenEval:** 54 pairs, detector 31.5%→50.0% (+18.5 pts), HPSv2 Z wins 53/54 (Δ +0.0358), runtime 6.16s→13.80s, 9-step compute-matched control still Z 42/54. → `tables/vlad_*.md`

All three are **Standard vs Z only** — no resampling arm yet.

## Lanes to write as in-progress / scoped-out (no numbers)

- **Resampling** (Gouri, Lenka) — required method, not yet run. Write as methodology + challenge until a scored result lands.
- **Radu, Bogdan** (SD2.1-DrawBench Z, Hunyuan-DiT) — no committed results.
- **Baselines & ablations** (AYS, Semantic-CFG, Diffusion-DPO, guidance-gap, zigzag-step) — descoped to future work, justified by the TA's "pick a reasonable subset" guidance.

## Minimum figures/tables

1. Method diagram: Standard vs Resampling vs Z-Sampling.
2. Implementation pipeline diagram.
3. Headline results table (Eugen + Daniel + Vlad).
4. One qualitative Standard-vs-Z image pair.
5. Optional: runtime / compute-matched control table.
