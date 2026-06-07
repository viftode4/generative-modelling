# End-term Presentation Draft — Group 14 Zigzag Diffusion Sampling

This is slide-ready markdown. Copy into PowerPoint/Google Slides/Overleaf Beamer as needed. Each slide has visible bullets plus speaker notes.

---

## 1. Project recap: why Z-Sampling?

**Visible bullets**

- Paper: *Zigzag Diffusion Sampling: Diffusion Models can Self-Improve via Self-Reflection*.
- Problem: diffusion images can look good while failing prompt details: count, color, spatial relations.
- Claim we test: inference-time denoise/invert/re-denoise steps can improve prompt alignment without retraining.

**Speaker notes**

We chose this paper because the method is simple to state but non-trivial to reproduce: it changes only sampling, not training. That made it realistic for a course project, while still giving us implementation, evaluation, and reproducibility issues to investigate.

**Source**: `knowledgebase/project-overview.md`; OpenReview `MKvQH1ekeY`.

---

## 2. Method: standard sampling vs zigzag sampling

**Visible bullets**

- Standard DDIM: denoise monotonically from noisy latent to image.
- Z-Sampling: for early steps, denoise with guidance `gamma_1`, invert with lower guidance `gamma_2`, then re-denoise.
- Intuition: the guidance gap injects prompt-related semantic signal step by step.

**Suggested visual**

```text
Standard:  x_T -> x_{T-1} -> ... -> x_0
Z-Sampling: x_t --denoise(gamma_1)--> x_{t-1}
                 --invert(gamma_2)--> x_t'
                 --denoise(gamma_1)--> refined x_{t-1}
```

**Speaker notes**

The important implementation point is that this is not just increasing CFG. It uses both a forward denoising scheduler and a DDIM inverse scheduler. The pair should be compatible, otherwise the zigzag loop drifts from the intended method.

**Source**: `knowledgebase/papers/zigzag-diffusion-sampling.md`; upstream `utils/pipeline_stable_diffusion_xl.py` at commit `513eb51`.

---

## 3. Scope: how the group split the project

**Visible bullets**

- Seven students, different model/dataset cells.
- Sampling methods in scope: standard sampling, RePaint-style resampling, Z-Sampling.
- Goal: show implementation effort and compare trends, not full benchmark-scale reproduction.

**Matrix to show**

| Person | Model | Dataset |
|---|---|---|
| Eugen | SD2.1 | Pick-a-Pic |
| Daniel | SDXL | DrawBench |
| Lenka | SDXL | Pick-a-Pic |
| Bogdan | Hunyuan-DiT | GenEval |
| Gouri | DreamShaper-xl-v2-turbo | DrawBench |
| Vlad | DreamShaper-xl-v2-turbo | GenEval |
| Radu | SD2.1 | DrawBench |

**Speaker notes**

This split makes the individual reports distinct. The shared part is the method and infrastructure; the individual part is the model/dataset cell, results, troubleshooting, and interpretation.

**Source**: `planning/role-assignments.md`; `planning/experiment-division.md`.

---

## 4. Implementation progress

**Visible bullets**

- Repo-local Z-Sampling implementation in `src/zigzag_project/zsampling.py`.
- Shared model handles keep pipeline, DDIM scheduler, inverse scheduler, device, dtype together.
- Resampling integration repaired with method-specific config validation.
- Fast tests cover Z-Sampling loop behavior without downloading models.

**Speaker notes**

A lot of the work was not just running notebooks. We had to port the upstream code to a newer diffusers stack, keep scheduler state aligned, support Apple Silicon for local runs, and prevent integration errors such as passing a Z-Sampling config into the resampling baseline.

**Source**: `src/zigzag_project/zsampling.py`; `src/zigzag_project/samplers/`; `tests/test_zsampling.py`; `tests/test_samplers_package.py`.

---

## 5. Verified result: SD2.1 + Pick-a-Pic

**Visible bullets**

- 300 paired generations: 100 Pick-a-Pic prompts × 3 seeds.
- Metric: HPSv2.1 preference score.
- Z-Sampling wins 213/300 pairs = 71.0%.
- Mean HPS delta: +0.0110 with 95% bootstrap CI [+0.0087, +0.0132].

**Small table**

| Pairs | Z wins | Standard wins | Mean delta |
|---:|---:|---:|---:|
| 300 | 213 | 87 | +0.0110 |

**Speaker notes**

This gives us one larger paired run. The effect is positive but not huge, which is useful: the presentation should not oversell Z-Sampling as magic. It looks like a modest but consistent preference gain.

**Source**: `report/tables/eugen_results.md`; `outputs/eugen_sd2.1_pickapic/summary.md`.

---

## 6. Verified result: DreamShaper + GenEval detector

**Visible bullets**

- Vlad lane: DreamShaper-xl-v2-turbo + balanced GenEval subset.
- 18 prompts × 3 seeds = 54 paired comparisons.
- GenEval detector pass rate: standard 31.5% vs Z-Sampling 50.0%.
- One-sided detector wins: 10 Z-only, 0 standard-only.

**Small table**

| Metric | Standard | Z-Sampling | Delta |
|---|---:|---:|---:|
| GenEval detector pass rate | 17/54 | 27/54 | +18.5 pp |
| HPSv2 pair wins | 1/54 | 53/54 | +0.0358 mean delta |

**Speaker notes**

This is important because HPS alone could reward aesthetics. The detector result checks compositional rules. It still is not full GenEval, but it answers the main local concern: on this subset, the detector also favors Z-Sampling.

**Source**: `report/tables/vlad_results.md`; `report/tables/vlad_geneval_detector.md`.

---

## 7. Vlad qualitative example and cost control

**Visible bullets**

- Biggest visual win: blue fire hydrant prompt becomes more recognizable and blue.
- Runtime: Z-Sampling is not free — 13.80s/image vs 6.16s/image for 4-step standard.
- Compute control: 9-step standard is slower than Z but Z still wins 42/54 HPS pairs.

**Suggested visual**

Use side-by-side image pair:

- `report/figures/vlad_geneval/geneval_259_colors_seed42_standard.png`
- `report/figures/vlad_geneval/geneval_259_colors_seed42_zigzag.png`

**Speaker notes**

The cost control is the strongest response to the obvious criticism: maybe Z only wins because it does more work. A stronger 9-step standard baseline narrows the margin, but does not erase it. So the fair claim is not that Z is free; the fair claim is that the gain is not explained only by extra denoising steps on this subset.

**Source**: `report/tables/vlad_compute_matched.md`; `report/tables/vlad_qualitative_audit.md`.

---

## 8. Limitations and honest interpretation

**Visible bullets**

- We did not reproduce the full paper-scale benchmark.
- HPSv2 is useful but not the same as detector-grounded compositional correctness.
- Environment differs from paper: modern `diffusers`, Apple Silicon/MPS and CPU detector path.
- Some prompt types remain hard: color attribution and most position prompts.

**Speaker notes**

This is where we should be explicit. Our strongest evidence is paired local reproduction with controlled seeds and documented scripts. The weakness is scale and environment mismatch. The presentation should ask for feedback on whether this level of implementation/evaluation is sufficient for the individual reports.

**Source**: `knowledgebase/course/requirements-summary.md`; `knowledgebase/research-log.md`; `report/tables/vlad_results.md`.

---

## 9. What remains before final reports

**Visible bullets**

- Each owner finalizes their lane: commands, environment, seeds, outputs, interpretation.
- Consolidate result tables without overclaiming unfinished cells.
- Keep figures minimal: one strong pair per report/presenter.
- Final writing focus: critical appraisal, not just positive results.

**Feedback request**

1. Is our small-but-controlled reproduction scope acceptable?
2. Should we prioritize more detector scoring, or clearer interpretation of existing results?
3. Are the runtime and metric caveats explicit enough?

**Speaker notes**

The next step is not to add random experiments. It is to turn the verified runs into clean individual reports and make sure each limitation is stated as a limitation, not hidden.

**Source**: `planning/report-spec.md`; `knowledgebase/course/requirements-summary.md`.
