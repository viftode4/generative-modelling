# Planning — Group 14

Project: reproducing **Zigzag Diffusion Sampling** (Bai et al., ICLR 2025) — comparing
Standard, Resampling, and Z-Sampling on text-to-image diffusion models.

## Current status (2026-06-09)

- **Deliverable:** one **group report**, scientific-paper style, with contribution shown in section titles (per the 2026-06-04 course clarification). *No longer individual reports.*
- **Report template:** TU Delft ACM `sigconf` — fixed sections `s1_introduction`, `s2_background`, `s3_methodology`, `s4_results`, `s5_conclusion`, `appendix`. Shared intro/background/conclusion; per-person subsections in methodology + results.
- **Results so far:** see [`experiments/results-index/coverage.md`](../experiments/results-index/coverage.md) — the audited source of truth. Three lanes have committed numbers (Eugen, Daniel, Vlad); resampling and the baseline/ablation experiments are not yet run.

## Confirmed dates

| Event | Date |
|-------|------|
| Midterm presentation | 2026-05-07 (done) |
| End-term presentation | 2026-06-09 — 7 min + 4 min feedback, graded 20% |
| Final report submission | **TBD — confirm with TA** |

## Files

| File | Contents |
|------|----------|
| [experiment-division.md](./experiment-division.md) | Who owns which model, dataset, and sampling method — ownership source of truth |
| [AGENTS.md](./AGENTS.md) | Agent instructions for this directory |

## Sources

- Paper: https://openreview.net/forum?id=MKvQH1ekeY
- Reference code: https://github.com/xie-lab-ml/zigzag-diffusion-sampling (mirrored in `external/`)
- Course docs + clarifications: `knowledgebase/course/`
- Paper notes: [`knowledgebase/papers/zigzag-diffusion-sampling.md`](../knowledgebase/papers/zigzag-diffusion-sampling.md)
