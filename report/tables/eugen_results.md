# Eugen's SD2.1 + Pick-a-Pic 100x3 — Results Table

Model: `sd2-community/stable-diffusion-2-1`. Subset: first 100 prompts from Pick-a-Pic v2 validation split x 3 seeds (42, 123, 2026) x {standard, Z-Sampling} = 300 paired generations. Metric: HPSv2.1 preference score.

## 1. Results Table (Absolute HPSv2 Scores x 100 prompts 3 seeds)

This table formats our results to match the absolute scoring values and scaling reported in Table 2 of the original Z-Sampling paper. 

| Model / Source | Method | Pick-a-Pic HPSv2 ↑ | Delta (Z - Std) |
| --- | --- | ---: | ---: |
| **SD-2.1 (Reproduction)** | Standard | 24.97 | — |
| | Z-Sampling (ours) | **26.07** | **+1.10** |
| *SD-2.1 (Original Paper)* | *Standard* | *23.05* | — |
| | *Z-Sampling* | ***24.53*** | ***+1.48*** |

## 2. Per-tag HPSv2 outcomes with 95% bootstrap CIs

Bootstrap confidence intervals computed with n=10000 resamples seeded with rng=42.

| Dataset / tag | Pairs | Std mean | Z mean | Mean delta | 95% CI (boot) | HPSv2.1 preference win rate |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| official | 300 | 0.2497 | 0.2607 | +0.0110 | [+0.0087, +0.0132] | 71.0% |
| **overall** | **300** | **0.2497** | **0.2607** | **+0.0110** | **[+0.0087, +0.0132]** | **71.0%** |

## Notes

- HPS deltas are signed (Z - standard). Positive means Z-Sampling preferred.
- Standard wins: 87 (29.0%), Z-Sampling wins: 213 (71.0%), Ties: 0 (0.0%).
- Model weights were executed on Colab GPU / CPU runtimes.
- In the original paper, HPSv2 values are reported scaled by 100 (e.g., `23.05` corresponds to raw score `0.2305`). Our absolute scores (`24.97` vs `26.07`) show a very similar and statistically significant quality improvement trend (+1.10 vs +1.48 reported in the paper).

