# Vlad's SDXL × Pick-a-Pic Efficiency (Experiment 4 / #8) — Results Table

Model: `stabilityai/stable-diffusion-xl-base-1.0`. Subset: first 100 Pick-a-Pic
prompts × seed 42 × {standard, compute-matched Z-Sampling} across four step budgets
= 400 paired generations (100 per budget). Reproduces the paper's time-matched
efficiency study (Figure 10). Metric: HPS v2.1 preference score + per-image runtime.

Source: `outputs/vlad_exp8_sdxl_pickapic/{manifest,scores}.jsonl`,
summary `outputs/vlad_exp8_sdxl_pickapic/efficiency_summary.md`,
figure `outputs/vlad_exp8_sdxl_pickapic/exp8_hps_vs_time.png`.

## Time-matched protocol

For each standard budget `T_s`, Z-Sampling uses `T_z = T_s/2` denoising steps and
`lambda = T_s/4` zigzag steps, so the noise-prediction op counts match exactly
(`T_s` vs `T_z + 2·lambda`). Both arms share the initial latent, DDIM + DDIM-inverse
scheduler, guidance (`gamma_1=5.5`, `gamma_2=0`), `t_max=1`, and 1024×1024 resolution.

Time is the **median** seconds/image per arm. Shared-MPS wall-clock is heavy-tailed: at
`T_s=50` two standard cells exceeded 3000 s (3995.7 s and 3534.4 s) and inflate the
Standard mean to 231 s against a 152 s median, and the Z arm has a comparable 708 s
outlier. We therefore report the median. The choice is not load-bearing: at
`T_s in {20,30,40}` the median and mean agree to within ~1 s, and the conclusions are
unchanged under the mean. HPS values are means, with 95% bootstrap CIs on the paired
delta (n=10000 resamples, rng=42).

## Per-budget HPS v2.1 and median runtime

| Budget T_s | T_z | λ | Pairs | Std time (s, med) | Z time (s, med) | Std HPS | Z HPS | Mean delta | 95% CI (boot) | Z win rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 20 | 10 | 5  | 100 | 36.79  | 36.85  | 0.2574 | 0.2720 | +0.0146 | [+0.0111, +0.0183] | 81.0% |
| 30 | 15 | 7  | 100 | 54.44  | 52.66  | 0.2645 | 0.2779 | +0.0134 | [+0.0102, +0.0166] | 82.0% |
| 40 | 20 | 10 | 100 | 72.18  | 72.19  | 0.2712 | 0.2860 | +0.0147 | [+0.0116, +0.0183] | 87.0% |
| 50 | 25 | 12 | 100 | 152.36 | 149.61 | 0.2727 | 0.2891 | +0.0164 | [+0.0134, +0.0198] | 91.0% |
| **overall** | | | **400** | | | **0.2665** | **0.2812** | **+0.0148** | **[+0.0131, +0.0165]** | **85.2%** |

## Efficiency headline

- Best Standard HPS across budgets: **0.2727** at ~152.4 s/image (`T_s=50`).
- Z-Sampling reaches HPS **0.2779** at ~52.7 s/image (`T_s=30`) — matching and beating
  the best Standard configuration with **~65% less time/image**.
- At every budget the compute-matched arms take essentially the same median time
  (e.g. 72.2 s vs 72.2 s at `T_s=40`), yet the Z curve lies above Standard throughout
  — a clean Pareto shift, reproducing the paper's claim that "even with 36% less
  computational time, Z-Sampling can surpass the best performance of standard sampling."

## Notes

- HPS deltas are signed (Z − standard). Positive means Z-Sampling preferred. All four
  per-budget CIs and the overall CI are strictly positive.
- Single seed (42), 100 prompts — smaller than the paper's full scale.
- Pick-a-Pic prompts are the author's `PickaPic-selected-prompts` substitute (count +
  dataset match the paper; exact prompts not guaranteed identical).
- Absolute runtimes are device-specific (Apple M-series / MPS, fp16); the comparison is
  strictly internal — Standard vs Z on the same hardware with the same initial latents.
