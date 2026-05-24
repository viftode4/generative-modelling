# Vlad GenEval Balanced 18x3 — Results Table

Model: `Lykon/dreamshaper-xl-v2-turbo`. Subset: first 3 GenEval prompts per
tag (18 prompts) x 3 seeds (42, 123, 2026) x {standard, Z-Sampling} = 54
paired generations. Metric: HPSv2.1 preference score. Runtime measured on
Apple M4 Max, MPS, fp16.

## Per-tag HPSv2 outcomes with 95% bootstrap CIs

Bootstrap confidence intervals computed with n=10000 resamples seeded with
rng=42, applied per-tag to the paired HPS deltas in `scores.jsonl`. "Win"
means higher HPSv2.1 score on the paired image — not the GenEval detector
score reported by Bai et al.

| GenEval tag   | Pairs | Std mean | Z mean  | Mean delta | 95% CI (boot)        | HPSv2.1 preference win rate |
| ---           | ---:  | ---:     | ---:    | ---:       | ---                  | ---:                        |
| color_attr    | 9     | 0.2247   | 0.2634  | +0.0387    | [+0.0300, +0.0476]   | 100.0%                      |
| colors        | 9     | 0.1967   | 0.2544  | +0.0577    | [+0.0405, +0.0759]   | 100.0%                      |
| counting      | 9     | 0.2235   | 0.2557  | +0.0322    | [+0.0196, +0.0503]   | 100.0%                      |
| position      | 9     | 0.2319   | 0.2585  | +0.0265    | [+0.0153, +0.0374]   | 88.9%                       |
| single_object | 9     | 0.2361   | 0.2582  | +0.0222    | [+0.0171, +0.0267]   | 100.0%                      |
| two_object    | 9     | 0.1878   | 0.2251  | +0.0373    | [+0.0293, +0.0466]   | 100.0%                      |
| **overall**   | **54**| **0.2168**| **0.2526** | **+0.0358** | **[+0.0304, +0.0418]** | **98.1%**             |

All per-tag 95% confidence intervals on the delta are strictly positive,
including `position`. At n=9 per tag the position effect is small but
statistically positive.

## Runtime cost per pair (seconds, MPS fp16, 768x768, 4 DDIM steps)

| Method     | Total runtime (s) | Mean per pair (s) | Ratio vs standard |
| ---        | ---:              | ---:              | ---:              |
| Standard   | 332.4             | 6.16              | 1.00x             |
| Z-Sampling | 745.5             | 13.80             | 2.24x             |

## Notes

- HPS deltas are signed (Z - standard). Positive means Z-Sampling preferred.
- The only standard-sampling win in the 54-pair run was
  `geneval_355_position seed 42` (prompt: "a photo of a couch below a cup",
  delta -0.0023).
- HPSv2 is a learned preference proxy, not the GenEval detector-grounded
  compositional score. These numbers should not be compared directly to
  the GenEval results in Bai et al.
- GenEval detector pass rates are *not* reported here. The official
  detector pipeline (`mmdet>=3` + Mask2Former + `mmcv` extension ops)
  did not install on the Apple Silicon (`torch==2.12.0`) environment.
  The runner (`scripts/run_geneval_detector.py`) and aggregator
  (`scripts/summarize_vlad_geneval.py`) are committed and ready for a
  CUDA-side rerun against the existing 108 committed images.
