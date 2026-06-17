# Vlad GenEval Balanced 18x3 — Results Table

Model: `Lykon/dreamshaper-xl-v2-turbo`. Subset: first 3 GenEval prompts per
tag (18 prompts) x 3 seeds (42, 123, 2026) x {standard, Z-Sampling} = 54
paired generations. Metrics: official GenEval detector pass/fail and HPSv2.1
preference score. Runtime measured on Apple M4 Max, MPS, fp16 for generation;
GenEval detector scoring ran on CPU.

## GenEval detector outcomes

Detailed detector table: [`vlad_geneval_detector.md`](vlad_geneval_detector.md).

| Metric | Standard | Z-Sampling | Delta (Z - standard) |
| --- | ---: | ---: | ---: |
| GenEval detector pass rate | 31.5% (17/54) | 50.0% (27/54) | +18.5% |

Pair categories: both pass 17; Z-only 10; standard-only 0; neither 27.

| Tag | Pairs | Std pass | Z pass | Std rate | Z rate | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| color_attr | 9 | 0 | 0 | 0.0% | 0.0% | +0.0% |
| colors | 9 | 5 | 8 | 55.6% | 88.9% | +33.3% |
| counting | 9 | 2 | 5 | 22.2% | 55.6% | +33.3% |
| position | 9 | 0 | 1 | 0.0% | 11.1% | +11.1% |
| single_object | 9 | 9 | 9 | 100.0% | 100.0% | +0.0% |
| two_object | 9 | 1 | 4 | 11.1% | 44.4% | +33.3% |

Detector command:

```bash
.venv/bin/python scripts/run_geneval_detector.py \
  --manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/geneval_scores.jsonl \
  --device cpu
```

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

## Prompt-level consistency and edge cases

The full 18-prompt aggregation is in
[`vlad_prompt_level.md`](vlad_prompt_level.md). Z-Sampling wins all three
seeds for 17/18 prompts. The only prompt with a standard-sampling win is
`geneval_355_position` (`a photo of a couch below a cup`), where standard
wins one seed by only 0.0023 HPS points.

Prompt-level extremes:

| Case | Prompt ID | Tag | Mean delta | Z wins / 3 | Note |
| --- | --- | --- | ---: | ---: | --- |
| Largest mean gain | `geneval_259_colors` | colors | +0.0808 | 3 | Standard often misses the hydrant object; Z-Sampling makes it recognizable. |
| Only mixed prompt | `geneval_355_position` | position | +0.0252 | 2 | The one standard win is a near-tie; both samples satisfy the coarse couch-below-cup relation. |
| Smallest mean gain | `geneval_000_single_object` | single_object | +0.0151 | 3 | Z-Sampling still wins all seeds, but visual gain is mostly sharper texture rather than clearer bench semantics. |

Manual visual notes for selected pairs are in
[`vlad_qualitative_audit.md`](vlad_qualitative_audit.md). They support the
main caveat: HPSv2.1 preference gains are real paired-score improvements,
but not a substitute for GenEval detector pass rates.

## Runtime cost per pair (seconds, MPS fp16, 768x768, 4 DDIM steps)

| Method     | Total runtime (s) | Mean per pair (s) | Ratio vs standard |
| ---        | ---:              | ---:              | ---:              |
| Standard   | 332.4             | 6.16              | 1.00x             |
| Z-Sampling | 745.5             | 13.80             | 2.24x             |

## Compute-matched HPSv2 control

Detailed cost-control table:
[`vlad_compute_matched.md`](vlad_compute_matched.md).

To check whether the HPS gain is only due to Z-Sampling spending more network
evaluations, we regenerated the standard branch with 9 DDIM steps from the same
prompt/seed/initial latent construction and compared those images against the
existing 4-step Z-Sampling images.

| Comparison | Pairs | Z wins | Standard wins | Mean standard HPS | Mean Z HPS | Mean delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-step standard vs 4-step Z | 54 | 53 | 1 | 0.2168 | 0.2526 | +0.0358 |
| 9-step standard vs 4-step Z | 54 | 42 | 12 | 0.2280 | 0.2526 | +0.0245 |

9-step standard averages 18.14 s/image on MPS, versus 13.80 s/image for the
existing 4-step Z-Sampling run. The cost-control therefore narrows the margin
but does not erase it under HPSv2.1.

## Notes

- HPS deltas are signed (Z - standard). Positive means Z-Sampling preferred.
- The only standard-sampling win in the 54-pair run was
  `geneval_355_position seed 42` (prompt: "a photo of a couch below a cup",
  delta -0.0023).
- HPSv2 is a learned preference proxy; GenEval is detector-grounded.
  Both metrics favor Z-Sampling on this subset, but the magnitudes differ.
- The detector ran with `mmdet==2.28.2`, `mmcv-full==1.7.2`, and CPU
  execution on Apple Silicon. This avoids CUDA, but it is still a small
  subset rather than the full paper benchmark.
- The compute-matched control is HPSv2-only. We did not rerun the GenEval
  detector on the 9-step standard images.
