# Vlad GenEval — Compute-Matched Standard Baseline

Purpose: test whether 4-step Z-Sampling only beats 4-step standard sampling
because it spends more compute. The comparison keeps the same DreamShaper
checkpoint, prompts, seeds, image size, CFG scale, and initial latent
construction, but regenerates the standard branch with 9 DDIM steps.

Source run: `outputs/vlad_geneval/balanced_18x3/`.
Compute-matched run: `outputs/vlad_geneval/compute_matched_standard9/`.

## HPSv2.1 outcome

| Comparison | Pairs | Z wins | Standard wins | Mean standard HPS | Mean Z HPS | Mean delta (Z - standard) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-step standard vs 4-step Z | 54 | 53 | 1 | 0.2168 | 0.2526 | +0.0358 |
| 9-step standard vs 4-step Z | 54 | 42 | 12 | 0.2280 | 0.2526 | +0.0245 |

The 9-step standard baseline improves over the original 4-step standard
baseline by +0.0112 mean HPS on the same 54 prompts/seeds, winning 33 of 54
direct standard9-vs-standard4 comparisons. However, Z-Sampling still keeps a
positive HPS margin against this stronger baseline.

## Runtime

| Method | Total runtime (s) | Mean per image (s) | Relative to 4-step Z |
| --- | ---: | ---: | ---: |
| 4-step standard | 332.4 | 6.16 | 0.45x |
| 4-step Z-Sampling | 745.5 | 13.80 | 1.00x |
| 9-step standard | 979.4 | 18.14 | 1.31x |

The chosen 9-step baseline is not merely compute-matched; on this MPS run it is
slower than the existing 4-step Z-Sampling images by about 31%.

## Per-tag HPSv2.1 against 9-step standard

| Tag | Pairs | Z wins | 9-step standard wins | Mean 9-step standard HPS | Mean Z HPS | Mean delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| color_attr | 9 | 9 | 0 | 0.2081 | 0.2634 | +0.0553 |
| colors | 9 | 8 | 1 | 0.2162 | 0.2544 | +0.0382 |
| counting | 9 | 6 | 3 | 0.2495 | 0.2557 | +0.0061 |
| position | 9 | 7 | 2 | 0.2421 | 0.2585 | +0.0164 |
| single_object | 9 | 7 | 2 | 0.2438 | 0.2582 | +0.0144 |
| two_object | 9 | 5 | 4 | 0.2084 | 0.2251 | +0.0167 |

Weakest prompt-level case: `geneval_000_single_object` (`a photo of a bench`)
has mean delta -0.0283 against 9-step standard, so this is where the
cost-matched baseline most clearly catches up. Strongest prompt-level case:
`geneval_453_color_attr` (`a photo of a purple wine glass and a black apple`)
has mean delta +0.0919.

## Reproduction commands

```bash
.venv/bin/python scripts/run_vlad_compute_matched_standard.py \
  --source-manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
  --standard-steps 9 \
  --out outputs/vlad_geneval/compute_matched_standard9

.venv/bin/python scripts/score_hpsv2_pairs.py \
  --manifest outputs/vlad_geneval/compute_matched_standard9/manifest.jsonl \
  --out outputs/vlad_geneval/compute_matched_standard9/scores.jsonl

.venv/bin/python scripts/summarize_vlad_results.py \
  --scores outputs/vlad_geneval/compute_matched_standard9/scores.jsonl \
  --out outputs/vlad_geneval/compute_matched_standard9/summary.md
```
