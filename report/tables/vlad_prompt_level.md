# Vlad GenEval Balanced 18x3 — Prompt-Level HPSv2 Table

Source: `outputs/vlad_geneval/balanced_18x3/scores.jsonl`. Each row aggregates three seeds for one prompt. Deltas are signed `Z-Sampling - standard`; positive means HPSv2.1 preferred Z-Sampling for that seed.

| Prompt ID | Tag | Prompt | Pairs | Z wins | Std wins | Std mean | Z mean | Mean delta | Min delta | Max delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| geneval_259_colors | colors | a photo of a blue fire hydrant | 3 | 3 | 0 | 0.1590 | 0.2398 | +0.0808 | +0.0600 | +0.1090 |
| geneval_260_colors | colors | a photo of a pink car | 3 | 3 | 0 | 0.2190 | 0.2672 | +0.0482 | +0.0259 | +0.0823 |
| geneval_180_counting | counting | a photo of two backpacks | 3 | 3 | 0 | 0.2121 | 0.2597 | +0.0476 | +0.0217 | +0.0947 |
| geneval_080_two_object | two_object | a photo of a bench and a sports ball | 3 | 3 | 0 | 0.2020 | 0.2476 | +0.0457 | +0.0203 | +0.0659 |
| geneval_261_colors | colors | a photo of a purple cup | 3 | 3 | 0 | 0.2121 | 0.2562 | +0.0441 | +0.0206 | +0.0676 |
| geneval_454_color_attr | color_attr | a photo of a green bus and a purple microwave | 3 | 3 | 0 | 0.2145 | 0.2565 | +0.0420 | +0.0302 | +0.0642 |
| geneval_455_color_attr | color_attr | a photo of a green skis and a brown airplane | 3 | 3 | 0 | 0.2057 | 0.2452 | +0.0395 | +0.0274 | +0.0523 |
| geneval_081_two_object | two_object | a photo of a toothbrush and a snowboard | 3 | 3 | 0 | 0.1777 | 0.2138 | +0.0361 | +0.0290 | +0.0414 |
| geneval_353_position | position | a photo of a dog right of a teddy bear | 3 | 3 | 0 | 0.2367 | 0.2721 | +0.0353 | +0.0209 | +0.0529 |
| geneval_453_color_attr | color_attr | a photo of a purple wine glass and a black apple | 3 | 3 | 0 | 0.2539 | 0.2886 | +0.0347 | +0.0161 | +0.0463 |
| geneval_082_two_object | two_object | a photo of a toaster and an oven | 3 | 3 | 0 | 0.1838 | 0.2140 | +0.0302 | +0.0263 | +0.0364 |
| geneval_179_counting | counting | a photo of two clocks | 3 | 3 | 0 | 0.2118 | 0.2385 | +0.0267 | +0.0172 | +0.0451 |
| geneval_001_single_object | single_object | a photo of a cow | 3 | 3 | 0 | 0.2625 | 0.2884 | +0.0259 | +0.0198 | +0.0306 |
| geneval_002_single_object | single_object | a photo of a bicycle | 3 | 3 | 0 | 0.2427 | 0.2682 | +0.0255 | +0.0206 | +0.0297 |
| geneval_355_position | position | a photo of a couch below a cup | 3 | 2 | 1 | 0.2412 | 0.2664 | +0.0252 | -0.0023 | +0.0434 |
| geneval_181_counting | counting | a photo of four handbags | 3 | 3 | 0 | 0.2465 | 0.2688 | +0.0223 | +0.0148 | +0.0371 |
| geneval_354_position | position | a photo of a wine glass above a kite | 3 | 3 | 0 | 0.2179 | 0.2370 | +0.0191 | +0.0086 | +0.0362 |
| geneval_000_single_object | single_object | a photo of a bench | 3 | 3 | 0 | 0.2030 | 0.2181 | +0.0151 | +0.0066 | +0.0251 |

## Reading notes

- Z-Sampling wins all three seeds for 17/18 prompts; the only prompt with a standard win is `geneval_355_position` (`a photo of a couch below a cup`), where standard wins one seed by `0.0023` HPS points.
- The largest mean gain is `geneval_259_colors` (`a photo of a blue fire hydrant`, mean delta `+0.0808`).
- The smallest mean gain is `geneval_000_single_object` (`a photo of a bench`, mean delta `+0.0151`), still with Z-Sampling winning all three seeds.