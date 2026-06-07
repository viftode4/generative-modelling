# Vlad GenEval Detector Summary

Overall pairs: 54
Standard pass rate: 31.5% (17/54)
Z-Sampling pass rate: 50.0% (27/54)
Delta (Z - standard): +18.5%
Both pass: 17; Z only: 10; standard only: 0; neither: 27

## Detector pass rate by GenEval tag

| Tag | Pairs | Std pass | Z pass | Std rate | Z rate | Delta | Z-only | Std-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| color_attr | 9 | 0 | 0 | 0.0% | 0.0% | +0.0% | 0 | 0 |
| colors | 9 | 5 | 8 | 55.6% | 88.9% | +33.3% | 3 | 0 |
| counting | 9 | 2 | 5 | 22.2% | 55.6% | +33.3% | 3 | 0 |
| position | 9 | 0 | 1 | 0.0% | 11.1% | +11.1% | 1 | 0 |
| single_object | 9 | 9 | 9 | 100.0% | 100.0% | +0.0% | 0 | 0 |
| two_object | 9 | 1 | 4 | 11.1% | 44.4% | +33.3% | 3 | 0 |

## Per-pair detector results

| prompt_id | seed | tag | std pass | Z pass | prompt |
| --- | ---: | --- | :---: | :---: | --- |
| geneval_000_single_object | 42 | single_object | yes | yes | a photo of a bench |
| geneval_000_single_object | 123 | single_object | yes | yes | a photo of a bench |
| geneval_000_single_object | 2026 | single_object | yes | yes | a photo of a bench |
| geneval_001_single_object | 42 | single_object | yes | yes | a photo of a cow |
| geneval_001_single_object | 123 | single_object | yes | yes | a photo of a cow |
| geneval_001_single_object | 2026 | single_object | yes | yes | a photo of a cow |
| geneval_002_single_object | 42 | single_object | yes | yes | a photo of a bicycle |
| geneval_002_single_object | 123 | single_object | yes | yes | a photo of a bicycle |
| geneval_002_single_object | 2026 | single_object | yes | yes | a photo of a bicycle |
| geneval_080_two_object | 42 | two_object | no | yes | a photo of a bench and a sports ball |
| geneval_080_two_object | 123 | two_object | yes | yes | a photo of a bench and a sports ball |
| geneval_080_two_object | 2026 | two_object | no | yes | a photo of a bench and a sports ball |
| geneval_081_two_object | 42 | two_object | no | no | a photo of a toothbrush and a snowboard |
| geneval_081_two_object | 123 | two_object | no | no | a photo of a toothbrush and a snowboard |
| geneval_081_two_object | 2026 | two_object | no | no | a photo of a toothbrush and a snowboard |
| geneval_082_two_object | 42 | two_object | no | yes | a photo of a toaster and an oven |
| geneval_082_two_object | 123 | two_object | no | no | a photo of a toaster and an oven |
| geneval_082_two_object | 2026 | two_object | no | no | a photo of a toaster and an oven |
| geneval_179_counting | 42 | counting | no | no | a photo of two clocks |
| geneval_179_counting | 123 | counting | no | yes | a photo of two clocks |
| geneval_179_counting | 2026 | counting | no | no | a photo of two clocks |
| geneval_180_counting | 42 | counting | yes | yes | a photo of two backpacks |
| geneval_180_counting | 123 | counting | no | yes | a photo of two backpacks |
| geneval_180_counting | 2026 | counting | no | no | a photo of two backpacks |
| geneval_181_counting | 42 | counting | no | yes | a photo of four handbags |
| geneval_181_counting | 123 | counting | yes | yes | a photo of four handbags |
| geneval_181_counting | 2026 | counting | no | no | a photo of four handbags |
| geneval_259_colors | 42 | colors | no | yes | a photo of a blue fire hydrant |
| geneval_259_colors | 123 | colors | yes | yes | a photo of a blue fire hydrant |
| geneval_259_colors | 2026 | colors | no | yes | a photo of a blue fire hydrant |
| geneval_260_colors | 42 | colors | no | yes | a photo of a pink car |
| geneval_260_colors | 123 | colors | yes | yes | a photo of a pink car |
| geneval_260_colors | 2026 | colors | yes | yes | a photo of a pink car |
| geneval_261_colors | 42 | colors | yes | yes | a photo of a purple cup |
| geneval_261_colors | 123 | colors | yes | yes | a photo of a purple cup |
| geneval_261_colors | 2026 | colors | no | no | a photo of a purple cup |
| geneval_353_position | 42 | position | no | no | a photo of a dog right of a teddy bear |
| geneval_353_position | 123 | position | no | yes | a photo of a dog right of a teddy bear |
| geneval_353_position | 2026 | position | no | no | a photo of a dog right of a teddy bear |
| geneval_354_position | 42 | position | no | no | a photo of a wine glass above a kite |
| geneval_354_position | 123 | position | no | no | a photo of a wine glass above a kite |
| geneval_354_position | 2026 | position | no | no | a photo of a wine glass above a kite |
| geneval_355_position | 42 | position | no | no | a photo of a couch below a cup |
| geneval_355_position | 123 | position | no | no | a photo of a couch below a cup |
| geneval_355_position | 2026 | position | no | no | a photo of a couch below a cup |
| geneval_453_color_attr | 42 | color_attr | no | no | a photo of a purple wine glass and a black apple |
| geneval_453_color_attr | 123 | color_attr | no | no | a photo of a purple wine glass and a black apple |
| geneval_453_color_attr | 2026 | color_attr | no | no | a photo of a purple wine glass and a black apple |
| geneval_454_color_attr | 42 | color_attr | no | no | a photo of a green bus and a purple microwave |
| geneval_454_color_attr | 123 | color_attr | no | no | a photo of a green bus and a purple microwave |
| geneval_454_color_attr | 2026 | color_attr | no | no | a photo of a green bus and a purple microwave |
| geneval_455_color_attr | 42 | color_attr | no | no | a photo of a green skis and a brown airplane |
| geneval_455_color_attr | 123 | color_attr | no | no | a photo of a green skis and a brown airplane |
| geneval_455_color_attr | 2026 | color_attr | no | no | a photo of a green skis and a brown airplane |

## HPSv2 head-to-head (for cross-reference)

Pairs: 54; Z win rate: 98.1%; mean HPS delta: +0.0358

| Tag | Pairs | Z wins | Standard wins | Ties | Z win rate | Mean delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| color_attr | 9 | 9 | 0 | 0 | 100.0% | +0.0387 |
| colors | 9 | 9 | 0 | 0 | 100.0% | +0.0577 |
| counting | 9 | 9 | 0 | 0 | 100.0% | +0.0322 |
| position | 9 | 8 | 1 | 0 | 88.9% | +0.0265 |
| single_object | 9 | 9 | 0 | 0 | 100.0% | +0.0222 |
| two_object | 9 | 9 | 0 | 0 | 100.0% | +0.0373 |

## Metric agreement (HPSv2 vs GenEval detector, per pair)

Pairs with both metrics available: 54/54
Detector-directional pairs (one sampler passes and the other fails): 10/54
Both metrics favor Z: 10
Both metrics favor standard: 0
HPS favors Z but detector favors standard: 0
HPS favors standard but detector favors Z: 0
Detector-tied pairs (both pass or both fail): 44
