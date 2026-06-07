# Vlad GenEval HPSv2 Summary

Overall pairs: 54
Overall Z win rate: 77.8%
Mean HPS delta (Z - standard): 0.0245

## By GenEval tag

| Tag | Pairs | Z wins | Standard wins | Ties | Z win rate | Mean delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| color_attr | 9 | 9 | 0 | 0 | 100.0% | 0.0553 |
| colors | 9 | 8 | 1 | 0 | 88.9% | 0.0382 |
| counting | 9 | 6 | 3 | 0 | 66.7% | 0.0061 |
| position | 9 | 7 | 2 | 0 | 77.8% | 0.0164 |
| single_object | 9 | 7 | 2 | 0 | 77.8% | 0.0144 |
| two_object | 9 | 5 | 4 | 0 | 55.6% | 0.0167 |

## Top Z-Sampling wins

### geneval_453_color_attr seed 42 (color_attr)
Prompt: a photo of a purple wine glass and a black apple
Delta: 0.1189
- Standard: `outputs/vlad_geneval/compute_matched_standard9/images/geneval_453_color_attr_seed42_standard9.png`
- Z-Sampling: `outputs/vlad_geneval/balanced_18x3/images/geneval_453_color_attr_seed42_zigzag.png`

### geneval_453_color_attr seed 123 (color_attr)
Prompt: a photo of a purple wine glass and a black apple
Delta: 0.1086
- Standard: `outputs/vlad_geneval/compute_matched_standard9/images/geneval_453_color_attr_seed123_standard9.png`
- Z-Sampling: `outputs/vlad_geneval/balanced_18x3/images/geneval_453_color_attr_seed123_zigzag.png`

### geneval_002_single_object seed 2026 (single_object)
Prompt: a photo of a bicycle
Delta: 0.0743
- Standard: `outputs/vlad_geneval/compute_matched_standard9/images/geneval_002_single_object_seed2026_standard9.png`
- Z-Sampling: `outputs/vlad_geneval/balanced_18x3/images/geneval_002_single_object_seed2026_zigzag.png`


## Top Z-Sampling losses

### geneval_000_single_object seed 2026 (single_object)
Prompt: a photo of a bench
Delta: -0.0756
- Standard: `outputs/vlad_geneval/compute_matched_standard9/images/geneval_000_single_object_seed2026_standard9.png`
- Z-Sampling: `outputs/vlad_geneval/balanced_18x3/images/geneval_000_single_object_seed2026_zigzag.png`

### geneval_000_single_object seed 123 (single_object)
Prompt: a photo of a bench
Delta: -0.0280
- Standard: `outputs/vlad_geneval/compute_matched_standard9/images/geneval_000_single_object_seed123_standard9.png`
- Z-Sampling: `outputs/vlad_geneval/balanced_18x3/images/geneval_000_single_object_seed123_zigzag.png`

### geneval_260_colors seed 42 (colors)
Prompt: a photo of a pink car
Delta: -0.0275
- Standard: `outputs/vlad_geneval/compute_matched_standard9/images/geneval_260_colors_seed42_standard9.png`
- Z-Sampling: `outputs/vlad_geneval/balanced_18x3/images/geneval_260_colors_seed42_zigzag.png`
