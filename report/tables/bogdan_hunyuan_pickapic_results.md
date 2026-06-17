# Bogdan Hunyuan-DiT + Pick-a-Pic 18x3 Results Table

Model: `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers`. Subset: first 18 prompts
from `experiments/prompts/pickapic_full.md`. Seeds: `42`, `123`, `2026`.
Sampling settings: `infer_step=50`, `gamma_1=5.0`, `gamma_2=0.0`,
`lambda_step=49`, `T_max=1`, `image_size=1024`.

Metric: HPSv2.1 pair score. Win means higher HPSv2.1 score on the paired image.

| Group | Pairs | Std mean | Z mean | Mean delta | Z wins | Standard wins | Z win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pick-a-Pic first 18 | 54 | 0.3195 | 0.3153 | -0.0042 | 23 | 31 | 42.6% |

Runtime on RTX 4090:

| Method | Total runtime | Mean runtime per image |
| --- | ---: | ---: |
| Standard | 1169.8 s | 21.66 s |
| Z-Sampling | 3438.8 s | 63.68 s |

Z-Sampling runtime ratio: `2.94x` standard.

Top Z-Sampling wins:

| Prompt ID | Seed | Delta | Prompt |
| --- | ---: | ---: | --- |
| `pickapic_009_official` | 2026 | +0.0319 | A beautiful UX design for a IoT web store |
| `pickapic_014_official` | 42 | +0.0229 | Kid eating gummy bears and sweets while watching tablet, sitting on green sofa |
| `pickapic_016_official` | 42 | +0.0193 | painting of Medieval talents crafts and skills |

Top Z-Sampling losses:

| Prompt ID | Seed | Delta | Prompt |
| --- | ---: | ---: | --- |
| `pickapic_010_official` | 42 | -0.0435 | A knitted forrest with foxes at sunset painted with pastel colors |
| `pickapic_011_official` | 42 | -0.0378 | older Harry Styles wearing feathers, riso, lithograph, illustration |
| `pickapic_010_official` | 2026 | -0.0330 | A knitted forrest with foxes at sunset painted with pastel colors |

Artifacts:

- Manifest: `outputs/bogdan_hunyuan_pickapic/subset_18x3/manifest.jsonl`
- Scores: `outputs/bogdan_hunyuan_pickapic/subset_18x3/scores.jsonl`
- Summary: `outputs/bogdan_hunyuan_pickapic/subset_18x3/summary.md`
