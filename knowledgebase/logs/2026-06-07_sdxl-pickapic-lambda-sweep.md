# SDXL Pick-a-Pic Lambda-Step Sweep

## Metadata

- **Date:** 2026-06-07
- **Owner:** Codex run for requested SDXL Pick-a-Pic lambda-step ablation
- **Goal:** Generate SDXL standard-vs-Z-Sampling image pairs for lambda-step values `0`, `3`, `7`, `15`, `25`, `35`, and `50`, then score win rate against standard sampling.
- **Source code commit:** `7c5d058` plus uncommitted local changes for the sweep scripts and metadata preservation.
- **Upstream commit:** Unavailable; `external/zigzag-diffusion-sampling/.git` is not initialized in this workspace, and the generation manifest records `upstream_commit: null`.
- **Environment:** `.venv`, Python 3.10.20, CUDA available.
- **Hardware:** NVIDIA GeForce RTX 4090.
- **Seed(s):** `42`, `123`, `2026`.

## Command(s)

```powershell
.venv\Scripts\python.exe scripts/run_sdxl_pickapic_lambda_sweep.py `
  --out outputs/sdxl_pickapic_lambda_sweep/subset_9x3 `
  --force

.venv\Scripts\python.exe scripts/score_hpsv2_pairs.py `
  --manifest outputs/sdxl_pickapic_lambda_sweep/subset_9x3/manifest.jsonl `
  --out outputs/sdxl_pickapic_lambda_sweep/subset_9x3/scores.jsonl

.venv\Scripts\python.exe scripts/summarize_sdxl_lambda_sweep.py `
  --scores outputs/sdxl_pickapic_lambda_sweep/subset_9x3/scores.jsonl `
  --out outputs/sdxl_pickapic_lambda_sweep/subset_9x3/summary.md
```

## Inputs

- Prompt file: `experiments/prompts/pickapic_full.md`
- Prompt subset: first 9 Pick-a-Pic prompts from the prompt file.
- Model: `stabilityai/stable-diffusion-xl-base-1.0`
- Inference steps: `51`
- Guidance: `gamma_1=5.5`, `gamma_2=0.0`
- `T_max`: `1`
- Image size: `1024`
- Lambda-step sweep: `0`, `3`, `7`, `15`, `25`, `35`, `50`

## Outputs

- Manifest: `outputs/sdxl_pickapic_lambda_sweep/subset_9x3/manifest.jsonl`
- Images: `outputs/sdxl_pickapic_lambda_sweep/subset_9x3/images/`
- Scores: `outputs/sdxl_pickapic_lambda_sweep/subset_9x3/scores.jsonl`
- Summary: `outputs/sdxl_pickapic_lambda_sweep/subset_9x3/summary.md`
- Plot: `outputs/sdxl_pickapic_lambda_sweep/subset_9x3/lambda_winrate.svg`

## Observations

- Generation wrote 189 manifest rows.
- Image directory contains 216 PNGs: 27 standard images and 189 Z-Sampling images.
- HPSv2 scoring wrote 189 score rows.
- Generation runtime was about 42.4 minutes based on the command wall time.
- HPSv2 scoring runtime was about 2.4 minutes based on the command wall time.
- Warnings: unauthenticated Hugging Face Hub requests; SDXL pipeline ignored `safety_checker` and `requires_safety_checker`; DDIM inverse scheduler ignored unexpected config attributes; upstream submodule commit unavailable.

## Results

| Lambda-step | Pairs | Z wins | Standard wins | Ties | Z win rate | Mean delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 27 | 0 | 0 | 27 | 0.0% | 0.0000 |
| 3 | 27 | 19 | 8 | 0 | 70.4% | 0.0051 |
| 7 | 27 | 19 | 7 | 1 | 70.4% | 0.0105 |
| 15 | 27 | 22 | 5 | 0 | 81.5% | 0.0121 |
| 25 | 27 | 24 | 3 | 0 | 88.9% | 0.0136 |
| 35 | 27 | 23 | 3 | 1 | 85.2% | 0.0156 |
| 50 | 27 | 23 | 4 | 0 | 85.2% | 0.0164 |

## Interpretation

Within this small 9-prompt x 3-seed subset, HPSv2 win rate improves strongly once Z-Sampling is enabled. The best win rate is at `lambda_step=25` (88.9%), while the largest mean HPS delta is at `lambda_step=50` (0.0164). This is an empirical result from the local HPSv2 scoring run, not a paper claim.

## Follow-up

- Inspect image grids for qualitative failures, especially `lambda_step=25`, `35`, and `50`.
- Consider a larger prompt subset if the report needs a more stable ablation curve.
