# SDXL Pick-a-Pic Lambda-Step Sweep Extension

## Metadata

- **Date:** 2026-06-16
- **Owner:** Codex run for requested SDXL Pick-a-Pic lambda-step extension
- **Goal:** Extend the existing 9-prompt SDXL Pick-a-Pic lambda-step sweep by generating and scoring 18 additional prompts, skipping the already generated first 9 prompts.
- **Source code commit:** `7c5d058` plus uncommitted local changes for `--start` support and plot options.
- **Upstream commit:** Unavailable; `external/zigzag-diffusion-sampling/.git` is not initialized in this workspace.
- **Environment:** `.venv`, Python 3.10.20, CUDA available.
- **Hardware:** NVIDIA GeForce RTX 4090.
- **Seed(s):** `42`, `123`, `2026`.

## Command(s)

```powershell
.venv\Scripts\python.exe scripts/run_sdxl_pickapic_lambda_sweep.py `
  --start 9 `
  --limit 18 `
  --out outputs/sdxl_pickapic_lambda_sweep/extra_18x3_from009 `
  --force

.venv\Scripts\python.exe scripts/score_hpsv2_pairs.py `
  --manifest outputs/sdxl_pickapic_lambda_sweep/extra_18x3_from009/manifest.jsonl `
  --out outputs/sdxl_pickapic_lambda_sweep/extra_18x3_from009/scores.jsonl

Get-Content `
  outputs/sdxl_pickapic_lambda_sweep/subset_9x3/scores.jsonl, `
  outputs/sdxl_pickapic_lambda_sweep/extra_18x3_from009/scores.jsonl |
  Set-Content outputs/sdxl_pickapic_lambda_sweep/combined_27x3/scores.jsonl

.venv\Scripts\python.exe scripts/summarize_sdxl_lambda_sweep.py `
  --scores outputs/sdxl_pickapic_lambda_sweep/combined_27x3/scores.jsonl `
  --out outputs/sdxl_pickapic_lambda_sweep/combined_27x3/summary.md `
  --plot-out outputs/sdxl_pickapic_lambda_sweep/combined_27x3/lambda_winrate.svg `
  --plot-y-min 0.5 `
  --plot-neutral-zero
```

The same combined plot was also written to the previously referenced path:
`outputs/sdxl_pickapic_lambda_sweep/subset_9x3/lambda_winrate.svg`.

## Inputs

- Existing scored run: `outputs/sdxl_pickapic_lambda_sweep/subset_9x3/scores.jsonl`
- New prompt range: `pickapic_009_official` through `pickapic_026_official`
- Model: `stabilityai/stable-diffusion-xl-base-1.0`
- Inference steps: `51`
- Guidance: `gamma_1=5.5`, `gamma_2=0.0`
- `T_max`: `1`
- Image size: `1024`
- Lambda-step sweep: `0`, `3`, `7`, `15`, `25`, `35`, `50`

## Outputs

- Extra manifest: `outputs/sdxl_pickapic_lambda_sweep/extra_18x3_from009/manifest.jsonl`
- Extra images: `outputs/sdxl_pickapic_lambda_sweep/extra_18x3_from009/images/`
- Extra scores: `outputs/sdxl_pickapic_lambda_sweep/extra_18x3_from009/scores.jsonl`
- Combined scores: `outputs/sdxl_pickapic_lambda_sweep/combined_27x3/scores.jsonl`
- Combined summary: `outputs/sdxl_pickapic_lambda_sweep/combined_27x3/summary.md`
- Combined plot: `outputs/sdxl_pickapic_lambda_sweep/combined_27x3/lambda_winrate.svg`
- Updated original plot path: `outputs/sdxl_pickapic_lambda_sweep/subset_9x3/lambda_winrate.svg`

## Observations

- Extra generation wrote 378 manifest rows for 18 prompts.
- Extra image directory contains 432 PNGs: 54 standard images and 378 Z-Sampling images.
- Extra HPSv2 scoring wrote 378 score rows.
- Combined score file contains 567 rows: 27 prompts x 3 seeds x 7 lambda-step settings.
- Each lambda-step setting has 81 scored pairs.

## Results

| Lambda-step | Pairs | Z wins | Standard wins | Ties | Z win rate | Mean delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 81 | 0 | 0 | 81 | 0.0% | 0.0000 |
| 3 | 81 | 53 | 25 | 3 | 65.4% | 0.0054 |
| 7 | 81 | 62 | 17 | 2 | 76.5% | 0.0102 |
| 15 | 81 | 65 | 16 | 0 | 80.2% | 0.0139 |
| 25 | 81 | 71 | 10 | 0 | 87.7% | 0.0168 |
| 35 | 81 | 72 | 8 | 1 | 88.9% | 0.0188 |
| 50 | 81 | 70 | 11 | 0 | 86.4% | 0.0191 |

## Interpretation

With the 27-prompt combined subset, the strongest HPSv2 win rate is at `lambda_step=35` (88.9%), while the largest mean HPS delta is at `lambda_step=50` (0.0191). The SVG plot uses a display-only neutral 50% baseline for `lambda_step=0`; the raw table keeps the exact HPS tie result.

## Follow-up

- Use `combined_27x3/summary.md` and `combined_27x3/lambda_winrate.svg` for report figures unless a larger prompt subset is generated.
