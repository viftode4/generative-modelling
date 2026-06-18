# Bogdan Hunyuan-DiT + Pick-a-Pic Experiment Config

## Identity

- **Owner:** Bogdan
- **Cell:** Hunyuan-DiT x Pick-a-Pic
- **Experiment:** Standard Sampling vs. Z-Sampling
- **Scope note:** This replaces the earlier `Hunyuan-DiT x GenEval` planning lane for Bogdan. The paper reports Hunyuan-DiT on Pick-a-Pic / DrawBench, while GenEval results are reported for SDXL, so Pick-a-Pic is the closer paper-facing choice.

## Model

- **Hugging Face ID:** `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers`
- **Architecture:** Hunyuan-DiT transformer pipeline with dual text encoders.
- **Pipeline class:** `HunyuanDiTPipeline`
- **Loader:** `zigzag_project.models.load_hunyuan_dit`

## Dataset

- **Source:** Pick-a-Pic prompt subset used by this repo.
- **Prompt file:** `experiments/prompts/pickapic_full.md`
- **Prompt count:** 100 prompts.

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `gamma_1` | 5.0 | Diffusers Hunyuan-DiT default guidance scale. |
| `gamma_2` | 0.0 | Paper/default inversion guidance. |
| `infer_step` | 50 | Paper/default non-turbo setting. |
| `lambda_step` | 49 | Full-path zigzag except final step (`infer_step - 1`). |
| `T_max` | 1 | Paper/default rounds per refined step. |
| `image_size` | 1024 | Native supported square Hunyuan-DiT resolution. |
| Seeds | 42, 123, 2026 | Match the repo's existing paired-run convention. |

## Expected Output

- **Pairs:** 100 prompts x 3 seeds = 300 pairs.
- **Total images:** 600 PNGs.
- **Output directory:** `outputs/bogdan_hunyuan_pickapic/full_100x3/`
- **Manifest:** `outputs/bogdan_hunyuan_pickapic/full_100x3/manifest.jsonl`

## Evaluation

- **Metric:** HPSv2.1 standard-vs-Z pair scoring, matching the existing Pick-a-Pic/HPS workflow.
- **Scripts:** `scripts/score_hpsv2_pairs.py`, `scripts/summarize_vlad_results.py`.
- **Caveat:** `z_sample_hunyuan_dit()` is newly ported and must pass a smoke run before full results are trusted.

## Commands

### Dry Run

```bash
.venv/bin/python scripts/run_bogdan_hunyuan_pickapic.py \
  --prompts experiments/prompts/pickapic_full.md \
  --seeds 42 \
  --limit 2 \
  --out outputs/dryrun/bogdan_hunyuan_pickapic \
  --dry-run
```

### Smoke Run

```bash
.venv/bin/python scripts/run_bogdan_hunyuan_pickapic.py \
  --prompts experiments/prompts/pickapic_full.md \
  --seeds 42 \
  --steps 2 \
  --cfg 5.0 \
  --inv-cfg 0 \
  --lambda-step 1 \
  --t-max 1 \
  --size 1024 \
  --limit 1 \
  --out outputs/bogdan_hunyuan_pickapic/smoke_1x1
```

### Full Evaluation Run

```bash
.venv/bin/python scripts/run_bogdan_hunyuan_pickapic.py \
  --prompts experiments/prompts/pickapic_full.md \
  --seeds 42 123 2026 \
  --steps 50 \
  --cfg 5.0 \
  --inv-cfg 0 \
  --lambda-step 49 \
  --t-max 1 \
  --size 1024 \
  --out outputs/bogdan_hunyuan_pickapic/full_100x3

.venv/bin/python scripts/score_hpsv2_pairs.py \
  --manifest outputs/bogdan_hunyuan_pickapic/full_100x3/manifest.jsonl \
  --out outputs/bogdan_hunyuan_pickapic/full_100x3/scores.jsonl

.venv/bin/python scripts/summarize_vlad_results.py \
  --scores outputs/bogdan_hunyuan_pickapic/full_100x3/scores.jsonl \
  --out outputs/bogdan_hunyuan_pickapic/full_100x3/summary.md
```
