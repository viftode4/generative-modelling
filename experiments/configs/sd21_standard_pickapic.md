# Eugen's SD2.1 + Pick-a-Pic Experiment Config

## Identity

- **Owner:** Eugen
- **Cell:** SD2.1 × Pick-a-Pic (from `planning/role-assignments.md`)
- **Experiment:** Standard Sampling vs. Z-Sampling (Experiment 1)

## Model

- **HuggingFace ID:** `sd2-community/stable-diffusion-2-1`
- **Architecture:** UNet-based Stable Diffusion v2.1 (single CLIP ViT-H/14 text encoder)
- **Pipeline class:** `StableDiffusionPipeline`

## Dataset

- **Source:** Pick-a-Pic v2 (`yuvalkirstain/pickapic_v2` validation split)
- **Prompt file:** `experiments/prompts/pickapic_full.md`
- **Prompt count:** 100 (Official raw dataset subset as used in the paper)

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `gamma_1` | 7.5 | SD2.1 recommended CFG scale |
| `gamma_2` | 0.0 | Paper default for inversion guidance |
| `infer_step` | 50 | Paper default; SD2.1 is non-distilled |
| `lambda_step` | 49 | Paper default (infer_step - 1) |
| `T_max` | 1 | Paper default rounds per zigzag step |
| `image_size` | 768 | SD2.1 native resolution |
| Seeds | 42, 123, 2026 | Matching Vlad's seed set |

## Expected Output

- **Pairs:** 100 prompts × 3 seeds = 300 pairs
- **Total images:** 600 PNGs (300 standard + 300 Z-Sampling)
- **Output directory:** `outputs/eugen_pickapic/full_100x3/`
- **Manifest:** `outputs/eugen_pickapic/full_100x3/manifest.jsonl`

## Evaluation

- **Metric:** HPSv2.1 (pair scoring)
- **Scripts:** `scripts/score_hpsv2_pairs.py`, `scripts/summarize_vlad_results.py`
- **Expected runtime ratio:** Z-Sampling should be ~2-3× standard (50 steps + zigzag overhead)

## Commands

### Full Evaluation Run (100 Prompts)
```bash
# Generation
.venv/bin/python scripts/run_eugen_pickapic.py \
  --prompts experiments/prompts/pickapic_full.md \
  --seeds 42 123 2026 \
  --steps 50 \
  --cfg 7.5 \
  --inv-cfg 0 \
  --lambda-step 49 \
  --t-max 1 \
  --size 768 \
  --out outputs/eugen_pickapic/full_100x3

# Scoring
.venv/bin/python scripts/score_hpsv2_pairs.py \
  --manifest outputs/eugen_pickapic/full_100x3/manifest.jsonl \
  --out outputs/eugen_pickapic/full_100x3/scores.jsonl

# Summary
.venv/bin/python scripts/summarize_vlad_results.py \
  --scores outputs/eugen_pickapic/full_100x3/scores.jsonl \
  --out outputs/eugen_pickapic/full_100x3/summary.md
```

### Smoke Test / Subset Run (e.g. 20 Prompts)
```bash
# Run only the first 20 prompts as a validation / smoke test
.venv/bin/python scripts/run_eugen_pickapic.py \
  --prompts experiments/prompts/pickapic_full.md \
  --seeds 42 \
  --steps 50 \
  --cfg 7.5 \
  --inv-cfg 0 \
  --lambda-step 49 \
  --t-max 1 \
  --size 768 \
  --limit 20 \
  --out outputs/eugen_pickapic/smoke_20x1
```
