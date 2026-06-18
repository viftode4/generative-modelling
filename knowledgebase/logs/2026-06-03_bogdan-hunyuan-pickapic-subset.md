# 2026-06-03 - Bogdan Hunyuan-DiT Pick-a-Pic subset run

## Metadata

- **Owner:** Bogdan / AI assistant
- **Goal:** Compare Hunyuan-DiT standard sampling vs. Z-Sampling on a report-sized Pick-a-Pic subset.
- **Model:** `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers`
- **Prompt file:** `experiments/prompts/pickapic_full.md`
- **Prompt subset:** first 18 prompts
- **Seeds:** `42`, `123`, `2026`
- **Hardware:** NVIDIA GeForce RTX 4090, CUDA-enabled `torch==2.12.0+cu130`
- **Output directory:** `outputs/bogdan_hunyuan_pickapic/subset_18x3/`

## Command

```powershell
.venv\Scripts\python.exe scripts\run_bogdan_hunyuan_pickapic.py `
  --prompts experiments\prompts\pickapic_full.md `
  --seeds 42 123 2026 `
  --steps 50 `
  --cfg 5.0 `
  --inv-cfg 0 `
  --lambda-step 49 `
  --t-max 1 `
  --size 1024 `
  --limit 18 `
  --device cuda `
  --out outputs\bogdan_hunyuan_pickapic\subset_18x3 `
  --force
```

Scoring:

```powershell
.venv\Scripts\python.exe scripts\score_hpsv2_pairs.py `
  --manifest outputs\bogdan_hunyuan_pickapic\subset_18x3\manifest.jsonl `
  --out outputs\bogdan_hunyuan_pickapic\subset_18x3\scores.jsonl

.venv\Scripts\python.exe scripts\summarize_vlad_results.py `
  --scores outputs\bogdan_hunyuan_pickapic\subset_18x3\scores.jsonl `
  --out outputs\bogdan_hunyuan_pickapic\subset_18x3\summary.md
```

## Results

- Pairs: 54
- Standard wins: 31
- Z-Sampling wins: 23
- Ties: 0
- Z-Sampling win rate: 42.6%
- Mean HPSv2.1 delta (`Z - standard`): -0.0042
- Standard mean HPSv2.1: 0.3195
- Z-Sampling mean HPSv2.1: 0.3153
- Min delta: -0.0435
- Max delta: +0.0319

Runtime:

- Standard total: 1169.8 s; mean 21.66 s per image.
- Z-Sampling total: 3438.8 s; mean 63.68 s per image.
- Z-Sampling runtime ratio: 2.94x standard.

## Interpretation

On this 18-prompt, 3-seed Pick-a-Pic subset, HPSv2.1 slightly favors standard
sampling over the current Hunyuan-DiT Z-Sampling port. This is an early
report-sized subset, not the full 100-prompt paper-style Pick-a-Pic run.

## Verification

- `.venv\Scripts\python.exe -m ruff check .` passed.
- `.venv\Scripts\python.exe -m pytest -q` passed: 31 tests, 2 warnings.

## Caveats

- The summary script name/header still says Vlad/GenEval because the repo's
  summarizer is shared and not yet parameterized. The data paths and result
  table are Bogdan/Hunyuan/Pick-a-Pic.
- The upstream submodule is not initialized in this checkout, so manifest rows
  record `upstream_commit: null`.
