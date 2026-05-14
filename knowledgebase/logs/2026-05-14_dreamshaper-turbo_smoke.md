# Smoke test: DreamShaper-xl-v2-turbo + DDIM on MPS

## Metadata

- **Date:** 2026-05-14
- **Owner:** Vlad
- **Goal:** Verify the end-to-end pipeline (model load → DDIM + DDIMInverseScheduler → generate) runs on the local environment before any Z-Sampling code lands. De-risks the W1 per-person gate for Vlad's cell (DreamShaper-xl-v2-turbo + GenEval) and confirms the inversion scheduler attaches cleanly to a turbo SDXL checkpoint.
- **Source code commit:** `854e24d`
- **Upstream commit:** `513eb51`
- **Environment:** Python 3.10.20, torch 2.12.0, diffusers 0.37.1, transformers 5.8.1 (relaxed from upstream pins; see research-log entry 2026-05-14).
- **Hardware:** Apple M4 Max, MPS backend, fp16.
- **Seed(s):** 42

## Command(s)

```bash
.venv/bin/python scripts/smoke_dreamshaper.py \
  --prompt "a small yellow dog sitting on grass, photorealistic" \
  --steps 6 --cfg 2.0 --size 768 --seed 42 \
  --out outputs/smoke/dreamshaper_smoke.png
```

## Inputs

- Model: `Lykon/dreamshaper-xl-v2-turbo` (Hugging Face)
- Scheduler: `DDIMScheduler` (denoising) + `DDIMInverseScheduler` (carried on the handle for later Z-Sampling use)
- Prompt: `"a small yellow dog sitting on grass, photorealistic"`
- Steps: 6 (turbo-appropriate; upstream uses 50 for SDXL base)
- CFG: 2.0 (turbo-appropriate; upstream uses 5.5 for SDXL base)
- Image size: 768×768

## Outputs

- `outputs/smoke/dreamshaper_smoke.png` (904 KB, gitignored).

## Observations

- Model load (first run, fetches ~6 GB to HF cache): **217.1 s**.
- Generation: **11.4 s** for 6 steps → ~1.3 s/step on MPS.
- Image is a recognizable small yellow dog on grass — quality is solid for 6 turbo steps.
- Two non-fatal warnings:
  - `DDIMInverseScheduler` ignored `sample_max_value` and `skip_prk_steps` (DreamShaper's scheduler config carries PNDM-style fields; harmless).
  - `upcast_vae` deprecation warning from diffusers — will need a one-line update if/when we re-pin to a newer diffusers version that removes the helper.

## Interpretation

- The full chain (DreamShaper turbo SDXL checkpoint + DDIM + DDIMInverseScheduler + MPS + fp16) composes without errors. This means Z-Sampling's inversion step has a viable scheduler on this cell — the highest-risk question for Vlad's cell is now answered yes.
- Performance budget is healthy: a per-prompt run (1 standard sample + 1 Z-Sampling pass, each ~6 turbo steps + some inversion overhead) should fit comfortably in seconds on this laptop. Running multi-seed, multi-prompt batches locally is realistic.
- The smoke prompt is qualitatively reasonable; this is not yet evidence of prompt-image alignment improvements from Z-Sampling — Z-Sampling hasn't been implemented yet.

## Follow-up

- When Z-Sampling lands in `src/zigzag_project/`, re-run on the same prompt + seed to confirm the inversion path doesn't blow up.
- For the eventual parity check vs. upstream, plan to also run on a CUDA box with strict upstream pins on at least one shared prompt + seed, so Vlad's relaxed-MPS results can be cross-referenced.
- When the group picks a metric (HPSv2 vs CLIP), wire it into a parameterized smoke script so each cell can produce a metric value on a known prompt before the real experiment matrix starts.
