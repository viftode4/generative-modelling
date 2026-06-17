# Daniel's Experiment Plan — SDXL × DrawBench × Standard Sampling

## Assignment

- **Person:** Daniel
- **Model:** SDXL (`stabilityai/stable-diffusion-xl-base-1.0`)
- **Dataset:** DrawBench (200 prompts total, subset TBD with Radu + Gouri)
- **Method owned:** Standard Sampling (Experiment 5)
- **Also responsible for:** Helping set up the shared SDXL standard baseline

## Goal

Generate standard-sampling images for SDXL on a DrawBench subset, producing a manifest that can later be paired with Z-Sampling outputs and scored.

## Design decisions

### Separation of concerns (differs from Eugen/Vlad scripts)

Their scripts (`run_eugen_pickapic.py`, `run_vlad_geneval.py`) generate both standard AND zigzag in one loop because they own Z-Sampling for their cells. 

For this cell, **generation and evaluation are separate steps:**

1. **Generate standard** — save images + manifest with seeds
2. **Generate Z-Sampling** (later, by whoever owns it) — uses same seeds/latents
3. **Evaluate** — score pairs from the manifest

This works because the comparison is fair as long as both arms use the same initial latent (= same seed). The seed is recorded in the manifest.

### Prompt subset

- DrawBench has 200 prompts across 11 categories.
- Run all 200 prompts.

### Hyperparameters (from paper's implementation)

| Parameter | Value | Source |
|-----------|-------|--------|
| model_id | `stabilityai/stable-diffusion-xl-base-1.0` | Upstream code |
| num_inference_steps | 50 | Upstream code |
| guidance_scale (gamma_1) | 5.5 | Upstream code |
| height | 1024 | Upstream code |
| width | 1024 | Upstream code |
| scheduler | DDIM | Upstream code |
| seeds | 42, 123, 2026 | Group convention |

### Compute estimate

- Device: Apple M4 MPS, float16
- ~5s/step × 50 steps = ~4.5 min/image
- 200 prompts × 3 seeds = 600 images → **~45 hours**

## Pipeline steps

### Step 1: Prepare DrawBench prompts

- Load all 200 prompts from `data/prompts/draw-bench.csv`
- Save as `experiments/prompts/drawbench_daniel.md` (fenced jsonl, same format as Eugen/Vlad)

### Step 2: Generation script

Create `scripts/run_daniel_drawbench.py`:

- Loads SDXL via `load_pipeline_standard` (with MPS support)
- Iterates prompts × seeds
- Saves images to `outputs/daniel_drawbench/images/`
- Writes `manifest.jsonl` per row (prompt_id, seed, image path, hyperparams, commit hash, runtime)
- Supports `--dry-run`, `--limit`, `--device`

### Step 3: Run generation

```bash
.venv/bin/python scripts/run_daniel_drawbench.py \
  --prompts experiments/prompts/drawbench_daniel.md \
  --seeds 42 123 2026 \
  --out outputs/daniel_drawbench \
  --device mps
```

### Step 4: Evaluation (after Z-Sampling is available)

- Score with shared metric (HPSv2 or CLIP, group decision pending)
- Compare standard vs Z using `scripts/score_hpsv2_pairs.py` or equivalent
- Report winning rate

## Open questions (to ask the group)

- [ ] Who generates Z-Sampling for SDXL × DrawBench? (Daniel later, or someone else?)
- [ ] Which metric: HPSv2 or CLIP?

## Files to create

- `experiments/prompts/drawbench_daniel.md` — prompt subset
- `scripts/run_daniel_drawbench.py` — generation script
- `knowledgebase/logs/YYYY-MM-DD_daniel-sdxl-drawbench.md` — experiment log (after run)
