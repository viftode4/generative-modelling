# Vlad — Experiment #8: SDXL × Pick-a-Pic Efficiency Comparison

Owner: Vlad. Experiment #8 from the plan diagram (Z-Sampling lane).

Reproduces the paper's **time-matched efficiency study** (Figure 10, p.10): HPS v2 vs. average
time/image for Standard sampling vs. compute-matched Z-Sampling on SDXL × Pick-a-Pic.

## Sources

- Paper PDF: `docs/papers/zigzag-diffusion-sampling.pdf` (efficiency: §4.2 / Figure 10, p.10)
- Upstream code commit: `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4` (`external/zigzag-diffusion-sampling/infer.py`)
- Fidelity audit: `experiments/results-index/exp8_fidelity.md`

## The time-matched protocol

Noise-prediction ops: Standard = `T_s`; Z-Sampling = `T_z + 2·λ`. To equalize compute, the paper
sets **`T_z = T_s / 2`** and **`λ = T_z / 2 = T_s / 4`**, so Z ops = `T_s` = Standard ops.

For each budget `T_s` in the sweep we generate, per (prompt, seed), a Standard image at `T_s` steps
and a Z-Sampling image at `T_z` steps with `λ = T_s//4`, both from the **same initial latent**.

## Fixed settings (paper-faithful)

| Field | Value | Source |
| --- | --- | --- |
| Model | `stabilityai/stable-diffusion-xl-base-1.0` | paper §4.1 / `infer.py:45` |
| Dataset | Pick-a-Pic first 100 prompts (`experiments/prompts/pickapic_full.md`) | Appendix A.1 (provenance caveat below) |
| Seeds | `42` (add `123 2026` if compute allows) | group convention |
| Image size | `1024` | `infer.py:16` |
| `gamma_1` | `5.5` | paper §4.1 / `infer.py:12` |
| `gamma_2` | `0.0` | paper §4.1 / `infer.py:13` |
| `t_max` | `1` | `infer.py:17` |
| Scheduler | DDIM + DDIMInverse | `infer.py:48-52`; `models.py:load_sdxl` |
| Budgets `T_s` | `20, 30, 40, 50` (Z: `T_z=T_s//2`, `λ=T_s//4`) | Figure 10 sweep |
| Metrics | HPS v2 + runtime/image | paper §4.2 / Figure 10 |

Provenance caveat: the Pick-a-Pic prompts are the author's `PickaPic-selected-prompts` substitute
(original v1/v2 unavailable); same dataset + 100 count match the paper, exact prompts not guaranteed.

## Commands

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q

# 0) Calibration: measure real per-image time before committing the full sweep.
.venv/bin/python scripts/run_vlad_exp8_efficiency.py \
  --ts-list 50 --seeds 42 --limit 1 \
  --out outputs/vlad_exp8_sdxl_pickapic --device mps

# 1) Full sweep (seed count chosen from calibration).
.venv/bin/python scripts/run_vlad_exp8_efficiency.py \
  --ts-list 20 30 40 50 --seeds 42 \
  --out outputs/vlad_exp8_sdxl_pickapic --device mps

# 2) HPS v2 scoring (unchanged shared scorer).
.venv/bin/python scripts/score_hpsv2_pairs.py \
  --manifest outputs/vlad_exp8_sdxl_pickapic/manifest.jsonl \
  --out outputs/vlad_exp8_sdxl_pickapic/scores.jsonl

# 3) Efficiency summary (Figure-10 table: HPS vs time per budget).
.venv/bin/python scripts/summarize_vlad_efficiency.py \
  --manifest outputs/vlad_exp8_sdxl_pickapic/manifest.jsonl \
  --scores outputs/vlad_exp8_sdxl_pickapic/scores.jsonl \
  --out outputs/vlad_exp8_sdxl_pickapic/efficiency_summary.md
```

## Expected outputs

- `manifest.jsonl` — one row per (prompt, seed, budget): both image paths, both runtimes, `T_s`/`T_z`/`λ`, hyperparams, commits. Resumable (rows are appended; recorded cells are skipped).
- `scores.jsonl` — HPS v2 standard/Z scores + delta (tag = `ts{T_s}` carries the budget).
- `efficiency_summary.md` — per-budget mean time + mean HPS for both arms, Z win rate, and the "Z at lower compute beats best standard" check.

Generated PNGs stay ignored under `outputs/`; small evidence files may be committed for the report.
