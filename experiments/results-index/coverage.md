# Experiment Coverage — Verified Against Committed Repo Evidence

Date: 2026-06-09. This file is the audited source of truth for what results actually
exist in the repo, mapped against the 8-experiment plan. "Committed" = a scores/summary
file or result table is checked in, not just a config, notebook, or plan sentence.

## Lanes with real, committed results

| Owner | Model × Dataset | Methods | Metrics with numbers | Pairs | Evidence |
|---|---|---|---|---|---|
| **Eugen** | SD2.1 × Pick-a-Pic | Standard vs Z | HPSv2 (Z 71.0% win, +0.0110, CI [+0.0087,+0.0132]) | 300 (100×3) | `report/tables/eugen_results.md`, `outputs/eugen_sd2.1_pickapic/` |
| **Daniel** | SDXL × DrawBench | Standard vs Z | HPSv2 (Z 540/600), AES (Z 420/600), PickScore (~tie). **IR = nan (failed)** | 600 (200×3) | `notebooks/daniel-sdxl-drawbench/*.csv,*.jsonl` |
| **Vlad** | DreamShaper-xl-v2-turbo × GenEval | Standard vs Z | GenEval detector (31.5%→50.0%), HPSv2 (Z 53/54, +0.0358), runtime + 9-step compute-matched control | 54 (18×3) | `report/tables/vlad_*.md`, `outputs/vlad_geneval/` |

All three lanes are **Standard vs Z-Sampling only**. None includes a Resampling arm.

## The 8-experiment plan vs. what's saved

| # | Plan experiment | Status | Notes |
|---|---|---|---|
| 1 | **Basic Sampling Comparison (MAIN)** — Std/Resampling/Z × {SD2.1,SDXL,DreamShaper,Hunyuan} × {DrawBench,Pick-a-Pic} × {HPS,PickScore,IR,AES} | ⚠️ Partial | Std-vs-Z on **3 of 4 models** (Eugen, Daniel, Vlad). **Missing: Resampling arm, Hunyuan-DiT, working ImageReward** |
| 2 | Additional Sampling — SDXL × GenEval, per-category | ⚠️ Mismatch | Only DreamShaper×GenEval per-category exists (Vlad), **not SDXL** |
| 3 | AYS + Z-Sampling (DreamShaper) | ❌ None | No AYS code or results |
| 4 | Semantic-aware CFG (SD2.1) | ❌ None | Nothing |
| 5 | Diffusion-DPO + Z (SDXL) | ❌ None | Nothing. Daniel's config is *labeled* "Exp 5" but is actually the standard SDXL baseline feeding #1 |
| 6 | Guidance-gap ablation (γ₂) | ❌ None | Nothing |
| 7 | Zigzag-step ablation (λ) | ❌ None | Nothing |
| 8 | Efficiency comparison (runtime/img) | ⚠️ Mismatch | Vlad has runtime + compute-matched control on **DreamShaper, not SDXL** |

## Biggest gaps (in priority order)

1. **Resampling has zero results anywhere** — a required method (TA: "go over all of them"). Only `notebooks/04_resampling_example.ipynb` + `04_exp2_colab_runner.ipynb` as code; no scored output. Owners: Gouri, Lenka.
2. **Radu, Bogdan** (Z-Sampling lanes #6/#7, Hunyuan): no committed results.
3. **Baselines/ablations (#3–#7)**: none. Candidates to formally descope as future work per TA's "pick a reasonable subset" guidance.

## Defensible framing

- **Done:** Experiment #1 (main comparison), Standard vs Z, on 3 models / 3 datasets / multiple metrics.
- **In progress:** Resampling; broader GenEval/efficiency coverage.
- **Scoped out → future work:** AYS, Semantic-CFG, Diffusion-DPO, guidance-gap + zigzag-step ablations.
