# DreamShaper Turbo GenEval — Mac-only Compute-Matched Baseline

Purpose: strengthen Vlad's DreamShaper-xl-v2-turbo × GenEval lane with a compute-control baseline. The official GenEval detector now runs on Apple Silicon via CPU (`mmdet==2.28.2`, `mmcv-full==1.7.2`), so this is no longer a detector fallback. It tests a different counterfactual: **does 4-step Z-Sampling beat a standard DDIM baseline given roughly the same wall-clock compute?**

## Source run

- Existing run: `outputs/vlad_geneval/balanced_18x3/`
- Existing manifest: `outputs/vlad_geneval/balanced_18x3/manifest.jsonl`
- Existing HPS scores: `outputs/vlad_geneval/balanced_18x3/scores.jsonl`
- Existing Z-Sampling settings: DreamShaper-xl-v2-turbo, `T=4`, `gamma_1=3.5`, `gamma_2=0`, `lambda_step=3`, `t_max=1`, image size 768, seeds `42 123 2026`.

## Compute-matched baseline

The recorded MPS runtime ratio is 2.24x: 4-step standard averages 6.16 s/image and 4-step Z-Sampling averages 13.80 s/image. A compute-matched standard baseline is therefore approximately `4 * 2.24 = 8.96` DDIM steps. Use **9 standard DDIM steps** as the main comparison.

## Commands

```bash
.venv/bin/python scripts/run_vlad_compute_matched_standard.py \
  --source-manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
  --standard-steps 9 \
  --out outputs/vlad_geneval/compute_matched_standard9

.venv/bin/python scripts/score_hpsv2_pairs.py \
  --manifest outputs/vlad_geneval/compute_matched_standard9/manifest.jsonl \
  --out outputs/vlad_geneval/compute_matched_standard9/scores.jsonl

.venv/bin/python scripts/summarize_vlad_results.py \
  --scores outputs/vlad_geneval/compute_matched_standard9/scores.jsonl \
  --out outputs/vlad_geneval/compute_matched_standard9/summary.md
```

Optional smoke before the full run:

```bash
.venv/bin/python scripts/run_vlad_compute_matched_standard.py \
  --source-manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
  --standard-steps 9 \
  --limit 1 \
  --out outputs/vlad_geneval/compute_matched_standard9_smoke
```

## Completed run

Completed on 2026-06-04:

- Output directory: `outputs/vlad_geneval/compute_matched_standard9/`
- Rows: 54 paired comparisons.
- HPSv2.1 result: Z-Sampling wins 42/54 pairs against 9-step standard; 9-step
  standard wins 12/54; mean delta (Z - standard9) is +0.0245.
- Mean scores: 9-step standard 0.2280, existing 4-step Z-Sampling 0.2526.
- Runtime: 9-step standard 18.14 s/image (979.4 s total), existing 4-step
  Z-Sampling 13.80 s/image (745.5 s total). The 9-step baseline is therefore
  about 1.31x slower than Z-Sampling on this MPS run.
- Reference table: `report/tables/vlad_compute_matched.md`.

## Interpretation

Z-Sampling still wins against the stronger 9-step standard baseline under
HPSv2.1, so the improvement is not only from spending more network evaluations.
However, the margin shrinks from +0.0358 against 4-step standard to +0.0245
against 9-step standard, so part of the original margin is explained by extra
compute. This is still an HPSv2.1 comparison, not an official GenEval detector
score for the 9-step standard images.
