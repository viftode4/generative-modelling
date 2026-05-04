# Scope

We reproduce the paper's main experiment: Z-Sampling vs. standard sampling. Each person runs it on their own (model, dataset) pair. Per-person extras are optional on top.

The TA said we can pick any scope. This is what we picked.

## Must

Required for every report.

- A working Z-Sampling implementation (one shared codebase)
- Each person runs their assigned model + dataset
- Z-Sampling vs. standard sampling on every cell
- 3-page double-column report covering the sections required by the course PDF
- Run log per image: prompt, seed, hyperparameters, GPU, commit hash
- At least one standard-vs-Z image pair per cell

## Should

Group-level targets.

- One shared metric on every image (HPSv2 or CLIP, to be decided)
- Multiple seeds per (prompt, method) pair (target: 3)
- Critical appraisal section in every report
- Per-person troubleshooting log
- Standard-vs-Z winning rate per cell

## Could

Optional per-person extras. Pick at most one. Skipping is fine.

- Ablation sweep on one hyperparameter (guidance scales or zigzag step settings)
- Cross-cell aggregation table
- Failure-mode gallery
- Comparison vs. one baseline
- Theory / derivation section
- Per-prompt-category breakdown
- Second metric
- Port Z-Sampling to a model where the upstream code does not ship it

## Won't

Out of scope.

- All 5 baselines from the paper
- All 4 models x all 3 datasets fully crossed
- Retraining or fine-tuning any diffusion model
- Implementing baseline papers from scratch
