<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# experiments

## Purpose

Experiment specifications, prompt sets, and a lightweight index of completed runs. **This directory holds plans and pointers, not raw outputs.** Generated images, batches, and large artifacts live under gitignored `outputs/` / `runs/` / `results/`.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Top-level conventions for running experiments. |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `configs/` | Per-cell or per-experiment markdown configs (model id, scheduler, hyperparameters, dataset slice, seeds). |
| `prompts/` | Prompt sets — both upstream's default prompts and curated subsets of Pick-a-Pic / DrawBench / GenEval. |
| `results-index/` | One small markdown per completed run pointing at outputs, metrics, and the experiment log entry in `knowledgebase/logs/`. |

## For AI Agents

### Working In This Directory

- **Do not commit generated images, model checkpoints, or large CSVs.** They go to `outputs/` (gitignored).
- An "experiment" here = one (model, dataset, sampler, hyperparameter set) combination. Each gets a markdown config under `configs/` and one entry in `results-index/` once it has run.
- Use stable, descriptive filenames: `<cell>_<sampler>_<short-tag>.md` (e.g. `dreamshaper-turbo_zsampling_geneval-singles.md`).
- Prompt sets should be reproducible: include exact prompt text + source dataset + index in source.

### Common Patterns

- Configs are markdown, not YAML/JSON — easier to read and grep in PRs.
- Every config lists: model HF id, scheduler, `gamma_1`, `gamma_2`, `infer_step`, `lambda_step`, `T_max`, prompt set ref, seeds, image size.

## Dependencies

### Internal
- `scripts/` — the runners that consume these configs.
- `knowledgebase/logs/` — long-form post-run write-ups using `knowledgebase/templates/experiment-log-template.md`.

<!-- MANUAL: -->
