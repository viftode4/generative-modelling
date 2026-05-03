# TA Meeting Brief

Walk into the TA meeting with two scoping options. Let the TA pick. This makes us look like we have a plan AND we're listening to feedback (both rubric-positive for "Motivation and Work Plan", 30%).

## What to present (in order)

1. **One sentence**: "We're studying Z-Sampling, a sampling-time modification of diffusion models. We want your input on scope before we lock our plan."
2. **What we know**: paper claims, our preliminary task split, our compute reality.
3. **The two options below.**
4. **Our questions** (from `planning/ta-questions.md`).

## Option A — Full paper reproduction (broad, ambitious)

| What | How much |
|------|---------|
| Models | 4 (SD2.1, SDXL, Hunyuan-DiT, DreamShaper-xl-v2-turbo) |
| Datasets | 3 (Pick-a-Pic, DrawBench, GenEval) |
| Methods | Z-Sampling + standard + 5 baselines (Semantic-aware CFG, Diffusion-DPO, AYS, SEG, CFG++) |
| Metrics | 4 (HPSv2, PickScore, ImageReward, AES) |
| Ablations | 3 (guidance scale, zigzag steps, efficiency) |
| Approx. experiment cells | ~288 |
| Approx. images | 5000+ |

**Pros:** Closest to the paper's actual experiments. Highest ceiling for Content (40%) rubric.
**Cons:** Probably infeasible in 8 weeks with 7 students. Each baseline is a separate paper — implementing 5 of them is a major lift. Compute budget likely exceeded by 5-10×.

## Option B — Focused reproduction (achievable, deep)

| What | How much |
|------|---------|
| Models | 4 (one per person/pair) |
| Datasets | 1-2 (TA picks: Pick-a-Pic and/or GenEval) |
| Methods | Z-Sampling vs. standard sampling only (no other baselines) |
| Metrics | 2-3 (HPSv2 + CLIP score, optional PickScore) |
| Ablations | 2 (guidance scale, zigzag steps) on ONE model only |
| Approx. experiment cells | ~24-36 |
| Approx. images | 500-800 |

**Pros:** Achievable in 8 weeks. Each person gets a deep, distinct story for their report. Time left for analysis (rewards Writing 30%).
**Cons:** Less than the full paper. We don't validate Z-Sampling against other baselines.

## Our recommendation to the TA

**Option B** unless the TA's grading expectation is "reproduce the full paper exactly." In our reading of the rubric, depth of analysis (Writing 30%, Content 40%) outweighs breadth of experiments. Option B preserves analysis time; Option A risks shallow results dumped into reports.

## Direct questions to put to the TA

1. Which option is closer to your expectation?
2. Are baselines (Semantic-aware CFG, DPO, AYS, SEG, CFG++) required, or optional?
3. Is comparing across 4 models (breadth) better, or 1-2 models with deeper ablation (depth)?
4. What's the minimum acceptable result: matching paper numbers, directional reproduction, or critical appraisal of why our results differ?
5. Is the upstream codebase OK as a reference while we reimplement from scratch?

## What we'll commit to regardless of option

- All 7 of us implement a clean Z-Sampling from scratch (the TA explicitly said don't copy upstream)
- Reproducible logging: every image tagged with prompt, seed, hyperparameters, GPU, commit hash → `experiments/runs.jsonl`
- 7 distinct individual reports with non-overlapping figures (see `planning/role-assignments.md`)
- Mid-term presentation: 7 minutes, each person presents their own piece

## Backup plan if compute fails

If GPUs are unavailable: we run Option B on Google Colab Pro / RunPod / Kaggle. Document the constraint as a finding in reports (rubric explicitly rewards "strong troubleshooting").
