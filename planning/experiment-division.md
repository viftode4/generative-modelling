# Experiment Division & Project Plan

Source: group planning doc (TA/group notes 2026-05-14, Drive notes from the 2026-05-04 TA
meeting, and the work-split diagram). This is the group's intended plan, captured verbatim
in structure. For what results actually exist in the repo, see
[`experiments/results-index/coverage.md`](../experiments/results-index/coverage.md).

## Work split — one experiment per person

Each person owns one of the 8 experiment boxes from the plan diagram. In the original notes
the **`+N` is the experiment number** (not "plus a model"); the leading label is that
person's primary sampling-method responsibility.

| Owner | Method responsibility | Exp # | Experiment |
| --- | --- | :--: | --- |
| Eugen | Standard | 1 | Basic Sampling Methods Comparison **(MAIN)** |
| Gouri | Resampling | 2 | Additional Sampling Methods Comparison |
| Lenka | Resampling | 3 | AYS Sampling + Z-Sampling |
| Daniel | Standard | 5 | Diffusion-DPO + Z-Sampling |
| Radu | Z-Sampling | 6 | Guidance Gap Ablation *(assignment marked uncertain — "Radu??")* |
| Bogdan | Z-Sampling | 7 | Zigzag Diffusion Step Ablation |
| Vlad | Z-Sampling | 8 | Efficiency Comparison |

- **Experiment #4 (Semantic-aware CFG)** is unassigned.
- Experiments **1, 5, 6, 8 share the same Standard-Sampling SDXL pipeline as baseline**, so the two Standard-Sampling owners (Eugen, Daniel) are responsible for setting that shared baseline up.
- Implementation interpretation: keep shared baseline code configurable; for valid method comparisons, compare each method against a matched standard baseline using the same model, prompts, seeds, scheduler family, image size, and step count unless an experiment explicitly studies cross-model differences.

## Model × dataset ownership

7 students, 4 models, 3 datasets. Matrix set so no two people share both their model and dataset.

| Person | Model | Dataset |
| --- | --- | --- |
| Eugen | SD2.1 | Pick-a-Pic |
| Radu | SD2.1 | DrawBench |
| Daniel | SDXL | DrawBench |
| Lenka | SDXL | Pick-a-Pic |
| Bogdan | Hunyuan-DiT | GenEval |
| Gouri | DreamShaper-xl-v2-turbo | DrawBench |
| Vlad | DreamShaper-xl-v2-turbo | GenEval |

## Full experiment matrix (paper §4.1)

- **Paper:** https://openreview.net/pdf?id=MKvQH1ekeY · **Code:** https://github.com/xie-lab-ml/zigzag-diffusion-sampling
- **Sampling methods:** Standard, Resampling, Z-Sampling.
- **Models:** SD2.1 (Rombach 2022) — Eugen, Radu · SDXL (Podell 2023) — Daniel, Lenka · Hunyuan-DiT (Li 2024) — Bogdan · DreamShaper-xl-v2-turbo — Gouri, Vlad.
- **Datasets:** Pick-a-Pic (Kirstain 2023) — Lenka, Eugen · DrawBench (Saharia 2022) — Daniel, Gouri, Radu · GenEval (Ghosh 2024) — Bogdan, Vlad.
- **Metrics:** HPS v2 (Wu 2023c), PickScore (Kirstain 2023), ImageReward / IR (Xu 2024a), AES (Schuhmann 2022).
- **Baselines:** Semantic-aware CFG (Shen 2024), Diffusion-DPO (Wallace 2024), AYS-Sampling (Sabour 2024), Smoothed Energy Guidance (Hong 2024), CFG++ (Chung 2024).
- **Ablations:** impact of guidance scale; number of zigzag steps; efficiency comparison.

## TA guidance

From the questions put to the TA and the answers received:

- **How many datasets?** Divide depending on workload and number of people.
- **Best way to divide tasks / be non-redundant / fair assignment?** For Thursday, don't implement everything — justify your choices. Focus on the main contribution across different datasets and models instead of expanding into new things.
- **Can we focus on one sampling method and ignore the other baselines?** No — go over all of them (Standard, Resampling, Z-Sampling are the in-scope set).
- **What to focus on in the final presentation? Implementation/experimentation only?** No need to have all results for the end-term, but at least some — highlight challenges, what was done so far, and the plan to be ready to submit by the due date.
- **How important is the theoretical foundation? Who grades us?** The TA is the grader and is familiar with the paper, so no need to go deep into the math.
- **General:** Implement the inference-time behaviour; do not retrain models. Not expected to implement every model — pick a subset, divide reasonably, make sure the paper's behaviour is represented. Torch/Adam and standard tooling are fine. The level of abstraction the paper uses is the level expected of us.

## TA meeting — 2026-05-04

- It is a group report, but with individual sections.
- A timeline is expected; some tasks have sequential dependencies.
- Thursday gives feedback on the plan and whether the workload is reasonable for the group size.
- Each submits their own document, with the group sections shared across the team plus individual reflections. *(Superseded by the 2026-06-04 clarification → single group report; see `knowledgebase/course/2026-06-04_presentation-report-clarification.md`.)*

## Preliminary plan

1. Everyone gets comfortable with the models (application) and the datasets; make sure all models/datasets are available.
2. Split into sampling-specific subgroups that implement Standard, Resampling, and Z-Sampling in parallel and test each separately to confirm they work.
3. Then split experiment types among the group to get as many done as possible.

### Tasks

- Understand the motivation for the paper.
- Everyone learns to load and access the datasets and models.
- High-level understanding of the paper and of Z-Sampling.
- Explore the existing codebase.
- Split into 3 subgroups to implement the 3 sampling methods and confirm sensible results.
- Once everything works, split experiment types to maximize coverage.
- Create the presentation.

## Resampling implementation note

Drive notes compare two resampling approaches against RePaint. The RePaint-aligned implementation is the **Lenka-style** non-monotonic schedule: true timestep backtracking, forward-diffusion re-noising between noisy states using alpha-ratio transitions, `jump_length` controlling excursion depth, and `jump_n_sample` controlling revisits. The earlier **Gouri-style** same-timestep perturbation loop (`x_t -> x_hat_0 -> x_t'`) is recorded as not RePaint-equivalent and should not be presented as the final Resampling baseline unless explicitly framed as a different ablation.

## Report & presentation structure

- **Report:** one introduction, one background, one conclusion (shared); methodology and results split per person (possibly merged into one "Method & Results per person" section). See [`report/report-plan.md`](../report/report-plan.md).
- **Presentation:** speaker split captured in [`presentation/final/outline.md`](../presentation/final/outline.md).
