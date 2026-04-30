# Project Work Plan

## Goal

Produce a strong individual report and supporting group project artifacts for Group 14 on Zigzag Diffusion Sampling.

## Success criteria mapped to rubric

### Motivation and Work Plan (30%)

- [ ] Paper and codebase understood well enough to explain to TA/group.
- [ ] Plan discussed with TA.
- [ ] Individual responsibilities documented.
- [ ] Mid-term presentation prepared with a research-project-level plan.
- [ ] Meeting notes and decisions recorded.

### Content (40%)

- [ ] Identify key result(s) to reproduce or analyze.
- [ ] Establish reproducible baseline using upstream code or a controlled adaptation.
- [ ] Run at least one meaningful experiment/comparison.
- [ ] Record troubleshooting and implementation details.
- [ ] Interpret results rather than only showing outputs.

### Writing (30%)

- [ ] Report follows double-column 3-page constraint.
- [ ] Clear motivation and method explanation.
- [ ] Results are shown with figures/tables.
- [ ] Analysis includes limitations, uncertainty, and rationale.
- [ ] Individual contribution is clear.

## Initial milestones

| Phase | Outcome | Repo artifacts |
| --- | --- | --- |
| Understand | Paper/code notes and glossary | `knowledgebase/papers/`, `knowledgebase/glossary.md`, `knowledgebase/upstream-code.md` |
| Plan | TA-discussed implementation/experiment plan | `planning/work-plan.md`, `planning/task-board.md`, `knowledgebase/meetings/` |
| Reproduce | Runnable baseline or documented blocker | `scripts/`, `experiments/logs`, `outputs/` ignored |
| Analyze | Result interpretation and limitations | `report/tables/`, `report/figures/`, `knowledgebase/logs/` |
| Present/write | Slides and individual report outline | `presentation/`, `report/outline.md` |

## Immediate next actions

1. Read PDF and extract algorithm/equations into `knowledgebase/papers/zigzag-diffusion-sampling.md`.
2. Inspect upstream `infer.py` and pipeline implementation; map standard sampling vs. Z-Sampling call flow.
3. Confirm available hardware/CUDA setup.
4. Decide the smallest reproducible experiment for the report.
5. Prepare TA discussion questions and group task division.
## Planning files

- `planning/task-board.md` - active task tracker.
- `planning/group-roles.md` - group and individual contribution tracking.
- `planning/timeline.md` - milestone tracker.
- `planning/ta-questions.md` - questions for TA feedback.
