# Project Work Plan

## Goal

Set up a clean, reproducible workspace for the Zigzag Diffusion Sampling course project and support the final individual report with clear evidence, experiments, and notes.

## Success criteria mapped to rubric

### Motivation and Work Plan (30%)

- [ ] Paper and codebase understood well enough to explain the project plan.
- [ ] Implementation and experimentation plan documented.
- [ ] Mid-term presentation prepared with a research-project-level plan.
- [ ] Key decisions recorded.

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

## Initial milestones

| Phase | Outcome | Repo artifacts |
| --- | --- | --- |
| Understand | Paper/code notes and glossary | `knowledgebase/papers/`, `knowledgebase/glossary.md`, `knowledgebase/upstream-code.md` |
| Plan | Implementation/experiment plan | `planning/work-plan.md`, `planning/task-board.md` |
| Reproduce | Runnable baseline or documented blocker | `scripts/`, `knowledgebase/logs/`, ignored `outputs/` |
| Analyze | Result interpretation and limitations | `report/tables/`, `report/figures/`, `knowledgebase/logs/` |
| Present/write | Slides and individual report outline | `presentation/`, `report/outline.md` |

## Immediate next actions

1. Read PDF and extract algorithm/equations into `knowledgebase/papers/zigzag-diffusion-sampling.md`.
2. Inspect upstream `infer.py` and pipeline implementation; map standard sampling vs. Z-Sampling call flow.
3. Confirm available hardware/CUDA setup.
4. Decide the smallest reproducible experiment for the report.
5. Draft the experiment plan and report outline from the evidence collected.

## Planning files

- `planning/task-board.md` - active task tracker.
- `planning/timeline.md` - milestone tracker.
- `planning/ta-questions.md` - questions for project feedback.
