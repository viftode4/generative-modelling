# Course Requirements Summary

Source documents:

- `docs/course/project-description-brightspace.pdf`
- `docs/course/project-rubric.pdf`
- Text extracts:
  - `knowledgebase/course/project-description-brightspace.txt`
  - `knowledgebase/course/project-rubric.txt`

## Assessment shape

- The project output is an **individual report** worth **30% of the assessment**.
- Work happens in a **group of 5** on a TA-proposed paper/topic per the official PDF.
  - **Our group is 7 students** (registered deviation). Whether expectations adjust for breadth vs. depth is an open question for the TA.
- The goal is to **implement, experiment with, and/or extend** parts of the paper (any combination — extension is not required).
- Existing code may be referenced when available — verbatim from the PDF: **"Some projects have existing code implementation. You have the choice to reference the existing codebase."**

## Feedback milestones

- Mid-term presentation: feedback on project planning.
- End-term presentation: feedback on implementation.
- Presentations are not graded directly and attendance is not mandatory, but they affect the motivation/work-plan part of the individual report rubric.

## Early-course plan expectations

Within weeks 1-4, focus on:

1. Select the paper.
2. Understand the paper.
3. Explore the existing codebase if applicable.
4. Prepare implementation and experimentation plan with the TA.
5. Define division of tasks: who does what.
6. Account for data, training, evaluation, and the method itself.

## Individual report format

- 3 pages.
- Double-column format.
- Use the course/template format when available.

Required report content:

- Introduction and motivation.
- Group and individual perspective on the paper/project.
- Approach, experiment setting, implementation details, and results.
- Interpretation of results.
- Critical appraisal of the paper and experimental design.
- Discussion of unclear parts, key results to reproduce, findings from experiments.

## Rubric weights

| Category | Weight | What excellent work needs |
| --- | ---: | --- |
| Motivation and Work Plan | 30% | Thorough proactive TA-discussed plan, active group contribution, research-project-level presentation. |
| Content | 40% | Strong implementation/experimentation effort, high-quality results with explanations, strong troubleshooting. |
| Writing | 30% | Clear research-paper-like structure, strong result demonstration/explanation, deep analysis and rationale. |

## Knock-out / pass constraints

- A written individual report must be submitted for the project to be considered.
- Passing the course requires at least 5.0 on each component and at least 5.8 weighted final grade.
- Project marks below 5.0 may have a repair/resubmission path under the stated constraints.

## Implications for this repo

This repo should make it easy to show:

- Motivation and plan: `planning/`, `knowledgebase/decision-log.md`, `knowledgebase/meetings/`.
- Content: `experiments/`, `scripts/`, `src/`, `notebooks/`, submodule commit pinning.
- Writing: `report/outline.md`, result tables/figures, interpretation notes, presentation drafts.
- Reproducibility: exact commands, seeds, environment, hardware, output locations, and upstream commit hashes.
