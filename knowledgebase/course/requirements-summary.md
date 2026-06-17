# Course Requirements Summary

Source documents:

- `docs/course/project-description-brightspace.pdf`
- `docs/course/project-rubric.pdf`
- Text extracts:
  - `knowledgebase/course/project-description-brightspace.txt`
  - `knowledgebase/course/project-rubric.txt`

## Assessment shape

- The original project PDF described an individual report, but the 2026-06-04 course clarification says the submitted report is a **group report** structured like a scientific paper, with section titles mentioning who did what.
- Work happens in a **group of 5** on a TA-proposed paper/topic per the official PDF.
  - **Our group is 7 students** (registered deviation). Whether expectations adjust for breadth vs. depth is an open question for the TA.
- The goal is to **implement, experiment with, and/or extend** parts of the paper (any combination — extension is not required).
- Existing code may be referenced when available — verbatim from the PDF: **"Some projects have existing code implementation. You have the choice to reference the existing codebase."**

## Feedback milestones

- Mid-term presentation: feedback on project planning.
- End-term presentation: graded implementation/results presentation.
- The 2026-06-04 course clarification says presentation is a **graded component worth 20%**, should cover methodology, results, and challenges, and lasts **7 minutes per group** plus **4 minutes feedback**.

## Early-course plan expectations

Within weeks 1-4, focus on:

1. Select the paper.
2. Understand the paper.
3. Explore the existing codebase if applicable.
4. Prepare implementation and experimentation plan with the TA.
5. Define division of tasks: who does what.
6. Account for data, training, evaluation, and the method itself.

## Group report format

- Group scientific-paper-style report.
- Mention who did what in the section titles.
- Use the course/template format when available.
- Earlier notes about 3-page individual reports may reflect the original PDF, but the 2026-06-04 announcement supersedes them for the final submission shape.

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

- A written group report must be submitted for the project to be considered.
- Passing the course requires at least 5.0 on each component and at least 5.8 weighted final grade.
- Project marks below 5.0 may have a repair/resubmission path under the stated constraints.

## Implications for this repo

This repo should make it easy to show:

- Motivation and plan: `planning/`, `knowledgebase/decision-log.md`, `knowledgebase/meetings/`.
- Content: `experiments/`, `scripts/`, `src/`, `notebooks/`, submodule commit pinning.
- Writing: `report/report-plan.md`, result tables/figures, interpretation notes, presentation drafts, and contribution-marked group-report sections.
- Reproducibility: exact commands, seeds, environment, hardware, output locations, and upstream commit hashes.

## 2026-06-04 clarification update

See `knowledgebase/course/2026-06-04_presentation-report-clarification.md`. This announcement updates the older understanding in two important ways:

1. Group 14 presents on **2026-06-09**.
2. The end-term presentation is **graded (20%)**, lasts **7 minutes per group**, and must cover methodology, results, and challenges.
3. The report is a **group report** in scientific-paper style, with section titles indicating who did what.
