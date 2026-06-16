# End-term Presentation Outline — Plan-to-Result Version

Source update: the 2026-06-04 course announcement says Group 14 presents on 2026-06-09. The presentation is graded (20%), lasts 7 minutes per group, and should cover methodology, results, and challenges.


## Source anchors

- Course timing/requirements: `knowledgebase/course/2026-06-04_presentation-report-clarification.md`.
- Midterm plan framing: `knowledgebase/course/2026-06-07_midterm-plan-to-final-framing.md` and `planning/experiment-division.md`.
- Verified result tables: `report/tables/eugen_results.md`, `report/tables/vlad_geneval_detector.md`, `report/tables/vlad_compute_matched.md`.

## One-sentence story

> We planned a broad reproduction of Z-Sampling, implemented the core inference-time pipeline, verified positive paired results in a smaller subset, and learned that the main limits are integration effort, metrics, runtime, and benchmark scale.

## What changed from midterm

Midterm plan:

- Compare Standard, Resampling, and Z-Sampling.
- Split work by model/dataset/method lanes.
- Use several metrics where possible.
- Add ablations or efficiency comparisons if time allowed.

Final presentation stance:

- Keep the methodology broad enough to mention the planned three sampling methods.
- Spend most time on verified implementation and results.
- Be explicit that not every planned cell/metric became report-ready.
- Present challenges as real project findings, not excuses.

## Timing rule

- 6 slides maximum.
- About 60 seconds per slide.
- Leave ~45 seconds buffer for handoffs.
- Every slide answers one question: **why, how, what did we build, what happened, what blocked us, what do we conclude?**

## Slide plan (agreed speaker split)

Presenters: **the Romanians.** Template suggestion: https://canva.link/rr17l2qsi7hoglv

| # | Section | Owner | Content |
|---:|---|---|---|
| 1 | Paper + background | **Vlad** | Paper overview; quick intuition about Z-Sampling; models/datasets used — **no Hunyuan dataset** |
| 2 | The sampling methods | **Radu** | Discuss each: regular sampling, resampling, Z-Sampling — and compare them |
| 3 | Challenges | **Eugen** | Resampling: understanding, implementation, running |
| 4 | Results | **Bogdan** | Eugen's results; Lenka — AYS + Z-Sampling (one table); Bogdan — z-sampling step-ablation graph |
| 5 | Q&A | — | 4-minute feedback slot |

## Design principle

- One headline per slide.
- Maximum 4 bullets per slide.
- No paragraph speaker notes on slides.
- Use one plan-to-actual table, one method diagram, one implementation pipeline, one result table, and one optional image pair.
- No dense tables; keep detailed results in backup/notes.
