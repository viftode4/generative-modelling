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

## Slide plan

| # | Time | Title | Purpose | Owner |
|---:|---:|---|---|---|
| 1 | 0:00-1:00 | From paper question to our plan | Connect paper motivation to midterm plan | TBD |
| 2 | 1:00-2:05 | Methodology: three sampling ideas | Explain Standard / Resampling / Z-Sampling simply | TBD |
| 3 | 2:05-3:15 | What we actually built | Plan vs actual implementation pipeline | Implementation lead / Vlad |
| 4 | 3:15-4:35 | Results we can stand behind | Show only verified headline numbers | Eugen + Vlad |
| 5 | 4:35-5:55 | What worked, what did not | Scope, compatibility, metrics, runtime | Vlad / implementation lead |
| 6 | 5:55-7:00 | Final takeaway | Honest conclusion + report direction | TBD |

## Design principle

- One headline per slide.
- Maximum 4 bullets per slide.
- No paragraph speaker notes on slides.
- Use one plan-to-actual table, one method diagram, one implementation pipeline, one result table, and one optional image pair.
- No dense tables; keep detailed results in backup/notes.
