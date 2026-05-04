# Midterm Presentation

- Date: Thursday 2026-05-07
- Length: 7 minutes total, 7 students, about 50 seconds each
- Purpose (per the course PDF): feedback from the professors and TAs on our plan. We are presenting the plan, not results.

## Frame

We present the plan and ask for feedback. Each slide should leave room for the audience to push back.

## Slide draft

| # | Slide | What it shows |
|---|-------|---------------|
| 1 | Problem & paper | Z-Sampling paper (Bai et al., ICLR 2025). Failure modes it targets: counting, text rendering, compositional, spatial. |
| 2 | Core idea | Standard sampling vs. zigzag (denoise then invert then continue). Guidance gap. |
| 3 | Our scope | Must / Should / Could / Won't from [../../planning/scope.md](../../planning/scope.md) |
| 4 | Task division | The 7-person model and dataset matrix from [../../planning/role-assignments.md](../../planning/role-assignments.md) |
| 5 | Implementation plan | Shared Z-Sampling code, diffusers as a black box, parity check vs. upstream |
| 6 | Evaluation plan | Metric (HPSv2 or CLIP, still open), winning rate, multiple seeds |
| 7 | Timeline & risks | Week buckets W0 to W8, top risks, what we want feedback on |

## What we want feedback on

- Is the scope (Must / Should / Could) reasonable for 7 people on this paper?
- Is the per-person uniqueness mechanism (different model+dataset pairs plus optional individual extra) acceptable for distinct reports?
- Are there metrics or evaluation choices we are missing?
- Anything in the timeline that looks unrealistic?

## Speaking ownership

7 slides, 7 people. Pick at the next group sync. Not preassigned here.

## Practice notes

- Hard cap: 7 minutes. 7 slides x 50 seconds = 5:50, leaving about 70 seconds for handoffs and overruns.
- Slides should be light. No dense math, no code blocks, no full tables.
- Each slide ends with one open question or "we'd like feedback on X".

## Backup slides (only if asked)

- Pseudocode of the zigzag loop
- Upstream codebase structure
- Could-tier menu
- Detailed gates from [../../planning/timeline.md](../../planning/timeline.md)
