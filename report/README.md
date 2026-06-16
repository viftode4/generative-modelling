# Report

Materials for the **final group report** — scientific-paper style, with contribution shown
in section titles (per the 2026-06-04 course clarification).

## Template

TU Delft ACM `sigconf`. Fixed section files: `s1_introduction`, `s2_background`,
`s3_methodology`, `s4_results`, `s5_conclusion`, `appendix`.

- **Shared sections** (one author each): introduction, background, conclusion.
- **Per-person subsections** in methodology + results, named with the owner (e.g. `4.2 DreamShaper + GenEval (Vlad)`).

## Files

| Path | Contents |
|------|----------|
| `report-plan.md` | Section-by-section plan mapped onto the ACM template + per-person ownership |
| `tables/` | Committed result tables (the only quantitative claims allowed in the report) |
| `figures/` | Curated, publication-grade figures (originals live in gitignored `outputs/`) |

## Rules

- Quantitative claims must trace to a committed table in `tables/` or `outputs/`. See [`experiments/results-index/coverage.md`](../experiments/results-index/coverage.md) for what's actually verified.
- Do not present unrun or unverified teammate lanes as results.
- Keep figures legible at column width; captions state prompt + seed + sampler.
- Don't commit compiled PDFs unless asked.
