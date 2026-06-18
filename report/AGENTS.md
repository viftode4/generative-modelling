<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# report

## Purpose

Materials for the **final group report**. The 2026-06-04 course clarification says the final report is group scientific-paper-style writing with contribution ownership visible in section titles. Older individual drafts remain source material.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Quick orientation + template/section conventions. |
| `report-plan.md` | Section map onto the ACM `sigconf` template, per-person ownership, and the verified results to write. |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `figures/` | Image assets used in reports. Source the originals from `outputs/` (gitignored) and copy curated versions here. |
| `tables/` | Tabular data used in reports (CSV/TSV/markdown). |

## For AI Agents

### Working In This Directory

- Keep the report concise and scientific-paper-like. Every figure costs space — prefer one well-chosen comparison over a gallery.
- All factual claims about the paper, method, or upstream code must trace back to a source (the paper PDF, an upstream file, or a recorded experiment in `knowledgebase/logs/`).
- Figures here must be **publication-grade**: legible at column width, captions explain prompt + seed + sampler. Source paths and seeds go in the caption or a footnote.
- Drafts and revisions live under git. Do not commit binary PDFs of compiled reports unless explicitly requested.
- Because the final report is group-level, do not present unverified teammate lanes as completed results. Use committed tables/logs/figures as the threshold for quantitative claims.

### Common Patterns

- Cite using `(Author, Year)` short form in markdown drafts; convert to the course's required style at compile time.
- Reference experiments by their `knowledgebase/logs/...` filename, not by hand-typed parameter dumps.

## Dependencies

### Internal
- `knowledgebase/course/` — the rubric and clarifications the report must satisfy.
- `experiments/results-index/coverage.md` — audited status of which results actually exist.
- `knowledgebase/logs/` — source-of-truth for any numbers/observations cited.
- `experiments/` — for the prompt sets referenced.

<!-- MANUAL: -->
