<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# report

## Purpose

Materials for the **3-page double-column individual report** required by the course. Each student writes their own; this folder holds the shared outline, the figure/table assets, and (eventually) draft LaTeX/Markdown sources.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Quick orientation. |
| `outline.md` | Section-by-section structure all 7 reports follow (Intro / Method / Setting / Results / Critical Appraisal / Conclusion). |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `figures/` | Image assets used in reports. Source the originals from `outputs/` (gitignored) and copy curated versions here. |
| `tables/` | Tabular data used in reports (CSV/TSV/markdown). |

## For AI Agents

### Working In This Directory

- The **page budget is hard:** 3 pages, double-column. Every figure costs space — prefer one well-chosen comparison over a gallery.
- All factual claims about the paper, method, or upstream code must trace back to a source (the paper PDF, an upstream file, or a recorded experiment in `knowledgebase/logs/`).
- Figures here must be **publication-grade**: legible at column width, captions explain prompt + seed + sampler. Source paths and seeds go in the caption or a footnote.
- Drafts and revisions live under git. Do not commit binary PDFs of compiled reports unless explicitly requested.

### Common Patterns

- Cite using `(Author, Year)` short form in markdown drafts; convert to the course's required style at compile time.
- Reference experiments by their `knowledgebase/logs/...` filename, not by hand-typed parameter dumps.

## Dependencies

### Internal
- `planning/report-spec.md` — the rubric the report must satisfy.
- `knowledgebase/logs/` — source-of-truth for any numbers/observations cited.
- `experiments/` — for the prompt sets referenced.

<!-- MANUAL: -->
