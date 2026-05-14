<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# knowledgebase

## Purpose

The **single source of truth** for everything not in source code: project context, paper notes, course requirements, decisions, research log, glossary, meeting notes, and per-run experiment logs. Read `README.md` here before doing research or writing.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Index of what's where and how the knowledgebase is organized. |
| `project-overview.md` | Group identity, paper, primary sources, deliverable scope. |
| `setup.md` | How to set up the environment + submodule. Status: PowerShell-centric; Vlad has macOS notes in `research-log.md`. |
| `glossary.md` | Terms: diffusion, inversion, guidance, CFG, etc. |
| `decision-log.md` | Append-only log of group decisions. Date every entry. |
| `research-log.md` | Append-only log of research findings, env changes, dead ends. Date every entry. |
| `upstream-code.md` | Notes on the upstream reference implementation. |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `course/` | Extracted text of the official course PDFs + a distilled requirements summary. |
| `papers/` | Per-paper notes (currently just the ZigZag paper). |
| `templates/` | Reusable templates: experiment log, paper note. |
| `logs/` | One file per experimental run, created from `templates/experiment-log-template.md`. |
| `meetings/` | Notes from group + TA meetings. |

## For AI Agents

### Working In This Directory

- **Append-only logs.** `research-log.md`, `decision-log.md`, and `logs/*.md` should grow; do not rewrite history. Add new dated sections at the bottom. Convert relative dates to absolute (`2026-05-14`, not "today").
- **Cite sources.** Any factual claim about the paper, upstream code, or external research gets a URL or file reference. Mark inferences explicitly.
- **Provenance for adapted code.** If a snippet here is paraphrasing or adapting upstream code, name the upstream file + commit hash.
- When recording an experiment, **copy** `templates/experiment-log-template.md` into `logs/`, don't edit the template.

### Common Patterns

- Filename pattern for run logs: `logs/YYYY-MM-DD_<cell>_<short-tag>.md`.
- Paper notes follow `templates/paper-note-template.md`.

## Dependencies

### Internal
- `planning/` — task tracking and scope. The knowledgebase records what *was* done; planning records what *will* be done.
- `experiments/` — configs and the run index. The knowledgebase logs hold the prose write-up of each run.

<!-- MANUAL: -->
