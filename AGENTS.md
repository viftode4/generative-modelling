# AGENTS.md - Generative Modelling Course Repo

This file gives local instructions to AI assistants working in this repository.

## Mission

Support the course project for **Group 14: ZigZag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection**.

Primary references:

- Paper/forum: https://openreview.net/forum?id=MKvQH1ekeY
- PDF: https://openreview.net/pdf?id=MKvQH1ekeY
- Reference code: https://github.com/xie-lab-ml/zigzag-diffusion-sampling

## Working rules

1. **Start with the knowledgebase.** Before research, implementation, or summarization, read `knowledgebase/README.md` and the relevant paper/project notes.
2. **Preserve provenance.** Any factual claim about the paper, code, experiment, or result should link to a source or identify that it is an inference.
3. **Do not commit large artifacts.** Keep datasets, checkpoints, generated images, outputs, and run logs out of Git unless explicitly requested.
4. **Treat upstream code as external.** The reference implementation is tracked as a Git submodule at `external/zigzag-diffusion-sampling/`; do not vendor-copy or silently modify it. If changes are needed, make them in this repository's own `src/` or scripts unless explicitly working on the submodule.
5. **Prefer reproducible notes.** When running experiments, record commands, environment, seed, hardware, inputs, outputs, and interpretation in `knowledgebase/logs/`.
6. **Separate source from generated work.** Put reusable code in `src/`, notebooks in `notebooks/`, and generated outputs in ignored directories like `outputs/` or `runs/`.
7. **Update project memory.** If a decision, assumption, or important finding is made, update `knowledgebase/decision-log.md`, `knowledgebase/research-log.md`, or the relevant paper note.
8. **Verify before claiming completion.** Run the smallest meaningful checks and report exactly what passed and what remains untested.

## Repository map

Each top-level directory has its own `AGENTS.md` with local rules and conventions. Read the one for the area you're editing before making changes.

| Directory | What lives there |
|-----------|------------------|
| `src/` | Importable Python package (`zigzag_project`). See `src/AGENTS.md`. |
| `tests/` | Fast `pytest` suite — no model downloads. See `tests/AGENTS.md`. |
| `scripts/` | Runnable CLIs and smoke tests. See `scripts/AGENTS.md`. |
| `experiments/` | Per-experiment configs, prompt sets, run index. See `experiments/AGENTS.md`. |
| `knowledgebase/` | Single source of truth: notes, logs, decisions, paper notes. See `knowledgebase/AGENTS.md`. |
| `planning/` | Forward-looking scope, timeline, roles, open decisions. See `planning/AGENTS.md`. |
| `report/` | 3-page individual report assets + outline. See `report/AGENTS.md`. |
| `presentation/` | Midterm + end-term presentation outlines. See `presentation/AGENTS.md`. |
| `docs/` | External documents (course PDFs, unmodified). See `docs/AGENTS.md`. |
| `external/` | Upstream ZigZag implementation as a Git submodule. Read-only reference. |

## AI collaboration style

- Be explicit about uncertainty and assumptions.
- Prefer small, reviewable changes.
- Reuse existing templates under `knowledgebase/templates/`.
- Use `knowledgebase/setup.md` for environment and submodule setup.
- If modifying or adapting upstream code, document the upstream commit hash and license considerations.

## Commit expectations

Use clear commits that explain why the change was made. Include verification details in the commit body when useful.
