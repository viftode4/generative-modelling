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
3. **Do not commit large artifacts.** Keep datasets, checkpoints, generated images, outputs, and upstream clones out of Git unless explicitly requested.
4. **Prefer reproducible notes.** When running experiments, record commands, environment, seed, hardware, inputs, outputs, and interpretation in `knowledgebase/logs/`.
5. **Separate source from generated work.** Put reusable code in `src/`, notebooks in `notebooks/`, and generated outputs in ignored directories like `outputs/` or `runs/`.
6. **Update project memory.** If a decision, assumption, or important finding is made, update `knowledgebase/decision-log.md`, `knowledgebase/research-log.md`, or the relevant paper note.
7. **Verify before claiming completion.** Run the smallest meaningful checks and report exactly what passed and what remains untested.

## AI collaboration style

- Be explicit about uncertainty and assumptions.
- Prefer small, reviewable changes.
- Reuse existing templates under `knowledgebase/templates/`.
- If external code is needed, use `scripts/fetch-upstream.ps1` to clone it into ignored `external/zigzag-diffusion-sampling/` rather than copying it into this repo.
- If modifying or adapting upstream code, document the upstream commit hash and license considerations.

## Commit expectations

Use clear commits that explain why the change was made. Include verification details in the commit body when useful.
