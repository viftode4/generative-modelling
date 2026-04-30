# Knowledgebase

This folder is the shared memory for the generative modelling course project. It is designed for humans and AI assistants to keep context, sources, decisions, and experiment notes in one place.

## Start here

1. [`project-overview.md`](project-overview.md) - current project identity, source links, and open questions.
2. [`papers/zigzag-diffusion-sampling.md`](papers/zigzag-diffusion-sampling.md) - structured reading notes for the main paper.
3. [`setup.md`](setup.md) - reproducible submodule, environment, and inference setup.
4. [`ai-workflow.md`](ai-workflow.md) - rules for using AI tools responsibly in this repo.
5. [`decision-log.md`](decision-log.md) - durable decisions and rationale.
6. [`research-log.md`](research-log.md) - chronological research notes.
7. [`glossary.md`](glossary.md) - technical terms and definitions.

## Conventions

- Add source URLs next to claims whenever possible.
- Mark uncertain notes with `TODO` or `Needs verification`.
- Keep summaries short enough to review, and link to detailed notes or source documents.
- Put experiment entries in `logs/` using `templates/experiment-log-template.md`.
- Put new paper notes in `papers/` using `templates/paper-note-template.md`.

## Suggested workflow

```text
Question -> source lookup -> notes -> decision/experiment -> reproducible artifact -> commit
```

Before asking an AI assistant to implement or analyze, paste or point it to `.ai/context.md` plus the relevant knowledgebase file.
