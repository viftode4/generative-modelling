# AI Workflow for This Repository

Use AI as a research and implementation assistant, not as an unquestioned source of truth.

## Context packet

When starting a new AI chat, provide:

1. `.ai/context.md`
2. `knowledgebase/project-overview.md`
3. The specific paper, experiment, or code note relevant to the task
4. Any current command output or error logs

## Good AI tasks

- Summarize a section of the paper into bullet points with equations preserved.
- Convert notes into presentation slides or speaking points.
- Explain unfamiliar diffusion-model terminology.
- Draft experiment plans and checklists.
- Help debug environment or inference scripts.
- Generate small utility scripts with tests or verification steps.

## Required AI hygiene

- Ask the AI to separate **source-backed facts**, **inferences**, and **speculation**.
- Ask for citations or exact source locations for claims about the paper/code.
- Never paste private tokens, API keys, student IDs, or unreleased data.
- Verify generated commands before running expensive GPU jobs.
- Record useful outputs in the knowledgebase.

## Prompt pattern

```text
You are helping with a generative modelling course project.
Read this context first: [paste .ai/context.md]
Task: [specific action]
Constraints: cite sources, mark uncertainty, keep outputs reproducible.
Expected output: [notes/table/code/checklist/etc.]
```

## Verification checklist for AI-generated work

- [ ] Does it cite paper/code sources for factual claims?
- [ ] Does it avoid committing large/generated artifacts?
- [ ] Are assumptions stated?
- [ ] Are commands reproducible?
- [ ] Is there a clear next action or conclusion?
