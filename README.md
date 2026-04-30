# Generative Modelling

Course workspace for generative modelling projects, experiments, notes, and assignments.

## Project group

- **Group:** 14
- **Topic:** ZigZag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection
- **Course group ID:** Group_774744_62489_16
- **Paper:** <https://openreview.net/forum?id=MKvQH1ekeY>
- **PDF:** <https://openreview.net/pdf?id=MKvQH1ekeY>
- **Reference code:** <https://github.com/xie-lab-ml/zigzag-diffusion-sampling>

## Knowledgebase

The project knowledgebase lives in [`knowledgebase/`](knowledgebase/README.md). Start there before adding notes, experiments, or AI-generated work.

Key files:

- [`AGENTS.md`](AGENTS.md) - repository instructions for AI coding/research assistants.
- [`knowledgebase/project-overview.md`](knowledgebase/project-overview.md) - project scope, links, and current research questions.
- [`knowledgebase/papers/zigzag-diffusion-sampling.md`](knowledgebase/papers/zigzag-diffusion-sampling.md) - paper reading notes and implementation anchors.
- [`knowledgebase/ai-workflow.md`](knowledgebase/ai-workflow.md) - how to collaborate with AI tools in this repo.
- [`.ai/context.md`](.ai/context.md) - compact context to paste into AI chats.

## Repository structure

Add course materials as they are created, for example:

- `knowledgebase/` - project memory, paper notes, decisions, research logs, and templates
- `.ai/` - AI-agent context and reusable prompts
- `notebooks/` - exploratory notebooks
- `src/` - reusable Python modules
- `data/` - local datasets (ignored by Git)
- `outputs/` - generated samples, plots, and reports (ignored by Git)
- `models/` - trained checkpoints/weights (ignored by Git)
- `external/` - local clones/submodules of upstream code (ignored by Git by default)

## Getting started

1. Create or activate your Python environment.
2. Read [`knowledgebase/README.md`](knowledgebase/README.md) and [`AGENTS.md`](AGENTS.md).
3. Add project code, notebooks, and documentation.
4. Commit source files, notes, and small reproducible artifacts; keep large generated data/checkpoints out of Git.

To fetch the upstream reference implementation locally without committing it:

```powershell
./scripts/fetch-upstream.ps1
```
