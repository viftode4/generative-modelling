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

- [`docs/course/`](docs/course/README.md) - official project description and rubric PDFs.
- [`AGENTS.md`](AGENTS.md) - repository instructions for AI coding/research assistants.
- [`knowledgebase/project-overview.md`](knowledgebase/project-overview.md) - project scope, links, and current research questions.
- [`knowledgebase/papers/zigzag-diffusion-sampling.md`](knowledgebase/papers/zigzag-diffusion-sampling.md) - paper reading notes and implementation anchors.
- [`knowledgebase/ai-workflow.md`](knowledgebase/ai-workflow.md) - how to collaborate with AI tools in this repo.
- [`.ai/context.md`](.ai/context.md) - compact context to paste into AI chats.

## Repository structure

Add course materials as they are created, for example:

- `knowledgebase/` - project memory, paper notes, decisions, research logs, and templates
- `planning/` - work plan, task board, TA questions, and meeting notes
- `experiments/` - experiment configs, prompt sets, and lightweight result indexes
- `report/` - individual report outline and final report assets
- `presentation/` - mid-term and final presentation outlines
- `.ai/` - AI-agent context and reusable prompts
- `notebooks/` - exploratory notebooks
- `src/` - reusable Python modules
- `data/` - local datasets (ignored by Git)
- `outputs/` - generated samples, plots, and reports (ignored by Git)
- `models/` - trained checkpoints/weights (ignored by Git)
- `external/` - external reference code tracked as Git submodules

## Setup

See [knowledgebase/setup.md](knowledgebase/setup.md) for reproducible submodule, Python environment, and inference setup.

## Getting started

1. Create or activate your Python environment.
2. Read [`knowledgebase/README.md`](knowledgebase/README.md) and [`AGENTS.md`](AGENTS.md).
3. Add project code, notebooks, and documentation.
4. Commit source files, notes, and small reproducible artifacts; keep large generated data/checkpoints out of Git.

To initialize or update the upstream reference implementation submodule:

```powershell
./scripts/fetch-upstream.ps1
# or
git submodule update --init --recursive
```
