# Generative Modelling

Course workspace for the Group 14 generative modelling project.

## Project

- Group: 14 (7 students)
- Topic: ZigZag Diffusion Sampling (Bai et al., ICLR 2025)
- Course group ID: Group_774744_62489_16
- Paper: https://openreview.net/forum?id=MKvQH1ekeY
- Reference code: https://github.com/xie-lab-ml/zigzag-diffusion-sampling

## Start here

- [planning/README.md](planning/README.md) - project status, dates, and index
- [planning/experiment-division.md](planning/experiment-division.md) - who owns which model, dataset, and sampling method
- [experiments/results-index/coverage.md](experiments/results-index/coverage.md) - audited status of what results actually exist
- [report/report-plan.md](report/report-plan.md) - group-report section plan (ACM template)
- [presentation/final/endterm_deck.md](presentation/final/endterm_deck.md) - end-term presentation content

## Repository layout

- `planning/` - ownership and current status
- `presentation/` - midterm and endterm decks
- `knowledgebase/` - paper notes, course requirements, AI-workflow notes
- `docs/course/` - official project description and rubric PDFs
- `external/` - upstream reference code (Git submodule)
- `experiments/`, `report/`, `src/`, `notebooks/`, `data/`, `outputs/`, `models/` - created as work progresses

## Setup

See [knowledgebase/setup.md](knowledgebase/setup.md) for submodule, Python, and inference setup.

```powershell
./scripts/fetch-upstream.ps1
# or
git submodule update --init --recursive
```
