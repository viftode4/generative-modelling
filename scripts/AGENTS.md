<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# scripts

## Purpose

Runnable entry points: environment setup, fetching upstream, smoke tests, batch generation, evaluation runs. Anything with a `__main__`-style CLI lives here, not in `src/`.

## Key Files

| File | Description |
|------|-------------|
| `setup-zigzag-env.ps1` | **Windows / PowerShell only.** Creates the `.venv` and installs the upstream pinned requirements. Does not work on macOS/Linux. |
| `fetch-upstream.ps1` | **Windows / PowerShell only.** Pulls the upstream submodule. |
| `check-zigzag-setup.ps1` | **Windows / PowerShell only.** Verifies submodule + Python + CUDA visibility. |
| `run-zigzag-inference.ps1` | **Windows / PowerShell only.** Runs `external/zigzag-diffusion-sampling/infer.py`. |
| `smoke_dreamshaper.py` | Cross-platform. Loads `DreamShaper-xl-v2-turbo` and generates one image. First run downloads ~6 GB. |

## For AI Agents

### Working In This Directory

- **Cross-platform parity is a known gap.** The original setup scripts are PowerShell because the project was bootstrapped on Windows. macOS/Linux users currently use direct commands (`uv venv .venv --python 3.10`, `uv pip install ...`). When adding new operational scripts, prefer **Python** (`.py`) over shell — it works for everyone.
- Every script must accept the relevant knobs as CLI flags (seed, device, prompt, output dir, model id, hyperparameters). No hard-coded paths or magic constants — agents and other students need to override them.
- Outputs (`*.png`, generated batches, logs) must go under `outputs/` / `runs/` / `results/` — all gitignored. Do not write into tracked directories.
- For every long-running script, log: repo commit hash, upstream submodule commit, command line, device, seed, runtime. Use `knowledgebase/templates/experiment-log-template.md` as the recording template.

### Common Patterns

- `argparse` for CLI parsing.
- `from zigzag_project.<module> import ...` (works because the package is editable-installed).
- Default `--device None` and resolve via `zigzag_project.models.resolve_device()`.

## Dependencies

### Internal
- `zigzag_project` (the project package).

### External
- Whatever the specific script needs. Keep scripts narrow.

<!-- MANUAL: -->
