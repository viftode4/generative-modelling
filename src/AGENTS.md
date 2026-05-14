<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# src

## Purpose

Reusable Python code for the Group 14 ZigZag Diffusion Sampling reproduction. Layout follows the `src/`-layout convention declared in `pyproject.toml`. Install editable with `uv pip install -e .` (or `pip install -e .`) so packages here are importable without `PYTHONPATH`.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `zigzag_project/` | The project's importable Python package (see `zigzag_project/AGENTS.md`). |

## For AI Agents

### Working In This Directory

- **Do not vendor-copy upstream code here.** Upstream lives in `external/zigzag-diffusion-sampling/` as a Git submodule. If we need an adapted version of upstream logic, write a thin wrapper or reimplementation here and record the upstream commit hash + reasoning in `knowledgebase/decision-log.md`.
- Keep modules small and focused. The shared codebase will be used by 7 students across 7 (model, dataset) cells — APIs that are easy to reuse beat clever abstractions.
- Prefer pure functions over classes for sampling logic. State (pipeline, scheduler) gets passed in.
- `from __future__ import annotations` is fine for forward-reference typing; project targets Python 3.10.

### Testing Requirements

- Any code added here should be importable and unit-testable. Pair non-trivial functions with a test in `tests/`.
- Smoke tests that need a model loaded belong in `scripts/`, not `tests/` — keep `pytest` fast.

### Common Patterns

- Device resolution: use `zigzag_project.models.resolve_device(...)` rather than hard-coding `"cuda"` / `"mps"` / `"cpu"`. The group runs on a mix of CUDA boxes and Apple Silicon.
- Use `dataclass(frozen=True)` for handles/configs that get passed around (see `ModelHandle`).

## Dependencies

### Internal
- `external/zigzag-diffusion-sampling/` — read-only reference for porting Z-Sampling logic. Adapt, don't import.

### External
- `torch`, `diffusers`, `transformers` — installed in `.venv`. See `requirements-zigzag.txt` for upstream pins (note: Vlad's local env diverges, see `knowledgebase/research-log.md`).

<!-- MANUAL: -->
