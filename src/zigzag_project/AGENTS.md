<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# zigzag_project

## Purpose

The project's installable Python package. Holds shared utilities used across all 7 cells: model loading, (eventually) the Z-Sampling implementation, evaluation hooks, and data/prompt handling.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package marker. Exposes `__version__`. |
| `models.py` | `load_dreamshaper_xl_turbo()` + `resolve_device()`. Returns a `ModelHandle` containing the pipeline, inverse scheduler, device, and dtype. |
| `README.md` | One-paragraph reminder: don't vendor-copy upstream here. |

## For AI Agents

### Working In This Directory

- The Z-Sampling implementation does **not yet live here**. When it lands, it should be a function/module that takes a `ModelHandle` plus hyperparameters (`gamma_1`, `gamma_2`, `infer_step`, `lambda_step`, `T_max`) and returns generated latents/images — not a monolithic pipeline subclass like upstream does.
- Every model loader should:
  1. Accept `device` and `dtype` kwargs (default sensibly via `resolve_device`).
  2. Configure `DDIMScheduler` + `DDIMInverseScheduler` (the inversion step requires a deterministic invertible scheduler).
  3. Return a `ModelHandle`, not the raw pipeline — keep the call site uniform across cells.
- New loaders (e.g. for SD2.1, Hunyuan-DiT, SDXL base) go alongside `load_dreamshaper_xl_turbo` and follow the same signature shape.

### Testing Requirements

- Import-level tests live in `tests/`. Run with `pytest`.
- For changes that touch generation, the manual smoke check is `scripts/smoke_dreamshaper.py` (downloads ~6 GB on first run).

### Common Patterns

- Use `dataclass(frozen=True)` for handles passed across module boundaries.
- Do not import `argparse` or read CLI args from inside this package — that belongs in `scripts/`.

## Dependencies

### Internal
- None yet — this is the leaf package.

### External
- `torch >= 2.3` (MPS-capable build for Apple Silicon).
- `diffusers >= 0.30` (newer than upstream's pin; needed for working schedulers + turbo support).

<!-- MANUAL: -->
