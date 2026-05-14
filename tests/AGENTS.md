<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# tests

## Purpose

Fast, deterministic unit tests for `src/zigzag_project/`. Run with `pytest` from the repo root.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Brief notes on what belongs here. |
| `.gitkeep` | Placeholder; remove when first real test lands. |

## For AI Agents

### Working In This Directory

- **Keep tests fast.** No model downloads, no real diffusion calls. Use small synthetic tensors or mocked pipelines for any code that would otherwise touch a GPU/MPS backend.
- Anything that needs the actual model (visual check, end-to-end smoke) goes in `scripts/`, not here.
- Test files match `test_*.py`. Mirror the structure of `src/zigzag_project/` (so `models.py` → `tests/test_models.py`).
- Prefer pure-function tests with explicit inputs/outputs over fixture sprawl.

### Testing Requirements

- `pytest` (lint via `ruff check`). Both should pass before commit.
- No network access in tests. Skip with `pytest.mark.skipif` if a test truly needs to hit Hugging Face.

### Common Patterns

- For tensor-shape tests, parametrize over a few representative shapes rather than enumerating exhaustively.
- For device tests, use `monkeypatch` against `torch.backends.mps.is_available` / `torch.cuda.is_available` to keep the suite portable.

## Dependencies

### Internal
- `zigzag_project` (the package under test).

### External
- `pytest`.

<!-- MANUAL: -->
