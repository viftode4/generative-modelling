<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# planning

## Purpose

Forward-looking project management: ownership and current status. **Not a log** — pages here represent *current intent* and get edited in place as the plan evolves.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Project status, confirmed dates, and index. |
| `experiment-division.md` | Ownership source of truth: model × dataset matrix, sampling-method/experiment ownership, TA guidance. |

## For AI Agents

### Working In This Directory

- **Edit in place.** Unlike the knowledgebase, planning files are *not* append-only. Edit directly — the prior intent is recoverable from git history.
- When a decision is resolved, add a dated entry to `knowledgebase/decision-log.md` with the resolution and rationale.
- Ownership is owned by the group — do not silently reassign cells in `experiment-division.md`.

### Common Patterns

- Markdown tables for matrices (roles, timeline gates).
- Cross-link related docs with relative paths.

## Dependencies

### Internal
- `knowledgebase/decision-log.md` — the durable record of resolved decisions.
- `experiments/results-index/coverage.md` — audited status of what results actually exist.
- `report/`, `presentation/` — downstream consumers of the plan.

<!-- MANUAL: -->
