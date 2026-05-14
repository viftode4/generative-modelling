<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-14 | Updated: 2026-05-14 -->

# planning

## Purpose

Forward-looking project management: scope, timeline, role assignments, open decisions, and the report specification. **Not a log** — pages here represent *current intent* and get edited in place as the plan evolves.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Quick index of planning docs. |
| `scope.md` | Must / Should / Could / Won't for the reproduction. |
| `timeline.md` | Week buckets (W0-W8) and per-week gates. |
| `role-assignments.md` | The 7-student × (model, dataset) matrix. |
| `decisions.md` | Open decisions awaiting the next group sync (metric, seeds, compute, etc.). Move items to `knowledgebase/decision-log.md` once resolved. |
| `report-spec.md` | What the 3-page individual report must cover. |

## For AI Agents

### Working In This Directory

- **Edit in place.** Unlike the knowledgebase, planning files are *not* append-only. If `scope.md` changes, edit it directly — the prior intent is recoverable from git history.
- When a decision in `decisions.md` is resolved, **delete it from `decisions.md` and add a dated entry to `knowledgebase/decision-log.md`** with the resolution and rationale.
- When the timeline slips, update `timeline.md` and note the reason in `knowledgebase/research-log.md`.
- Role assignments are owned by the group — do not silently reassign cells. Open seats live in `role-assignments.md` and graduate to assignments once accepted.

### Common Patterns

- Markdown tables for matrices (roles, timeline gates).
- Cross-link related docs with relative paths.

## Dependencies

### Internal
- `knowledgebase/decision-log.md` — the durable record of resolved decisions.
- `report/`, `presentation/` — downstream consumers of `report-spec.md` and the timeline.

<!-- MANUAL: -->
