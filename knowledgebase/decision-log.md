# Decision Log

Record durable choices that future humans or AI agents should not have to rediscover.

## 2026-04-30 - Knowledgebase structure

**Decision:** Use `knowledgebase/` for project memory and `.ai/` for compact AI context/prompts.

**Rationale:** The course project needs both human-readable notes and reusable AI context. Keeping these separate avoids mixing durable research notes with prompt snippets.

**Status:** Active

## 2026-04-30 - Track upstream code as a submodule

**Decision:** Track the Zigzag Diffusion Sampling reference implementation as a Git submodule at `external/zigzag-diffusion-sampling/`.

**Rationale:** The course repo needs reproducible access to the exact upstream code while preserving upstream history and avoiding vendored copies of external source files.

**Status:** Active
