# Source Package

Use this package for small utilities or adaptations written by this project.

The established Z-Sampling implementation path remains
`zigzag_project.zsampling` so existing teammate notebooks and scripts keep the
same import surface.

There is also a small optional `zigzag_project.samplers` facade:

- `standard.py` — plain pipeline calls.
- `resampling.py` — facade over the canonical RePaint-style resampling
  baseline in `zigzag_project.zsampling`.
- `registry.py` — sampler names, config types, and mismatch validation.

Do not move existing working experiment code into the facade unless there is a
clear compatibility reason. Keep old imports working.

Do not copy upstream source code here unless there is a deliberate adaptation
with provenance documented in `knowledgebase/decision-log.md`.
