# Group 14 Role Assignments

7 students. 4 models. 3 datasets. Goal: zero redundancy across individual reports.

## Layered ownership

Each person owns three things, layered so no two reports tell the same story:

1. **A model** (assigned)
2. **A dataset** (assigned)
3. **An angle / role** (to confirm in next sync)

Each person's report is naturally framed as:
> "Z-Sampling on **[my model]** using **[my dataset]** — **[my angle]**"

That is 7 distinct sentences, which means 7 distinct reports.

## Current assignments

### Layer 1 — Models

| Model | People |
|-------|--------|
| SD2.1 (Rombach et al. 2022) | Eugen, Radu |
| SDXL (Podell et al. 2023) | Daniel, Lenka |
| Hunyuan-DiT (Li 2024) | Bogdan |
| DreamShaper-xl-v2-turbo | Gouri, Vlad |

### Layer 2 — Datasets

| Dataset | People |
|---------|--------|
| Pick-a-Pic (Kirstain et al. 2023) | Lenka, Eugen |
| DrawBench (Saharia et al. 2022) | Daniel, Gouri, Radu |
| GenEval (Ghosh et al. 2024) | Bogdan, Vlad |

### Resulting (Person → Model × Dataset) intersections

| Person | Model | Dataset | Natural distinction |
|--------|-------|---------|--------------------|
| Eugen | SD2.1 | Pick-a-Pic | Same model as Radu, different dataset |
| Radu | SD2.1 | DrawBench | Same model as Eugen, different dataset |
| Daniel | SDXL | DrawBench | Same model as Lenka, different dataset |
| Lenka | SDXL | Pick-a-Pic | Same model as Daniel, different dataset |
| Bogdan | Hunyuan-DiT | GenEval | Only one on this model — naturally unique |
| Gouri | DreamShaper | DrawBench | Same model as Vlad, different dataset |
| Vlad | DreamShaper | GenEval | Same model as Gouri, different dataset |

**No two people share both model AND dataset.** The base layer is already non-redundant.

## Layer 3 — Angles / cross-cutting roles (TO ASSIGN)

These need owners. Each person picks one. Some are "shared infrastructure" (others depend on it), others are "individual analysis" (only feeds one report).

### Shared infrastructure roles (2)

These produce code/data everyone uses. Whoever takes them is also doing their per-person experiment, but they have an extra obligation to the team.

- **Implementation Lead** — owns `src/zsample/core.py` (the Z-Sampling function from scratch). Sets up the diffusers pipeline harness. **Best fit:** someone on SDXL since the upstream reference code targets SDXL → **Daniel or Lenka**.
- **Evaluation Lead** — owns `src/zsample/metrics.py`. Installs and verifies HPSv2, PickScore, ImageReward, AES. Provides one common scoring function everyone calls. **Best fit:** someone with patience for fragile academic libraries → could be anyone.

### Individual angle roles (5)

Each is a distinct narrative angle that produces ONE figure or table that ONLY appears in that person's report.

| # | Angle | Unique artifact for the report | Notes |
|---|-------|-------------------------------|-------|
| A1 | **Theory & method derivation** | Theoretical diagram + derivation of guidance gap from CFG; latent-trajectory PCA | Lighter on compute, heavier on math/writing. Good fit for Bogdan since Hunyuan-DiT is architecturally different (transformer-based) and motivates a theory angle. |
| A2 | **Hyperparameter ablation: guidance scales** | γ₁/γ₂ sensitivity heatmap | Runs ablation on ONE model (probably SDXL). Owner shares results with everyone. |
| A3 | **Hyperparameter ablation: zigzag steps** | λ_step / T_max sweep curve | Same model as A2 for consistency. |
| A4 | **Per-dataset / per-category analysis** | Bar chart: where does Z-Sampling help most? (compositional, style, counting, spatial, text) | Owner pulls together results from all 7 model×dataset experiments. |
| A5 | **Failure-mode analysis** | Annotated gallery of prompts where Z-Sampling DEGRADES quality | Owner runs diagnostic prompts; finds where the paper's claims break. |
| A6 | **Compute / efficiency analysis** | Cost-vs-quality Pareto curve across all 4 models | Owner wraps timing/memory instrumentation; compares wall-clock cost across architectures. |
| A7 | **Cross-model comparison** | Table: HPSv2 winning rate per (model, dataset) cell | Owner aggregates everyone's results into a single comparison matrix. |

(7 angles for 7 people. Pick one each.)

## Suggested first-pass angle assignment

Open for discussion. Adjust based on what people want to write about:

| Person | Suggested angle | Why |
|--------|----------------|-----|
| Daniel | Implementation Lead (cross-cutting) + A2 ablation | SDXL has the upstream reference code; he's natural to lead implementation |
| Lenka | A5 Failure-mode analysis | SDXL + Pick-a-Pic gives a known-strong setup to find where it BREAKS |
| Eugen | Evaluation Lead (cross-cutting) + A4 per-category | Pick-a-Pic dataset is designed for the paper's primary metrics (HPSv2/PickScore) |
| Radu | A3 zigzag-steps ablation | Different angle from Eugen so the SD2.1 pair is non-redundant |
| Bogdan | A1 Theory + Hunyuan-DiT deep-dive | Hunyuan-DiT is architecturally unique (DiT vs UNet); naturally a theoretical/comparative angle |
| Gouri | A6 Compute/efficiency | DreamShaper-turbo is the fast model — natural fit for an efficiency story |
| Vlad | A7 Cross-model comparison | DreamShaper + GenEval gives a balanced view; aggregator role |

Reshuffle freely — these are starting points.

## Coordination contracts

To avoid people stepping on each other:

1. **Single source of truth for results** = `experiments/runs.jsonl` (one row per generated image, all metadata logged). Everyone appends to it.
2. **Implementation Lead delivers `z_sample()` by Day 14.** Until then, others build their model loaders against a stub that returns dummy images.
3. **Evaluation Lead delivers `compute_metric(image, prompt) -> dict` by Day 14.** Until then, others log raw images and metrics are computed in batch later.
4. **Per-person experiment owner** runs their (model, dataset) generation, calls the shared `z_sample()` and shared metric function. They DO NOT reimplement these.
5. **Angle owner** does the cross-cutting analysis on top of everyone's results. They do NOT generate their own data unless their angle requires it (e.g., A5 failure-mode hunting).

## Open seats / decisions

- [ ] Confirm Implementation Lead (Daniel or Lenka)
- [ ] Confirm Evaluation Lead
- [ ] Each person picks their angle (A1-A7)
- [ ] Decide compute pooling: who runs what on whose GPU
- [ ] Confirm midterm presentation date
- [ ] Confirm TA meeting date
