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

## 2026-04-30 - Structure project around rubric evidence

**Decision:** Store official project PDFs in `docs/course/`, extracted searchable text in `knowledgebase/course/`, and organize work into `planning/`, `experiments/`, `report/`, and `presentation/`.

**Rationale:** The rubric rewards motivation/work plan, implementation/experimentation content, and writing. The directory structure maps directly to those evidence categories.

**Status:** Active

## 2026-05-14 - Keep Z-Sampling as repo-local utility instead of submodule edits

**Decision:** Implement the first SDXL Z-Sampling path in `src/zigzag_project/zsampling.py` as a repo-local utility that takes `ModelHandle`, rather than modifying `external/zigzag-diffusion-sampling/` or copying its full custom pipeline into this repository.

**Rationale:** The upstream implementation is tracked as an external submodule and should remain a read-only reference. A small utility keeps the shared project API reusable across model cells while preserving provenance to upstream commit `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`.

**Status:** Active

## 2026-05-14 - Use updated sampling-method experiment division

**Decision:** Treat the TA/group notes provided on 2026-05-14 as the current experiment-division source of truth. Vlad owns `Z-Sampling + 8` and uses the prior model/dataset lane `DreamShaper-xl-v2-turbo + GenEval` for the concrete run config.

**Rationale:** The newer notes reflect TA feedback to divide work by sampling method and experiment number, rather than having every person independently run the same Z-vs-standard setup. Capturing this in `planning/experiment-division.md` keeps the runnable Vlad config aligned with the group plan.

**Status:** Active

## 2026-05-14 - Vlad uses balanced GenEval subset with HPSv2 win-rate scoring

**Decision:** Vlad's first report-ready experiment uses 18 GenEval prompts (first 3 from each of 6 tags), 3 seeds (`42`, `123`, `2026`), DreamShaper-xl-v2-turbo settings from the paper (`T=4`, `gamma_1=3.5`, `gamma_2=0`, `lambda_step=3`, `T_max=1`), and HPSv2.1 standard-vs-Z win rate as the primary metric.

**Rationale:** This is small enough to run locally while covering all GenEval categories and matching the paper's preference-metric reporting style. Full GenEval detector evaluation remains out of scope for the first pass.

**Status:** Active

## 2026-05-16 - Implement z_sample_sd21 as a separate function

**Decision:** Create `z_sample_sd21()` in `zsampling.py` as a parallel implementation to `z_sample_sdxl()`, rather than generalizing the SDXL function with conditionals.

**Rationale:** SD2.1 uses `StableDiffusionPipeline` (single CLIP ViT-H/14 text encoder, no `add_text_embeds`/`add_time_ids`, simpler VAE decode) while SDXL uses dual text encoders with additional conditioning kwargs. A clean separation avoids error-prone conditionals and follows the project's "small and focused modules" principle from `src/AGENTS.md`. The core zigzag loop logic is identical — only the pipeline plumbing differs.

**Status:** Active

## 2026-05-16 - Eugen SD2.1 hyperparameters: gamma_1=7.5, 50 steps, 768px

**Decision:** Use `gamma_1=7.5`, `gamma_2=0.0`, `infer_step=50`, `lambda_step=49`, `T_max=1`, `image_size=768` for the SD2.1 + Pick-a-Pic experiment.

**Rationale:** SD2.1's recommended guidance scale is 7.5 (vs SDXL's 5.5). Using the model's native guidance scale isolates the Z-Sampling effect from guidance scale changes. SD2.1 is non-distilled (unlike DreamShaper-turbo) so requires the paper's default 50 DDIM steps. Native resolution is 768×768.

**Status:** Active

## 2026-05-16 - Eugen Pick-a-Pic prompt subset: 20 prompts, 5 categories

**Decision:** Curate 20 prompts from Pick-a-Pic across 5 categories (simple, style, spatial, multi_object, detail), 4 per category, with seeds `42`, `123`, `2026` — producing 60 pairs total.

**Rationale:** This is comparable to Vlad's 18×3=54 pairs while covering diverse prompt difficulty levels representative of real-world user submissions. The paper uses Pick-a-Pic as a benchmark, making this a direct reproduction of their evaluation methodology.

**Status:** Active

