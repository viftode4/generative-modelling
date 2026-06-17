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

## 2026-06-07 - SDXL Pick-a-Pic lambda sweep uses 51 DDIM steps

**Decision:** For the SDXL Pick-a-Pic lambda-step ablation, default to `infer_step=51` when sweeping `lambda_step` values `0`, `3`, `7`, `15`, `25`, `35`, and `50`.

**Rationale:** The repo-local Z-Sampling loop in `src/zigzag_project/zsampling.py` validates `0 <= lambda_step < infer_step` because every refined step needs a following timestep for the DDIM inversion call. Using 51 DDIM steps makes the requested `lambda_step=50` valid while preserving the intended interpretation that 50 early denoising steps receive zigzag refinement. This is an implementation-boundary decision based on the local port, with provenance to the upstream SDXL Z-Sampling adaptation recorded in `src/zigzag_project/zsampling.py`.

**Status:** Active

## 2026-06-03 - Keep local validation environment on NumPy 2.x

**Decision:** Use `numpy>=2,<3` in `pyproject.toml` for the currently validated macOS/MPS environment.

**Rationale:** The active `.venv` already runs `numpy==2.2.6` with `torch==2.12.0`, `diffusers==0.37.1`, `transformers==5.8.1`, and `hpsv2==1.2.0`; `pip check`, `ruff`, and the pytest suite pass after refreshing editable package metadata. The previous `numpy<2` declaration was stale and did not match the environment used for Vlad's scored run or validation tests.

**Status:** Active

## 2026-06-04 - Treat CUDA detector as out of scope for Vlad's local lane

**Decision:** Because Vlad only has the Apple Silicon Mac/MPS environment and no CUDA machine, do not block the individual report on official GenEval detector scoring. Keep the detector gap explicit, and use a 9-step standard DDIM compute-matched baseline as the best local validation if one more experiment is needed.

**Rationale:** The GenEval detector stack requires CUDA-oriented `mmcv`/`mmdet` extension support that failed on the Mac environment. A 9-step standard baseline directly tests the strongest Mac-feasible counterargument: that 4-step Z-Sampling only wins HPS because it spends roughly 2.24x more compute than 4-step standard sampling.

## 2026-06-04 - Use CPU mmdet-2.x path for Vlad GenEval detector

**Decision:** For Vlad's Apple Silicon environment, run GenEval detector scoring with `mmdet==2.28.2`, `mmcv-full==1.7.2`, and `--device cpu`; keep mmdet-3.x support in the runner but do not require it.

**Rationale:** Upstream GenEval instructs users to install mmdetection 2.x. The Mac environment has a working compiled Darwin `mmcv-full` extension for that stack. The previous failure came from targeting mmdet 3.x output format, not from an impossible Mac detector path.

**Result:** Official detector-style scoring completed for Vlad's 54 paired images: Z-Sampling 27/54 vs standard 17/54, +18.5 percentage points.

## 2026-06-04 - Report Vlad's compute-matched baseline as a completed HPS control

**Decision:** Include the 9-step standard DDIM baseline in Vlad's report as a
completed cost-control experiment, but label it HPSv2-only and keep it separate
from the GenEval detector result.

**Rationale:** The 9-step standard baseline is slower than the existing 4-step
Z-Sampling run on MPS (18.14 s/image vs 13.80 s/image), so it is a strong
counterfactual for the "Z only wins because it spends more compute" concern. It
narrows the HPS margin but does not erase it: Z-Sampling wins 42/54 pairs with
mean delta +0.0245 against 9-step standard.

**Status:** Active

## 2026-06-07 - Centralize sampler implementations under zigzag_project.samplers

**Decision:** Move sampler implementations to `src/zigzag_project/samplers/` with separate modules for plain standard sampling, Z-Sampling, and RePaint-style resampling. Keep `src/zigzag_project/zsampling.py` only as a backward-compatible import shim for existing notebooks and scripts.

**Rationale:** The branch integration mixed resampling code into `zsampling.py` and made it easy to pass a `ZSamplingConfig` into a resampling function. A sampler package plus registry keeps method-specific configs explicit while preserving old imports for reproducibility of Vlad's validated artifacts.

**Status:** Active

## 2026-06-07 - Keep zsampling.py as canonical compatibility path

**Decision:** Keep `src/zigzag_project/zsampling.py` as the canonical implementation module for Z-Sampling because teammates already use that import path in notebooks and scripts. Use `src/zigzag_project/samplers/` only as a lightweight facade/guardrail surface for sampler registry/config validation and new resampling helpers.

**Rationale:** A broad move of existing code would create unnecessary churn during group integration. The immediate bug to prevent is cross-method config misuse (for example passing `ZSamplingConfig` to a resampling sampler), not a full module migration.

**Status:** Active

## 2026-06-07 - Preserve merged resampling implementation after rebase

**Decision:** After rebasing onto `origin/main`, keep the merged `resampling_sdxl()` implementation in `src/zigzag_project/zsampling.py` and keep `src/zigzag_project/resampling_utils.py` as its helper/config module. Use `zigzag_project.samplers` only as a thin facade/registry that delegates to those canonical paths.

**Rationale:** The group branch already introduced resampling through the established `zigzag_project.zsampling` path. Duplicating or moving that implementation would create unnecessary integration risk for teammates. The real fix is to prevent config mixups and repair the experiment script to pass `ResamplingConfig` to resampling and `ZSamplingConfig` to Z-Sampling.

**Status:** Active
