# Research Log

Chronological research notes. Prefer short entries with sources and next actions.

## 2026-04-30

- Initialized project knowledgebase for Group 14.
- Recorded primary paper and code links.
- Source URLs:
  - https://openreview.net/forum?id=MKvQH1ekeY
  - https://openreview.net/pdf?id=MKvQH1ekeY
  - https://github.com/xie-lab-ml/zigzag-diffusion-sampling

Next actions:

- Read PDF and extract equations/algorithm.
- Inspect upstream `infer.py` after cloning reference code locally.
- Decide whether the course deliverable needs reproduction, presentation, or both.

## 2026-04-30 - Upstream code fetched

- Cloned the upstream reference implementation into ignored local path `external/zigzag-diffusion-sampling/`.
- Upstream commit recorded at clone time: `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`.
- This local clone is intentionally not tracked by this repository; use it for inspection/reproduction and record the commit hash in experiment logs.

## 2026-04-30 - Repo setup formalized

- Converted upstream reference access from an ignored local clone into a tracked Git submodule.
- Added Python 3.10 environment setup helpers and inference/check scripts.
- Created a local `.venv` with Python 3.10.18 using `./scripts/setup-zigzag-env.ps1 -SkipPackageInstall`.
- Package installation was intentionally skipped during setup verification to avoid large PyTorch/CUDA downloads until a GPU-ready environment is confirmed.

## 2026-04-30 - Course PDFs added

- Added official project description and project rubric PDFs from Downloads into `docs/course/`.
- Extracted text versions into `knowledgebase/course/` for search and AI-assisted planning.
- Summarized assessment requirements in `knowledgebase/course/requirements-summary.md`.
- Initialized planning, experiments, report, and presentation folders based on the rubric.

## 2026-05-14 - Vlad's local env set up (M4 Max, MPS)

- Hardware: Apple M4 Max (arm64, no CUDA). Using MPS (Metal) as the inference device.
- Python: created `.venv` with CPython 3.10.20 via `uv venv .venv --python 3.10`.
- Submodule `external/zigzag-diffusion-sampling` initialized at upstream commit `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`.
- Installed packages (relaxed from upstream pins to get working MPS support):
  - `torch==2.12.0` (upstream pins `2.0.1`; that version's MPS support is too weak for SDXL inversion).
  - `diffusers==0.37.1` (upstream pins `0.22.3`).
  - `transformers==5.8.1` (upstream pins `4.41.0`).
  - Also: `accelerate`, `safetensors`, `einops`, `torchsde`, `tqdm`, `huggingface_hub`.
- Verified `torch.backends.mps.is_available()` is True and a small MPS matmul runs.
- Added a minimal model loader at `src/zigzag_project/models.py` (`load_dreamshaper_xl_turbo`) and a smoke script at `scripts/smoke_dreamshaper.py`.
- Editable-installed the project: `uv pip install -e .` so `import zigzag_project` works without `PYTHONPATH`.
- Open risk: dependency divergence from upstream pins. Any parity-check vs. upstream needs to account for this. Group should decide whether final results runs use a CUDA box with strict pins or accept MPS divergence.

## 2026-05-14 - DreamShaper-xl-v2-turbo smoke test passes on MPS

- First end-to-end run of `scripts/smoke_dreamshaper.py` on the M4 Max. Full log: `knowledgebase/logs/2026-05-14_dreamshaper-turbo_smoke.md`.
- Load: 217.1 s (first run, includes ~6 GB Hugging Face cache fetch). Generation: 11.4 s for 6 DDIM steps at 768×768 fp16.
- Generated image is a recognizable photoreal small yellow dog on grass — pipeline is qualitatively healthy.
- DDIM + DDIMInverseScheduler attach cleanly to a turbo SDXL checkpoint. This was the open risk for Vlad's cell; Z-Sampling now has a viable scheduler path to build on.
- W1 per-person gate (your model loader works) is met for Vlad. Group-level W1 items still open: metric choice, seeds per (prompt, method), implementation/evaluation lead seats.
- Added a regression net: 7 unit tests in `tests/test_models.py` covering device resolution and `ModelHandle` invariants; pytest + ruff configured via `[project.optional-dependencies].dev` in `pyproject.toml` (`uv pip install -e .[dev]`).

## 2026-05-14 - Repo-local SDXL Z-Sampling smoke path added

- Added a repo-local Z-Sampling utility in `src/zigzag_project/zsampling.py`, adapted from upstream `utils/pipeline_stable_diffusion_xl.py::z_sampling_call` at commit `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`.
- Kept the upstream submodule unchanged, following the project rule to treat `external/zigzag-diffusion-sampling/` as read-only reference code.
- Extended `scripts/smoke_dreamshaper.py` with `--sampler standard|zigzag|both`, shared deterministic initial latents, and Z-Sampling hyperparameter flags.
- Verified fast checks: `ruff check .` passed and `pytest -q` passed with 15 tests.
- Verified a minimal runtime probe on MPS: `scripts/smoke_dreamshaper.py --sampler zigzag --steps 2 --lambda-step 1 --size 512 --seed 42` generated `outputs/smoke/dreamshaper_zigzag_probe.png` in 3.8 s after model load.
- Full smoke log: `knowledgebase/logs/2026-05-14_dreamshaper-zsampling_probe.md`.

Next actions:

- Generate a standard-vs-Z pair with `--sampler both` and report-relevant settings.
- Run parity-oriented validation against upstream SDXL-base on CUDA before making reproduction-quality claims.

## 2026-05-14 - Vlad experiment runner and prompt subset implemented

- Added `planning/experiment-division.md` to capture the updated TA/group division: Vlad owns `Z-Sampling + 8`.
- Added Vlad's balanced GenEval prompt subset at `experiments/prompts/geneval_vlad_balanced.md`, using the first 3 prompts from each GenEval tag in `https://github.com/djghosh13/geneval/blob/main/prompts/evaluation_metadata.jsonl`.
- Added `experiments/configs/dreamshaper_turbo_geneval.md` with the concrete DreamShaper + GenEval settings for Vlad: seeds `42`, `123`, `2026`; `T=4`; `gamma_1=3.5`; `gamma_2=0`; `lambda_step=3`; `T_max=1`; `768x768`; HPSv2.1 scoring.
- Added reusable experiment helpers in `src/zigzag_project/experiment_io.py` for prompt parsing, manifest JSONL writing, HPS row construction, and HPS summary aggregation.
- Added scripts:
  - `scripts/run_vlad_geneval.py` for dry-run/full generation manifest creation.
  - `scripts/score_hpsv2_pairs.py` for optional HPSv2 scoring (`uv pip install hpsv2` required).
  - `scripts/summarize_vlad_results.py` for Markdown result summaries and win/loss gallery pointers.
- Verification: `ruff check .` passed and `pytest -q` passed with 21 tests.
- Dry-run verification: `scripts/run_vlad_geneval.py --prompts experiments/prompts/geneval_vlad_balanced.md --seeds 42 --out outputs/dryrun/vlad_geneval --dry-run` wrote 18 manifest rows without loading the model.

Next actions:

- Run the one-prompt smoke pair at `T=4`, `gamma_1=3.5`, `lambda_step=3`.
- Run the full balanced 18×3 generation job when ready; outputs stay under `outputs/vlad_geneval/balanced_18x3/`.
- Install `hpsv2` only for scoring, then run `scripts/score_hpsv2_pairs.py` and `scripts/summarize_vlad_results.py`.

## 2026-05-14 - Vlad balanced GenEval experiment completed and scored

- Ran Vlad's balanced GenEval experiment with DreamShaper-xl-v2-turbo: 18 prompts × 3 seeds × standard/Z-Sampling = 108 generated PNG images.
- Output directory: `outputs/vlad_geneval/balanced_18x3/` (gitignored, about 91 MB).
- Manifest rows: 54; score rows: 54.
- HPSv2.1 results: Z-Sampling wins 53/54 pairs (98.1% win rate), mean HPS delta +0.0358.
- Per-tag Z win rates: color_attr 100%, colors 100%, counting 100%, position 88.9%, single_object 100%, two_object 100%.
- Runtime totals: standard 332.4 s total / 6.16 s mean per pair; Z-Sampling 745.5 s total / 13.80 s mean per pair (~2.24× standard runtime).
- HPSv2 install note: `uv pip install hpsv2` downgraded `pytest` to 7.2.0 and the package missed `hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz`; downloading the OpenAI CLIP vocab into that venv package path fixed scoring.
- Verification after install/scoring: `ruff check .` passed and `pytest -q` passed with 21 tests / 2 warnings.
- Full run log: `knowledgebase/logs/2026-05-14_vlad-dreamshaper-geneval-balanced.md`.

Next actions:

- Manually inspect the top wins/losses in `outputs/vlad_geneval/balanced_18x3/summary.md` and choose report figures.
- Record report interpretation carefully: these are HPSv2 preference scores on a balanced GenEval subset, not official GenEval detector scores.

## 2026-05-16 - Eugen SD2.1 + Pick-a-Pic infrastructure implemented

- Added `load_sd21()` model loader in `models.py` for `stabilityai/stable-diffusion-2-1`.
- Implemented `z_sample_sd21()` in `zsampling.py` — a parallel Z-Sampling function adapted for SD2.1's simpler architecture (single text encoder, no `add_text_embeds`/`add_time_ids`, simpler VAE decode). Provenance: upstream commit `513eb51`, translated to `StableDiffusionPipeline` internals.
- Made `PromptSpec.prompt_id` prefix configurable (default `"geneval"` preserved for Vlad's backward compat; Eugen uses `"pickapic"`).
- Created balanced Pick-a-Pic prompt subset: 20 prompts across 5 categories (simple, style, spatial, multi_object, detail), 4 per category.
- Created experiment config: `experiments/configs/sd21_standard_pickapic.md` with `gamma_1=7.5`, `infer_step=50`, `lambda_step=49`, `T_max=1`, `768×768`, seeds `42/123/2026`.
- Created runner script: `scripts/run_eugen_pickapic.py` (mirrors `run_vlad_geneval.py`).
- Created smoke test: `scripts/smoke_sd21.py`.
- Added tests: `test_sd21_loader_function_exists()` and `test_z_sample_sd21_is_callable()`.
- Verification: `ruff check .` passed; all 14 Python files parse cleanly; prompt parsing validated (20 prompts, 5 tags balanced).

Next actions:

- Set up local `.venv` with torch + diffusers and run `pytest -q`.
- Run `scripts/smoke_sd21.py --sampler both` to verify SD2.1 pipeline health.
- Run the full experiment: `scripts/run_eugen_pickapic.py` with the config in `experiments/configs/sd21_standard_pickapic.md`.
- Score with HPSv2 and generate summary.
- Write experiment log using `knowledgebase/templates/experiment-log-template.md`.

## 2026-05-25 - Z-Sampling hardening pass: review, fixes, A/B, report tightening

Adversarial critic + code-reviewer pass over the whole repo produced a mixed
set of findings: false alarms (claimed nonexistent torch 2.12.0 / transformers
5.8.1 / Python 3.10.20 — all verified to exist in the live venv) and several
real bugs.

Real fixes applied:

- `pyproject.toml` now declares runtime dependencies (was dev-only). New
  optional `[scoring]` extra pins `hpsv2==1.2.0` separately because it
  downgrades pytest and needs the OpenCLIP BPE vocab patched in by hand.
  Mirrored to `requirements-vlad.txt` for non-editable installs.
- `src/zigzag_project/models.py`: the inverse scheduler is now derived from
  the DDIM config (after the forward scheduler is replaced), not from the
  default Euler/PNDM config. Both `load_dreamshaper_xl_turbo` and `load_sd21`
  fixed.
- `src/zigzag_project/zsampling.py`: `_step_index = None` is reset before
  every `scheduler.step` and `inverse_scheduler.step` call (six sites total
  across `z_sample_sdxl` and `z_sample_sd21`), mirroring upstream
  `pipeline_stable_diffusion_xl.py:1455/1486/1517`. Entry-time assertion
  that `pipe.unet.config.time_cond_proj_dim is None` so LCM-distilled
  checkpoints cannot silently disable CFG. `_decode_latents` stashes and
  restores `pipe.vae.dtype` so it does not mutate the pipe across calls.
- `src/zigzag_project/experiment_io.py`: `write_jsonl` is atomic
  (`.tmp` + `os.replace`); new `append_jsonl` flushes + fsyncs every row so
  long runs preserve partial evidence on crash.
- `scripts/run_vlad_geneval.py`: streams the manifest one row at a time
  via `append_jsonl`, refuses to write into a non-empty `--out` without
  `--force`, calls `torch.{cuda,mps}.synchronize()` around the per-pair
  `time.time()` measurements, gained `--limit` for A/B smokes.
- `scripts/score_hpsv2_pairs.py`: `np.asarray(...).reshape(-1).tolist()`
  normalizes the `hpsv2.score` return shape across point releases; docstring
  now spells out the OpenCLIP BPE vocab curl line.
- `tests/test_vlad_scripts.py`: `sys.executable` replaces hardcoded
  `.venv/bin/python` so the tests work on any venv layout.
- Notebooks `01..03` had output cells stripped (removed embedded
  `/Users/dicisnerosacev/...` and `/home/lenka-hake/...` paths from a
  teammate's earlier execution). `notebooks/03_SD2.1_pickapick_experiment.ipynb`
  dropped the `sys.path.append('../src')` hack and now derives
  `lambda_step = steps - 1` instead of hardcoding 49.
- `scripts/repro-vlad.sh` (new, executable): one-shot driver running
  `run_vlad_geneval.py` → `score_hpsv2_pairs.py` → `summarize_vlad_results.py`.
- `.gitignore`: added comment explaining the 3-level `outputs/<family>/<run>/`
  evidence-re-include convention; re-included `geneval_scores.jsonl`;
  ignored throwaway `outputs/ab_postfix/` directory.
- `knowledgebase/setup.md`: new Apple Silicon / MPS variant section
  pointing at `requirements-vlad.txt` and `scripts/repro-vlad.sh`.
- `experiments/prompts/geneval_vlad_balanced.md`: explicit pin note —
  `source_index` is the upstream `evaluation_metadata.jsonl` line number
  so the set is reconstructible without re-fetching.

A/B verification (Phase B of the fix plan): re-ran `geneval_259_colors`
seed 42 ("a photo of a blue fire hydrant", the top Z-Sampling win) under
the fixed code. Both standard and zigzag PNGs are *bit-identical* to the
committed images (MAE = 0.000, max diff = 0, 100% identical pixels). The
scheduler-config-order and `_step_index` bugs were real but had no
perceptual impact on the 4-step turbo configuration. The committed 108
images and the published HPSv2.1 numbers stand.

GenEval detector attempt (Phase C of the fix plan): the official detector
pipeline (`mmdet>=3` + `mmcv` + Mask2Former) does not build on Apple
Silicon with `torch==2.12.0`. `mim install` polluted the venv (downgraded
setuptools below the torch constraint, triggered `pkg_resources` import
failures during the mmcv wheel build); cleanup restored a clean
`pip check`. The runner script `scripts/run_geneval_detector.py`, the
prompt-rule loader `src/zigzag_project/geneval_io.py`, the summary tool
`scripts/summarize_vlad_geneval.py`, and a mocked smoke test are committed
ready for a CUDA-side rerun. Full attempt log:
`knowledgebase/logs/2026-05-25_geneval-detector-attempt.md`.

Phase D report tightening: `report/draft.md` regenerated with HPSv2.1-vs-
GenEval branding clarified in §4.1, per-tag 95% bootstrap CIs added
(all strictly positive, including position [+0.0153, +0.0374]), figure
pairs converted to two-image side-by-side, HPSv2 saturation reward-hacking
caveat in §5, MPS non-determinism caveat in §5, position-tag explanation
rewritten with absolute standard means (refuting the "less headroom"
alternative — position has the second-highest standard mean), FLOPs-matched
counterfactual made explicit (2.24× cost ≈ 8-9 step standard), new §1.1
Group context section, new §5.5 Unclear-parts-of-the-paper section, and
§2 implementation-details paragraph including the scheduler-order finding.

Verification:

- `.venv/bin/python -m ruff check .` — All checks passed.
- `.venv/bin/python -m pytest -q` — 28 passed (was 23 before the new
  GenEval smoke tests).
- `.venv/bin/python -m pip check` — No broken requirements.

Next actions:

- Find a CUDA box (Colab / runpod / university cluster) and run
  `scripts/run_geneval_detector.py` on the existing 108 images. Update
  the report's §4.1.2 and §5.2 once the detector pass-rate table exists.
- Optionally run a FLOPs-matched standard baseline (8-step standard vs
  4-step Z, 18 prompts × 1 seed) to close the counterfactual gap in §5.4.

## 2026-05-20 - Local macOS dev/test environment repaired after teammate pull

- Pulled `origin/main` to `afb1cda`; teammate branches
  `origin/gourigkrishnan0204-hue-patch-1` and
  `origin/gourigkrishnan0204-hue-patch-2` were already merged into main
  according to `git branch -r --merged origin/main`.
- Existing repo `.venv` had Python 3.10.20 but no `pip`; repaired it with
  `.venv/bin/python -m ensurepip --upgrade`.
- Installed the repo in editable dev mode with `.venv/bin/python -m pip install -e '.[dev]'`.
- Re-aligned package versions after resolver conflicts from the already-installed
  experiment/evaluation stack:
  - `setuptools==81.0.0` satisfies the installed `torch 2.12.0` constraint
    (`setuptools<82`).
  - `pytest==7.2.0` satisfies the installed `hpsv2 1.2.0` and
    `pytest-split 0.8.0` constraints; `pyproject.toml` now pins the dev
    extra to that verified version.
- Cleaned lint-only issues from the pulled code/notebooks: import ordering, star
  imports in notebooks, and long lines. No model-running logic was intentionally changed.
- Verification after cleanup:
  - `.venv/bin/python -m pip check` passed with no broken requirements.
  - `.venv/bin/python -m pytest -q` passed: 23 tests, 2 warnings.
  - `.venv/bin/ruff check .` passed.

## 2026-06-03 - Bogdan Hunyuan-DiT loader scaffold added

- Added `load_hunyuan_dit()` in `src/zigzag_project/models.py` for
  `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers`, using Diffusers'
  `HunyuanDiTPipeline` with DDIM + DDIMInverse schedulers and returning the
  shared `ModelHandle`.
- Added `load_hunyan` as a spelling-tolerant alias for Bogdan's local use.
- This is a loader scaffold only: Hunyuan-DiT still needs a Hunyuan-specific
  standard smoke script and a separate Z-Sampling port before it can run the
  full standard-vs-Z experiment.
- Verification in the current Windows sandbox:
  - `.venv\Scripts\python.exe -m py_compile src\zigzag_project\models.py`
    passed.
  - `.venv\Scripts\python.exe -m py_compile tests\test_models.py` passed.
  - `pytest` was not available in this Windows venv, so the test suite was not
    rerun.

## 2026-06-03 - Bogdan Hunyuan-DiT + Pick-a-Pic comparison scaffold

- Switched Bogdan's practical lane from the earlier planning matrix
  `Hunyuan-DiT + GenEval` toward `Hunyuan-DiT + Pick-a-Pic`, because the paper
  reports Hunyuan-DiT on Pick-a-Pic / DrawBench while GenEval tables are SDXL
  only.
- Added experimental `z_sample_hunyuan_dit()` in
  `src/zigzag_project/zsampling.py`, following Diffusers'
  `HunyuanDiTPipeline.__call__` plumbing: dual text encoders, transformer noise
  prediction, Hunyuan rotary image embeddings, style ids, and image metadata.
- Added `scripts/run_bogdan_hunyuan_pickapic.py`, a dry-run/full-run paired
  standard-vs-Z runner for `experiments/prompts/pickapic_full.md`.
- Added `experiments/configs/hunyuan_dit_pickapic.md` with the concrete command
  set: `gamma_1=5.0`, `gamma_2=0`, `infer_step=50`, `lambda_step=49`,
  `T_max=1`, `image_size=1024`, seeds `42/123/2026`.
- Added lightweight callable/dry-run tests for the new Hunyuan path.
- Verification in the current Windows environment:
  - `py_compile` passed for `src/zigzag_project/zsampling.py`,
    `scripts/run_bogdan_hunyuan_pickapic.py`, `tests/test_zsampling.py`, and
    `tests/test_vlad_scripts.py`.
  - Dry-run passed outside the sandbox:
    `.venv\Scripts\python.exe scripts\run_bogdan_hunyuan_pickapic.py --prompts experiments\prompts\pickapic_full.md --seeds 42 --limit 1 --out outputs\dryrun\bogdan_hunyuan_pickapic --dry-run --force`.
  - Import check passed outside the sandbox for `z_sample_hunyuan_dit` and
    `load_hunyuan_dit`.
- Open risk: no real Hunyuan image generation has been run yet. The next step is
  a 1-prompt, 1-seed, 2-step smoke run on a CUDA box before trusting full
  50-step results.

## 2026-06-03 - Bogdan Hunyuan-DiT smoke blocked on CPU-only hardware

- Attempted the 1-prompt, 1-seed, 2-step Hunyuan-DiT Pick-a-Pic smoke run from
  `experiments/configs/hunyuan_dit_pickapic.md`.
- The Hunyuan-DiT snapshot downloaded successfully from Hugging Face.
- Pipeline loading required extra tokenizer packages; installed
  `sentencepiece`, `tiktoken`, and `protobuf`, and added them to
  `pyproject.toml`.
- Hardware check showed `cuda False` and `mps False`. The smoke then ran on CPU
  for more than 15 minutes without producing images, reached roughly 18 GB RAM,
  and was manually stopped.
- Added a CPU guard to `scripts/run_bogdan_hunyuan_pickapic.py`: real Hunyuan
  generation now refuses CPU by default unless `--allow-cpu` is passed.
- No image pair and no HPSv2 scores were produced. Full log:
  `knowledgebase/logs/2026-06-03_bogdan-hunyuan-pickapic-smoke-attempt.md`.

Next action:

- Move the Hunyuan smoke/full run to a CUDA machine, then score the generated
  manifest with `scripts/score_hpsv2_pairs.py`.

## 2026-06-03 - Bogdan Hunyuan-DiT Pick-a-Pic 18x3 subset completed

- Replaced CPU-only PyTorch with CUDA-enabled `torch==2.12.0+cu130` and
  installed `torchvision==0.27.0+cu130`; verified CUDA sees the NVIDIA GeForce
  RTX 4090.
- Restored `numpy==1.26.4` after the CUDA wheel install pulled `numpy>=2`;
  `uv pip check` then passed.
- Ran a 1-prompt smoke successfully, then ran a report-sized Hunyuan-DiT
  Pick-a-Pic subset: first 18 prompts x seeds `42/123/2026`, standard vs.
  Z-Sampling, 50 DDIM steps, `gamma_1=5.0`, `gamma_2=0`, `lambda_step=49`,
  `T_max=1`, `image_size=1024`.
- Output directory: `outputs/bogdan_hunyuan_pickapic/subset_18x3/`.
- HPSv2.1 results over 54 pairs: Z-Sampling wins 23, standard wins 31, ties 0;
  Z win rate 42.6%; mean HPS delta `Z - standard` = -0.0042. Standard mean
  HPS 0.3195, Z mean HPS 0.3153.
- Runtime: standard total 1169.8 s / 21.66 s mean; Z-Sampling total 3438.8 s /
  63.68 s mean; runtime ratio 2.94x.
- Added report table `report/tables/bogdan_hunyuan_pickapic_results.md` and log
  `knowledgebase/logs/2026-06-03_bogdan-hunyuan-pickapic-subset.md`.
- Verification: `.venv\Scripts\python.exe -m ruff check .` passed and
  `.venv\Scripts\python.exe -m pytest -q` passed with 31 tests / 2 warnings.

## 2026-06-03 - Official ZigZag PDF downloaded and paper notes validated

- Downloaded the official OpenReview PDF for *Zigzag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection* from https://openreview.net/pdf?id=MKvQH1ekeY.
- Saved the unmodified PDF as `docs/papers/zigzag-diffusion-sampling.pdf` and added `docs/papers/README.md` for provenance.
- Validated `knowledgebase/papers/zigzag-diffusion-sampling.md` against the downloaded PDF. The old note was directionally correct but too incomplete for report-critical claims.
- Updated the note with paper-backed algorithm details, experiment settings, main benchmark distinctions, limitations, and a parity/caveat checklist for Vlad's DreamShaper + GenEval run.
- Important finding for report wording: the paper's main GenEval table is SDXL, not DreamShaper; Vlad's DreamShaper + GenEval setup should be framed as the group-assigned cross-cell stress test, not a direct reproduction of the paper's Table 2.

## 2026-06-03 - Group Drive planning and resampling notes incorporated

- Added `knowledgebase/meetings/2026-05-04_ta-meeting-and-drive-notes.md` from user-provided group Drive notes.
- Updated `planning/experiment-division.md` to clarify the sampling-method ownership, TA guidance, and the interpretation of the shared standard-sampling baseline.
- Important interpretation: standard-sampling owners provide baseline infrastructure, but method comparisons still need matched standard baselines for the same model/prompt/seed/settings unless explicitly studying cross-model differences.
- Added the Drive note conclusion that Lenka's non-monotonic RePaint-style schedule is the paper-aligned Resampling implementation, while the earlier Gouri same-timestep perturbation/refinement loop should not be presented as RePaint-equivalent.

## 2026-06-03 - Z-Sampling math/behavior validation pass

- Re-read the paper note, upstream `external/zigzag-diffusion-sampling/utils/pipeline_stable_diffusion_xl.py::z_sampling_call`, and project-local `src/zigzag_project/zsampling.py`.
- Added model-free behavior tests in `tests/test_zsampling.py` that execute the real `z_sample_sdxl` loop with a fake SDXL pipeline. The tests verify:
  - for `T=4`, `lambda_step=T-1`, and `t_max=1`, the forward scheduler sees timesteps `[3, 3, 2, 2, 1, 1, 0]` and the inverse scheduler sees `[2, 1, 0]`, matching the upstream denoise -> inverse to next timestep -> re-denoise pattern;
  - denoise/re-denoise calls use `gamma_1`, while inverse calls use `gamma_2`, proving the classifier-free guidance gap behavior in the loop;
  - `t_max>1` repeats inverse/re-denoise rounds at each refined timestep.
- Audited Vlad's saved run artifacts under `outputs/vlad_geneval/balanced_18x3/`: 54 manifest rows, 54 score rows, 18 prompts, 3 seeds, balanced 9 rows per GenEval tag, no missing images, all images 768x768, no identical standard/Z image pairs, score deltas internally consistent, recomputed Z win rate 53/54 and mean delta +0.035773, runtime ratio 2.242x.
- Fixed report wording found during validation:
  - the algorithm description now states that the inverse/re-denoise pair happens after the first denoise;
  - the scheduler-config note now says the inverse scheduler is cloned from the DDIM forward scheduler rather than the original checkpoint scheduler config;
  - the `geneval_355_position` explanation now preserves the prompt relation "couch below cup" instead of reversing it.
- Reconciled the stale `numpy<2` project dependency with the validated environment by changing it to `numpy>=2,<3` and reinstalling the project editable with `.[dev,scoring]`.
- Verification after changes:
  - `.venv/bin/python -m pip install -e '.[dev,scoring]'` completed and refreshed dependency metadata.
  - `.venv/bin/ruff check .` passed.
  - `.venv/bin/python -m pytest -q` passed: 30 tests, 2 warnings.
  - `.venv/bin/python -m pytest tests/test_zsampling.py -q` passed: 11 tests, 2 warnings.
  - `.venv/bin/python -m pip check` passed with no broken requirements.
  - `git diff --check` passed.
- Remaining validation gap: this proves local loop behavior against upstream structure and validates saved HPS artifacts, but it still does not produce official GenEval detector pass rates or a FLOPs/time-matched standard baseline.

Additional fix from the same validation pass: adding the SD2.1 fake-pipeline behavior test exposed that `z_sample_sd21()` returned `StableDiffusionPipelineOutput(images=image)` without the required `nsfw_content_detected` field under the current `diffusers==0.37.1` API. Fixed `src/zigzag_project/zsampling.py` to return `StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)`. Full verification after the fix: `.venv/bin/python -m pytest -q` passed with 31 tests and 2 warnings; `.venv/bin/ruff check .` passed; `.venv/bin/python -m pip check` passed; `git diff --check` passed.

## 2026-06-03 - Vlad prompt-level and qualitative result strengthening

- Added `report/tables/vlad_prompt_level.md`, aggregating `outputs/vlad_geneval/balanced_18x3/scores.jsonl` by prompt across three seeds.
- Added `report/tables/vlad_qualitative_audit.md`, a manual visual audit of selected top-win and edge-case pairs from `outputs/vlad_geneval/balanced_18x3/images/`.
- Main prompt-level finding: Z-Sampling wins all three seeds for 17/18 prompts; the only mixed prompt is `geneval_355_position` (`a photo of a couch below a cup`), where standard wins one seed by 0.0023 HPS points.
- Manual audit conclusion: the largest HPS wins are visually meaningful (`geneval_259_colors_seed42`, `geneval_180_counting_seed2026`), but small positive HPS deltas can be mostly perceptual sharpening on semantically ambiguous images (`geneval_000_single_object_seed2026`, `geneval_354_position_seed123`). This strengthens the report's metric-mismatch caveat.
- Updated `report/draft.md` and `report/tables/vlad_results.md` to use the safer claim: Vlad's run is a validated DreamShaper-turbo HPSv2.1 preference experiment, not an official GenEval detector reproduction.

## 2026-06-04 - No-CUDA constraint: Mac-only validation path

- User confirmed there is no CUDA machine available for Vlad's lane; only the Apple Silicon Mac/MPS environment is available.
- Consequence: the official GenEval detector remains a documented environment limitation rather than an actionable local next step. The detector runner stays useful for future CUDA access, but it should not be presented as something Vlad can complete locally.
- Added `scripts/run_vlad_compute_matched_standard.py`, a Mac-feasible helper that generates a higher-step standard DDIM baseline from the same prompt/seed/initial latent and writes a comparison manifest compatible with `scripts/score_hpsv2_pairs.py`.
- Added `experiments/configs/dreamshaper_turbo_geneval_compute_matched.md`. Main recommended comparison: existing 4-step Z-Sampling vs 9-step standard DDIM, because the measured MPS runtime ratio is 2.24x and `4 * 2.24 ≈ 9`.
- Updated report wording: CUDA detector is not queued for Vlad; the practical local next experiment is the 9-step compute-matched HPSv2 comparison.

## 2026-06-04 - GenEval detector made to work on Apple Silicon CPU

- Re-diagnosed the GenEval detector instead of accepting the prior CUDA-only conclusion.
- Found current `.venv` had `mmcv-full==1.7.2` with a compiled Darwin extension (`mmcv/_ext.cpython-310-darwin.so`) but lacked `mmdet`.
- Installed `mmdet==2.28.2`, matching upstream GenEval's `mmdetection` 2.x instruction in `external/geneval/README.md`.
- Reproduced the real runner failure: `AttributeError: 'tuple' object has no attribute 'pred_instances'`. Cause: project runner expected mmdet 3.x `DetDataSample`, while working/upstream mmdet 2.x returns `(bbox_results, segm_results)`.
- Patched `scripts/run_geneval_detector.py` to support both mmdet 2.x tuple/list outputs and mmdet 3.x `DetDataSample` outputs. Added `tests/test_run_geneval_detector_smoke.py::test_extract_detections_supports_mmdet2_tuple_output`.
- Ran full detector on CPU: 54 rows, 566.4 s, output `outputs/vlad_geneval/balanced_18x3/geneval_scores.jsonl`.
- Result: standard detector pass rate 31.5% (17/54), Z-Sampling pass rate 50.0% (27/54), delta +18.5pp, with 10 Z-only pairs and 0 standard-only pairs.
- Updated `report/draft.md`, `report/tables/vlad_results.md`, and added `report/tables/vlad_geneval_detector.md` so Vlad's report now includes detector-grounded results.

## 2026-06-04 - Vlad compute-matched standard9 baseline completed

- Completed the Mac/MPS compute-control baseline for Vlad's DreamShaper-turbo
  GenEval lane using `scripts/run_vlad_compute_matched_standard.py`.
- Generated 54 new 9-step standard DDIM images in
  `outputs/vlad_geneval/compute_matched_standard9/`, seeded and initialized to
  match the original prompt/seed rows from
  `outputs/vlad_geneval/balanced_18x3/manifest.jsonl`.
- Scored HPSv2.1 with `scripts/score_hpsv2_pairs.py` and summarized with
  `scripts/summarize_vlad_results.py`.
- Result: existing 4-step Z-Sampling still beats 9-step standard under HPSv2.1
  on 42/54 pairs with mean delta +0.0245. The original 4-step-standard margin
  was 53/54 and +0.0358, so the stronger standard baseline narrows but does not
  erase the effect.
- Runtime result: 9-step standard averages 18.14 s/image versus 13.80 s/image
  for the existing 4-step Z-Sampling run, making the 9-step baseline about
  1.31x slower than Z on this MPS setup.
- Updated `report/draft.md`, `report/tables/vlad_results.md`, and added
  `report/tables/vlad_compute_matched.md` so the report no longer says the
  compute-matched baseline is missing.

## 2026-06-07 - Sampler implementations centralized

- Refactored sampler code into `src/zigzag_project/samplers/`:
  - `standard.py` for plain pipeline calls;
  - `zigzag.py` for SDXL and SD2.1 Z-Sampling;
  - `resampling.py` for the RePaint-style resampling baseline;
  - `registry.py` for sampler names, config classes, and config mismatch validation.
- Kept `src/zigzag_project/zsampling.py` as a backward-compatible import shim so existing notebooks/scripts and Vlad's validated result provenance keep resolving.
- Added `tests/test_samplers_package.py` to lock the central import surface and reject the specific integration error where `ZSamplingConfig` is passed to the resampling sampler.
- Updated current-facing report/source notes to reference `src/zigzag_project/samplers/zigzag.py` as the implementation path.
- Verification after the refactor:
  - `.venv/bin/ruff check .` passed.
  - `.venv/bin/python -m pytest -q` passed: 37 tests, 2 warnings.
  - `.venv/bin/python -m pip check` passed.
  - `git diff --check` passed.

## 2026-06-07 - Sampler centralization scope reduced for teammate compatibility

- Restored `src/zigzag_project/zsampling.py` as the real Z-Sampling implementation path so existing teammate scripts and notebooks do not need import changes.
- Kept the new `zigzag_project.samplers` package as a small facade/registry and resampling guardrail surface, not as a replacement for established imports.
- Reverted current runnable scripts to import Z-Sampling from `zigzag_project.zsampling`.
- Kept tests that lock the optional facade and catch the config mismatch that caused the resampling integration risk.
- Verification after reducing the sampler refactor scope:
  - `.venv/bin/ruff check .` passed.
  - `.venv/bin/python -m pytest -q` passed: 37 tests, 2 warnings.
  - `.venv/bin/python -m pip check` passed.
  - `git diff --check` passed.

## 2026-06-07 - Rebased Vlad work onto merged group main

- Fast-forwarded local `main` from `7c5d058` to `origin/main` at `ee0a260` after stashing Vlad's local work.
- Resolved the only content conflict in `src/zigzag_project/zsampling.py` by preserving the merged SDXL resampling implementation and retaining the SD2.1 return fix: `StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)`.
- Found and fixed a real integration issue in `scripts/run_exp2_sdxl_geneval.py`: the script created `ZSamplingConfig` and passed it into `resampling_sdxl()`, but the merged resampling implementation expects `ResamplingConfig` with `resolved_jump_params()`.
- Added a direct type guard in canonical `resampling_sdxl()` so `ZSamplingConfig` misuse fails before model/pipeline access.
- Reduced `zigzag_project.samplers.resampling` to a facade over the merged canonical implementation rather than a duplicate resampling implementation.
