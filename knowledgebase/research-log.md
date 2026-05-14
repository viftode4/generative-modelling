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
