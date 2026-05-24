# Vlad Experiment Log — DreamShaper-xl-v2-turbo + GenEval (Balanced 18x3)

## Metadata

- **Date:** 2026-05-14
- **Owner:** Vlad Iftode
- **Goal:** Reproduce the qualitative direction of Zigzag Diffusion Sampling
  (Bai et al., openreview `MKvQH1ekeY`) on `Lykon/dreamshaper-xl-v2-turbo`
  using a balanced 18-prompt subset of GenEval, three seeds per prompt, and
  HPSv2.1 preference scoring as a stand-in metric. Compare standard DDIM
  sampling against Z-Sampling from shared initial latents.
- **Source code commit:** working tree based on `7fea0cc` (`source_commit`
  field in `manifest.jsonl`).
- **Upstream commit:** `external/zigzag-diffusion-sampling` pinned at
  `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`.
- **Environment:** `.venv` CPython 3.10.20; `torch==2.12.0`;
  `diffusers==0.37.1`; `transformers==5.8.1`; `hpsv2==1.2.0` (vendored
  OpenCLIP BPE file patched in from `openai/CLIP`); `pytest==7.2.0`.
  Upstream pins `torch==2.0.1`, `diffusers==0.22.3`, `transformers==4.41.0`
  on CUDA — this run is deliberately divergent for MPS support.
- **Hardware:** Apple M4 Max, arm64, MPS backend, fp16 inference. No CUDA.
- **Seed(s):** `42`, `123`, `2026`. Initial latents are created once per
  seed and shared by the standard and Z-Sampling paths within a pair.

## Command(s)

Generation:

```bash
.venv/bin/python scripts/run_vlad_geneval.py \
  --prompts experiments/prompts/geneval_vlad_balanced.md \
  --seeds 42 123 2026 \
  --steps 4 \
  --cfg 3.5 \
  --inv-cfg 0 \
  --lambda-step 3 \
  --t-max 1 \
  --size 768 \
  --out outputs/vlad_geneval/balanced_18x3
```

HPSv2.1 scoring and summary:

```bash
uv pip install hpsv2
curl -L -s --fail https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
  -o .venv/lib/python3.10/site-packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
.venv/bin/python scripts/score_hpsv2_pairs.py \
  --manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/scores.jsonl
.venv/bin/python scripts/summarize_vlad_results.py \
  --scores outputs/vlad_geneval/balanced_18x3/scores.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/summary.md
```

Verification:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

## Inputs

- Prompt subset: `experiments/prompts/geneval_vlad_balanced.md`. 18 prompts
  total; the first 3 prompts from each of the 6 GenEval tags
  (`single_object`, `two_object`, `counting`, `colors`, `position`,
  `color_attr`).
- Model: `Lykon/dreamshaper-xl-v2-turbo` loaded by
  `zigzag_project.models.load_dreamshaper_xl_turbo` on MPS, fp16.
- Sampler config (matches the DreamShaper turbo setting in paper §4.1):
  `gamma_1=3.5`, `gamma_2=0.0`, `infer_step=4`, `lambda_step=3`,
  `t_max=1`, image size `768x768`. Z-Sampling refinement therefore runs
  along the full denoising path with one zag/zig round per refined step.
- Metric: `hpsv2.score(..., hps_version="v2.1")` per generated image.

## Outputs

- Manifest: `outputs/vlad_geneval/balanced_18x3/manifest.jsonl` (54 rows).
- Images: `outputs/vlad_geneval/balanced_18x3/images/` (108 PNGs, ~91 MB).
- Per-pair scores: `outputs/vlad_geneval/balanced_18x3/scores.jsonl`
  (54 rows; fields `standard_score`, `zigzag_score`, `delta`).
- Aggregated summary: `outputs/vlad_geneval/balanced_18x3/summary.md`.
- Result table for the report: `report/tables/vlad_results.md`.
- Selected figure pairs: `report/figures/vlad_geneval/`.

## Observations

- All 18 x 3 x 2 = 108 generations completed without crashes or NaNs.
- Pairwise HPSv2.1 result: Z-Sampling wins 53 / 54 pairs (98.1%). Mean
  HPS delta (Z - standard) is +0.0358; min -0.0023, max +0.1090.
- HPS score scale is small (standard mean 0.217, Z mean 0.253; both well
  under the ~0.30 ballpark HPSv2 typically reports for stronger SDXL
  configurations), so absolute scores should be treated as relative.
- Per-tag Z win rates: color_attr 100%, colors 100% (largest mean
  delta +0.0577), counting 100%, position 88.9% (the only loss tag),
  single_object 100%, two_object 100%.
- Runtime per pair: standard 6.16 s mean, Z-Sampling 13.80 s mean. Total
  generation wall time was 332.4 s standard vs 745.5 s Z-Sampling, giving
  a 2.24x cost ratio. This matches the upstream paper's complexity
  expectation (each refined step inverts then re-denoises).
- The single Z-Sampling loss is `geneval_355_position` at seed 42
  ("a photo of a couch below a cup", delta -0.0023). Spatial-relation
  prompts are also the lowest-mean-delta tag overall, consistent with
  the observation that HPSv2 is mostly an aesthetic/CLIP-like preference
  proxy and does not strongly check spatial layout.
- Tooling issues encountered and fixed:
  - `hpsv2==1.2.0` ships without `bpe_simple_vocab_16e6.txt.gz` in its
    vendored OpenCLIP path; downloaded the file from the OpenAI CLIP
    repo into the expected venv site-packages location.
  - Installing `hpsv2` downgraded `pytest` from 9.0.3 to 7.2.0; the
    project test suite still passes at that version.
  - Modern diffusers emits deprecation warnings about scheduler/config
    fields and `upcast_vae`; none affect output.

## Interpretation

The balanced GenEval subset strongly favors Z-Sampling under HPSv2.1
scoring on this turbo model, at the cost of roughly 2x runtime per
image. The direction matches Bai et al.'s claim that sampling-time
zigzag refinement improves perceptual quality, but two caveats are
critical:

1. HPSv2.1 is a learned human-preference proxy, not the GenEval
   detector-grounded compositional metric the paper reports. A 98.1%
   HPS win rate is not the same as a 98.1% GenEval score gain. The
   most likely overlap is with the paper's HPS-based ablations, not
   with the GenEval table.
2. The dependency stack diverges from upstream pins. Any quantitative
   comparison to the paper's numbers should be considered indicative
   rather than a strict reproduction.

The single position-tag loss and the lower mean delta on `position`
also suggest that Z-Sampling's gains here are largely perceptual /
texture-level and not specifically spatial-reasoning fixes.

## Follow-up

- Run the official GenEval detector pipeline on the same 108 images to
  produce a true GenEval comparison and avoid conflating HPS with
  detector scores in the report.
- If a CUDA machine becomes available, re-run the same prompt subset
  with upstream pins (`torch==2.0.1`, `diffusers==0.22.3`) and compare
  against this MPS run to estimate environment-induced variance.
- Add at least one bigger sweep over `lambda_step` (e.g. 1, 2, 3) on
  the 18 prompts to confirm that "zigzag along the whole path" is
  necessary, or whether shallow zigzag is enough on a 4-step turbo.

## 2026-05-25 Addendum — code fixes verified bit-identical

A subsequent review pass (`knowledgebase/research-log.md` 2026-05-25) found
two real bugs in the Z-Sampling implementation:

1. `DDIMInverseScheduler.from_config(pipe.scheduler.config)` in
   `src/zigzag_project/models.py` was being called *before* the forward
   scheduler was replaced with `DDIMScheduler`, so inversion was being
   parameterized off the default Euler config.
2. `src/zigzag_project/zsampling.py` did not reset
   `scheduler._step_index = None` before repeated `.step(...)` calls at
   the same timestep, which can mis-advance internal state in
   `diffusers>=0.26`.

Both fixes are applied. A/B re-ran `geneval_259_colors` seed 42 (the
"blue fire hydrant" top-Z win) under the fixed code and compared the
PNGs against the committed images:

- Standard: MAE = 0.000, max diff = 0, 100% identical pixels.
- Z-Sampling: MAE = 0.000, max diff = 0, 100% identical pixels.

The two bugs were real but had **no perceptual impact** on this 4-step
turbo configuration. The committed 108-image set, the 53/54 HPSv2 win
count, and all numbers in `summary.md` therefore stand without revision.
