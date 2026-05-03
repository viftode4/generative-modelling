# RALPLAN-DR: Zigzag Diffusion Sampling -- Course Project Plan

**Date:** 2026-05-03
**Course:** Generative Modelling -- Group 14 (7 students)
**Paper:** Zigzag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection (ICLR 2025)
**Mode:** DELIBERATE (high-stakes graded work)

---

## 1. RALPLAN-DR Summary

### 1.1 Principles

1. **Experiment-heavy, code-light.** The rubric awards 40% for Content (results + analysis). Our reimplementation should be minimal and correct, freeing time for systematic experiments and deep interpretation.
2. **Differentiated narratives.** Seven individual reports must each tell a distinct story. Shared infrastructure, unique angles.
3. **Reproducibility as evidence.** Every generated image must be traceable to a specific (prompt, seed, method, hyperparameters, commit, GPU) tuple via a JSONL run log. This directly supports the rubric's "strong troubleshooting" criterion.
4. **Binary checkpoints, not hopes.** Each phase ends with a pass/fail gate. If a gate fails, the group pivots before wasting downstream effort.
5. **Minimal viable scope.** No architecture redesign, no model retraining. Z-Sampling is a sampling-time modification. Our reimplementation wraps diffusers' existing SDXL pipeline with a clean z-sampling loop, nothing more.

### 1.2 Decision Drivers (Top 3)

| # | Driver | Why it dominates |
|---|--------|-----------------|
| 1 | **Rubric alignment** | Content (40%) + Writing (30%) = 70% of the grade comes from experiments, analysis, and clear writing -- not from code volume. |
| 2 | **Compute scarcity** | SDXL at 1024px costs ~20s/image on A100. Z-Sampling triples UNet calls per step. GPU access across 7 students is uncertain. Every experiment-design decision must respect a ~500-image total budget. |
| 3 | **TA expectation of from-scratch code** | The TA explicitly said not to copy upstream. We must reimplement the core loop ourselves, using the upstream only as a correctness oracle. |

### 1.3 Viable Options

#### Option A: Pipeline-wrapping approach (SELECTED)

Write a standalone `z_sample()` function that accepts a loaded `diffusers.StableDiffusionXLPipeline` and performs the zigzag loop externally, using the pipeline's UNet, scheduler, and encoder as black boxes.

| Pros | Cons |
|------|------|
| Clean separation: our code vs. library code | Cannot intercept internal pipeline state without some scheduler manipulation |
| Easy to unit-test the zigzag logic in isolation | Must manually manage `scheduler._step_index` reset (a known diffusers quirk) |
| Portable to future diffusers versions | Slightly more boilerplate for prompt encoding setup |
| TA-friendly: clearly our own work | |

#### Option B: Custom pipeline subclass

Subclass `StableDiffusionXLPipeline` and override `__call__` to insert the zigzag steps inline, similar to what the upstream repo does.

| Pros | Cons |
|------|------|
| Closer structural match to upstream (easier correctness comparison) | Couples us to a specific diffusers version (0.22.3 is old) |
| Single entry point for both standard and z-sampling | Hard to unit-test the zigzag logic independently |
| | TA might view it as too close to upstream's approach |
| | Brittle to diffusers API changes |

**Invalidation rationale for Option B:** The TA explicitly warned against copying upstream structure. A subclass of the same pipeline class, overriding the same method, risks looking derivative even if rewritten. Option A creates a visibly different architecture (external function vs. internal method) while being easier to test and more portable.

### 1.4 Pre-mortem (5 failure scenarios)

#### Scenario 1: Implementation parity fails at Day 14

**What happens:** Our z_sample() produces visually different images from upstream on the same (prompt, seed, gamma_1, gamma_2, lambda_step) tuple.
**Root cause:** Subtle differences in scheduler state management, noise prediction ordering, or floating-point accumulation.
**Mitigation:** Compare intermediate latents at each timestep (not just final images). Log L2 distance between our latents and upstream latents per step. The bug will localize to the first divergent step. Budget 3 days of debugging time for this.

#### Scenario 2: GPU access collapses

**What happens:** Only 1-2 students have reliable GPU access. The experiment matrix cannot be completed in time.
**Root cause:** University cluster queue saturation, personal GPU failures, or cloud credits exhausted.
**Mitigation:** (a) Pre-compute a "reference bank" of ~50 standard+z-sampling image pairs on one GPU early (Week 2-3). (b) Design experiments so that analysis-heavy roles (failure-mode analysis, prompt-category analysis) can work from the shared image bank. (c) Reduce image resolution to 512px for ablation sweeps (document the trade-off). (d) Use Google Colab T4 as fallback (slower but free).

#### Scenario 3: Reports converge to the same story

**What happens:** Despite assigned angles, multiple reports end up with the same figures, same analysis, same conclusions. TA flags lack of differentiation.
**Root cause:** Angles were too vaguely defined; students defaulted to "here are my results, they look better."
**Mitigation:** Each angle is assigned a UNIQUE figure/table type that no other report may include. The angle owner produces that artifact; others may reference it but not reproduce it. Enforce this at the Week 5 draft-review checkpoint.

#### Scenario 4: Cross-version parity impossible

**What happens:** Even with correct logic, modern diffusers' DDIMScheduler differs from 0.22.3 (e.g., `_step_index` semantics changed between 0.22 and 0.27). Our z_sample() passes unit tests but produces different images from upstream because the scheduler internals have shifted.
**Root cause:** Diffusers version drift -- internal implementation details that are not part of the public API have changed across releases.
**Mitigation:** Dual-environment strategy (see Section 6 Phase 1 and Risk 1). Run the parity test in `upstream-env` (pinned to `diffusers==0.22.3`) to isolate version drift from logic bugs. If parity passes in upstream-env but fails in project-env, the difference is a version issue, not a logic bug -- document and move on.

#### Scenario 5: Paper claims don't replicate

**What happens:** Z-Sampling shows <=50% winning rate on our 30-prompt set, contradicting the paper's claimed improvements.
**Root cause:** The paper's prompt distribution may be cherry-picked, or improvements may depend on specific hyperparameter tuning not fully described.
**Mitigation:** This is a publishable FINDING, not a failure. The rubric explicitly rewards critical appraisal. Reframe each report's narrative: "We found Z-Sampling does/does not generalize beyond the paper's prompt distribution because X." Document in `knowledgebase/decision-log.md` as soon as it appears. A rigorous negative result with analysis is more valuable than an uncritical positive replication.

### 1.5 Expanded Test Plan

#### Unit tests

| Test | What it verifies | Pass criterion |
|------|-----------------|----------------|
| `test_inv_scheduler_round_trip` | DDIMInverseScheduler.step() followed by DDIMScheduler.step() (or vice versa) returns the original latent within fp16 tolerance | L2 < 1e-3 |
| `test_z_sample_reduces_to_standard` | When `lambda_step=0`, z_sample output matches standard DDIM output | L2 < 1e-3 between latent tensors (or LPIPS < 0.01 on decoded images) |
| `test_z_sample_deterministic` | Same (seed, prompt, params) always produces same output | Byte-identical PNG |
| `test_guidance_gap_zero` | When gamma_1 == gamma_2, z-sampling degenerates gracefully | No NaN, no crash |
| `test_t_max_iterations` | T_max=3 produces different output from T_max=1 | Images differ (L2 > threshold) |

#### Integration tests

| Test | What it verifies |
|------|-----------------|
| `test_upstream_parity` | Our z_sample on reference prompt+seed produces image within LPIPS < 0.05 of upstream output |
| `test_run_log_written` | Every generate call appends a valid JSON object to `experiments/runs.jsonl` with all required fields |
| `test_clip_score_computed` | CLIP score computation runs end-to-end and produces a float in [0, 1] |

#### End-to-end tests

| Test | What it verifies |
|------|-----------------|
| `test_full_experiment_run` | `run_experiment.py --prompt "a cat" --methods standard,zsample --seeds 42` produces 2 images, 2 JSONL entries, and 2 metric entries |
| `test_ablation_sweep` | Sweep over 3 gamma_1 values completes and produces a comparison grid |

#### Observability

- Every image generation logs: prompt, seed, method, all hyperparameters, wall-clock seconds, peak GPU memory MB, commit hash, GPU name.
- `experiments/runs.jsonl` is the single source of truth for all results (see Section 9).

---

## 2. Selected Approach: Pipeline-Wrapping with External Z-Sample Function

### Architecture decision record (ADR)

**Decision:** Implement Z-Sampling as a standalone function `z_sample()` that operates on a loaded diffusers SDXL pipeline, rather than subclassing the pipeline.

**Drivers:**
1. TA requirement to not copy upstream code structure.
2. Testability -- the zigzag logic can be unit-tested with mock UNet forward passes.
3. Portability -- works with any diffusers version that exposes `DDIMScheduler` and `UNet2DConditionModel`.

**Alternatives considered:**
- Pipeline subclass (Option B): rejected because it mirrors upstream structure too closely and couples to a specific diffusers version.
- Pure from-scratch (no diffusers): rejected because reimplementing VAE, UNet, text encoders, scheduler from scratch adds no learning value for the course and is infeasible in the timeline.

**Why chosen:** Cleanest separation of concerns. Our original work (the zigzag loop, experiment harness, evaluation) is clearly distinguishable from library code. Easiest to explain in reports.

**Consequences:**
- We must handle prompt encoding, scheduler setup, and latent preparation in our wrapper code.
- We depend on diffusers' `DDIMScheduler.step()` and `DDIMInverseScheduler.step()` for the forward/backward DDIM steps.
- Scheduler `_step_index` must be manually reset before each step call (a known diffusers implementation detail).

**Follow-ups:**
- Pin `diffusers==0.30.0` in `environments/project-env.yml` (see dual-environment strategy in Section 6 Phase 1).
- Lock specific `torch`, `transformers`, and `accelerate` versions in `environments/project-env.yml` for reproducibility.
- Document the scheduler quirk in code comments and in the implementation-focused report.

---

## 3. Detailed Implementation Plan

### 3.1 File/module layout

```
src/
  zsample/
    __init__.py
    core.py              # z_sample(), standard_sample()
    pipeline_loader.py   # load_sdxl_pipeline(), encode_prompt()
    metrics.py           # compute_clip_score(), compute_hpsv2()
    utils.py             # seed_everything(), get_gpu_info(), timer context manager, log_run()

scripts/
  run_experiment.py      # CLI: --prompts-file --methods --seeds --params -> images + JSONL log
  run_ablation.py        # CLI: --param gamma_1 --range 3,5.5,7,9 -> sweep + grid
  run_evaluation.py      # CLI: reads runs.jsonl, computes metrics, appends scores to JSONL
  compare_upstream.py    # CLI: runs upstream and ours side-by-side for parity check (upstream-env only)

environments/
  upstream-env.yml       # pins diffusers==0.22.3, torch==2.0.1 (for parity checks only)
  project-env.yml        # pins diffusers==0.30.0, torch + transformers + accelerate (for all other work)

experiments/
  prompts/
    prompts.csv          # prompt_id, text, category, source
  runs.jsonl             # THE results ledger (see Section 9)
  images/                # {method}_{prompt_id}_{seed}_{param_hash}.png

tests/
  test_core.py
  test_metrics.py
  conftest.py            # shared fixtures (mock UNet, small latents)
```

### 3.2 What is reimplemented vs. reused from diffusers

| Component | Source | Rationale |
|-----------|--------|-----------|
| UNet2DConditionModel | **diffusers** (pretrained SDXL) | No value in reimplementing a 3.5B-parameter architecture |
| VAE encoder/decoder | **diffusers** (pretrained SDXL) | Same |
| CLIP text encoders | **diffusers** (pretrained SDXL) | Same |
| DDIMScheduler.step() | **diffusers** | Standard DDIM step is well-established math |
| DDIMInverseScheduler.step() | **diffusers** | Standard DDIM inversion is well-established math |
| `z_sample()` loop | **Reimplemented** | The entire zigzag logic is our original work |
| `standard_sample()` | **Reimplemented** | Baseline comparator; thin wrapper around scheduler loop |
| Prompt encoding wrapper | **Reimplemented** | Handles dual CLIP encoders for SDXL |
| JSONL run logging | **Reimplemented** | Experiment infrastructure |
| Metric computation | **Reimplemented** (wrapping `clip-score`, `hpsv2` libraries) | Evaluation layer |

> **Note on `backward_ddim()`:** The upstream repo defines `_backward_ddim()` at `pipeline_stable_diffusion_xl.py:74` but NEVER calls it. The inversion step uses `DDIMInverseScheduler.step()` from diffusers. We follow the same approach -- our zigzag zag-step calls `inv_scheduler.step()` directly. There is no separate `backward_ddim()` function in our codebase.

### 3.3 Pseudocode of the core algorithm

> **CFG mechanics note:** Each `unet_predict` call below is an abstraction. Internally it: (1) doubles the latent via `torch.cat([latents, latents])` for classifier-free guidance, (2) calls `scheduler.scale_model_input(latent_model_input, t)`, (3) runs the UNet, (4) returns the doubled noise prediction (unconditional + conditional). See upstream `pipeline_stable_diffusion_xl.py` lines 1430-1443 for the explicit form.

```python
def z_sample(
    unet, scheduler, inv_scheduler,
    prompt_embeds, negative_prompt_embeds,
    added_cond_kwargs,
    latents,              # x_T: initial noise
    timesteps,            # [T, T-1, ..., 1] from scheduler
    gamma_1: float,       # denoising guidance scale (default 5.5)
    gamma_2: float,       # inversion guidance scale (default 0.0)
    lambda_step: int,     # zigzag cutoff (default 49 of 50)
    t_max: int,           # zigzag iterations per step (default 1)
) -> Tensor:
    """
    Zigzag Diffusion Sampling.

    For each timestep t in [T..1]:
      1. DENOISE: x_t -> x_{t-1}  with guidance gamma_1
      2. If step_index < lambda_step, repeat t_max times:
         a. ZAG (invert):  predict noise at t using x_{t-1}, apply inv_scheduler
                           with guidance gamma_2 to get x_t'
         b. ZIG (denoise): predict noise at t using x_t', apply scheduler
                           with guidance gamma_1 to get x_{t-1}'
         c. x_{t-1} = x_{t-1}'
      3. x_t = x_{t-1}   (advance to next timestep)
    """
    for i, t in enumerate(timesteps):
        # --- STANDARD DENOISE STEP ---
        # CFG: double latents for unconditional + conditional pass
        # matches upstream lines 1430-1435
        latent_model_input = torch.cat([latents, latents])
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,
                          added_cond_kwargs=added_cond_kwargs).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + gamma_1 * (noise_pred_cond - noise_pred_uncond)
        # matches upstream lines 1436-1443

        scheduler._step_index = None          # diffusers quirk: reset index
        x_denoised = scheduler.step(noise_pred, t, latents).prev_sample

        # --- ZIGZAG REFINEMENT ---
        # matches upstream lines 1445-1470
        if i < lambda_step:
            for _ in range(t_max):
                # ZAG: invert x_{t-1} back to x_t' using gamma_2
                # matches upstream lines 1448-1455
                latent_model_input_inv = torch.cat([x_denoised, x_denoised])
                latent_model_input_inv = scheduler.scale_model_input(latent_model_input_inv, t)
                noise_pred_inv = unet(latent_model_input_inv, t,
                                      encoder_hidden_states=prompt_embeds,
                                      added_cond_kwargs=added_cond_kwargs).sample
                noise_pred_inv_uncond, noise_pred_inv_cond = noise_pred_inv.chunk(2)
                noise_pred_inv = noise_pred_inv_uncond + gamma_2 * (noise_pred_inv_cond - noise_pred_inv_uncond)

                inv_scheduler._step_index = None
                x_inverted = inv_scheduler.step(
                    noise_pred_inv, timesteps[i + 1], x_denoised
                ).prev_sample

                # ZIG: denoise x_t' back to x_{t-1}' using gamma_1
                # matches upstream lines 1458-1468
                latent_model_input_zig = torch.cat([x_inverted, x_inverted])
                latent_model_input_zig = scheduler.scale_model_input(latent_model_input_zig, t)
                noise_pred_zig = unet(latent_model_input_zig, t,
                                      encoder_hidden_states=prompt_embeds,
                                      added_cond_kwargs=added_cond_kwargs).sample
                noise_pred_zig_uncond, noise_pred_zig_cond = noise_pred_zig.chunk(2)
                noise_pred_zig = noise_pred_zig_uncond + gamma_1 * (noise_pred_zig_cond - noise_pred_zig_uncond)

                scheduler._step_index = None
                x_denoised = scheduler.step(
                    noise_pred_zig, t, x_inverted
                ).prev_sample

        latents = x_denoised

    return latents  # decode with VAE to get image
```

**Key insight for reports:** Each zigzag step at timestep t costs 2 additional UNet forward passes (one for zag-inversion, one for zig-denoising). With T_max=1 and lambda_step=49 of 50, this means 49*2 = 98 extra UNet calls on top of the 50 standard calls, roughly tripling inference time. This is the core compute/quality trade-off the paper claims is worthwhile.

---

## 4. Experiment Plan

### 4.1 Prompt dataset design

**Total prompts: 30** (manageable within compute budget)

| Category | Count | Sourcing | Rationale |
|----------|-------|----------|-----------|
| Compositional | 6 | DrawBench subset | Tests multi-object composition ("a red cube on top of a blue sphere") |
| Style transfer | 5 | Custom | Tests artistic style adherence ("in the style of Van Gogh") |
| Fine-grained text | 4 | Custom | Tests text rendering ("a sign that says HELLO") |
| Counting | 4 | Custom | Tests numerical accuracy ("exactly three cats") |
| Spatial reasoning | 4 | T2I-CompBench subset | Tests spatial relations ("to the left of") |
| Simple/baseline | 4 | Upstream repo prompts + simple nouns | Sanity checks ("a small yellow dog") |
| Adversarial/hard | 3 | Custom | Deliberate failure-prone prompts ("a horse riding an astronaut") |

Stored in `experiments/prompts/prompts.csv` with columns: `prompt_id, text, category, source`.

### 4.2 Experiment matrix

**Methods (2):**
- Standard DDIM sampling (baseline)
- Z-Sampling with paper defaults (gamma_1=5.5, gamma_2=0, lambda_step=49, T_max=1)

**Seeds (3):** 42, 123, 7 (for statistical variance)

**Core matrix:** 30 prompts x 2 methods x 3 seeds = **180 images**

Budget allocation:
- Core matrix: 180 images
- Ablation sweeps: ~250 images
- Debugging/parity checks: ~50 images
- Buffer: ~20 images
- **Total: ~500 images** (feasible across 7 students sharing GPU time)

### 4.3 Ablation plan

Each ablation varies ONE parameter while holding others at paper defaults. Each ablation uses **2 seeds** (42, 123) for statistical variance.

| Ablation | Parameter | Values | Images per value |
|----------|-----------|--------|-----------------|
| A1: Guidance gap | gamma_2 | {0, 1, 2.75, 4, 5.5} | 5 prompts x 2 seeds = 10 |
| A2: Denoising guidance | gamma_1 | {3, 5.5, 7.5, 10, 15} | 5 prompts x 2 seeds = 10 |
| A3: Zigzag cutoff | lambda_step | {10, 25, 35, 49} | 5 prompts x 2 seeds = 10 |
| A4: Iteration count | T_max | {1, 2, 3, 5} | 5 prompts x 2 seeds = 10 |
| A5: Inference steps | num_inference_steps | {20, 30, 50} | 5 prompts x 2 seeds = 10 |
| **Ablation total** | | | **~250 images** |

For each ablation, the 5 prompts are drawn from different categories to avoid category bias.

### 4.4 Evaluation metrics

| Metric | What it measures | Implementation | Used by paper? |
|--------|-----------------|----------------|----------------|
| **HPSv2** | Human preference alignment | `hpsv2` library (pre-trained scoring model) | YES (primary metric) |
| **CLIP score** | Text-image alignment | `torchmetrics` or `clip-score` library | YES |
| **Qualitative grid** | Visual inspection of quality, artifacts, coherence | Side-by-side grids (standard vs. z-sampling) | YES |
| **Wall-clock time** | Computational cost | `time.perf_counter()` around generation | Discussed but not tabled |
| **Peak GPU memory** | Memory footprint | `torch.cuda.max_memory_allocated()` | Not measured |
| **LPIPS** (optional) | Perceptual distance between standard and z-sampling | `lpips` library | No, but useful for analysis |

**Winning rate calculation:** For each (prompt, seed) pair, compare HPSv2 of z-sampling vs. standard. Report % of pairs where z-sampling wins. This directly replicates the paper's primary result format.

---

## 5. Seven-Person Task Allocation

Each person owns a distinct narrative angle that yields one unique figure or table for their individual report. All seven share the same codebase and run log data.

| Person | Role Title | Narrative Angle | Unique Report Artifact | Implementation Responsibility |
|--------|-----------|-----------------|----------------------|-------------------------------|
| **P1** | Theory Lead | Mathematical derivation of guidance gap and why zigzag accumulates semantic information | **Figure: Theoretical diagram** showing the zigzag trajectory in latent space with guidance vectors, plus derivation of Eq. 4-7 from the paper AND a **latent-trajectory PCA visualization** (3 prompts, intermediate latents at each zigzag step projected to 2D, ~15 images absorbed into buffer allocation) | Write `knowledgebase/papers/` deep notes; derive the math for the group |
| **P2** | Implementation Lead | Engineering deep-dive into the reimplementation: scheduler quirks, diffusers internals, code design decisions | **Table: Implementation decision log** (what we built vs. upstream, why each choice, what broke and how we fixed it) | Owns `src/zsample/core.py` and `src/zsample/pipeline_loader.py`; drives parity testing |
| **P3** | Hyperparameter Analyst | Ablation study across gamma_1, gamma_2, lambda_step, T_max | **Figure: Ablation heatmap/grid** showing HPSv2 and CLIP score as functions of hyperparameter pairs | Owns `scripts/run_ablation.py`; runs ablations A1-A4 |
| **P4** | Prompt-Category Analyst | How Z-Sampling's benefit varies across prompt categories (compositional, style, counting, spatial, text) | **Figure: Per-category bar chart** of winning rates and metric deltas | Designs prompt dataset (`experiments/prompts/prompts.csv`); runs category analysis |
| **P5** | Failure-Mode Analyst | When and why Z-Sampling fails, degrades, or shows no improvement; includes edge cases | **Figure: Failure case gallery** with annotated examples showing artifacts, mode collapse, or degradation | Identifies failure prompts; runs diagnostic experiments with intermediate latent logging |
| **P6** | Compute/Efficiency Analyst | Wall-clock cost, memory footprint, and cost-benefit analysis of zigzag steps | **Table + Figure: Cost-performance Pareto curve** (HPSv2 improvement vs. added inference time across lambda_step and T_max) | Owns timing/memory instrumentation; runs ablation A5 (inference steps) |
| **P7** | Evaluation/Metrics Lead | Metric reliability analysis: how well do CLIP score and HPSv2 agree? Are automated metrics trustworthy? | **Figure: CLIP vs HPSv2 scatter plot** with correlation analysis; includes a small human-preference validation (5 people rank 10 pairs) | Owns `src/zsample/metrics.py` and `scripts/run_evaluation.py`; designs mini human eval |

### Midterm presentation ownership

Each person presents ~1 minute on their angle during the 7-minute midterm (see Section 7).

### Report differentiation enforcement

Rule: each person's unique artifact (column 4 above) MUST NOT appear in any other person's report. All seven reports share the same Introduction and Method sections (paraphrased individually), but diverge on Results and Analysis.

---

## 6. Phase Plan with Binary Checkpoints

### Phase 0: GPU & Environment (Days 1-3)

**Tasks:**
- Install PyTorch with CUDA in .venv
- Download SDXL model weights (~7GB)
- Run upstream `infer.py` on one prompt to verify GPU works
- Survey all 7 members' GPU access (personal, university cluster, Colab)

**Day-3 checkpoint (PASS/FAIL):**
- [ ] At least 2 students can generate one SDXL image on a GPU
- [ ] SDXL model weights are cached and accessible
- [ ] GPU inventory document written (who has what, estimated availability)

**FAIL action:** Immediately set up Google Colab notebooks as primary compute. Reduce experiment scope by 50%.

---

### Phase 1: Reimplementation (Days 4-14)

**Tasks -- P2 (Implementation Lead):**
- P2 implements `core.py` (z_sample, standard_sample)
- P2 implements `pipeline_loader.py` (load pipeline, encode prompts)
- P2 sets up dual environments by Day-7: `environments/upstream-env.yml` (diffusers==0.22.3, torch==2.0.1) and `environments/project-env.yml` (diffusers==0.30.0, pinned torch + transformers + accelerate)
- P1 writes unit tests from the test plan

**Tasks -- P3 through P7 (scaffold deliverables, Days 4-14):**
- **P3:** `scripts/run_ablation.py` scaffold -- CLI argument parsing via argparse, sweep loop structure, no zigzag logic. Calls a stub `generate()` function that returns a placeholder image. `python scripts/run_ablation.py --help` must work.
- **P4:** Finalize `experiments/prompts/prompts.csv` with all 30 prompts, categorized per Section 4.1, with sourcing attribution (DrawBench ID, T2I-CompBench ID, or "custom"). Reviewed by Day-10.
- **P5:** Run upstream `infer.py` on 5 prompts (must include known-hard categories: counting, text rendering, spatial). Document observed failure modes in `knowledgebase/failure-modes-baseline.md` with screenshots. Seed material for their report's failure-mode analysis.
- **P6:** Implement `src/zsample/utils.py` timer context manager (`class Timer`) + `torch.cuda.max_memory_allocated()` instrumentation as a decorator (`@track_memory`). Standalone, no dependency on `z_sample()`. Unit-testable on CPU with mock CUDA.
- **P7:** Implement `src/zsample/metrics.py::compute_clip_score()` standalone function. Test on any 2 image+prompt pairs with known CLIP scores. CLIP-only first; HPSv2 added in Phase 2.

**Day-10 sub-gate:**
- [ ] P3-P7 each have a draft PR (or branch) with their scaffold code
- [ ] `--help` works on all CLI scripts (P3's `run_ablation.py`)
- [ ] Standalone functions importable (`from zsample.utils import Timer; from zsample.metrics import compute_clip_score`)

**Day-14 checkpoint -- Implementation Parity (PASS/FAIL):**
- [ ] Parity verified using the dual-environment protocol:
  1. Activate upstream-env: `conda activate upstream-env`
  2. Run upstream `infer.py`: `python upstream/infer.py --prompt "A small yellow dog." --seed 42 --output upstream_ref.png`
  3. Activate project-env: `conda activate project-env`
  4. Run our compare script: `python scripts/compare_upstream.py --prompt "A small yellow dog." --seed 42 --upstream-image upstream_ref.png`
  5. Compute LPIPS: script uses the `lpips` library to compare, asserts LPIPS < 0.05
- [ ] Unit tests pass: `pytest tests/ -v` shows all green
- [ ] `z_sample()` with `lambda_step=0` produces output within L2 < 1e-3 of `standard_sample()` (fp16 tolerance; byte-identical is unrealistic with fp16)

**FAIL action:** If parity fails on identical environment (upstream-env), debug zigzag logic -- diff intermediate latents step-by-step. If it passes in upstream-env but fails in project-env, the difference is version drift -- document as a finding and port carefully. Budget 3 extra days.

---

### Phase 2: Experiment Infrastructure (Days 15-21)

**Tasks:**
- P4 finalizes `prompts.csv` review feedback (if any from Day-10)
- P6 integrates timing and memory instrumentation into the generation loop
- P7 implements HPSv2 scoring (CLIP already done in Phase 1)
- P2 implements `log_run()` in `src/zsample/utils.py` (JSONL writer with file lock)
- P3 connects `run_ablation.py` scaffold to real `z_sample()` / `standard_sample()` calls
- All run a small smoke test: 3 prompts x 2 methods x 1 seed = 6 images

**Day-21 checkpoint (PASS/FAIL):**
- [ ] `run_experiment.py` generates images AND writes valid JSONL entries to `experiments/runs.jsonl`
- [ ] `run_evaluation.py` reads `runs.jsonl` and appends CLIP + HPSv2 scores keyed by `run_id`
- [ ] JSONL entries contain all required fields (see Section 9)
- [ ] `pandas.read_json('experiments/runs.jsonl', lines=True)` loads without errors

**FAIL action:** Fix the broken pipeline component. Defer ablation complexity until core path works.

---

### Phase 3: Main Experiments (Days 22-35)

**Tasks:**
- Run core experiment matrix: 30 prompts x 2 methods x 3 seeds = 180 images
- Run ablation sweeps A1-A5: ~250 images
- P5 runs failure-mode diagnostic experiments with intermediate latent logging
- P6 collects timing data across all runs
- P7 computes metrics for all images

**Day-28 checkpoint -- Midterm readiness (PASS/FAIL):**
- [ ] At least 60 core images generated (10 prompts x 2 methods x 3 seeds)
- [ ] Midterm slides drafted (see Section 7)
- [ ] Each person can present their angle for 1 minute
- [ ] **Quality gate:** Z-Sampling vs Standard winning rate (by CLIP score, since HPSv2 may not be installed yet) on a 5-prompt subset (seeds 42 + 123) > 50%, OR qualitative inspection confirms visible improvement on at least 3 of 5 prompts

**Quality gate FAIL action:** Pause new experiment runs. Debug the z_sample() implementation -- compare latent trajectories step-by-step against upstream. Do not proceed to ablations until quality is confirmed or the negative result is understood and documented.

**Day-35 checkpoint -- Data completeness (PASS/FAIL):**
- [ ] Core matrix 100% complete (180 images)
- [ ] At least 3 of 5 ablations complete
- [ ] `experiments/runs.jsonl` has >= 350 valid entries
- [ ] HPSv2 winning rate computed for z-sampling vs. standard
- [ ] **Report outline gate:** Each of the 7 students has a 1-page report outline at `report/outlines/personN.md` with their unique artifact placeholder (figure/table caption + 3-5 bullet point analysis plan). No outline = blocking for that student until produced.

**FAIL action:** Prioritize core matrix over ablations. Cut ablation A5 (inference steps) first, then A4 (T_max) if needed.

---

### Phase 4: Analysis & Figures (Days 36-45)

**Tasks:**
- Each person produces their unique artifact (see Section 5, column 4)
- P1 writes the theory section draft + produces latent-trajectory PCA visualization
- P3 produces ablation heatmaps
- P4 produces per-category analysis
- P5 annotates failure cases
- P6 produces Pareto curves
- P7 produces metric correlation analysis + mini human eval

**Day-42 checkpoint (PASS/FAIL):**
- [ ] Each person has their unique figure/table in draft form
- [ ] Group review: no two artifacts overlap
- [ ] Shared Introduction and Method section drafted

**FAIL action:** Any person without their artifact gets pair-programmed by P2 to produce it.

---

### Phase 5: Report Writing (Days 46-56)

**Tasks:**
- Each person writes their 3-page individual report
- Peer review: each report reviewed by 2 other group members
- Final figures polished and inserted
- Critical appraisal section: each person writes their own interpretation

**Day-49 checkpoint -- Draft review (PASS/FAIL):**
- [ ] All 7 drafts exist (at least 2 pages each)
- [ ] Each draft contains the unique artifact
- [ ] No two drafts have the same Results section content

**Day-56 checkpoint -- Submission readiness (PASS/FAIL):**
- [ ] All 7 reports are 3 pages, double-column, complete
- [ ] All figures are legible at print resolution
- [ ] References are complete and formatted consistently

**FAIL action:** Triage: focus on completing the Results and Analysis sections first (they carry the Content 40% weight). Introduction and Method can be refined last.

---

## 7. Midterm Presentation Skeleton

**Total: 7 minutes, 7 students, 7 slides (+ 30s Q&A buffer)**

| Slide | Owner | Content | Duration |
|-------|-------|---------|----------|
| 1: Title + Problem | P1 | Paper title, authors, venue. Problem statement: diffusion models fail on challenging prompts. Why self-reflection? | 50s |
| 2: Z-Sampling Method | P1 | Algorithm diagram: standard vs. zigzag trajectory. Key equations. Guidance gap concept. | 50s |
| 3: Our Reimplementation Plan | P2 | Architecture diagram (pipeline-wrapping approach). What we reimplement vs. reuse. Code design decisions. | 50s |
| 4: Dataset & Experiment Design | P4 | Prompt categories (table). Experiment matrix. Why these categories. | 50s |
| 5: Evaluation Plan | P7 | Metrics (HPSv2, CLIP, qualitative). Winning rate methodology. Mini human eval plan. | 50s |
| 6: Ablation & Analysis Plan | P3 + P5 | Hyperparameter sweep plan (which params, which ranges). Failure-mode investigation plan. | 60s |
| 7: Timeline & Compute | P6 | Compute budget. GPU inventory. Phase plan with checkpoints. Risk mitigations. | 60s |
| Q&A buffer | All | | 30s |

**Format requirement satisfaction:** Each student presents the part they will be responsible for implementing (matches instructor's explicit instruction).

---

## 8. Risks and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| 1 | **Diffusers version incompatibility.** Upstream uses 0.22.3 (2+ years old). Modern diffusers may have different scheduler APIs (e.g., `_step_index` semantics changed between 0.22 and 0.27). | MEDIUM | HIGH | Dual-environment strategy: `environments/upstream-env.yml` pins `diffusers==0.22.3, torch==2.0.1`; `environments/project-env.yml` pins `diffusers==0.30.0` with locked dependencies. P2 sets up both envs by Day-7. `scripts/compare_upstream.py` runs in upstream-env only and produces parity reference images. All other scripts run in project-env. Parity test uses upstream-env to isolate version drift from logic bugs. If parity passes in upstream-env but fails in project-env, debug the porting; if it fails in upstream-env, debug zigzag logic. Document any porting issues as a finding. |
| 2 | **HPSv2 library unavailable or broken.** HPSv2 is an academic library, not a stable PyPI package. | MEDIUM | MEDIUM | Test HPSv2 installation in Day-3 phase. Fallback: use ImageReward (similar purpose, better maintained) or CLIP score only. |
| 3 | **SDXL model download fails or is too slow.** The model is ~7GB; university networks may throttle HuggingFace downloads. | LOW | HIGH | Download once, share the cache directory across students via external drive or university storage. Set `HF_HOME` env variable. |
| 4 | **Floating-point non-determinism across GPUs.** Different GPU architectures (A100 vs. T4 vs. RTX 3090) may produce slightly different outputs for the same seed. | HIGH | LOW | Document which GPU produced each result in the JSONL log. Run parity checks on a single GPU. Accept that cross-GPU images may differ slightly; this is a known property of CUDA, not a bug. |
| 5 | **Time overrun on report writing.** 3 pages sounds short but double-column is dense. Students underestimate writing time. | HIGH | MEDIUM | Start report outlines in Phase 3 (Day 22), not Phase 5. Each person maintains a running "observations log" during experiments that feeds directly into their report. |
| 6 | **Prompt dataset quality.** Poorly chosen prompts may not reveal interesting differences between standard and z-sampling. | MEDIUM | MEDIUM | Include the 3 upstream prompts as sanity checks. Include known-hard categories (counting, text rendering) where the paper claims the largest improvement. Review prompt quality at Day-21 checkpoint. |
| 7 | **Student availability variance.** 7 students have different schedules, motivation levels, and compute access. | HIGH | MEDIUM | Assign each person a self-contained deliverable with a clear artifact. No task depends on more than 2 other people's output. P2 (Implementation Lead) is the only single point of failure; assign P1 as backup who pair-programs during Phase 1. **If P2 is unavailable >3 consecutive days, P1 takes over `core.py` ownership. P1 must have a working dev environment by Day-5 as a precaution.** |

---

## 9. Run Log Specification (JSONL)

Each generation appends one JSON object to `experiments/runs.jsonl`. The object contains: `prompt_id`, `prompt_text`, `category`, `seed`, `method`, `gamma_1`, `gamma_2`, `lambda_step`, `t_max`, `num_inference_steps`, `image_width`, `image_height`, `commit_hash`, `gpu_name`, `wall_clock_s`, `peak_memory_mb`, `image_path`, `timestamp_iso`, `run_id` (uuid). Metric scores (`clip_score`, `hpsv2_score`) are appended post-hoc to the same row keyed by `run_id`. Loaded for analysis with `pandas.read_json('experiments/runs.jsonl', lines=True)`. No formal validator.

A utility function in `src/zsample/utils.py` handles appending:

```python
def log_run(run_dict: dict) -> None:
    """Append a single run record to experiments/runs.jsonl with file lock for concurrent writes."""
```

---

## 10. Open Questions

1. **Which diffusers version is current stable?** We have pinned `diffusers==0.30.0` in the plan. If a newer stable release exists at project start, update `environments/project-env.yml` accordingly. Check via `pip index versions diffusers` or the PyPI page.
2. **Which 5 group members will own which of the 7 angles?** The plan assigns P1-P7 roles but does not map them to actual student names. This must be decided at the Day-1 kickoff meeting.
3. **What is the actual TA meeting date?** The midterm presentation date drives Phase 3 deadlines. Confirm the exact date and adjust day counts if needed.
4. **Which GPU sources are confirmed available?** The Day-3 GPU inventory will answer this, but if anyone has pre-existing access to A100s, V100s, or cloud credits, surface it before Day-1 to inform compute budget planning.
