# Experiment #8 — Paper-Fidelity Audit (SDXL × Pick-a-Pic Efficiency)

Audited 2026-06-09 against the paper PDF (`docs/papers/zigzag-diffusion-sampling.pdf`, ICLR 2025)
and the upstream reference code (`external/zigzag-diffusion-sampling/infer.py`). This is the
source of truth for what #8 must reproduce.

## What the paper did (Experiment #8 = Efficiency Comparison, Figure 10, p.10)

NOT a single 50-step Standard-vs-Z timing. A **time-matched sweep**:

- Standard op count = `T_s`; Z-Sampling op count = `T_z + 2·λ`.
- Paper sets **`T_z = T_s/2`, `λ = T_z/2 = T_s/4`** → Z ops = `T_s` (matched compute).
- Sweep several time budgets; plot **mean HPS v2 (Pick-a-Pic, SDXL) vs avg time/image** for both arms.
- Headline: **"even with 36% less computational time, Z-Sampling can surpass the best performance of standard sampling."**
- Note: `λ = T_s/4` here, NOT the default `λ = T−1` used in the main experiments.

## Fidelity table (ours vs paper/upstream)

| Input | Ours | Paper / upstream | Verdict |
| --- | --- | --- | --- |
| SDXL model_id | `stabilityai/stable-diffusion-xl-base-1.0` (`model_utils.py:17`, `models.py:load_sdxl`) | same (`infer.py:45`) | ✅ |
| Steps T / γ1 / γ2 | 50 / 5.5 / 0 (defaults) | 50 / 5.5 / 0 (`infer.py:12-15`, §4.1) | ✅ |
| Image size | 1024 | 1024 (`infer.py:16`) | ✅ |
| t_max | 1 | 1 (`infer.py:17`) | ✅ |
| Scheduler | DDIM + DDIMInverse (`models.py:load_sdxl`) | DDIM + DDIMInverse (`infer.py:48-52`) | ✅ |
| Z-Sampling loop | `zsampling.py:z_sample_sdxl` (port at pinned commit) | upstream `pipeline_stable_diffusion_xl.py:z_sampling_call` | ✅ port |
| Pick-a-Pic count | first 100 | first 100 (Appendix A.1) | ✅ count |
| Pick-a-Pic exact prompts | author `PickaPic-selected-prompts` substitute | original v1/v2 unavailable | ⚠️ same dataset, **not guaranteed identical** |
| Metric | HPS v2 + runtime/image | HPS v2 + time/image (Fig 10) | ✅ |

## Gaps found and fixed

1. **SDXL inverse-scheduler bug.** `model_utils.load_pipeline_standard` returns a raw pipe with no
   inverse scheduler; the only prior SDXL runner (`run_exp2_sdxl_geneval.py`) set
   `inverse_scheduler = pipe.inv_scheduler if hasattr(...) else pipe.scheduler` → silently the
   **forward** DDIM scheduler, which would corrupt Z-Sampling inversion. **Fixed:** added
   `models.py:load_sdxl` that pairs a real `DDIMInverseScheduler` (mirrors `load_dreamshaper_xl_turbo`,
   matches `infer.py:48-52`). Vlad's #8 runner uses `load_sdxl`, not that wrapper.
2. **Pick-a-Pic overclaim.** `pickapic_full.md` and `dataset_utils.py` claimed the prompts "directly
   match" the paper. **Fixed:** both now state the honest provenance (substitute set; count + dataset
   match, exact prompts not guaranteed).
3. **#8 mis-specified.** Repo recorded #8 as DreamShaper×GenEval; corrected to SDXL×Pick-a-Pic
   time-matched sweep per the diagram and Figure 10.

## Not guaranteed identical to the paper (report these as caveats)

- Pick-a-Pic exact prompts (substitute set).
- Environment: Apple Silicon / MPS + modern diffusers vs the paper's CUDA stack — absolute runtimes
  are device-specific (the comparison is internal: Standard vs Z on the **same** hardware).
- Image size 1024 is the SDXL/upstream default; the paper text does not separately pin it.
