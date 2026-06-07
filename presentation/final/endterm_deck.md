# Group 14 End-term Presentation — 7-minute Content

Format: 7 minutes total, graded 20%. Keep this to **6 slides**.

Core story:

> At midterm we planned a broad reproduction of the paper across sampling methods, models, datasets, and metrics. In the final version, we should show what actually survived implementation: a working paired pipeline, verified positive Z-Sampling results in two lanes, and clear limitations around scale, metrics, and runtime.

---

## Slide 1 — From paper question to our plan

**Headline:** We tested whether inference-time “self-reflection” improves prompt alignment.

**Bullets:**

- Paper claim: improve diffusion outputs without retraining.
- Midterm plan: compare Standard, Resampling, and Z-Sampling.
- Group split: different model/dataset/method lanes.
- Final focus: what we implemented, what worked, what did not.

**Visual:** one-line plan:

```text
Paper idea → sampling methods → model/dataset lanes → paired evaluation
```

**Say:**

- This connects directly to what we promised in the midterm.
- We should not pretend we reproduced every paper table.

---

## Slide 2 — Methodology: three sampling ideas

**Headline:** The main change is in the sampling loop, not model training.

**Bullets:**

- **Standard:** denoise step by step.
- **Resampling:** denoise, add noise back, denoise again.
- **Z-Sampling:** denoise strongly, invert weakly, re-denoise.
- Key mechanism: guidance gap `γ₁ − γ₂` injects prompt information.

**Visual:** simple comparison:

```text
Standard:   x_t → denoise → x_{t-1}
Resampling: x_t → denoise → noise back → denoise
Z-Sampling: x_t → denoise γ₁ → invert γ₂ → re-denoise γ₁
```

**Say:**

- No need to go deep into equations.
- Important point: all methods are inference-time changes.

---

## Slide 3 — What we actually built

**Headline:** We turned the paper idea into controlled paired experiments.

**Bullets:**

- Shared repo pipeline around `diffusers`, not retraining.
- Standard vs Z matched by prompt, seed, scheduler, and initial latent.
- Result tables generated from reproducible manifests and scoring scripts.
- Integration cleanup added guardrails so sampler configs are not mixed.

**Visual:** implementation pipeline:

```text
prompts + seeds → sampler pair → images → HPSv2 / detector → result table
```

**Plan vs actual mini-table:**

| Midterm plan | Final reality |
|---|---|
| Many model/dataset/method cells | Verified subset + cleaner shared code |
| Multiple metrics | HPSv2 + GenEval detector where feasible |
| Ablations/efficiency if time | One compute-matched runtime control |

**Say:**

- The useful project work is partly engineering: making comparisons fair and reproducible.
- This also explains why implementation took longer than the plan suggested.

---

## Slide 4 — Results we can stand behind

**Headline:** Z-Sampling improved our verified paired comparisons, but not at full paper scale.

**Bullets:**

- SD2.1 + Pick-a-Pic: Z wins 213/300 HPS pairs = 71.0%.
- DreamShaper + GenEval: detector improves 31.5% → 50.0%.
- Same DreamShaper run: HPS favors Z in 53/54 pairs.
- Strongest Vlad gains: colors, counting, two-object prompts.

**Visual:** compact table:

| Lane | Metric | Result |
|---|---|---|
| SD2.1 + Pick-a-Pic | HPSv2 | 71.0% Z wins, +0.0110 |
| DreamShaper + GenEval | Detector | 31.5% → 50.0% |
| DreamShaper + GenEval | HPSv2 | 53/54 Z wins, +0.0358 |

**Optional image pair:**

- `report/figures/vlad_geneval/geneval_259_colors_seed42_standard.png`
- `report/figures/vlad_geneval/geneval_259_colors_seed42_zigzag.png`

**Say:**

- These are controlled paired comparisons, not cherry-picked single images.
- But they are still smaller than the paper benchmarks.

---

## Slide 5 — What worked, what did not

**Headline:** The main challenges were scope, compatibility, metrics, and runtime.

**Bullets:**

- **Worked:** paired setup, HPS scoring, Vlad GenEval detector, compute control.
- **Hard:** old CUDA-oriented paper stack vs local modern/MPS environment.
- **Messy:** several implementation waves needed integration cleanup.
- **Cost:** Z-Sampling was slower: 13.80s/image vs 6.16s/image in Vlad’s run.

**Visual:** four cards:

```text
Scope | Dependencies | Metrics | Runtime
```

**Say:**

- This is where we should be honest: the plan was broader than what fully landed.
- The compute-control helps: 9-step standard was slower than Z, but Z still won 42/54 HPS pairs.

---

## Slide 6 — Final takeaway

**Headline:** Our conclusion is positive, but bounded.

**Bullets:**

- We implemented the core inference-time behaviour.
- We verified meaningful gains in the lanes we can defend.
- We did not reproduce the full paper benchmark matrix.
- Final report should emphasize method, evidence, and limitations.

**End sentence:**

> Z-Sampling can improve prompt alignment in our controlled local runs, but the evidence is bounded by runtime cost, metric choice, environment differences, and benchmark scale.

**Feedback ask:**

- Is this enough evidence for the final group report?
- Should remaining effort go to more scoring or clearer interpretation?

---

## Source anchors for slide claims

- Course timing/requirements: `knowledgebase/course/2026-06-04_presentation-report-clarification.md`.
- Midterm plan framing: `knowledgebase/course/2026-06-07_midterm-plan-to-final-framing.md` and `planning/experiment-division.md`.
- Paper method/settings: `knowledgebase/papers/zigzag-diffusion-sampling.md`.
- SD2.1 Pick-a-Pic result: `report/tables/eugen_results.md`.
- Vlad GenEval detector result: `report/tables/vlad_geneval_detector.md`.
- Vlad compute/runtime control: `report/tables/vlad_compute_matched.md`.
