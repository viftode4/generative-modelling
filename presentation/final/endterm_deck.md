# Group 14 End-term Presentation — Short Slide Content

Format: 7 minutes total, graded 20%. Keep this to 6 slides.

---

## Slide 1 — Why this paper?

**Headline:** Can diffusion models improve prompt alignment without retraining?

**Bullets:**

- Diffusion images can look good but miss prompt details.
- Failures: object count, color, spatial relations.
- Z-Sampling changes only inference, not training.
- Our goal: reproduce the core idea and test if it helps locally.

**Visual:** paper title + 3 failure words: `count / color / position`.

**Say:**

- We picked this because it is realistic to implement but still scientifically interesting.
- The question is not “can we train a better model?” but “can the sampling loop help?”

---

## Slide 2 — Method: the zigzag step

**Headline:** Z-Sampling adds a small denoise → invert → re-denoise loop.

**Bullets:**

- Standard DDIM: denoise step-by-step.
- Z-Sampling: denoise with strong guidance `γ₁`.
- Then invert with weak guidance `γ₂`.
- Re-denoise and keep the guidance-gap signal.

**Visual:**

```text
Standard:   x_t ──denoise──> x_{t-1}
Z-Sampling: x_t ──denoise γ₁──> x_{t-1}
                 └─invert γ₂──> x'_t ──re-denoise γ₁──> refined x_{t-1}
```

**Say:**

- The important thing is that this is not just increasing CFG.
- It needs a compatible DDIM inverse scheduler.

---

## Slide 3 — What we implemented

**Headline:** We built a shared, tested reproduction pipeline.

**Bullets:**

- Ported upstream Z-Sampling into `src/zigzag_project/zsampling.py`.
- Kept standard vs Z pairs matched by prompt, seed, and initial latent.
- Added sampler guardrails for standard / resampling / Z-Sampling.
- Split experiments by model and dataset across the group.

**Visual:** simple pipeline:

```text
prompts + seeds → standard / Z-Sampling → images → HPSv2 / detector → tables
```

**Say:**

- Much of the work was making the method runnable in our environment.
- We avoided editing the upstream submodule and kept our changes reproducible.

---

## Slide 4 — Results we can stand behind

**Headline:** Verified lanes show positive but not universal gains.

**Bullets:**

- SD2.1 + Pick-a-Pic: Z wins 213/300 HPS pairs = 71.0%.
- DreamShaper + GenEval: detector improves 31.5% → 50.0%.
- Same DreamShaper run: HPS favors Z in 53/54 pairs.
- Strongest gains: color, counting, two-object prompts.

**Visual:** compact table:

| Lane | Metric | Result |
|---|---|---|
| SD2.1 + Pick-a-Pic | HPSv2 | 71.0% Z wins, +0.0110 |
| DreamShaper + GenEval | Detector | 31.5% → 50.0% |
| DreamShaper + GenEval | HPSv2 | 53/54 Z wins, +0.0358 |

**Say:**

- This is not full benchmark-scale reproduction.
- But these are controlled paired comparisons, not cherry-picked single images.

---

## Slide 5 — Challenges and what went wrong

**Headline:** The hard parts were reproducibility, metrics, and cost.

**Bullets:**

- Paper stack was CUDA/older `diffusers`; local runs used modern MPS stack.
- GenEval detector needed `mmdet` 2.x compatibility work.
- HPSv2 is preference-based, not the same as detector correctness.
- Z-Sampling is slower: 13.80s/image vs 6.16s/image in Vlad’s run.

**Visual:** 4 warning cards:

```text
Dependencies | Hardware | Metrics | Runtime
```

**Say:**

- We should present these as real project work, not embarrassment.
- The compute-control matters: 9-step standard was slower than Z, but Z still won 42/54 HPS pairs.

---

## Slide 6 — Final takeaway

**Headline:** Z-Sampling helped in our controlled runs, but it is not magic.

**Bullets:**

- We implemented the core method and verified it with tests and paired runs.
- Results are positive in pushed/verified lanes.
- Gains depend on prompt type and metric.
- Final report should be honest: useful method, runtime cost, limited scale.

**End sentence:**

> Our conclusion is a controlled positive reproduction: Z-Sampling can improve prompt alignment locally, but the evidence is bounded by runtime, metrics, environment, and benchmark scale.

**Feedback ask:**

- Is this scope enough for the group report?
- Should we prioritize more detector scoring or clearer interpretation?
