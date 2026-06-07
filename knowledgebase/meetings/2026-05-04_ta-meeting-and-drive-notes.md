# 2026-05-04 — TA Meeting and Drive Planning Notes

Source: user-provided notes from group Drive, pasted into the Codex session on 2026-06-03.

## TA guidance

- The project is group work, but each student submits an individual document with shared group sections plus individual reflections.
- A timeline is expected; some tasks have sequential dependencies.
- The team should implement inference-time behaviour and **not retrain models**.
- The implementation abstraction level expected by the course is roughly the level used by the paper/reference code; using standard tools such as PyTorch is acceptable.
- The team is **not expected to implement every model**. Pick a subset, divide reasonably, and make sure the paper's behavior is represented.
- For the end-term presentation, complete results are not required, but the group should show implementation/experimentation progress, challenges, what has been done, and the plan to finish by the report deadline.
- The theory/math behind Z-Sampling should be explained enough to justify the method; do not avoid the theoretical foundation.
- The group asked whether it can focus on one sampling method and ignore the rest of the baselines. The recorded answer was no: go over the relevant sampling methods/baselines.

## Experiment division from Drive notes

| Sampling method / experiment | Owner | Notes |
| --- | --- | --- |
| Standard Sampling + 1 | Eugen | Could add more models if too easy. |
| Standard Sampling + 5 | Daniel | Standard baseline setup support. |
| Resampling + 2 | Gouri |  |
| Resampling + 3 | Lenka |  |
| Z-Sampling + 6 | Radu | Assignment uncertain in notes. |
| Z-Sampling + 7 | Bogdan |  |
| Z-Sampling + 8 | Vlad |  |

The Drive notes say experiments 1, 5, 6, and 8 use the same standard-sampling SDXL pipeline as baseline, so the Standard Sampling owners should help set that up. Technical interpretation to preserve in implementation: shared baseline code should be configurable and each method result must be compared to a matched standard baseline for the same model/prompt/seed/settings, unless an experiment explicitly studies cross-model differences.

## Paper experiment axes recorded in Drive notes

Sampling methods:

- Standard
- Resampling
- Z-Sampling

Models:

- SD2.1 — Eugen, Radu
- SDXL — Daniel, Lenka
- Hunyuan-DiT — Bogdan
- DreamShaper-xl-v2-turbo — Gouri, Vlad

Datasets:

- Pick-a-Pic — Lenka, Eugen
- DrawBench — Daniel, Gouri, Radu
- GenEval — Bogdan, Vlad

Metrics listed from paper:

- HPS v2
- PickScore
- ImageReward (IR)
- AES

Additional paper baselines/ablations listed for context:

- Semantic-aware CFG
- Diffusion-DPO
- AYS-Sampling
- Smoothed Energy Guidance
- CFG++
- impact of guidance scale
- number of zigzag steps
- efficiency comparison

## Preliminary plan from Drive notes

1. Everyone gets comfortable loading/accessing assigned models and datasets.
2. Everyone builds a high-level understanding of the paper motivation, Z-Sampling, and the existing codebase.
3. Split into sampling-method subgroups for standard sampling, resampling, and Z-Sampling. Implement/test methods separately until outputs are sensible.
4. Once methods work, split experiment types among the team to cover as much useful paper-aligned evidence as possible.
5. Create presentation materials showing method understanding, implementation, experiments, challenges, and finish plan.

## Resampling implementation comparison from Drive notes

The Drive notes compare two resampling implementations against RePaint-style resampling.

### RePaint target behavior

- RePaint defines a back-and-forth schedule with jump length `j` and repetition count `r`.
- It “undoes `j` diffusion steps for `r` times”.
- The schedule is non-monotonic: reverse diffusion when timestep decreases and forward diffusion when timestep increases.
- Timestep genuinely increases again during backtracking, e.g. `40 -> 50 -> 49 -> ...`.
- Re-noising should follow the forward diffusion transition between noisy states, not repeated perturbation at the same timestep through predicted `x_hat_0`.

### Lenka implementation conclusion in Drive notes

The Drive notes mark Lenka's implementation as the RePaint-aligned one:

- non-monotonic true backtracking schedule
- forward-diffusion re-noising between states using alpha ratios
- jump-depth controlled by `jump_length`
- repetition controlled by `jump_n_sample`
- no predicted-`x_hat_0` perturbation loop
- no in-place same-timestep repeat as the final RePaint implementation

### Gouri implementation issue in Drive notes

The Drive notes describe Gouri's earlier implementation as repeated perturbation/refinement at the same timestep:

- `lambda_step` controls where repeated refinement starts
- `t_max` controls repeat count
- timestep never genuinely backtracks to a noisier index
- re-noising uses `x_t -> predicted x_hat_0 -> x_t'`

The Drive notes mark this as not RePaint-equivalent; the final resampling code should use the Lenka/RePaint-style non-monotonic schedule.
