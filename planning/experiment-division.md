# Experiment Division

Updated source of truth from TA/group notes provided on 2026-05-14, plus Drive notes from the 2026-05-04 TA meeting pasted into the session on 2026-06-03.

## Sampling-method ownership

| Sampling method / experiment | Owner | Notes |
| --- | --- | --- |
| Standard Sampling + 1 | Eugen | Could add more models if too easy. |
| Standard Sampling + 5 | Daniel | Also helps set up shared standard-sampling baseline infrastructure. |
| Resampling + 2 | Gouri |  |
| Resampling + 3 | Lenka |  |
| Z-Sampling + 6 | Radu | Assignment marked uncertain in notes. |
| Z-Sampling + 7 | Bogdan |  |
| Z-Sampling + 8 | Vlad | Vlad's implementation/run lane. |

Drive notes say experiments 1, 5, 6, and 8 use the same standard-sampling SDXL pipeline as baseline, so the Standard Sampling owners should help set that up. Implementation interpretation: keep shared baseline code configurable; for scientifically valid method comparisons, compare each method against a matched standard baseline using the same model, prompts, seeds, scheduler family, image size, and step count unless an experiment explicitly studies cross-model differences.

## TA guidance captured

- Implement inference-time behavior; do not retrain models.
- Do not implement every model; pick a subset, divide reasonably, and ensure the paper's behavior is represented.
- Focus on the main contribution across datasets/models rather than expanding too broadly.
- End-term presentation should show implementation and experimentation progress, challenges, and the plan to finish by the report deadline; complete results are not required yet.
- Explain enough theory/math to justify Z-Sampling; do not avoid the theoretical foundation entirely.
- The group should cover the relevant sampling methods/baselines rather than only Z-Sampling: standard sampling, resampling, and Z-Sampling are the in-scope sampling-method set for the scoped project.

## Vlad lane

- Method: Z-Sampling.
- Experiment: 8.
- Model/dataset from previous assignment matrix: DreamShaper-xl-v2-turbo + GenEval.
- Concrete config: `experiments/configs/dreamshaper_turbo_geneval.md`.


## Resampling implementation note

Drive notes compare two resampling approaches against RePaint. The RePaint-aligned implementation is the Lenka-style non-monotonic schedule: true timestep backtracking, forward-diffusion re-noising between noisy states using alpha-ratio transitions, `jump_length` controlling excursion depth, and `jump_n_sample` controlling revisits. The earlier Gouri-style same-timestep perturbation loop (`x_t -> x_hat_0 -> x_t'`) is recorded as not RePaint-equivalent and should not be presented as the final Resampling baseline unless explicitly framed as a different ablation.
