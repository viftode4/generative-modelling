# Experiment Division

Updated source of truth from TA/group notes provided on 2026-05-14.

## Sampling-method ownership

| Sampling method / experiment | Owner | Notes |
| --- | --- | --- |
| Standard Sampling + 1 | Eugen | Could add more models if too easy. |
| Standard Sampling + 5 | Daniel | Also helps set up shared SDXL standard baseline. |
| Resampling + 2 | Gouri |  |
| Resampling + 3 | Lenka |  |
| Z-Sampling + 6 | Radu | Assignment marked uncertain in notes. |
| Z-Sampling + 7 | Bogdan |  |
| Z-Sampling + 8 | Vlad | Vlad's implementation/run lane. |

Experiments 1, 5, 6, and 8 use the same standard SDXL sampling pipeline as baseline, so the Standard Sampling owners should also help set that up.

## TA guidance captured

- Implement inference-time behavior; do not retrain models.
- Do not implement every model; pick a subset, divide reasonably, and ensure the paper's behavior is represented.
- Focus on the main contribution across datasets/models rather than expanding too broadly.
- End-term presentation should show implementation and experimentation progress, challenges, and the plan to finish by the report deadline; complete results are not required yet.
- Explain enough theory/math to justify Z-Sampling; do not avoid the theoretical foundation entirely.

## Vlad lane

- Method: Z-Sampling.
- Experiment: 8.
- Model/dataset from previous assignment matrix: DreamShaper-xl-v2-turbo + GenEval.
- Concrete config: `experiments/configs/dreamshaper_turbo_geneval.md`.
