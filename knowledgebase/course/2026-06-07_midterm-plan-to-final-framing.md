# 2026-06-07 Midterm Plan to Final Presentation Framing

Source: Vlad pasted the midterm planning notes into the workspace on 2026-06-07. A local untracked `Midterm.pptx` was also inspected on 2026-06-07 to recover the slide-level midterm narrative.

## Midterm plan that should be closed in the final presentation

- Compare three sampling methods:
  - Standard Sampling;
  - Resampling;
  - Z-Sampling.
- Split work across model/dataset/method lanes so the group is non-redundant.
- Use the paper's main experiment structure as reference:
  - models: SD2.1, SDXL, Hunyuan-DiT, DreamShaper-xl-v2-turbo;
  - datasets: Pick-a-Pic, DrawBench, GenEval;
  - metrics considered: HPSv2, PickScore, ImageReward, AES, CLIP-style metrics, GenEval detector where applicable.
- Implement inference-time behaviour; do not retrain models.
- Focus on the paper's main contribution and justify scoped choices instead of expanding without control.
- Use ablations / efficiency comparisons only if time permits.

## Final presentation framing derived from that plan

- Do not present the final as a full-paper benchmark reproduction.
- Present it as a scoped implementation and experimentation project:
  - planned broad reproduction;
  - implemented controlled paired pipeline;
  - verified positive result lanes;
  - documented limitations and challenges.
- Include methodology, results, and challenges explicitly because the 2026-06-04 course announcement made those required for the graded 7-minute presentation.

## Midterm PPTX narrative recovered locally

The local `Midterm.pptx` slide text had this flow:

1. background diffusion / classifier-free guidance;
2. strong-guidance quality trade-off;
3. ZigZag paper and key observation;
4. Standard vs Resampling vs Z-Sampling;
5. main paper experiment matrix;
6. metrics and evaluation methodology;
7. project timeline and task division.

The end-term deck should mirror this enough that it feels like a completed project story, but shorten it to fit 7 minutes and foreground verified results.
