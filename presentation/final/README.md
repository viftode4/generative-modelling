# End-term Presentation Prep

Use this folder for the graded Group 14 end-term presentation.

## Constraints

- Date: 2026-06-09.
- Time: 7 minutes per group + 4 minutes feedback.
- Required content: methodology, results, and challenges.
- Target: 6 slides maximum.

## Files to use

- `endterm_deck.md` — actual slide content to transfer into PowerPoint/Google Slides.
- `outline.md` — timing, slide purpose, and source anchors.

Everything else should stay out unless the group explicitly needs it. Do not add presenter-specific speaker notes here; keep those private or in the slide tool.

## Story spine

```text
midterm plan → implemented pipeline → verified results → challenges → bounded conclusion
```

Say this plainly:

- We planned a broad reproduction across sampling methods, models, datasets, and metrics.
- We implemented a controlled paired pipeline and cleaned up integration issues.
- We have verified positive evidence from Eugen and Vlad lanes.
- We did not reproduce the full paper benchmark matrix.
- The honest conclusion is positive but bounded by scale, metrics, runtime, and environment.

## Slide build checklist

1. Use exactly the six slides in `endterm_deck.md`.
2. Keep each slide to one headline and at most four bullets.
3. Use one method diagram and one compact results table.
4. If using an image pair, use only one clear pair.
5. Keep backup details out of the main 7-minute deck.

## Visual style

- Light background.
- Standard sampling: gray.
- Resampling: orange or dashed gray.
- Z-Sampling: blue or purple accent.
- Avoid dense tables and long paper quotes.

## Recommended assets

Main qualitative pair:

- Standard: `report/figures/vlad_geneval/geneval_259_colors_seed42_standard.png`
- Z-Sampling: `report/figures/vlad_geneval/geneval_259_colors_seed42_zigzag.png`
- Caption: `Prompt: "a photo of a blue fire hydrant". Standard vs Z-Sampling, DreamShaper-xl-v2-turbo, GenEval colors tag, seed 42.`

Backup qualitative pair:

- Standard: `report/figures/vlad_geneval/geneval_180_counting_seed2026_standard.png`
- Z-Sampling: `report/figures/vlad_geneval/geneval_180_counting_seed2026_zigzag.png`
- Caption: `Prompt: "a photo of two backpacks". Standard vs Z-Sampling, GenEval counting tag, seed 2026.`

Tables to rebuild visually, not paste densely:

- `report/tables/eugen_results.md`
- `report/tables/vlad_geneval_detector.md`
- `report/tables/vlad_compute_matched.md`

## Do not claim

- Do not say we fully reproduced the paper.
- Do not present unverified teammate lanes as quantitative results.
- Do not put every package/install issue on slides.
- Do not include raw manifests or large galleries.
