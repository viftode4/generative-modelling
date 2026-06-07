# End-term Presentation Asset Checklist

## Recommended visuals

### Method diagram

Create manually in slides:

```text
Standard:  x_T -> x_{T-1} -> ... -> x_0
Z-Sampling: denoise(gamma_1) -> invert(gamma_2) -> re-denoise(gamma_1)
```

### Vlad figure pair

Use the clearest qualitative example:

- Standard: `report/figures/vlad_geneval/geneval_259_colors_seed42_standard.png`
- Z-Sampling: `report/figures/vlad_geneval/geneval_259_colors_seed42_zigzag.png`
- Caption: `Prompt: "a photo of a blue fire hydrant" (GenEval colors tag, seed 42). Z-Sampling was the largest HPSv2 win in Vlad's subset and also improves visual object/color clarity.`

Backup qualitative pair:

- Standard: `report/figures/vlad_geneval/geneval_180_counting_seed2026_standard.png`
- Z-Sampling: `report/figures/vlad_geneval/geneval_180_counting_seed2026_zigzag.png`
- Caption: `Prompt: "a photo of two backpacks" (counting tag, seed 2026). Z-Sampling separates the two objects more clearly.`

### Tables to screenshot or rebuild

- Vlad detector summary: `report/tables/vlad_geneval_detector.md`
- Vlad compute control: `report/tables/vlad_compute_matched.md`
- Eugen result: `report/tables/eugen_results.md`

## What not to include

- Full dense result tables: too much for slides.
- Raw JSONL manifests.
- Large galleries of generated images.
- Any claim that the full paper benchmark was reproduced.
