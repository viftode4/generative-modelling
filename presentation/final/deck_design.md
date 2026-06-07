# End-term Deck Design Brief

## Tone

- Direct, technical, not flashy.
- Scientific honesty over hype.
- Short bullets only.

## Visual style

- White or very light background.
- One accent color for Z-Sampling, e.g. blue or purple.
- Standard sampling = gray.
- Z-Sampling = accent color.
- Use the same colors in diagrams and result tables.

## Slide layout

1. **Title / motivation**: big question + three failure keywords.
2. **Method diagram**: arrows only, no equations except `γ₁`, `γ₂`.
3. **Implementation pipeline**: prompts/seeds → samplers → metrics.
4. **Results table**: three rows maximum.
5. **Challenge cards**: dependencies, hardware, metrics, runtime.
6. **Takeaway**: one sentence + feedback ask.

## Must-use assets

- Optional image pair for Slide 4 or backup:
  - `report/figures/vlad_geneval/geneval_259_colors_seed42_standard.png`
  - `report/figures/vlad_geneval/geneval_259_colors_seed42_zigzag.png`
- Caption: `Prompt: "a photo of a blue fire hydrant". Standard vs Z-Sampling, DreamShaper-xl-v2-turbo, GenEval colors tag.`

## What to avoid

- More than 6 slides.
- Full experiment matrix unless it is visually tiny.
- Full per-tag tables.
- Long quotes from the paper.
- Explaining every package install issue.
- Saying “we fully reproduced the paper”.
