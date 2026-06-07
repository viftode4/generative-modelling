# End-term Deck Design Brief

## Main narrative

Use a **plan → implementation → evidence → limitations** story.

The final presentation should feel like a project closure:

1. What we planned at midterm.
2. What we actually implemented.
3. What results we can defend.
4. What blocked the full planned scope.
5. What the final report should honestly claim.

## Tone

- Direct, technical, not flashy.
- Scientific honesty over hype.
- Short bullets only.
- Avoid sounding like the whole paper was fully reproduced.

## Visual style

- White or very light background.
- One accent color for Z-Sampling, e.g. blue or purple.
- Standard sampling = gray.
- Resampling = orange or dashed gray.
- Z-Sampling = accent color.
- Use the same colors in diagrams and result tables.

## Slide layout

1. **Plan/motivation**: paper question + midterm plan in one line.
2. **Method diagram**: Standard vs Resampling vs Z-Sampling, arrows only.
3. **Plan vs actual**: small 3-row table + pipeline.
4. **Results table**: three rows maximum + optional image pair.
5. **Challenge cards**: scope, dependencies, metrics, runtime.
6. **Takeaway**: one bounded conclusion + feedback ask.

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
