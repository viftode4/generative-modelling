# Vlad GenEval — Figure Pairs

Standard vs Z-Sampling pairs selected from the balanced GenEval run
(`outputs/vlad_geneval/balanced_18x3/`). Provenance for each pair is in
`../../outputs/vlad_geneval/balanced_18x3/manifest.jsonl` and HPSv2 deltas
in `scores.jsonl`.

| File stem | Prompt | Tag | Seed | HPS delta (Z − std) |
| --- | --- | --- | ---: | ---: |
| `geneval_259_colors_seed42` | a photo of a blue fire hydrant | colors | 42 | +0.1090 |
| `geneval_180_counting_seed2026` | a photo of two backpacks | counting | 2026 | +0.0947 |
| `geneval_355_position_seed42` | a photo of a couch below a cup | position | 42 | −0.0023 |

The first two are top Z-Sampling wins; the third is the single pair on
which standard sampling beat Z-Sampling in the 54-pair run.
