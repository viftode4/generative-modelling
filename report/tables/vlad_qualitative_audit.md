# Vlad GenEval Balanced 18x3 — Qualitative Audit Notes

Source images: `outputs/vlad_geneval/balanced_18x3/images/`. Scores: `outputs/vlad_geneval/balanced_18x3/scores.jsonl`. This is a manual visual audit of selected pairs, not an official GenEval detector evaluation.

| Pair | Prompt | Tag | Seed | HPS delta | Manual observation | Interpretation |
| --- | --- | --- | ---: | ---: | --- | --- |
| `geneval_259_colors_seed42` | a photo of a blue fire hydrant | colors | 42 | +0.1090 | Standard sampling mostly produces pavement with a small blue smear; Z-Sampling produces a clear hydrant-like object with strong blue coloration. | HPS win aligns with both prompt faithfulness and image quality. This is the cleanest pro-Z example. |
| `geneval_180_counting_seed2026` | a photo of two backpacks | counting | 2026 | +0.0947 | Standard sampling shows a vague scene with one backpack/person-like object; Z-Sampling shows two distinct backpack-like objects. | HPS win plausibly reflects real object/count improvement, not only aesthetics. |
| `geneval_355_position_seed42` | a photo of a couch below a cup | position | 42 | -0.0023 | Both samples place a cup above/on a couch, so both satisfy the coarse couch-below-cup relation. Z-Sampling is sharper, but HPS gives standard a tiny edge. | The only standard win is effectively a metric tie, not strong evidence that Z-Sampling failed spatial composition. |
| `geneval_000_single_object_seed2026` | a photo of a bench | single_object | 2026 | +0.0066 | Both samples look more like close-up wooden planks than a recognizable bench. Z-Sampling is sharper/cleaner but not clearly more semantically correct. | Small HPS gains can be mostly perceptual refinement. Do not treat every positive delta as a GenEval-style success. |
| `geneval_354_position_seed123` | a photo of a wine glass above a kite | position | 123 | +0.0086 | Both samples render a wine glass; the kite is at best tiny/ambiguous. Z-Sampling produces a cleaner glass/scene. | Supports the metric-mismatch caveat: HPS can reward image quality even when relation/object grounding remains ambiguous. |

## Takeaway

The manual audit strengthens the report in two directions at once: the largest HPS wins are visually convincing, but edge cases show that HPSv2.1 alone does not reliably certify GenEval-style compositional correctness. The follow-up detector run in `report/tables/vlad_geneval_detector.md` resolves that gap for this subset: Z-Sampling improves detector pass rate from 31.5% to 50.0%, but color attribution and most position prompts remain hard.
