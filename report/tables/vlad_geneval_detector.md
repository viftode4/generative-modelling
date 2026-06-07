# Vlad GenEval Balanced 18x3 — Official Detector Results

Source: `outputs/vlad_geneval/balanced_18x3/geneval_scores.jsonl`, produced on 2026-06-04 with:

```bash
.venv/bin/python scripts/run_geneval_detector.py \
  --manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/geneval_scores.jsonl \
  --device cpu
```

Detector stack: upstream GenEval Mask2Former evaluator logic using `mmdet==2.28.2`, `mmcv-full==1.7.2`, `open-clip-torch==3.3.0`, and CPU execution on Apple Silicon. The local runner supports both mmdet-2.x tuple output and mmdet-3.x `DetDataSample` output.

## Overall detector pass rates

| Metric | Standard | Z-Sampling | Delta (Z - standard) |
| --- | ---: | ---: | ---: |
| GenEval detector pass rate | 31.5% (17/54) | 50.0% (27/54) | +18.5% |

Pair categories: both pass 17; Z-only 10; standard-only 0; neither 27.

## Detector pass rate by GenEval tag

| Tag | Pairs | Std pass | Z pass | Std rate | Z rate | Delta | Z-only | Std-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| color_attr | 9 | 0 | 0 | 0.0% | 0.0% | +0.0% | 0 | 0 |
| colors | 9 | 5 | 8 | 55.6% | 88.9% | +33.3% | 3 | 0 |
| counting | 9 | 2 | 5 | 22.2% | 55.6% | +33.3% | 3 | 0 |
| position | 9 | 0 | 1 | 0.0% | 11.1% | +11.1% | 1 | 0 |
| single_object | 9 | 9 | 9 | 100.0% | 100.0% | +0.0% | 0 | 0 |
| two_object | 9 | 1 | 4 | 11.1% | 44.4% | +33.3% | 3 | 0 |

## Agreement with HPSv2.1

HPSv2.1 was available for the same 54 pairs. On the 10 detector-directional pairs where exactly one sampler passed GenEval, HPSv2 also favored Z-Sampling in all 10 cases. The remaining 44 pairs are detector-tied: either both pass or both fail.

## Interpretation

This resolves the previous “detector missing” gap for Vlad's 18-prompt subset. It does **not** make the run a full paper-scale GenEval reproduction: the subset is small, the model is DreamShaper-xl-v2-turbo rather than the SDXL/SD2.1 headline table setting, and the detector ran on CPU/Mac with a modernized wrapper. But it is now a real GenEval detector result for the exact existing Vlad images.
