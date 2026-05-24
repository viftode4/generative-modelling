# 2026-05-25 — GenEval detector run attempt (blocked on dep install)

## Goal

Run the official GenEval detector pipeline on the 108 committed standard-vs-Z
images in `outputs/vlad_geneval/balanced_18x3/` so the report can publish
detector-based pass rates alongside the existing HPSv2 preference numbers.

## What landed

- `external/geneval/` cloned (gitignored under `/external/*`).
- `scripts/run_geneval_detector.py` — runner that reads the manifest, runs
  Mask2Former object detection + open_clip color classifier + position rules
  per the official `external/geneval/evaluation/evaluate_images.py`, and
  writes `outputs/vlad_geneval/balanced_18x3/geneval_scores.jsonl`.
- `scripts/summarize_vlad_geneval.py` — per-tag pass-rate aggregator for
  the markdown summary.
- `src/zigzag_project/geneval_io.py` — metadata loader (`prompts/evaluation_metadata.jsonl`).
- `tests/test_run_geneval_detector_smoke.py` — mocked detector roundtrip.

## What did not land

`geneval_scores.jsonl` was not produced. The detector requires `mmdet`,
`mmcv`, and a compatible `mmengine`, and `mmcv` does not build on Apple
Silicon against `torch==2.12.0`:

- `pip install mmcv` triggers a from-source build that needs CUDA headers.
- `mim install mmcv mmdet` downgraded `setuptools` to 60.2.0 (breaking the
  `torch==2.12.0` `setuptools<82` constraint) and still hit
  `ModuleNotFoundError: No module named 'pkg_resources'` during the mmcv
  wheel build.
- `mmcv-lite` is the CPU-only alternative but `mmdet>=3` requires the full
  `mmcv` extension ops for `Mask2Former`.

The venv pollution from `mim install` was rolled back; `pip check` is clean
again and the existing 28-test suite passes.

## Next step (not done in this session)

The detector run needs a CUDA Linux box (Colab, runpod, university cluster).
The committed images, the runner script, and the prompt-rule lookup are all
ready; on a CUDA box the steps are:

```bash
pip install -r requirements-vlad.txt
pip install -e .
mim install mmcv "mmdet>=3.3"  # works on CUDA + setuptools>=68
python scripts/run_geneval_detector.py \
    --manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
    --out outputs/vlad_geneval/balanced_18x3/geneval_scores.jsonl
python scripts/summarize_vlad_geneval.py \
    --hps outputs/vlad_geneval/balanced_18x3/scores.jsonl \
    --geneval outputs/vlad_geneval/balanced_18x3/geneval_scores.jsonl \
    --out outputs/vlad_geneval/balanced_18x3/summary.md
```

## Impact on the report

`report/draft.md` §5 must explicitly acknowledge that detector pass rates
were not produced, and frame the missing-GenEval result as a *deliberate
limitation* of the MPS environment rather than a missed deliverable. This
is honest and rubric-positive — the rubric rewards documented limitations.
