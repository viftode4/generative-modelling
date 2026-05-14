# Vlad DreamShaper + GenEval Balanced Run

## Metadata

- **Date:** 2026-05-14
- **Owner:** Vlad / AI assistant
- **Goal:** Run Vlad's report-ready balanced GenEval standard-vs-Z-Sampling experiment for DreamShaper-xl-v2-turbo and score pairs with HPSv2.
- **Source code commit:** Working tree based on `7fea0cc` with uncommitted local experiment-runner changes.
- **Upstream commit:** `external/zigzag-diffusion-sampling` at `513eb51`.
- **Environment:** `.venv` CPython 3.10; `torch==2.12.0`; `diffusers==0.37.1`; `hpsv2==1.2.0`; `pytest==7.2.0` after HPSv2 install.
- **Hardware:** Apple M4 Max, MPS device, fp16 generation.
- **Seed(s):** `42`, `123`, `2026`

## Command(s)

Generation:

```bash
.venv/bin/python scripts/run_vlad_geneval.py \
  --prompts experiments/prompts/geneval_vlad_balanced.md \
  --seeds 42 123 2026 \
  --steps 4 \
  --cfg 3.5 \
  --inv-cfg 0 \
  --lambda-step 3 \
  --t-max 1 \
  --size 768 \
  --out outputs/vlad_geneval/balanced_18x3
```

Scoring and summary:

```bash
uv pip install hpsv2
curl -L -s --fail https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
  -o .venv/lib/python3.10/site-packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
.venv/bin/python scripts/score_hpsv2_pairs.py \
  --manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/scores.jsonl
.venv/bin/python scripts/summarize_vlad_results.py \
  --scores outputs/vlad_geneval/balanced_18x3/scores.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/summary.md
```

Verification:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

## Inputs

- Prompt file: `experiments/prompts/geneval_vlad_balanced.md`
- Prompts: 18 total, first 3 prompts from each GenEval tag (`single_object`, `two_object`, `counting`, `colors`, `position`, `color_attr`).
- Model: `Lykon/dreamshaper-xl-v2-turbo`
- Z-Sampling config: `gamma_1=3.5`, `gamma_2=0.0`, `infer_step=4`, `lambda_step=3`, `T_max=1`
- Metric: HPSv2 via `hpsv2.score(..., hps_version="v2.1")`

## Outputs

- Manifest: `outputs/vlad_geneval/balanced_18x3/manifest.jsonl`
- Images: `outputs/vlad_geneval/balanced_18x3/images/`
- Scores: `outputs/vlad_geneval/balanced_18x3/scores.jsonl`
- Summary: `outputs/vlad_geneval/balanced_18x3/summary.md`
- Output size: about 91 MB.
- Generated image count: 108 PNGs.
- Manifest rows: 54 paired rows.
- Score rows: 54 paired rows.

## Results

Overall HPSv2 pair summary:

- Pairs: 54
- Z-Sampling wins: 53
- Standard wins: 1
- Ties: 0
- Z-Sampling win rate: 98.1%
- Mean HPS delta (Z - standard): 0.0358

Per-tag Z win rates from `summary.md`:

| Tag | Pairs | Z win rate | Mean delta |
| --- | ---: | ---: | ---: |
| color_attr | 9 | 100.0% | 0.0387 |
| colors | 9 | 100.0% | 0.0577 |
| counting | 9 | 100.0% | 0.0322 |
| position | 9 | 88.9% | 0.0265 |
| single_object | 9 | 100.0% | 0.0222 |
| two_object | 9 | 100.0% | 0.0373 |

Runtime totals from manifest:

- Standard generation total: 332.4 s; mean 6.16 s per pair.
- Z-Sampling generation total: 745.5 s; mean 13.80 s per pair.
- Z-Sampling was about 2.24× the standard generation runtime in this run.

## Observations

- Generation completed successfully for all 18 prompts × 3 seeds × 2 methods.
- HPSv2 package installation downgraded `pytest` from 9.0.3 to 7.2.0 in the local venv; the test suite still passed afterward.
- `hpsv2==1.2.0` was missing `bpe_simple_vocab_16e6.txt.gz` in its vendored OpenCLIP path. Downloading the OpenAI CLIP vocab file to the expected venv package path fixed scoring.
- Diffusers/HPSv2 emitted deprecation/future warnings; none blocked generation or scoring.

## Interpretation

The balanced subset strongly favors Z-Sampling under HPSv2 scoring, with one standard win in the `position` category. This is enough for Vlad's first quantitative report table, but it should be paired with manual image inspection because HPSv2 is a preference proxy and this is a small subset, not full official GenEval detector evaluation.

## Follow-up

- Manually inspect the summary's top wins/losses and choose qualitative examples for the report.
- Consider adding the HPSv2 vocab workaround to setup notes or replacing HPSv2 with a more reproducible install path if teammates need to run scoring.
- If time allows, run official GenEval detector scoring separately; do not conflate HPSv2 results with official GenEval object-correctness scores.
