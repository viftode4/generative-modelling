# Vlad DreamShaper Turbo + GenEval Experiment Config

Owner: Vlad

Purpose: compare standard sampling against Z-Sampling for Vlad's assigned lane under the updated experiment division: **Z-Sampling + Experiment 8**.

## Sources

- Paper/forum: https://openreview.net/forum?id=MKvQH1ekeY
- Paper PDF: https://openreview.net/pdf?id=MKvQH1ekeY
- Upstream code commit: `513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4`
- GenEval prompt source: https://github.com/djghosh13/geneval/blob/main/prompts/evaluation_metadata.jsonl
- HPSv2 scorer: https://github.com/tgxs002/HPSv2

## Fixed settings

| Field | Value |
| --- | --- |
| Model | `Lykon/dreamshaper-xl-v2-turbo` |
| Dataset subset | `experiments/prompts/geneval_vlad_balanced.md` |
| Seeds | `42`, `123`, `2026` |
| Image size | `768` |
| Denoising steps | `4` |
| Scheduler | DDIM + DDIMInverseScheduler |
| `gamma_1` | `3.5` |
| `gamma_2` | `0.0` |
| `lambda_step` | `3` |
| `T_max` | `1` |
| Metrics | GenEval detector pass rate; HPSv2.1 standard-vs-Z win rate |
| Output dir | `outputs/vlad_geneval/balanced_18x3/` |

The DreamShaper settings follow the paper's §4.1 description: turbo model with `T=4`, recommended DreamShaper guidance `gamma_1=3.5`, `gamma_2=0`, and zigzag along the full path (`lambda_step=T-1`).

## Commands

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q

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

uv pip install hpsv2
# hpsv2==1.2.0 may miss this vendored OpenCLIP tokenizer file in a fresh venv.
curl -L -s --fail https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
  -o .venv/lib/python3.10/site-packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
.venv/bin/python scripts/score_hpsv2_pairs.py \
  --manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/scores.jsonl

.venv/bin/python scripts/summarize_vlad_results.py \
  --scores outputs/vlad_geneval/balanced_18x3/scores.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/summary.md

# GenEval detector on Apple Silicon: use CPU, not MPS.
.venv/bin/python -m pip install -e '.[geneval]'
.venv/bin/python scripts/run_geneval_detector.py \
  --manifest outputs/vlad_geneval/balanced_18x3/manifest.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/geneval_scores.jsonl \
  --device cpu

.venv/bin/python scripts/summarize_vlad_geneval.py \
  --geneval-scores outputs/vlad_geneval/balanced_18x3/geneval_scores.jsonl \
  --hps-scores outputs/vlad_geneval/balanced_18x3/scores.jsonl \
  --out outputs/vlad_geneval/balanced_18x3/geneval_summary.md
```

## Expected outputs

- `manifest.jsonl`: one row per `(prompt, seed)` pair with paths to standard and Z-Sampling images.
- `scores.jsonl`: HPSv2 standard/Z scores and `delta = Z - standard`.
- `geneval_scores.jsonl`: GenEval detector pass/fail rows for the same pairs.
- `summary.md`: overall win rate, per-tag win rates, and qualitative gallery pointers.
- `geneval_summary.md`: detector pass-rate summary plus HPS cross-reference.

Generated images stay ignored under `outputs/`; small evidence files
(`manifest.jsonl`, `scores.jsonl`, `geneval_scores.jsonl`, summaries) may be
committed when they support the report.
