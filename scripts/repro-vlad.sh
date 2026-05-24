#!/usr/bin/env bash
# One-shot pipeline driver for Vlad's balanced GenEval experiment.
#
# Generates 18 prompts x 3 seeds x {standard, Z-Sampling} = 108 images with
# DreamShaper-xl-v2-turbo on MPS/CUDA, scores them with HPSv2.1, and writes
# a markdown summary. Outputs land under outputs/vlad_geneval/balanced_18x3/.
#
# Prerequisites:
#   .venv/bin/pip install -e '.[dev,scoring]'
#   # then patch the OpenCLIP BPE vocab per scripts/score_hpsv2_pairs.py
#
# Usage:
#   scripts/repro-vlad.sh [extra args passed to run_vlad_geneval.py]
#
# Pass --force to overwrite an existing run directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-.venv/bin/python}"
OUT="outputs/vlad_geneval/balanced_18x3"
PROMPTS="experiments/prompts/geneval_vlad_balanced.md"
SEEDS=(42 123 2026)

echo "[1/3] generate standard + Z-Sampling pairs -> $OUT"
"$PY" scripts/run_vlad_geneval.py \
    --prompts "$PROMPTS" \
    --seeds "${SEEDS[@]}" \
    --out "$OUT" \
    "$@"

echo "[2/3] score with HPSv2.1"
"$PY" scripts/score_hpsv2_pairs.py \
    --manifest "$OUT/manifest.jsonl" \
    --out "$OUT/scores.jsonl"

echo "[3/3] summarize"
"$PY" scripts/summarize_vlad_results.py \
    --scores "$OUT/scores.jsonl" \
    --out "$OUT/summary.md"

echo "done. summary: $OUT/summary.md"
