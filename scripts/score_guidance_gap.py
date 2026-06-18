"""Score a guidance-gap manifest with the three main-env metrics (HPS v2, AES,
PickScore). ImageReward is scored separately (transformers-v4 env) and merged in
the summarize step. Writes one row per image with hps/aes/pick filled and ir=NaN.
"""
from __future__ import annotations

import argparse
import sys

sys.path.append("src")

from zigzag_project.experiment_io import read_jsonl, write_jsonl
from zigzag_project.metrics import build_main_scorers, make_metric_score_rows
from zigzag_project.models import resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="e.g. outputs/.../scores_main.jsonl")
    p.add_argument("--device", default=None, help="cpu recommended on Mac for metric models")
    args = p.parse_args()

    device = str(resolve_device(args.device))
    # HPS uses its own device internally; AES/Pick are small — cpu is safe and avoids
    # MPS dtype quirks. Override with --device mps to try the GPU.
    score_device = args.device or "cpu"
    scorers = build_main_scorers(device=score_device)

    manifest = read_jsonl(args.manifest)
    rows = make_metric_score_rows(manifest, scorers)  # ir defaults to NaN
    write_jsonl(args.out, rows)
    print(f"scored {len(rows)} images (HPS/AES/PickScore) on {device} -> {args.out}")


if __name__ == "__main__":
    main()
