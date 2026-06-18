"""Score guidance-gap manifest with HPS v2, AES, PickScore one metric at a time.

Each metric loads, scores all 381 images, writes to disk, then unloads before
the next metric loads.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

sys.path.append("src")

from zigzag_project.experiment_io import read_jsonl
from zigzag_project.models import resolve_device

METRICS = ("hps", "aes", "pick")
PASSTHROUGH = ("prompt_id", "source_index", "tag", "prompt", "seed",
               "method", "gamma_2", "gamma_1", "image_path")


def _load_done(path: Path) -> dict[str, dict]:
    """Return {image_path: row} for already-scored rows in path."""
    if not path.exists():
        return {}
    return {r["image_path"]: r for r in read_jsonl(str(path))}


def _score_metric(metric: str, manifest: list[dict], out: Path, device: str) -> None:
    done = _load_done(out)
    todo = [r for r in manifest if r["image_path"] not in done]
    if not todo:
        print(f"  {metric}: all {len(manifest)} already scored, skipping")
        return

    print(f"  {metric}: loading model...")
    if metric == "hps":
        from zigzag_project.metrics import HpsV2Scorer
        scorer = HpsV2Scorer()
    elif metric == "aes":
        from zigzag_project.metrics import AesScorer
        scorer = AesScorer(device=device)
    else:
        from zigzag_project.metrics import PickScorer
        scorer = PickScorer(device=device)

    print(f"  {metric}: scoring {len(todo)} images ({len(done)} already done)...")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for i, row in enumerate(todo, 1):
            val = float(scorer(row["image_path"], row["prompt"]))
            rec = {k: row[k] for k in PASSTHROUGH if k in row}
            rec[metric] = val
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if i % 25 == 0:
                print(f"    {i}/{len(todo)} (last {metric}={val:.4f})")

    del scorer
    gc.collect()
    print(f"  {metric}: done -> {out}")


def _merge(per_metric: dict[str, Path], manifest: list[dict], out: Path) -> None:
    """Join per-metric files on image_path into one row per image."""
    by_path: dict[str, dict] = {}
    for row in manifest:
        ip = row["image_path"]
        by_path[ip] = {k: row[k] for k in PASSTHROUGH if k in row}
        for m in METRICS:
            by_path[ip][m] = math.nan

    for metric, path in per_metric.items():
        for row in read_jsonl(str(path)):
            ip = row["image_path"]
            if ip in by_path:
                by_path[ip][metric] = row[metric]

    rows = list(by_path.values())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"merged {len(rows)} rows -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="final merged scores_main.jsonl")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or "cpu"
    resolve_device(device)

    manifest = read_jsonl(args.manifest)
    out = Path(args.out)
    scratch = out.parent / "scratch"

    per_metric = {m: scratch / f"scores_{m}.jsonl" for m in METRICS}
    for metric, path in per_metric.items():
        _score_metric(metric, manifest, path, device)

    _merge(per_metric, manifest, out)
    print(f"scored {len(manifest)} images (HPS/AES/PickScore) -> {out}")


if __name__ == "__main__":
    main()
