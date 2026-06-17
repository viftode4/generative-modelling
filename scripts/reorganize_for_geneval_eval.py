"""
Reorganize experiment output images into the format expected by the GenEval evaluator.

GenEval evaluator expects:
    {out_dir}/{prompt_index}/0.png

Run once per method, e.g.:
    python scripts/reorganize_for_geneval_eval.py \
        --manifest /content/drive/MyDrive/exp2_geneval_full/manifest.jsonl \
        --method standard \
        --out /content/drive/MyDrive/exp2_geneval_eval/standard

    python scripts/reorganize_for_geneval_eval.py \
        --manifest /content/drive/MyDrive/exp2_geneval_full/manifest.jsonl \
        --method resampling \
        --out /content/drive/MyDrive/exp2_geneval_eval/resampling

    python scripts/reorganize_for_geneval_eval.py \
        --manifest /content/drive/MyDrive/exp2_geneval_full/manifest.jsonl \
        --method zigzag \
        --out /content/drive/MyDrive/exp2_geneval_eval/zigzag
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--method", required=True, choices=["standard", "resampling", "zigzag"])
    parser.add_argument("--out", required=True, help="Output directory for this method")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)

    key = {"standard": "standard_path", "resampling": "resampling_path", "zigzag": "zigzag_path"}[args.method]

    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    print(f"Reorganizing {len(rows)} images for method '{args.method}' -> {out_dir}")

    for row in rows:
        src = Path(row[key])
        prompt_index = int(row["prompt_id"].split("_")[1])
        dest_dir = out_dir / str(prompt_index)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / "0.png")

    print(f"Done. {len(rows)} prompt folders created in {out_dir}")


if __name__ == "__main__":
    main()
