"""
Reorganize experiment output images into the format expected by the GenEval evaluator.

GenEval evaluator expects:
    {out_dir}/{prompt_index}/metadata.jsonl   <- single JSON object with tag/include/exclude
    {out_dir}/{prompt_index}/samples/0.png

Run once per method, e.g.:
    python scripts/reorganize_for_geneval_eval.py \
        --manifest /content/drive/MyDrive/exp2_geneval_full/manifest.jsonl \
        --prompts data/prompts/evaluation_metadata.jsonl \
        --method standard \
        --out /content/drive/MyDrive/exp2_geneval_eval/standard

    python scripts/reorganize_for_geneval_eval.py \
        --manifest /content/drive/MyDrive/exp2_geneval_full/manifest.jsonl \
        --prompts data/prompts/evaluation_metadata.jsonl \
        --method resampling \
        --out /content/drive/MyDrive/exp2_geneval_eval/resampling

    python scripts/reorganize_for_geneval_eval.py \
        --manifest /content/drive/MyDrive/exp2_geneval_full/manifest.jsonl \
        --prompts data/prompts/evaluation_metadata.jsonl \
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
    parser.add_argument("--prompts", default="data/prompts/evaluation_metadata.jsonl",
                        help="Path to GenEval evaluation_metadata.jsonl")
    parser.add_argument("--method", required=True, choices=["standard", "resampling", "zigzag"])
    parser.add_argument("--out", required=True, help="Output directory for this method")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)

    # Build lookup: prompt text -> full metadata row (tag, include, exclude, prompt)
    meta_rows = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    prompt_to_meta = {row["prompt"]: row for row in meta_rows}

    key = {"standard": "standard_path", "resampling": "resampling_path", "zigzag": "zigzag_path"}[args.method]

    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    print(f"Reorganizing {len(rows)} images for method '{args.method}' -> {out_dir}")

    missing = 0
    for row in rows:
        src = Path(row[key])
        prompt_text = row["prompt"]
        prompt_index = int(row["prompt_id"].split("_")[1])

        meta = prompt_to_meta.get(prompt_text)
        if meta is None:
            print(f"  WARNING: no metadata found for prompt: {prompt_text!r}")
            missing += 1
            continue

        dest_dir = out_dir / str(prompt_index)
        samples_dir = dest_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata as single JSON object (evaluator uses json.load, not jsonl parsing)
        (dest_dir / "metadata.jsonl").write_text(json.dumps(meta))
        shutil.copy2(src, samples_dir / "0.png")

    print(f"Done. {len(rows) - missing} prompt folders created in {out_dir}")
    if missing:
        print(f"WARNING: {missing} prompts had no matching metadata.")


if __name__ == "__main__":
    main()
