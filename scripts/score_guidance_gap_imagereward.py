"""Score a guidance-gap manifest with ImageReward ONLY.

Run with the isolated env: `.venv-ir/bin/python ...` (transformers==4.44.2).
Reads the same manifest as the main scorer, writes one row per image with the
ir field; merged on image_path by the summarize step. Resumable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="e.g. outputs/.../scores_ir.jsonl")
    p.add_argument("--device", default="cpu", help="cpu is safest for BLIP on Mac")
    args = p.parse_args()

    import ImageReward as RM

    model = RM.load("ImageReward-v1.0", device=args.device)

    done = set()
    if Path(args.out).exists():
        done = {r["image_path"] for r in read_jsonl(args.out)}
        print(f"resume: {len(done)} already scored")

    manifest = read_jsonl(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("a") as fh:
        for row in manifest:
            ip = row["image_path"]
            if ip in done:
                continue
            ir = float(model.score(row["prompt"], ip))
            rec = {"image_path": ip, "prompt_id": row["prompt_id"], "seed": row["seed"],
                   "method": row["method"], "gamma_2": row["gamma_2"], "ir": ir}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n += 1
            if n % 25 == 0:
                print(f"  scored {n} images (last ir={ir:.4f})")
    print(f"ImageReward scored {n} new images -> {out}")


if __name__ == "__main__":
    main()
