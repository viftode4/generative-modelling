"""Smoke test: load DreamShaper-xl-v2-turbo and generate one tiny image.

Run from repo root with the venv active:

    .venv/bin/python scripts/smoke_dreamshaper.py

First run will download ~6 GB of weights to the Hugging Face cache.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from zigzag_project.models import load_dreamshaper_xl_turbo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a small yellow dog sitting on grass")
    parser.add_argument("--steps", type=int, default=6, help="turbo models work at ~4-8 steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="turbo models want low CFG")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/smoke/dreamshaper_smoke.png")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    t0 = time.time()
    handle = load_dreamshaper_xl_turbo(device=args.device)
    print(f"loaded {handle.hf_id} on {handle.device} ({handle.dtype}) in {time.time()-t0:.1f}s")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t1 = time.time()
    image = handle.pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.size,
        width=args.size,
        generator=generator,
    ).images[0]
    print(f"generated in {time.time()-t1:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
