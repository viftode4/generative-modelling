"""Smoke test: load DreamShaper-xl-v2-turbo and generate one tiny image.

Run from repo root with the venv active:

    .venv/bin/python scripts/smoke_dreamshaper.py

First run will download ~6 GB of weights to the Hugging Face cache. Use
``--sampler both`` to create a standard-vs-Z-Sampling pair from the same
initial latents.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from zigzag_project.models import load_dreamshaper_xl_turbo
from zigzag_project.zsampling import ZSamplingConfig, make_initial_latents, z_sample_sdxl


def _paired_output_path(base: Path, suffix: str, sampler: str) -> Path:
    if sampler != "both":
        return base
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a small yellow dog sitting on grass")
    parser.add_argument("--steps", type=int, default=6, help="turbo models work at ~4-8 steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="denoising guidance / gamma_1")
    parser.add_argument("--inv-cfg", type=float, default=0.0, help="inversion guidance / gamma_2")
    parser.add_argument("--lambda-step", type=int, default=None)
    parser.add_argument("--t-max", type=int, default=1)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/smoke/dreamshaper_smoke.png")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--sampler",
        choices=("standard", "zigzag", "both"),
        default="standard",
        help="Generate with standard DDIM, repo-local Z-Sampling, or both.",
    )
    args = parser.parse_args()

    t0 = time.time()
    handle = load_dreamshaper_xl_turbo(device=args.device)
    print(f"loaded {handle.hf_id} on {handle.device} ({handle.dtype}) in {time.time() - t0:.1f}s")

    latents = make_initial_latents(
        seed=args.seed,
        height=args.size,
        width=args.size,
        dtype=handle.dtype,
        device=handle.device,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sampler in {"standard", "both"}:
        t1 = time.time()
        image = handle.pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            height=args.size,
            width=args.size,
            latents=latents.clone(),
        ).images[0]
        standard_path = _paired_output_path(out_path, "standard", args.sampler)
        image.save(standard_path)
        print(f"standard generated in {time.time() - t1:.1f}s -> {standard_path}")

    if args.sampler in {"zigzag", "both"}:
        t1 = time.time()
        config = ZSamplingConfig(
            gamma_1=args.cfg,
            gamma_2=args.inv_cfg,
            infer_step=args.steps,
            lambda_step=args.lambda_step,
            t_max=args.t_max,
        )
        image = z_sample_sdxl(
            handle,
            prompt=args.prompt,
            config=config,
            height=args.size,
            width=args.size,
            latents=latents.clone(),
        ).images[0]
        zigzag_path = _paired_output_path(out_path, "zigzag", args.sampler)
        image.save(zigzag_path)
        print(f"zigzag generated in {time.time() - t1:.1f}s -> {zigzag_path}")


if __name__ == "__main__":
    main()
