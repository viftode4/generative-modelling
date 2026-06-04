
## Run from root with:
## PYTHONPATH=src python -m scripts.smoke_resampling_sdxl --device cpu --size 512 --steps 50
## (use cuda instead of cpu if available)

from __future__ import annotations

import argparse
import time
from pathlib import Path

from zigzag_project.model_utils import load_pipeline_standard
from zigzag_project.zsampling import make_initial_latents, resampling_sdxl
from zigzag_project.resampling_utils import ResamplingConfig


class ModelHandleWrapper:
    def __init__(self, pipeline):
        self.pipe = pipeline
        self.inverse_scheduler = pipeline.scheduler
        self.dtype = pipeline.unet.dtype
        self.device = pipeline.device
        self.hf_id = "stabilityai/stable-diffusion-xl-base-1.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a small yellow dog sitting on grass")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.5)
    parser.add_argument("--jump-length", type=int, default=10)
    parser.add_argument("--jump-n-sample", type=int, default=10)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="outputs/smoke/resampling_sdxl.png")
    args = parser.parse_args()

    pipe = load_pipeline_standard("sdxl", device=args.device)
    handle = ModelHandleWrapper(pipe)

    latents = make_initial_latents(
        seed=args.seed,
        height=args.size,
        width=args.size,
        dtype=handle.dtype,
        device=handle.device,
    )

    config = ResamplingConfig(
        gamma_1=args.cfg,
        infer_step=args.steps,
        jump_length=args.jump_length,
        jump_n_sample=args.jump_n_sample,
    )

    effective_j, effective_r, was_clamped = config.resolved_jump_params()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    image = resampling_sdxl(
        handle,
        prompt=args.prompt,
        config=config,
        height=args.size,
        width=args.size,
        latents=latents.clone(),
    ).images[0]
    runtime_s = time.time() - t0

    image.save(out_path)

    print(f"saved -> {out_path}")
    print(f"runtime: {runtime_s:.1f}s")
    print(
        f"requested j={config.jump_length}, r={config.jump_n_sample}; "
        f"effective j={effective_j}, r={effective_r}; "
        f"clamped={was_clamped}"
    )


if __name__ == "__main__":
    main()