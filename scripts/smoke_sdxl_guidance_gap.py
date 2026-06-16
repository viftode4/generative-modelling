"""Tiny SDXL guidance-gap smoke: 1 prompt, standard + gamma_2 in {0, 5.5}, prints
per-image wall-clock so the MPS time budget can be extrapolated before the full run.
"""
from __future__ import annotations

import sys
import time

sys.path.append("src")

from zigzag_project.models import load_sdxl_base
from zigzag_project.zsampling import ZSamplingConfig, make_initial_latents, z_sample_sdxl

PROMPT = "An elephant in a room full of glassware"


def main() -> None:
    handle = load_sdxl_base()
    print("device", handle.device, "dtype", handle.dtype)
    latents = make_initial_latents(seed=42, height=1024, width=1024,
                                   dtype=handle.dtype, device=handle.device)

    t0 = time.time()
    handle.pipe(prompt=PROMPT, num_inference_steps=50, guidance_scale=5.5,
                height=1024, width=1024, latents=latents.clone()).images[0]
    t_std = time.time() - t0
    print(f"standard: {t_std:.1f}s")

    for g2 in (0.0, 5.5):
        cfg = ZSamplingConfig(gamma_1=5.5, gamma_2=g2, infer_step=50, lambda_step=49, t_max=1)
        t0 = time.time()
        z_sample_sdxl(handle, prompt=PROMPT, config=cfg, height=1024, width=1024,
                      latents=latents.clone()).images[0]
        print(f"zigzag gamma_2={g2}: {time.time() - t0:.1f}s")

    print("\nExtrapolation: 1 std + 10 z per (prompt,seed). "
          "24 prompts x 1 seed ~= 24*std + 240*z seconds.")


if __name__ == "__main__":
    main()
