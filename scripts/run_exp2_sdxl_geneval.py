"""
Run Gouri's Experiment #2: SDXL on GenEval
Comparing Standard Sampling, Resampling, and Z-Sampling.
"""
from __future__ import annotations

import argparse
import time
import json
from pathlib import Path
import torch
import sys
import os

# Point Python to the src folder
sys.path.append(os.path.abspath("src"))

# Import team utilities
from zigzag_project.dataset_utils import load_geneval_prompts
from zigzag_project.model_utils import load_pipeline_standard
# importing resampling_sdxl function
from zigzag_project.zsampling import ZSamplingConfig, make_initial_latents, z_sample_sdxl, resampling_sdxl

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="data/prompts/evaluation_metadata.jsonl", help="Path to GenEval jsonl")
    parser.add_argument("--out", default="outputs/exp2_geneval", help="Output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds to run")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.5)
    parser.add_argument("--inv-cfg", type=float, default=0.0)
    parser.add_argument("--lambda-step", type=int, default=49)
    parser.add_argument("--t-max", type=int, default=1)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--device", default="cpu") 
    parser.add_argument("--test-run", action="store_true", help="Only run the first 2 prompts for testing")
    args = parser.parse_args()

    # 1. Load Prompts
    print(f"Loading GenEval prompts from {args.prompts}...")
    df_prompts = load_geneval_prompts(args.prompts)
    if args.test_run:
        df_prompts = df_prompts.head(2)
        print("TEST RUN: Only running the first 2 prompts.")

    out_dir = Path(args.out)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Model
    print(f"Loading SDXL Model on {args.device.upper()}...")
    pipe = load_pipeline_standard("sdxl", device=args.device)
    
    class ModelHandleWrapper:
        def __init__(self, pipeline):
            self.pipe = pipeline
            self.inverse_scheduler = pipeline.inv_scheduler if hasattr(pipeline, "inv_scheduler") else pipeline.scheduler
            self.dtype = pipeline.unet.dtype
            self.device = pipeline.device
    
    handle = ModelHandleWrapper(pipe)

    # 3. Generate Images
    manifest_rows = []
    
    for index, row in df_prompts.iterrows():
        prompt_text = row["prompt"]
        prompt_id = f"prompt_{index:04d}"
        print(f"\nProcessing [{index+1}/{len(df_prompts)}]: {prompt_text}")
          
            
        for seed in args.seeds:
            latents = make_initial_latents(
                seed=seed, height=args.size, width=args.size,
                dtype=handle.dtype, device=torch.device(args.device)
            )

            std_path = image_dir / f"{prompt_id}_seed{seed}_standard.png"
            res_path = image_dir / f"{prompt_id}_seed{seed}_resampling.png"
            zig_path = image_dir / f"{prompt_id}_seed{seed}_zigzag.png"

            # A. Standard Sampling
            print("  -> Running Standard...")
            t0 = time.time()
            std_image = pipe(
                prompt=prompt_text, num_inference_steps=args.steps,
                guidance_scale=args.cfg, height=args.size, width=args.size,
                latents=latents.clone()
            ).images[0]
            std_time = time.time() - t0
            std_image.save(std_path)
            print(f"  -> Standard done ({std_time:.1f}s)")

            #B. Resampling (lug, 2022)
            print("  -> Running Resampling...")
            config = ZSamplingConfig(
                gamma_1=args.cfg, gamma_2=args.inv_cfg, infer_step=args.steps,
                lambda_step=args.lambda_step, t_max=args.t_max
            )
            t0 = time.time()
            res_image = resampling_sdxl(
                handle, prompt=prompt_text, config=config,
                height=args.size, width=args.size, latents=latents.clone()
            ).images[0]
            res_time = time.time() - t0
            res_image.save(res_path)
            print(f"  -> Resampling done ({res_time:.1f}s)")

            # C. Z-Sampling 
            print("  -> Running Z-Sampling...")
            t0 = time.time()
            zig_image = z_sample_sdxl(
                handle, prompt=prompt_text, config=config,
                height=args.size, width=args.size, latents=latents.clone()
            ).images[0]
            zig_time = time.time() - t0
            zig_image.save(zig_path)
            print(f"  -> Z-Sampling done ({zig_time:.1f}s)")

            manifest_rows.append({
                "prompt_id": prompt_id, "prompt": prompt_text, "seed": seed,
                "standard_path": str(std_path), "resampling_path": str(res_path),
                "zigzag_path": str(zig_path), "standard_time_s": std_time,
                "resampling_time_s": res_time, "zigzag_time_s": zig_time
            })

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for r in manifest_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nExperiment Complete! Saved {len(manifest_rows)} rows to {manifest_path}")

if __name__ == "__main__":
    main()