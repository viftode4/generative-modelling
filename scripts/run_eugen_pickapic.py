"""Run Eugen's SD2.1 Pick-a-Pic standard-vs-Z experiment.

Generated images and manifests are written under an ignored output directory.
Use ``--dry-run`` to validate prompt parsing and manifest paths without loading the model.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from zigzag_project.experiment_io import GenerationPair, load_prompt_specs, write_jsonl
from zigzag_project.models import load_sd21
from zigzag_project.zsampling import ZSamplingConfig, make_initial_latents, z_sample_sd21

MODEL_ID = "sd2-community/stable-diffusion-2-1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        required=True,
        help="Markdown file with a fenced jsonl prompt block",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output directory, usually under outputs/")
    parser.add_argument("--steps", type=int, default=50, help="DDIM inference steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="denoising guidance / gamma_1")
    parser.add_argument("--inv-cfg", type=float, default=0.0, help="inversion guidance / gamma_2")
    parser.add_argument("--lambda-step", type=int, default=None)
    parser.add_argument("--t-max", type=int, default=1)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts to run")
    args = parser.parse_args()

    lambda_step = args.lambda_step if args.lambda_step is not None else args.steps - 1
    prompts = load_prompt_specs(args.prompts, prefix="pickapic")
    if args.limit is not None:
        prompts = prompts[:args.limit]
    out_dir = Path(args.out)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    source_commit = _git_rev_parse(["git", "rev-parse", "--short", "HEAD"])
    upstream_commit = _git_rev_parse(
        ["git", "-C", "external/zigzag-diffusion-sampling", "rev-parse", "--short", "HEAD"]
    )

    handle = None
    if not args.dry_run:
        handle = load_sd21(device=args.device)

    rows: list[dict[str, object]] = []
    for prompt in prompts:
        for seed in args.seeds:
            standard_path = image_dir / f"{prompt.prompt_id}_seed{seed}_standard.png"
            zigzag_path = image_dir / f"{prompt.prompt_id}_seed{seed}_zigzag.png"
            standard_runtime_s = None
            zigzag_runtime_s = None

            if standard_path.exists() and zigzag_path.exists():
                print(f"  -> Already done (seed={seed}), skipping.")
                standard_runtime_s = 0.0
                zigzag_runtime_s = 0.0
            elif handle is not None:
                latents = make_initial_latents(
                    seed=seed,
                    height=args.size,
                    width=args.size,
                    dtype=handle.dtype,
                    device=handle.device,
                )

                t0 = time.time()
                standard_image = handle.pipe(
                    prompt=prompt.prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    height=args.size,
                    width=args.size,
                    latents=latents.clone(),
                ).images[0]
                standard_runtime_s = time.time() - t0
                standard_image.save(standard_path)

                config = ZSamplingConfig(
                    gamma_1=args.cfg,
                    gamma_2=args.inv_cfg,
                    infer_step=args.steps,
                    lambda_step=lambda_step,
                    t_max=args.t_max,
                )
                t0 = time.time()
                zigzag_image = z_sample_sd21(
                    handle,
                    prompt=prompt.prompt,
                    config=config,
                    height=args.size,
                    width=args.size,
                    latents=latents.clone(),
                ).images[0]
                zigzag_runtime_s = time.time() - t0
                zigzag_image.save(zigzag_path)

            rows.append(
                GenerationPair(
                    prompt_id=prompt.prompt_id,
                    source_index=prompt.source_index,
                    tag=prompt.tag,
                    prompt=prompt.prompt,
                    seed=seed,
                    standard_path=str(standard_path),
                    zigzag_path=str(zigzag_path),
                    model_id=MODEL_ID,
                    image_size=args.size,
                    steps=args.steps,
                    gamma_1=args.cfg,
                    gamma_2=args.inv_cfg,
                    lambda_step=lambda_step,
                    t_max=args.t_max,
                    dry_run=args.dry_run,
                    standard_runtime_s=standard_runtime_s,
                    zigzag_runtime_s=zigzag_runtime_s,
                    source_commit=source_commit,
                    upstream_commit=upstream_commit,
                ).to_json_dict()
            )

    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(manifest_path, rows)
    print(f"wrote {len(rows)} manifest rows -> {manifest_path}")


def _git_rev_parse(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


if __name__ == "__main__":
    main()
