"""Run an SDXL Pick-a-Pic sweep over Z-Sampling lambda-step values.

The default run matches the requested compact ablation: 9 prompts x 3 seeds
for each lambda-step in ``0 3 7 15 25 35 50``. Generated images and manifests
are written under an ignored output directory. Use ``--dry-run`` to validate
prompt parsing and manifest paths without loading the model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from zigzag_project.experiment_io import GenerationPair, append_jsonl, load_prompt_specs
from zigzag_project.models import load_sdxl_base, resolve_device
from zigzag_project.zsampling import ZSamplingConfig, make_initial_latents, z_sample_sdxl

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SEEDS = [42, 123, 2026]
DEFAULT_LAMBDA_STEPS = [0, 3, 7, 15, 25, 35, 50]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        default="experiments/prompts/pickapic_full.md",
        help="Markdown file with a fenced jsonl prompt block",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--lambda-steps",
        nargs="+",
        type=int,
        default=DEFAULT_LAMBDA_STEPS,
        help="Number of early denoising steps refined with Z-Sampling",
    )
    parser.add_argument("--out", required=True, help="Output directory, usually under outputs/")
    parser.add_argument(
        "--steps",
        type=int,
        default=51,
        help="DDIM inference steps. Must be greater than max --lambda-steps.",
    )
    parser.add_argument("--cfg", type=float, default=5.5, help="denoising guidance / gamma_1")
    parser.add_argument("--inv-cfg", type=float, default=0.0, help="inversion guidance / gamma_2")
    parser.add_argument("--t-max", type=int, default=1)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow real generation on CPU. SDXL CPU runs are extremely slow.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty --out directory (overwrites manifest and images).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=9,
        help="Process only N prompts after --start. Defaults to the requested 9-prompt subset.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip the first N loaded prompts before applying --limit.",
    )
    args = parser.parse_args()

    lambda_steps = sorted(dict.fromkeys(args.lambda_steps))
    invalid_lambda_steps = [value for value in lambda_steps if value < 0 or value >= args.steps]
    if invalid_lambda_steps:
        raise SystemExit(
            "--lambda-steps values must satisfy 0 <= lambda_step < --steps. "
            f"Invalid values for --steps {args.steps}: {invalid_lambda_steps}"
        )

    prompts = load_prompt_specs(args.prompts, prefix="pickapic")
    if args.start < 0:
        raise SystemExit("--start must be non-negative.")
    prompts = prompts[args.start :]
    if args.limit is not None:
        prompts = prompts[: args.limit]

    out_dir = Path(args.out)
    image_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.jsonl"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"--out {out_dir} is non-empty. Pass --force to overwrite, or pick a new path."
        )
    image_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest_path.unlink()

    source_commit = _git_rev_parse(["git", "rev-parse", "--short", "HEAD"])
    upstream_commit = _submodule_rev_parse("external/zigzag-diffusion-sampling")
    if source_commit is None:
        print("WARNING: could not resolve repo HEAD; manifest source_commit will be null.")
    if upstream_commit is None:
        print("WARNING: could not resolve upstream submodule HEAD; upstream_commit will be null.")

    handle = None
    sync_device = None
    if not args.dry_run:
        resolved_device = resolve_device(args.device)
        if resolved_device.type == "cpu" and not args.allow_cpu:
            raise SystemExit(
                "SDXL generation on CPU is not practical. Use a CUDA/MPS machine, "
                "pass --device cuda, or pass --allow-cpu if intentionally running slowly."
            )
        handle = load_sdxl_base(device=args.device)
        sync_device = handle.device

    rows_written = 0
    standard_cache: dict[tuple[str, int], tuple[str, float | None]] = {}
    for prompt in prompts:
        for seed in args.seeds:
            standard_path = image_dir / f"{prompt.prompt_id}_seed{seed}_standard.png"
            latents = None
            standard_runtime_s = None

            if handle is not None:
                latents = make_initial_latents(
                    seed=seed,
                    height=args.size,
                    width=args.size,
                    dtype=handle.dtype,
                    device=handle.device,
                    channels=handle.pipe.unet.config.in_channels,
                    vae_scale_factor=handle.pipe.vae_scale_factor,
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
                _device_synchronize(sync_device)
                standard_runtime_s = time.time() - t0
                standard_image.save(standard_path)

            standard_cache[(prompt.prompt_id, seed)] = (str(standard_path), standard_runtime_s)

            for lambda_step in lambda_steps:
                zigzag_path = (
                    image_dir / f"{prompt.prompt_id}_seed{seed}_lambda{lambda_step:02d}_zigzag.png"
                )
                zigzag_runtime_s = None

                if handle is not None:
                    config = ZSamplingConfig(
                        gamma_1=args.cfg,
                        gamma_2=args.inv_cfg,
                        infer_step=args.steps,
                        lambda_step=lambda_step,
                        t_max=args.t_max,
                    )
                    t0 = time.time()
                    zigzag_image = z_sample_sdxl(
                        handle,
                        prompt=prompt.prompt,
                        config=config,
                        height=args.size,
                        width=args.size,
                        latents=latents.clone(),
                    ).images[0]
                    _device_synchronize(sync_device)
                    zigzag_runtime_s = time.time() - t0
                    zigzag_image.save(zigzag_path)

                cached_standard_path, cached_standard_runtime_s = standard_cache[
                    (prompt.prompt_id, seed)
                ]
                row = GenerationPair(
                    prompt_id=prompt.prompt_id,
                    source_index=prompt.source_index,
                    tag=prompt.tag,
                    prompt=prompt.prompt,
                    seed=seed,
                    standard_path=cached_standard_path,
                    zigzag_path=str(zigzag_path),
                    model_id=MODEL_ID,
                    image_size=args.size,
                    steps=args.steps,
                    gamma_1=args.cfg,
                    gamma_2=args.inv_cfg,
                    lambda_step=lambda_step,
                    t_max=args.t_max,
                    dry_run=args.dry_run,
                    standard_runtime_s=cached_standard_runtime_s,
                    zigzag_runtime_s=zigzag_runtime_s,
                    source_commit=source_commit,
                    upstream_commit=upstream_commit,
                ).to_json_dict()
                append_jsonl(manifest_path, row)
                rows_written += 1

    print(f"wrote {rows_written} manifest rows -> {manifest_path}")


def _device_synchronize(device: torch.device | None) -> None:
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _git_rev_parse(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"WARNING: {' '.join(cmd)} failed: {exc}", file=sys.stderr)
        return None


def _submodule_rev_parse(path: str) -> str | None:
    submodule = Path(path)
    if not (submodule / ".git").exists():
        print(f"WARNING: upstream submodule is not initialized at {path}.", file=sys.stderr)
        return None
    return _git_rev_parse(["git", "-C", path, "rev-parse", "--short", "HEAD"])


if __name__ == "__main__":
    main()
