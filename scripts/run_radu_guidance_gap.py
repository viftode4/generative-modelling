"""Experiment #6 — SDXL guidance-gap generation. Sweeps gamma_2 for standard vs Z-Sampling.
Writes images + manifest to --out. Resumable.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.append("src")

from zigzag_project.experiment_io import append_jsonl, load_prompt_specs, read_jsonl, write_jsonl
from zigzag_project.guidance_gap import expand_variants, image_filename
from zigzag_project.models import load_sdxl_base
from zigzag_project.zsampling import ZSamplingConfig, make_initial_latents, z_sample_sdxl

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SWEEP = [-20.0, -10.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.5, 7.5, 10.0]


def _git_rev(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="experiments/prompts/pickapic_full.md")
    p.add_argument("--out", required=True, help="Run dir, e.g. outputs/radu_guidance_gap/mps_24x1")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--gamma1", type=float, default=5.5)
    p.add_argument("--gamma2-sweep", nargs="+", type=float, default=DEFAULT_SWEEP)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lambda-step", type=int, default=49)
    p.add_argument("--t-max", type=int, default=1)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None, help="First N prompts only")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip model load; only build manifest paths",
    )
    args = p.parse_args()

    prompts = load_prompt_specs(args.prompts, prefix="pickapic")
    if args.limit is not None:
        prompts = prompts[: args.limit]
    variants = expand_variants(gamma_1=args.gamma1, gamma2_sweep=args.gamma2_sweep)

    out_dir = Path(args.out)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    source_commit = _git_rev(["git", "rev-parse", "--short", "HEAD"])
    upstream_commit = _git_rev(
        ["git", "-C", "external/zigzag-diffusion-sampling", "rev-parse", "--short", "HEAD"]
    )

    existing = (
        {r["image_path"]: r for r in read_jsonl(manifest_path)}
        if manifest_path.exists()
        else {}
    )
    rows: dict[str, dict] = dict(existing)

    handle = None if args.dry_run else load_sdxl_base(device=args.device)
    dtype = handle.dtype if handle is not None else None
    device = handle.device if handle is not None else None

    total = len(prompts) * len(args.seeds) * len(variants)
    done = 0
    for spec in prompts:
        for seed in args.seeds:
            latents = None
            if handle is not None:
                latents = make_initial_latents(
                    seed=seed, height=args.size, width=args.size, dtype=dtype, device=device
                )
            for v in variants:
                done += 1
                fname = image_filename(spec.prompt_id, seed, v.method, v.gamma_2)
                img_path = image_dir / fname
                tag = f"[{done}/{total}] {spec.prompt_id} seed{seed} {v.label}"

                if img_path.exists():
                    print(f"  skip (exists) {tag}")
                    runtime = rows.get(str(img_path), {}).get("runtime_s")
                elif handle is None:
                    runtime = None  # dry-run: just record the intended path
                else:
                    print(f"  gen  {tag}")
                    t0 = time.time()
                    if v.method == "standard":
                        image = handle.pipe(
                            prompt=spec.prompt, num_inference_steps=args.steps,
                            guidance_scale=args.gamma1, height=args.size, width=args.size,
                            latents=latents.clone(),
                        ).images[0]
                    else:
                        cfg = ZSamplingConfig(
                            gamma_1=args.gamma1, gamma_2=v.gamma_2, infer_step=args.steps,
                            lambda_step=args.lambda_step, t_max=args.t_max,
                        )
                        image = z_sample_sdxl(
                            handle, prompt=spec.prompt, config=cfg,
                            height=args.size, width=args.size, latents=latents.clone(),
                        ).images[0]
                    runtime = time.time() - t0
                    image.save(img_path)
                    print(f"       done {runtime:.1f}s")

                row = {
                    "prompt_id": spec.prompt_id, "source_index": spec.source_index,
                    "tag": spec.tag, "prompt": spec.prompt, "seed": seed,
                    "method": v.method, "gamma_1": args.gamma1, "gamma_2": v.gamma_2,
                    "image_path": str(img_path), "steps": args.steps,
                    "lambda_step": args.lambda_step, "t_max": args.t_max,
                    "image_size": args.size, "model_id": MODEL_ID, "runtime_s": runtime,
                    "device": str(device) if device else args.device,
                    "source_commit": source_commit, "upstream_commit": upstream_commit,
                }
                if str(img_path) not in rows:
                    append_jsonl(manifest_path, row)
                rows[str(img_path)] = row

    write_jsonl(manifest_path, list(rows.values()))
    print(f"\nwrote {len(rows)} manifest rows -> {manifest_path}")


if __name__ == "__main__":
    main()
