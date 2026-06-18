"""Run Vlad's Experiment #8: SDXL x Pick-a-Pic efficiency comparison.

Reproduces the paper's time-matched efficiency study (Figure 10): for each
standard step budget ``T_s`` we run standard sampling at ``T_s`` steps and a
compute-matched Z-Sampling at ``T_z = T_s // 2`` steps with ``lambda = T_s // 4``
(so the noise-prediction op counts match: ``T_s`` vs ``T_z + 2*lambda``). HPS v2
vs. average time/image across budgets gives the Figure-10 curve.

Both arms of a (prompt, seed, budget) cell share the same initial latent, so
pairwise differences come from the sampler, not initialization.

Generated images are written under an ignored output directory. The manifest is
appended row-by-row and used for resume: a cell already present in the manifest
is skipped, which preserves its measured runtime across restarts (runtime is the
experiment's metric, so it must never be lost to a resume).

Use ``--dry-run`` to validate prompt parsing, the sweep math, and paths without
loading the model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from zigzag_project.experiment_io import append_jsonl, load_prompt_specs, read_jsonl
from zigzag_project.models import load_sdxl
from zigzag_project.zsampling import ZSamplingConfig, make_initial_latents, z_sample_sdxl

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def matched_z_budget(t_s: int) -> tuple[int, int]:
    """Return the compute-matched (T_z, lambda_step) for a standard budget T_s.

    Paper Figure 10 / efficiency section: T_z = T_s/2, lambda = T_z/2 = T_s/4,
    giving Z-Sampling op count T_z + 2*lambda = T_s == standard op count.
    """
    t_z = t_s // 2
    lambda_step = t_s // 4
    return t_z, lambda_step


def validate_budget(t_s: int) -> None:
    t_z, lambda_step = matched_z_budget(t_s)
    if t_z < 2:
        raise SystemExit(f"--ts-list value {t_s} too small: T_z={t_z} must be >= 2")
    if not 0 <= lambda_step < t_z:
        raise SystemExit(
            f"--ts-list value {t_s} invalid: lambda={lambda_step} "
            f"must satisfy 0 <= lambda < T_z={t_z}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        default="experiments/prompts/pickapic_full.md",
        help="Markdown file with a fenced jsonl prompt block",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--out", required=True, help="Output directory, usually under outputs/")
    parser.add_argument(
        "--ts-list",
        nargs="+",
        type=int,
        default=[20, 30, 40, 50],
        help="Standard step budgets T_s to sweep; Z uses T_z=T_s//2, lambda=T_s//4.",
    )
    parser.add_argument("--cfg", type=float, default=5.5, help="denoising guidance / gamma_1")
    parser.add_argument("--inv-cfg", type=float, default=0.0, help="inversion guidance / gamma_2")
    parser.add_argument("--t-max", type=int, default=1)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts to run")
    args = parser.parse_args()

    for t_s in args.ts_list:
        validate_budget(t_s)

    prompts = load_prompt_specs(args.prompts, prefix="pickapic")
    if args.limit is not None:
        prompts = prompts[: args.limit]

    out_dir = Path(args.out)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Resume: keys already recorded in the manifest are skipped so their
    # measured runtime survives restarts.
    done: set[tuple[str, int, int]] = set()
    if manifest_path.exists():
        for row in read_jsonl(manifest_path):
            done.add((str(row["prompt_id"]), int(row["seed"]), int(row["budget_ts"])))
        print(f"Resume: {len(done)} cells already in {manifest_path}, will skip them.")

    source_commit = _git_rev_parse(["git", "rev-parse", "--short", "HEAD"])
    upstream_commit = _git_rev_parse(
        ["git", "-C", "external/zigzag-diffusion-sampling", "rev-parse", "--short", "HEAD"]
    )

    handle = None
    sync_device = None
    if not args.dry_run:
        handle = load_sdxl(device=args.device)
        sync_device = handle.device

    written = 0
    skipped = 0
    for t_s in args.ts_list:
        t_z, lambda_step = matched_z_budget(t_s)
        for prompt in prompts:
            for seed in args.seeds:
                key = (prompt.prompt_id, seed, t_s)
                if key in done:
                    skipped += 1
                    continue

                standard_path = image_dir / f"{prompt.prompt_id}_seed{seed}_ts{t_s}_standard.png"
                zigzag_path = (
                    image_dir / f"{prompt.prompt_id}_seed{seed}_ts{t_s}_tz{t_z}_zigzag.png"
                )
                standard_runtime_s = None
                zigzag_runtime_s = None

                if handle is not None:
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
                        num_inference_steps=t_s,
                        guidance_scale=args.cfg,
                        height=args.size,
                        width=args.size,
                        latents=latents.clone(),
                    ).images[0]
                    _device_synchronize(sync_device)
                    standard_runtime_s = time.time() - t0
                    standard_image.save(standard_path)

                    config = ZSamplingConfig(
                        gamma_1=args.cfg,
                        gamma_2=args.inv_cfg,
                        infer_step=t_z,
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

                row = {
                    "prompt_id": prompt.prompt_id,
                    "source_index": prompt.source_index,
                    # tag encodes the budget so it flows through HPS scoring and
                    # aggregate_hps_pairs groups by budget for the Figure-10 curve.
                    "tag": f"ts{t_s}",
                    "prompt": prompt.prompt,
                    "seed": seed,
                    "budget_ts": t_s,
                    "standard_path": str(standard_path),
                    "standard_steps": t_s,
                    "standard_runtime_s": standard_runtime_s,
                    "zigzag_path": str(zigzag_path),
                    "zigzag_steps": t_z,
                    "zigzag_lambda_step": lambda_step,
                    "zigzag_t_max": args.t_max,
                    "zigzag_runtime_s": zigzag_runtime_s,
                    "gamma_1": args.cfg,
                    "gamma_2": args.inv_cfg,
                    "model_id": MODEL_ID,
                    "image_size": args.size,
                    "dry_run": args.dry_run,
                    "source_commit": source_commit,
                    "upstream_commit": upstream_commit,
                }
                append_jsonl(manifest_path, row)
                done.add(key)
                written += 1
                if handle is not None:
                    print(
                        f"  T_s={t_s} (T_z={t_z}, lambda={lambda_step}) {prompt.prompt_id} "
                        f"seed={seed}: std {standard_runtime_s:.1f}s / Z {zigzag_runtime_s:.1f}s"
                    )

    print(
        f"wrote {written} new manifest rows ({skipped} skipped) -> {manifest_path}"
    )


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


if __name__ == "__main__":
    main()
