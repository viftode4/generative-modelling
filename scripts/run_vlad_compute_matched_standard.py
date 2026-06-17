"""Generate a compute-matched standard baseline for Vlad's DreamShaper run.

This is the Mac-feasible cost-control follow-up after Vlad's standard-vs-Z
run. It reads the original Vlad standard-vs-Z manifest, generates a
higher-step *standard* DDIM image from the same prompt/seed/initial latent,
and writes a comparison manifest where:

- ``standard_path`` points to the new higher-step standard image;
- ``zigzag_path`` points to the existing 4-step Z-Sampling image.

The output manifest is compatible with ``scripts/score_hpsv2_pairs.py``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from zigzag_project.experiment_io import append_jsonl, read_jsonl
from zigzag_project.models import load_dreamshaper_xl_turbo
from zigzag_project.zsampling import make_initial_latents

MODEL_ID = "Lykon/dreamshaper-xl-v2-turbo"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="outputs/vlad_geneval/balanced_18x3/manifest.jsonl",
        help="Original Vlad manifest containing existing 4-step Z-Sampling paths.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory, usually outputs/vlad_geneval/compute_matched_standard9/.",
    )
    parser.add_argument(
        "--standard-steps",
        type=int,
        default=9,
        help=(
            "DDIM steps for the compute-matched standard baseline. 9 roughly "
            "matches the 2.24x cost of 4-step Z-Sampling on Vlad's MPS run."
        ),
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="Standard guidance scale. Defaults to source row gamma_1.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Image size. Defaults to source row image_size.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty --out directory by replacing manifest rows.",
    )
    args = parser.parse_args()

    source_manifest = Path(args.source_manifest)
    source_rows = read_jsonl(source_manifest)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]

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
    upstream_commit = _git_rev_parse(
        ["git", "-C", "external/zigzag-diffusion-sampling", "rev-parse", "--short", "HEAD"]
    )

    handle = None
    sync_device = None
    if not args.dry_run:
        handle = load_dreamshaper_xl_turbo(device=args.device)
        sync_device = handle.device

    rows_written = 0
    for row in source_rows:
        standard_path = image_dir / _standard_filename(row, args.standard_steps)
        standard_runtime_s = None
        cfg = float(args.cfg if args.cfg is not None else row.get("gamma_1", 3.5))
        size = int(args.size if args.size is not None else row.get("image_size", 768))
        seed = int(row["seed"])

        if handle is not None:
            latents = make_initial_latents(
                seed=seed,
                height=size,
                width=size,
                dtype=handle.dtype,
                device=handle.device,
            )
            t0 = time.time()
            image = handle.pipe(
                prompt=str(row["prompt"]),
                num_inference_steps=args.standard_steps,
                guidance_scale=cfg,
                height=size,
                width=size,
                latents=latents.clone(),
            ).images[0]
            _device_synchronize(sync_device)
            standard_runtime_s = time.time() - t0
            image.save(standard_path)

        out_row = _comparison_row(
            row=row,
            source_manifest=source_manifest,
            standard_path=standard_path,
            standard_steps=args.standard_steps,
            standard_cfg=cfg,
            size=size,
            dry_run=args.dry_run,
            standard_runtime_s=standard_runtime_s,
            source_commit=source_commit,
            upstream_commit=upstream_commit,
        )
        append_jsonl(manifest_path, out_row)
        rows_written += 1

    print(f"wrote {rows_written} compute-matched comparison rows -> {manifest_path}")


def _standard_filename(row: dict[str, Any], standard_steps: int) -> str:
    return f"{row['prompt_id']}_seed{int(row['seed'])}_standard{standard_steps}.png"


def _comparison_row(
    *,
    row: dict[str, Any],
    source_manifest: Path,
    standard_path: Path,
    standard_steps: int,
    standard_cfg: float,
    size: int,
    dry_run: bool,
    standard_runtime_s: float | None,
    source_commit: str | None,
    upstream_commit: str | None,
) -> dict[str, Any]:
    return {
        "prompt_id": row["prompt_id"],
        "source_index": int(row["source_index"]),
        "tag": row["tag"],
        "prompt": row["prompt"],
        "seed": int(row["seed"]),
        "standard_path": str(standard_path),
        "zigzag_path": str(row["zigzag_path"]),
        "model_id": row.get("model_id", MODEL_ID),
        "image_size": size,
        "comparison": "compute_matched_standard_vs_existing_zigzag",
        "standard_steps": standard_steps,
        "standard_gamma": standard_cfg,
        "zigzag_steps": int(row.get("steps", 4)),
        "zigzag_gamma_1": float(row.get("gamma_1", 3.5)),
        "zigzag_gamma_2": float(row.get("gamma_2", 0.0)),
        "zigzag_lambda_step": int(row.get("lambda_step", 3)),
        "zigzag_t_max": int(row.get("t_max", 1)),
        "dry_run": dry_run,
        "standard_runtime_s": standard_runtime_s,
        "zigzag_runtime_s": row.get("zigzag_runtime_s"),
        "source_manifest": str(source_manifest),
        "source_standard_path": row.get("standard_path"),
        "source_zigzag_path": row.get("zigzag_path"),
        "source_commit": source_commit,
        "upstream_commit": upstream_commit,
    }


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
