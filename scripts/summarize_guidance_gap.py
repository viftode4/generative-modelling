"""Merge guidance-gap scores, aggregate by gamma_2, and output a Markdown table + figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append("src")

from zigzag_project.experiment_io import read_jsonl, write_jsonl
from zigzag_project.guidance_gap import aggregate_guidance_gap, merge_score_rows

METRIC_LABELS = {"hps": "HPS v2", "aes": "AES", "pick": "PickScore", "ir": "ImageReward"}
DISPLAY_SCALE = {"hps": 100.0, "aes": 1.0, "pick": 1.0, "ir": 1.0}  # HPS x100 to match paper


def render_table(agg: dict, gamma_1: float) -> str:
    lines = [
        "# SDXL Guidance Gap Ablation (Pick-a-Pic)", "",
        f"gamma_1 = {gamma_1} (fixed). Guidance gap delta_gamma = gamma_1 - gamma_2.",
        "HPS shown x100 to match the paper. Standard baseline means: "
        + ", ".join(f"{METRIC_LABELS[m]}={agg['standard'][m]['mean']*DISPLAY_SCALE[m]:.3f}"
                    for m in ('hps', 'aes', 'pick', 'ir')),
        "",
        "| gamma_2 | gap | HPS v2 | AES | PickScore | ImageReward | HPS win% | Pick win% |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g2, e in sorted(agg["by_gamma2"].items()):
        def cell(m, e=e):
            return f"{e[m]['mean']*DISPLAY_SCALE[m]:.3f}"
        def winp(m, e=e):
            wr = e[m]["win_rate"]
            return "n/a" if wr is None else f"{wr*100:.0f}%"
        lines.append(
            f"| {g2:g} | {e['gap']:g} | {cell('hps')} | {cell('aes')} | "
            f"{cell('pick')} | {cell('ir')} | {winp('hps')} | {winp('pick')} |"
        )
    return "\n".join(lines) + "\n"


def render_figure(agg: dict, gamma_1: float, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("hps", "aes", "pick", "ir")
    g2s = sorted(agg["by_gamma2"].keys())
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, m in zip(axes, metrics, strict=True):
        s = DISPLAY_SCALE[m]
        means = [agg["by_gamma2"][g][m]["mean"] * s for g in g2s]
        stds = [agg["by_gamma2"][g][m]["std"] * s for g in g2s]
        ax.errorbar(g2s, means, yerr=stds, marker="o", capsize=3, label="Z-Sampling")
        ax.axhline(agg["standard"][m]["mean"] * s, ls="--", color="gray", label="standard")
        ax.axvline(gamma_1, ls=":", color="red", alpha=0.6)
        ax.set_title(METRIC_LABELS[m])
        ax.set_xlabel("inversion guidance gamma_2")
        ax.legend(fontsize=8)
    fig.suptitle(f"Guidance gap ablation — SDXL × Pick-a-Pic (gamma_1={gamma_1})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote figure -> {out_png}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--main-scores", required=True)
    p.add_argument("--ir-scores", default=None, help="Optional scores_ir.jsonl to merge")
    p.add_argument("--gamma1", type=float, default=5.5)
    p.add_argument("--out-summary", required=True, help="summary.md path")
    p.add_argument("--out-figure", required=True, help="figure7_guidance_gap.png path")
    p.add_argument("--out-merged", default=None, help="optional merged scores.jsonl")
    args = p.parse_args()

    main_rows = read_jsonl(args.main_scores)
    rows = merge_score_rows(main_rows, read_jsonl(args.ir_scores)) if args.ir_scores else main_rows
    if args.out_merged:
        write_jsonl(args.out_merged, rows)

    agg = aggregate_guidance_gap(rows, metrics=("hps", "aes", "pick", "ir"), gamma_1=args.gamma1)

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(render_table(agg, args.gamma1))
    print(f"wrote summary -> {out_summary}")
    render_figure(agg, args.gamma1, Path(args.out_figure))


if __name__ == "__main__":
    main()
