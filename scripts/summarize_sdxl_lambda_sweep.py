"""Summarize an SDXL lambda-step sweep and write a win-rate plot."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="HPSv2 scores JSONL")
    parser.add_argument("--out", required=True, help="Markdown summary path")
    parser.add_argument(
        "--plot-out",
        default=None,
        help="SVG plot path. Defaults to a sibling of --out named lambda_winrate.svg.",
    )
    parser.add_argument(
        "--plot-y-min",
        type=float,
        default=0.0,
        help="Minimum y-axis value for the SVG plot, as a fraction. Use 0.5 for 50-100%.",
    )
    parser.add_argument(
        "--plot-neutral-zero",
        action="store_true",
        help="Display lambda_step=0 as a neutral 50% baseline in the SVG only.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    summary = aggregate_by_lambda_step(rows)
    out = Path(args.out)
    plot_out = Path(args.plot_out) if args.plot_out else out.with_name("lambda_winrate.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(render_markdown(summary, plot_out))
    plot_out.write_text(
        render_svg_plot(
            summary,
            y_min=args.plot_y_min,
            neutral_zero=args.plot_neutral_zero,
        )
    )
    print(f"wrote summary -> {out}")
    print(f"wrote plot -> {plot_out}")


def aggregate_by_lambda_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "lambda_step" not in row:
            raise ValueError(
                "Score rows must include lambda_step. Re-run scripts/score_hpsv2_pairs.py "
                "against a manifest that includes lambda_step."
            )
        grouped[int(row["lambda_step"])].append(row)

    summary = []
    for lambda_step, lambda_rows in sorted(grouped.items()):
        pairs = len(lambda_rows)
        zigzag_wins = 0
        standard_wins = 0
        ties = 0
        total_delta = 0.0
        for row in lambda_rows:
            delta = float(row["zigzag_score"]) - float(row["standard_score"])
            total_delta += delta
            if delta > 0:
                zigzag_wins += 1
            elif delta < 0:
                standard_wins += 1
            else:
                ties += 1
        summary.append(
            {
                "lambda_step": lambda_step,
                "pairs": pairs,
                "zigzag_wins": zigzag_wins,
                "standard_wins": standard_wins,
                "ties": ties,
                "zigzag_win_rate": zigzag_wins / pairs if pairs else 0.0,
                "mean_hps_delta": total_delta / pairs if pairs else 0.0,
            }
        )
    return summary


def render_markdown(summary: list[dict[str, Any]], plot_out: Path) -> str:
    lines = [
        "# SDXL Pick-a-Pic Lambda-Step Sweep",
        "",
        f"Plot: `{plot_out}`",
        "",
        "| Lambda-step | Pairs | Z wins | Standard wins | Ties | Z win rate | Mean delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {lambda_step} | {pairs} | {zigzag_wins} | {standard_wins} | {ties} | "
            "{zigzag_win_rate:.1%} | {mean_hps_delta:.4f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def render_svg_plot(
    summary: list[dict[str, Any]],
    *,
    y_min: float = 0.0,
    neutral_zero: bool = False,
) -> str:
    if not 0.0 <= y_min < 1.0:
        raise ValueError("--plot-y-min must satisfy 0 <= value < 1")
    width = 760
    height = 460
    left = 72
    right = 28
    top = 36
    bottom = 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [int(row["lambda_step"]) for row in summary]
    y_values = [
        0.5 if neutral_zero and int(row["lambda_step"]) == 0 else float(row["zigzag_win_rate"])
        for row in summary
    ]
    min_x = min(x_values) if x_values else 0
    max_x = max(x_values) if x_values else 1
    x_span = max(max_x - min_x, 1)

    def x_pos(value: int) -> float:
        return left + ((value - min_x) / x_span) * plot_width

    def y_pos(value: float) -> float:
        clamped = max(y_min, min(1.0, value))
        return top + ((1.0 - clamped) / (1.0 - y_min)) * plot_height

    points = " ".join(
        f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in zip(x_values, y_values, strict=True)
    )
    circles = "\n".join(
        f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="5" fill="#0f766e" />'
        for x, y in zip(x_values, y_values, strict=True)
    )
    labels = "\n".join(
        (
            f'<text x="{x_pos(x):.1f}" y="{y_pos(y) - 12:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#111827">{y:.1%}</text>'
        )
        for x, y in zip(x_values, y_values, strict=True)
    )
    x_ticks = "\n".join(
        (
            f'<line x1="{x_pos(x):.1f}" y1="{top + plot_height}" '
            f'x2="{x_pos(x):.1f}" y2="{top + plot_height + 6}" stroke="#374151" />'
            f'<text x="{x_pos(x):.1f}" y="{top + plot_height + 24}" text-anchor="middle" '
            f'font-size="12" fill="#374151">{x}</text>'
        )
        for x in x_values
    )
    y_ticks = "\n".join(
        (
            f'<line x1="{left - 6}" y1="{y_pos(value):.1f}" x2="{left}" '
            f'y2="{y_pos(value):.1f}" stroke="#374151" />'
            f'<line x1="{left}" y1="{y_pos(value):.1f}" x2="{left + plot_width}" '
            f'y2="{y_pos(value):.1f}" stroke="#e5e7eb" />'
            f'<text x="{left - 12}" y="{y_pos(value) + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#374151">{value:.0%}</text>'
        )
        for value in _plot_ticks(y_min)
    )
    title_x = width / 2
    x_label_x = left + plot_width / 2
    y_label_y = top + plot_height / 2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
  viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{title_x:.1f}" y="22" text-anchor="middle" font-size="18"
    font-weight="600" fill="#111827">SDXL Pick-a-Pic Z-Sampling Win Rate</text>
  {y_ticks}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}"
    y2="{top + plot_height}" stroke="#374151" />
  {x_ticks}
  <polyline points="{points}" fill="none" stroke="#0f766e" stroke-width="3" />
  {circles}
  {labels}
  <text x="{x_label_x:.1f}" y="{height - 18}" text-anchor="middle" font-size="13"
    fill="#111827">Z-Sampling steps (lambda-step)</text>
  <text x="18" y="{y_label_y:.1f}" text-anchor="middle"
    transform="rotate(-90 18 {y_label_y:.1f})" font-size="13" fill="#111827">
    Win rate vs. standard sampling</text>
</svg>
"""


def _plot_ticks(y_min: float) -> list[float]:
    if y_min == 0.5:
        return [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return [0.0, 0.25, 0.5, 0.75, 1.0]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
