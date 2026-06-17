"""Render the SDXL lambda-step sweep as a PNG plot."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Combined HPSv2 scores JSONL")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--y-min", type=float, default=0.5, help="Minimum y-axis value")
    parser.add_argument(
        "--neutral-zero",
        action="store_true",
        help="Display lambda_step=0 as a neutral 50% baseline.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    summary = aggregate_by_lambda_step(rows)
    write_png(summary, Path(args.out), y_min=args.y_min, neutral_zero=args.neutral_zero)
    print(f"wrote plot -> {args.out}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate_by_lambda_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["lambda_step"])].append(row)

    summary: list[dict[str, Any]] = []
    for lambda_step, lambda_rows in sorted(grouped.items()):
        pairs = len(lambda_rows)
        wins = 0
        total_delta = 0.0
        for row in lambda_rows:
            delta = float(row["zigzag_score"]) - float(row["standard_score"])
            total_delta += delta
            if delta > 0:
                wins += 1
        summary.append(
            {
                "lambda_step": lambda_step,
                "pairs": pairs,
                "win_rate": wins / pairs if pairs else 0.0,
                "mean_delta": total_delta / pairs if pairs else 0.0,
            }
        )
    return summary


def write_png(
    summary: list[dict[str, Any]],
    out: Path,
    *,
    y_min: float,
    neutral_zero: bool,
) -> None:
    scale = 3
    width, height = 1200, 760
    margin_left, margin_right = 130, 60
    margin_top, margin_bottom = 100, 120
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    canvas = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(canvas)

    def sx(value: float) -> int:
        return round(value * scale)

    def sy(value: float) -> int:
        return round(value * scale)

    font_title = _font(30 * scale, bold=True)
    font_axis = _font(21 * scale)
    font_tick = _font(18 * scale)
    font_label = _font(18 * scale, bold=True)
    color_axis = "#374151"
    color_grid = "#e5e7eb"
    color_line = "#0f766e"
    color_text = "#111827"

    x_values = [int(row["lambda_step"]) for row in summary]
    y_values = [
        0.5 if neutral_zero and int(row["lambda_step"]) == 0 else float(row["win_rate"])
        for row in summary
    ]
    min_x = min(x_values)
    max_x = max(x_values)
    x_span = max(max_x - min_x, 1)

    def x_pos(value: int) -> float:
        return plot_left + ((value - min_x) / x_span) * plot_width

    def y_pos(value: float) -> float:
        clamped = max(y_min, min(1.0, value))
        return plot_top + ((1.0 - clamped) / (1.0 - y_min)) * plot_height

    title = "SDXL Pick-a-Pic Z-Sampling Win Rate"
    _center_text(draw, title, width / 2, 36, font_title, color_text, scale)

    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if y_min == 0.5 else [0, 0.25, 0.5, 0.75, 1.0]
    for tick in ticks:
        y = y_pos(tick)
        draw.line(
            [(sx(plot_left), sy(y)), (sx(plot_right), sy(y))],
            fill=color_grid,
            width=sx(1),
        )
        draw.line(
            [(sx(plot_left - 8), sy(y)), (sx(plot_left), sy(y))],
            fill=color_axis,
            width=sx(2),
        )
        _right_text(draw, f"{tick:.0%}", plot_left - 16, y - 10, font_tick, color_axis, scale)

    draw.line(
        [(sx(plot_left), sy(plot_top)), (sx(plot_left), sy(plot_bottom))],
        fill=color_axis,
        width=sx(2),
    )
    draw.line(
        [(sx(plot_left), sy(plot_bottom)), (sx(plot_right), sy(plot_bottom))],
        fill=color_axis,
        width=sx(2),
    )

    for x in x_values:
        xpos = x_pos(x)
        draw.line(
            [(sx(xpos), sy(plot_bottom)), (sx(xpos), sy(plot_bottom + 8))],
            fill=color_axis,
            width=sx(2),
        )
        _center_text(draw, str(x), xpos, plot_bottom + 22, font_tick, color_axis, scale)

    points = [(sx(x_pos(x)), sy(y_pos(y))) for x, y in zip(x_values, y_values, strict=True)]
    if len(points) > 1:
        draw.line(points, fill=color_line, width=sx(5), joint="curve")

    for x, y in zip(x_values, y_values, strict=True):
        xpos = x_pos(x)
        ypos = y_pos(y)
        r = 7
        draw.ellipse(
            [sx(xpos - r), sy(ypos - r), sx(xpos + r), sy(ypos + r)],
            fill=color_line,
            outline="white",
            width=sx(2),
        )
        _center_text(draw, f"{y:.1%}", xpos, ypos - 34, font_label, color_text, scale)

    _center_text(
        draw,
        "Z-Sampling steps (lambda-step)",
        (plot_left + plot_right) / 2,
        height - 55,
        font_axis,
        color_text,
        scale,
    )

    y_axis = "Win rate vs. standard sampling"
    label_layer = Image.new("RGBA", (height * scale, 50 * scale), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text((0, 0), y_axis, fill=color_text, font=font_axis)
    rotated = label_layer.rotate(90, expand=True)
    canvas.paste(rotated, (sx(28), sy(260)), rotated)

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    canvas.save(out)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: float,
    y: float,
    font: ImageFont.ImageFont,
    fill: str,
    scale: int,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    draw.text((round(x * scale - width / 2), round(y * scale)), text, font=font, fill=fill)


def _right_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: float,
    y: float,
    font: ImageFont.ImageFont,
    fill: str,
    scale: int,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    draw.text((round(x * scale - width), round(y * scale)), text, font=font, fill=fill)


if __name__ == "__main__":
    main()
