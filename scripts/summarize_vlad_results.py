"""Summarize Vlad's HPSv2 standard-vs-Z result JSONL into Markdown."""
from __future__ import annotations

import argparse
from pathlib import Path

from zigzag_project.experiment_io import aggregate_hps_pairs, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gallery-count", type=int, default=3)
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    summary = aggregate_hps_pairs(rows)
    markdown = render_summary(rows, summary, gallery_count=args.gallery_count)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    print(f"wrote summary -> {out}")


def render_summary(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    *,
    gallery_count: int,
) -> str:
    overall = summary["overall"]
    by_tag = summary["by_tag"]
    sorted_rows = sorted(rows, key=lambda row: float(row["delta"]))
    losses = sorted_rows[:gallery_count]
    wins = list(reversed(sorted_rows[-gallery_count:]))

    lines = [
        "# Vlad GenEval HPSv2 Summary",
        "",
        f"Overall pairs: {overall['pairs']}",
        f"Overall Z win rate: {overall['zigzag_win_rate']:.1%}",
        f"Mean HPS delta (Z - standard): {overall['mean_hps_delta']:.4f}",
        "",
        "## By GenEval tag",
        "",
        "| Tag | Pairs | Z wins | Standard wins | Ties | Z win rate | Mean delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag, tag_summary in by_tag.items():
        lines.append(
            "| {tag} | {pairs} | {zw} | {sw} | {ties} | {zrate:.1%} | {delta:.4f} |".format(
                tag=tag,
                pairs=tag_summary["pairs"],
                zw=tag_summary["zigzag_wins"],
                sw=tag_summary["standard_wins"],
                ties=tag_summary["ties"],
                zrate=tag_summary["zigzag_win_rate"],
                delta=tag_summary["mean_hps_delta"],
            )
        )

    lines.extend(["", "## Top Z-Sampling wins", ""])
    lines.extend(_gallery_lines(wins))
    lines.extend(["", "## Top Z-Sampling losses", ""])
    lines.extend(_gallery_lines(losses))
    lines.append("")
    return "\n".join(lines)


def _gallery_lines(rows: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        lines.extend(
            [
                f"### {row['prompt_id']} seed {row['seed']} ({row['tag']})",
                f"Prompt: {row['prompt']}",
                f"Delta: {float(row['delta']):.4f}",
                f"- Standard: `{row['standard_path']}`",
                f"- Z-Sampling: `{row['zigzag_path']}`",
                "",
            ]
        )
    return lines


if __name__ == "__main__":
    main()
