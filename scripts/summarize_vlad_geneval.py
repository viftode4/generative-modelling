"""Summarize Vlad's GenEval object-detector pass/fail JSONL into Markdown.

Pairs with ``scripts/run_geneval_detector.py``. When ``--hps-scores`` is
provided, the existing HPSv2 win-rate table is appended so the report can
quote both detector and human-preference metrics side-by-side.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from zigzag_project.experiment_io import aggregate_hps_pairs, read_jsonl
from zigzag_project.geneval_io import aggregate_geneval_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geneval-scores", required=True)
    parser.add_argument("--hps-scores", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    geneval_rows = read_jsonl(args.geneval_scores)
    geneval_summary = aggregate_geneval_pairs(geneval_rows)
    markdown = render_summary(geneval_rows, geneval_summary)

    if args.hps_scores:
        hps_rows = read_jsonl(args.hps_scores)
        hps_summary = aggregate_hps_pairs(hps_rows)
        markdown += "\n" + render_hps_section(hps_summary)
        markdown += "\n" + render_agreement(geneval_rows, hps_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    print(f"wrote summary -> {out}")


def render_summary(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    overall = summary["overall"]
    by_tag = summary["by_tag"]
    lines = [
        "# Vlad GenEval Detector Summary",
        "",
        f"Overall pairs: {overall['pairs']}",
        f"Standard pass rate: {overall['standard_pass_rate']:.1%}"
        f" ({overall['standard_pass']}/{overall['pairs']})",
        f"Z-Sampling pass rate: {overall['zigzag_pass_rate']:.1%}"
        f" ({overall['zigzag_pass']}/{overall['pairs']})",
        f"Delta (Z - standard): {overall['delta_pass_rate']:+.1%}",
        f"Both pass: {overall['both_pass']}; "
        f"Z only: {overall['zigzag_only']}; "
        f"standard only: {overall['standard_only']}; "
        f"neither: {overall['neither_pass']}",
        "",
        "## Detector pass rate by GenEval tag",
        "",
        "| Tag | Pairs | Std pass | Z pass | Std rate | Z rate | Delta | Z-only | Std-only |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    row_template = (
        "| {tag} | {pairs} | {sp} | {zp} | {sr:.1%} | {zr:.1%} | "
        "{d:+.1%} | {zo} | {so} |"
    )
    for tag, tag_summary in by_tag.items():
        lines.append(
            row_template.format(
                tag=tag,
                pairs=tag_summary["pairs"],
                sp=tag_summary["standard_pass"],
                zp=tag_summary["zigzag_pass"],
                sr=tag_summary["standard_pass_rate"],
                zr=tag_summary["zigzag_pass_rate"],
                d=tag_summary["delta_pass_rate"],
                zo=tag_summary["zigzag_only"],
                so=tag_summary["standard_only"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-pair detector results",
            "",
            "| prompt_id | seed | tag | std pass | Z pass | prompt |",
            "| --- | ---: | --- | :---: | :---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {pid} | {seed} | {tag} | {sp} | {zp} | {prompt} |".format(
                pid=row["prompt_id"],
                seed=row["seed"],
                tag=row["tag"],
                sp="yes" if row["standard_pass"] else "no",
                zp="yes" if row["zigzag_pass"] else "no",
                prompt=row["prompt"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_hps_section(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    by_tag = summary["by_tag"]
    lines = [
        "## HPSv2 head-to-head (for cross-reference)",
        "",
        f"Pairs: {overall['pairs']}; Z win rate: {overall['zigzag_win_rate']:.1%}; "
        f"mean HPS delta: {overall['mean_hps_delta']:+.4f}",
        "",
        "| Tag | Pairs | Z wins | Standard wins | Ties | Z win rate | Mean delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag, tag_summary in by_tag.items():
        lines.append(
            "| {tag} | {pairs} | {zw} | {sw} | {ties} | {zr:.1%} | {d:+.4f} |".format(
                tag=tag,
                pairs=tag_summary["pairs"],
                zw=tag_summary["zigzag_wins"],
                sw=tag_summary["standard_wins"],
                ties=tag_summary["ties"],
                zr=tag_summary["zigzag_win_rate"],
                d=tag_summary["mean_hps_delta"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_agreement(
    geneval_rows: list[dict[str, Any]],
    hps_rows: list[dict[str, Any]],
) -> str:
    hps_lookup = {(r["prompt_id"], int(r["seed"])): r for r in hps_rows}
    agree_dir = 0
    z_wins_hps_loses_geneval = 0
    z_loses_hps_wins_geneval = 0
    both_z_wins = 0
    both_std_wins = 0
    inconclusive = 0
    for row in geneval_rows:
        key = (row["prompt_id"], int(row["seed"]))
        hps = hps_lookup.get(key)
        if hps is None:
            inconclusive += 1
            continue
        hps_z_wins = float(hps["zigzag_score"]) > float(hps["standard_score"])
        geneval_z_wins = bool(row["zigzag_pass"]) and not bool(row["standard_pass"])
        geneval_std_wins = bool(row["standard_pass"]) and not bool(row["zigzag_pass"])
        if hps_z_wins and geneval_z_wins:
            both_z_wins += 1
            agree_dir += 1
        elif (not hps_z_wins) and geneval_std_wins:
            both_std_wins += 1
            agree_dir += 1
        elif hps_z_wins and geneval_std_wins:
            z_loses_hps_wins_geneval += 1  # HPS says Z, detector says std
        elif (not hps_z_wins) and geneval_z_wins:
            z_wins_hps_loses_geneval += 1  # HPS says std, detector says Z
        else:
            inconclusive += 1
    total = len(geneval_rows)
    directional = total - inconclusive
    lines = [
        "## Metric agreement (HPSv2 vs GenEval detector, per pair)",
        "",
        f"Pairs with both metrics available: {total}/{total}",
        "Detector-directional pairs (one sampler passes and the other fails): "
        f"{directional}/{total}",
        f"Both metrics favor Z: {both_z_wins}",
        f"Both metrics favor standard: {both_std_wins}",
        f"HPS favors Z but detector favors standard: {z_loses_hps_wins_geneval}",
        f"HPS favors standard but detector favors Z: {z_wins_hps_loses_geneval}",
        f"Detector-tied pairs (both pass or both fail): {inconclusive}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
