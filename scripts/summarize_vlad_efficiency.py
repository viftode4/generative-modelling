"""Summarize Vlad's Experiment #8 efficiency sweep into Markdown.

Joins the generation manifest (per-image runtime, per budget) with the HPSv2
score rows (per-image quality) and reports, for each time budget T_s, the median
time/image and mean HPS for both Standard and the compute-matched Z-Sampling.
This is the data behind the paper's Figure 10 (HPS v2 vs. average time/image),
plus the headline check: does Z at a lower compute budget beat the best Standard?

The sweep ran on a laptop that was closed and moved mid-run, so a few cells carry
device-suspend wall-clock artifacts (e.g. two T_s=50 standard cells exceeded 3000 s).
These are not real sampler costs: suspending the process does not change the generated
image, only its measured wall-clock. We therefore exclude per-arm timings above
``OUTLIER_FACTOR`` times the per-budget median from the timing summary, while retaining
every pair for the HPS comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

from zigzag_project.experiment_io import read_jsonl

# A per-image timing above this multiple of the per-budget median is treated as a
# device-suspend artifact (laptop closed/moved mid-run), not a real sampler cost.
OUTLIER_FACTOR = 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="run_vlad_exp8_efficiency manifest.jsonl")
    parser.add_argument("--scores", required=True, help="HPSv2 scores.jsonl from score_hpsv2_pairs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    scores = read_jsonl(args.scores)

    # Join key: a (prompt, seed, budget) cell. The manifest carries budget_ts;
    # the score rows carry it inside tag = "ts{T_s}".
    hps_by_key = {
        (str(r["prompt_id"]), int(r["seed"]), _budget_from_tag(str(r["tag"]))): r for r in scores
    }

    budgets: dict[int, dict[str, list[float]]] = {}
    for row in manifest:
        if row.get("standard_runtime_s") is None or row.get("zigzag_runtime_s") is None:
            continue  # dry-run / unmeasured row
        t_s = int(row["budget_ts"])
        b = budgets.setdefault(
            t_s,
            {
                "std_time": [],
                "z_time": [],
                "std_hps": [],
                "z_hps": [],
                "deltas": [],
                "t_z": int(row["zigzag_steps"]),
                "lambda": int(row["zigzag_lambda_step"]),
            },
        )
        b["std_time"].append(float(row["standard_runtime_s"]))
        b["z_time"].append(float(row["zigzag_runtime_s"]))
        key = (str(row["prompt_id"]), int(row["seed"]), t_s)
        score = hps_by_key.get(key)
        if score is not None:
            b["std_hps"].append(float(score["standard_score"]))
            b["z_hps"].append(float(score["zigzag_score"]))
            b["deltas"].append(float(score["zigzag_score"]) - float(score["standard_score"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(budgets))
    print(f"wrote efficiency summary -> {out}")


def _budget_from_tag(tag: str) -> int:
    return int(tag[2:]) if tag.startswith("ts") else -1


def drop_suspend_artifacts(times: list[float], factor: float = OUTLIER_FACTOR) -> list[float]:
    """Return the timings with device-suspend artifacts removed.

    A suspend (laptop closed/moved) inflates a cell's wall-clock far above the
    per-image cost; such a value sits well above the per-budget median while real
    cells cluster within ~20% of it. We drop anything above ``factor`` x median.
    """
    if not times:
        return times
    cutoff = factor * median(times)
    return [t for t in times if t <= cutoff]


def _render(budgets: dict[int, dict[str, list[float]]]) -> str:
    excluded: list[str] = []
    table_rows: list[str] = []
    rows_for_claim: list[tuple[int, float, float, float, float]] = []

    for t_s in sorted(budgets):
        b = budgets[t_s]
        pairs = len(b["std_time"])
        std_times = drop_suspend_artifacts(b["std_time"])
        z_times = drop_suspend_artifacts(b["z_time"])
        n_drop = (pairs - len(std_times)) + (len(b["z_time"]) - len(z_times))
        if n_drop:
            excluded.append(
                f"T_s={t_s}: {pairs - len(std_times)} std + {len(b['z_time']) - len(z_times)} Z"
            )
        std_time = median(std_times) if std_times else 0.0
        z_time = median(z_times) if z_times else 0.0
        std_hps = mean(b["std_hps"]) if b["std_hps"] else 0.0
        z_hps = mean(b["z_hps"]) if b["z_hps"] else 0.0
        wins = sum(1 for d in b["deltas"] if d > 0)
        win_rate = wins / len(b["deltas"]) if b["deltas"] else 0.0
        delta = mean(b["deltas"]) if b["deltas"] else 0.0
        table_rows.append(
            f"| {t_s} | {b['t_z']} | {b['lambda']} | {pairs} | {std_time:.2f} | {z_time:.2f} | "
            f"{std_hps:.4f} | {z_hps:.4f} | {win_rate:.1%} | {delta:+.4f} |"
        )
        rows_for_claim.append((t_s, std_time, std_hps, z_time, z_hps))

    note = (
        "Time is the **median** seconds/image per arm, computed after removing "
        f"device-suspend artifacts (per-arm timings above {OUTLIER_FACTOR:g}x the per-budget "
        "median). The sweep ran on a laptop that was closed and moved mid-run; the affected "
        "cells (e.g. two T_s=50 standard cells exceeded 3000 s) reflect wall-clock pauses, "
        "not sampler cost. The generated images are unaffected, so all pairs are kept for the "
        "HPS comparison. Excluded timing cells: " + ("; ".join(excluded) if excluded else "none")
        + "."
    )

    lines = [
        "# Vlad Experiment #8 — SDXL x Pick-a-Pic Efficiency (HPS v2 vs time)",
        "",
        "Time-matched comparison (paper Figure 10): for standard budget `T_s`, "
        "Z-Sampling uses `T_z = T_s//2` steps with `lambda = T_s//4`, so the "
        "noise-prediction op counts match (`T_s` vs `T_z + 2*lambda`).",
        "",
        note,
        "",
        "| Budget T_s | T_z | lambda | Pairs | Std time (s, med) | Z time (s, med) | "
        "Std HPS | Z HPS | Z win rate | Mean delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *table_rows,
        "",
        "## Efficiency headline",
        "",
        *_claim_lines(rows_for_claim),
        "",
    ]
    return "\n".join(lines)


def _claim_lines(rows: list[tuple[int, float, float, float, float]]) -> list[str]:
    """Reproduce the paper's claim: Z at lower compute can beat the best Standard."""
    if not rows:
        return ["No measured budgets available."]
    best_std_hps = max(r[2] for r in rows)
    best_std_time = min(r[1] for r in rows if r[2] == best_std_hps)
    # Cheapest Z point that still beats the best standard HPS.
    z_beats = [
        (t_s, z_time, z_hps)
        for (t_s, _st, _sh, z_time, z_hps) in rows
        if z_hps >= best_std_hps
    ]
    out = [
        f"- Best Standard HPS across budgets: **{best_std_hps:.4f}** "
        f"at ~{best_std_time:.2f}s/image.",
    ]
    if z_beats:
        t_s, z_time, z_hps = min(z_beats, key=lambda x: x[1])
        saving = (best_std_time - z_time) / best_std_time * 100 if best_std_time else 0.0
        out.append(
            f"- Z-Sampling reaches HPS **{z_hps:.4f}** at ~{z_time:.2f}s/image (budget T_s={t_s}), "
            f"matching/beating the best Standard with **{saving:.0f}% less time/image**."
        )
    else:
        out.append(
            "- No Z budget beat the best Standard HPS in this run; report the curve and discuss "
            "(scale, seeds, or budget range may need extending)."
        )
    return out


if __name__ == "__main__":
    main()
