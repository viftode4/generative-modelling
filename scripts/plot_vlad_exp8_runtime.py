#!/usr/bin/env python
"""Plot Experiment #8 runtime graphics from the (possibly partial) manifest.

Reads the resumable manifest and produces presentation-ready figures from whatever
budgets are present. Runtime-only: needs no HPS scoring and no GPU, so it is safe to
run while the generation sweep is still going.

Outputs (PNG, 150 dpi) into <out-dir>:
  - exp8_runtime_per_budget.png : median time/image per budget, Standard vs Z (matched compute)
  - exp8_runtime_scatter.png    : per-image Standard vs Z runtime, y=x reference

Each budget's compute is matched (Z ops = T_s), so equal-ish bars are the expected,
on-message result: Z buys its quality gains at no extra wall-clock on this hardware.

Time is summarized with the *median*, not the mean: absolute wall-clock on shared MPS
hardware is heavy-tailed (at T_s=50 two standard cells exceeded 3000 s and the Z arm has a
~708 s outlier, all OS-level stalls), and the mean of a compute-matched arm is not robust
to such outliers. The scatter axes are likewise clipped to a high percentile so the bulk
of the cells stays legible.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STANDARD_COLOR = "#4C72B0"
ZIGZAG_COLOR = "#DD8452"


def load_rows(manifest: Path):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return [r for r in rows if not r.get("dry_run")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/vlad_exp8_sdxl_pickapic/manifest.jsonl")
    ap.add_argument("--scores", default="outputs/vlad_exp8_sdxl_pickapic/scores.jsonl")
    ap.add_argument("--out", default="outputs/vlad_exp8_sdxl_pickapic")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(manifest)

    by_budget = defaultdict(list)
    for r in rows:
        by_budget[r["budget_ts"]].append(r)
    budgets = sorted(by_budget)

    # ---- Figure 1: median time/image per budget (grouped bars) ----
    std_med, z_med, labels = [], [], []
    for b in budgets:
        grp = by_budget[b]
        std = [r["standard_runtime_s"] for r in grp]
        z = [r["zigzag_runtime_s"] for r in grp]
        std_med.append(statistics.median(std))
        z_med.append(statistics.median(z))
        labels.append(f"T_s={b}\n(n={len(grp)})")

    x = range(len(budgets))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - w / 2 for i in x], std_med, w, label="Standard", color=STANDARD_COLOR)
    ax.bar([i + w / 2 for i in x], z_med, w, label="Z-Sampling (matched ops)", color=ZIGZAG_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median time / image (s)")
    ax.set_title("SDXL × Pick-a-Pic — compute-matched runtime per budget (MPS)")
    ax.legend()
    for i, (s, z) in enumerate(zip(std_med, z_med, strict=True)):
        ax.text(i - w / 2, s, f"{s:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, z, f"{z:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "exp8_runtime_per_budget.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---- Figure 2: per-image scatter Standard vs Z ----
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    cmap = plt.get_cmap("viridis")
    for idx, b in enumerate(budgets):
        grp = by_budget[b]
        sx = [r["standard_runtime_s"] for r in grp]
        zy = [r["zigzag_runtime_s"] for r in grp]
        ax.scatter(sx, zy, s=14, alpha=0.6, color=cmap(idx / max(1, len(budgets) - 1)),
                   label=f"T_s={b}")
    # Clip the view to a high percentile so OS-stall outliers (e.g. a ~3995 s cell)
    # don't blow out the axes; the y=x reference is drawn across the clipped range.
    all_times = sorted(t for r in rows for t in (r["standard_runtime_s"], r["zigzag_runtime_s"]))
    lo = all_times[0]
    hi = all_times[min(len(all_times) - 1, int(len(all_times) * 0.99))]
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (equal time)")
    ax.set_xlim(lo * 0.9, hi * 1.1)
    ax.set_ylim(lo * 0.9, hi * 1.1)
    ax.set_xlabel("Standard time / image (s)")
    ax.set_ylabel("Z-Sampling time / image (s)")
    ax.set_title("Per-image runtime: Standard vs Z (matched ops)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = out_dir / "exp8_runtime_scatter.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ---- Figure 3: HPS v2 vs time (the paper's Figure 10), if scores exist ----
    scores_path = Path(args.scores)
    if scores_path.exists():
        scores = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
        s_by_budget = defaultdict(list)
        for s in scores:
            b = int(s["tag"].replace("ts", ""))
            s_by_budget[b].append(s)
        def mean_ci(vals):
            m = statistics.mean(vals)
            n = len(vals)
            if n < 2:
                return m, 0.0
            sem = statistics.stdev(vals) / (n ** 0.5)
            return m, 1.96 * sem  # 95% CI half-width

        sb = sorted(set(s_by_budget) & set(by_budget))
        std_t, std_h, std_e, z_t, z_h, z_e, lbls = [], [], [], [], [], [], []
        for b in sb:
            std_t.append(statistics.median(r["standard_runtime_s"] for r in by_budget[b]))
            z_t.append(statistics.median(r["zigzag_runtime_s"] for r in by_budget[b]))
            mh, eh = mean_ci([s["standard_score"] for s in s_by_budget[b]])
            mz, ez = mean_ci([s["zigzag_score"] for s in s_by_budget[b]])
            std_h.append(mh)
            std_e.append(eh)
            z_h.append(mz)
            z_e.append(ez)
            lbls.append(b)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        # 95% CI error bars on each point (no shaded bands).
        ax.errorbar(std_t, std_h, yerr=std_e, fmt="o-", color=STANDARD_COLOR,
                    capsize=4, label="Standard")
        ax.errorbar(z_t, z_h, yerr=z_e, fmt="s-", color=ZIGZAG_COLOR,
                    capsize=4, label="Z-Sampling (matched ops)")
        # Breathing room on both axes.
        tmin, tmax = min(std_t + z_t), max(std_t + z_t)
        tpad = 0.08 * (tmax - tmin)
        ax.set_xlim(tmin - tpad, tmax + tpad)
        hi_vals = [h + e for h, e in zip(std_h + z_h, std_e + z_e, strict=True)]
        lo_vals = [h - e for h, e in zip(std_h + z_h, std_e + z_e, strict=True)]
        hpad = 0.12 * (max(hi_vals) - min(lo_vals))
        ax.set_ylim(min(lo_vals) - hpad, max(hi_vals) + hpad)
        for t, h, b in zip(std_t, std_h, lbls, strict=True):
            ax.annotate(f"T={b}", (t, h), textcoords="offset points", xytext=(4, -10), fontsize=7)
        for t, h, b in zip(z_t, z_h, lbls, strict=True):
            ax.annotate(f"T={b}", (t, h), textcoords="offset points", xytext=(4, 5), fontsize=7)
        best_std = max(std_h)
        ax.axhline(best_std, color="grey", ls=":", lw=1)
        ax.text(min(z_t), best_std, " best Standard HPS", color="grey", fontsize=8, va="bottom")
        ax.set_xlabel("Median time / image (s)")
        ax.set_ylabel("Mean HPS v2")
        ax.set_title("SDXL × Pick-a-Pic — HPS v2 vs time (paper Figure 10)")
        ax.legend()
        fig.tight_layout()
        p3 = out_dir / "exp8_hps_vs_time.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"wrote {p3}")

        # ---- Figure 4: HPS per budget as equally-spaced grouped bars (95% CI) ----
        xb = range(len(sb))
        wb = 0.38
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([i - wb / 2 for i in xb], std_h, wb, yerr=std_e, capsize=4,
               color=STANDARD_COLOR, label="Standard")
        ax.bar([i + wb / 2 for i in xb], z_h, wb, yerr=z_e, capsize=4,
               color=ZIGZAG_COLOR, label="Z-Sampling (matched ops)")
        ax.axhline(max(std_h), color="grey", ls=":", lw=1)
        ax.text(0, max(std_h), " best Standard HPS", color="grey", fontsize=8, va="bottom")
        ax.set_xticks(list(xb))
        ax.set_xticklabels([f"T_s={b}\n(n={len(s_by_budget[b])})" for b in sb])
        ax.set_ylabel("Mean HPS v2 (95% CI)")
        ax.set_ylim(min(std_h) - 0.015, max(z_h) + 0.02)
        ax.set_title("SDXL × Pick-a-Pic — HPS v2 per budget (compute-matched)")
        ax.legend()
        fig.tight_layout()
        p4 = out_dir / "exp8_hps_bars.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        print(f"wrote {p4}")

    print(f"budgets plotted: {budgets}")
    for b, s, z in zip(budgets, std_med, z_med, strict=True):
        print(f"  T_s={b}: std {s:.1f}s  Z {z:.1f}s  median (n={len(by_budget[b])})")
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
