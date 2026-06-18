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

Device-suspend wall-clock artifacts are removed before summarizing time: the sweep ran on
a laptop that was closed and moved mid-run (e.g. two T_s=50 standard cells exceeded 3000 s),
so per-arm timings above OUTLIER_FACTOR x the per-budget median are dropped as artifacts.
The remaining cells cluster tightly, so the per-budget median reflects the real per-image
cost. The generated images are unaffected, so quality (HPS) plots use every cell.
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

# A per-image timing above this multiple of the per-budget median is a device-suspend
# artifact (laptop closed/moved mid-run), not a real sampler cost.
OUTLIER_FACTOR = 2.0


def drop_suspend_artifacts(times, factor=OUTLIER_FACTOR):
    """Return timings with device-suspend artifacts (above factor x median) removed."""
    if not times:
        return times
    cutoff = factor * statistics.median(times)
    return [t for t in times if t <= cutoff]


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
        std = drop_suspend_artifacts([r["standard_runtime_s"] for r in grp])
        z = drop_suspend_artifacts([r["zigzag_runtime_s"] for r in grp])
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
    plotted: list[tuple[float, float]] = []
    for idx, b in enumerate(budgets):
        grp = by_budget[b]
        sx_all = [r["standard_runtime_s"] for r in grp]
        zy_all = [r["zigzag_runtime_s"] for r in grp]
        s_cut = OUTLIER_FACTOR * statistics.median(sx_all)
        z_cut = OUTLIER_FACTOR * statistics.median(zy_all)
        # Drop a cell if either arm's wall-clock is a device-suspend artifact.
        pts = [(s, z) for s, z in zip(sx_all, zy_all, strict=True) if s <= s_cut and z <= z_cut]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, alpha=0.6,
                   color=cmap(idx / max(1, len(budgets) - 1)), label=f"T_s={b}")
        plotted.extend(pts)
    # Artifacts are filtered above, so the axes follow the real (clustered) data.
    vals = [v for p in plotted for v in p]
    lo, hi = min(vals), max(vals)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (equal time)")
    ax.set_xlim(lo * 0.9, hi * 1.05)
    ax.set_ylim(lo * 0.9, hi * 1.05)
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
            std_t.append(statistics.median(drop_suspend_artifacts(
                [r["standard_runtime_s"] for r in by_budget[b]])))
            z_t.append(statistics.median(drop_suspend_artifacts(
                [r["zigzag_runtime_s"] for r in by_budget[b]])))
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
