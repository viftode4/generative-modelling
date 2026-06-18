"""Pure helpers for the SDXL guidance-gap ablation (Experiment #6).

Reproduces the variable in the paper's Figure 7: fix gamma_1, sweep gamma_2,
and study the guidance gap delta_gamma = gamma_1 - gamma_2.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any


@dataclass(frozen=True)
class Variant:
    """One generation variant for a (prompt, seed): the standard baseline or a
    Z-Sampling run at a specific inversion guidance gamma_2."""

    method: str           # "standard" | "zigzag"
    gamma_2: float | None  # None for the standard baseline
    label: str            # "standard" | "g2_<tag>"


def gamma_tag(value: float) -> str:
    """Filesystem-safe tag for a gamma value: -20 -> 'm20', 5.5 -> '5p5'."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def expand_variants(*, gamma_1: float, gamma2_sweep: Sequence[float]) -> list[Variant]:
    """Standard baseline first, then one Z-Sampling variant per unique gamma_2
    (ascending). gamma_1 is accepted for symmetry / future validation."""
    variants = [Variant(method="standard", gamma_2=None, label="standard")]
    for g2 in sorted({float(g) for g in gamma2_sweep}):
        variants.append(Variant(method="zigzag", gamma_2=g2, label=f"g2_{gamma_tag(g2)}"))
    return variants


def image_filename(prompt_id: str, seed: int, method: str, gamma_2: float | None) -> str:
    if method == "standard":
        return f"{prompt_id}_seed{seed}_standard.png"
    return f"{prompt_id}_seed{seed}_g2_{gamma_tag(float(gamma_2))}.png"


def _stats(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": mean(clean),
        "std": pstdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def aggregate_guidance_gap(
    score_rows: Iterable[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = ("hps", "aes", "pick", "ir"),
    gamma_1: float | None = None,
) -> dict[str, Any]:
    """Aggregate per-image metric rows by gamma_2.

    Each row must have: prompt_id, seed, method ("standard"|"zigzag"),
    gamma_2 (None for standard), and one numeric field per metric.

    Returns means/std per metric for the standard baseline and, per gamma_2,
    means/std plus a win-rate computed against the *matched* standard image
    (same prompt_id + seed). When ``gamma_1`` is given, each gamma_2 entry also
    carries the guidance gap ``gap = gamma_1 - gamma_2``. NaN metric values are
    skipped.
    """
    rows = list(score_rows)
    std_by_key = {
        (r["prompt_id"], r["seed"]): r for r in rows if r["method"] == "standard"
    }

    out: dict[str, Any] = {"standard": {}, "by_gamma2": {}}
    std_rows = list(std_by_key.values())
    for m in metrics:
        out["standard"][m] = _stats([float(r[m]) for r in std_rows])

    by_g2: dict[float, list[Mapping[str, Any]]] = {}
    for r in rows:
        if r["method"] != "zigzag":
            continue
        by_g2.setdefault(float(r["gamma_2"]), []).append(r)

    for g2, group in sorted(by_g2.items()):
        entry: dict[str, Any] = {"gamma_2": g2, "n_images": len(group)}
        entry["gap"] = (gamma_1 - g2) if gamma_1 is not None else None
        for m in metrics:
            entry[m] = _stats([float(r[m]) for r in group])
            wins = total = 0
            for r in group:
                std = std_by_key.get((r["prompt_id"], r["seed"]))
                if std is None:
                    continue
                zv, sv = float(r[m]), float(std[m])
                if math.isnan(zv) or math.isnan(sv):
                    continue
                total += 1
                if zv > sv:
                    wins += 1
            entry[m]["win_rate"] = (wins / total) if total else None
        out["by_gamma2"][g2] = entry
    return out


def merge_score_rows(
    main_rows: Iterable[Mapping[str, Any]],
    ir_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay ImageReward scores (keyed by image_path) onto the main score rows."""
    ir_by_path = {r["image_path"]: float(r["ir"]) for r in ir_rows}
    merged = []
    for r in main_rows:
        row = dict(r)
        if row["image_path"] in ir_by_path:
            row["ir"] = ir_by_path[row["image_path"]]
        merged.append(row)
    return merged
