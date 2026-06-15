import math

import pytest

from zigzag_project.guidance_gap import (
    Variant,
    aggregate_guidance_gap,
    expand_variants,
    gamma_tag,
    image_filename,
    merge_score_rows,
)


def test_gamma_tag_is_filesystem_safe():
    assert gamma_tag(-20) == "m20"
    assert gamma_tag(0) == "0"
    assert gamma_tag(5.5) == "5p5"
    assert gamma_tag(7.5) == "7p5"


def test_expand_variants_has_standard_first_then_sorted_unique_gamma2():
    variants = expand_variants(gamma_1=5.5, gamma2_sweep=[5.5, 0, 0, -20])
    assert variants[0] == Variant(method="standard", gamma_2=None, label="standard")
    z = variants[1:]
    assert [v.gamma_2 for v in z] == [-20.0, 0.0, 5.5]   # deduped + ascending
    assert all(v.method == "zigzag" for v in z)
    assert z[0].label == "g2_m20"


def test_image_filename_encodes_variant():
    std = image_filename("pickapic_003", 42, "standard", None)
    assert std == "pickapic_003_seed42_standard.png"
    assert image_filename("pickapic_003", 42, "zigzag", 5.5) == "pickapic_003_seed42_g2_5p5.png"


def _row(pid, seed, method, g2, **metrics):
    r = {"prompt_id": pid, "seed": seed, "method": method, "gamma_2": g2}
    r.update(metrics)
    return r


def test_aggregate_computes_means_and_winrate_vs_matched_standard():
    rows = [
        _row("p0", 42, "standard", None, hps=0.20, aes=5.0, pick=21.0, ir=0.0),
        _row("p1", 42, "standard", None, hps=0.20, aes=5.0, pick=21.0, ir=0.0),
        _row("p0", 42, "zigzag", 0.0, hps=0.30, aes=6.0, pick=22.0, ir=0.5),
        _row("p1", 42, "zigzag", 0.0, hps=0.10, aes=6.0, pick=22.0, ir=0.5),
    ]
    agg = aggregate_guidance_gap(rows, metrics=("hps", "aes", "pick", "ir"), gamma_1=5.5)
    assert agg["standard"]["hps"]["mean"] == pytest.approx(0.20)
    g0 = agg["by_gamma2"][0.0]
    assert g0["gap"] == pytest.approx(5.5)
    assert g0["hps"]["mean"] == pytest.approx(0.20)
    assert g0["hps"]["win_rate"] == pytest.approx(0.5)   # p0 wins, p1 loses
    assert g0["aes"]["win_rate"] == pytest.approx(1.0)


def test_aggregate_skips_nan_for_winrate_and_mean():
    rows = [
        _row("p0", 42, "standard", None, ir=0.0),
        _row("p0", 42, "zigzag", 0.0, ir=float("nan")),
    ]
    agg = aggregate_guidance_gap(rows, metrics=("ir",), gamma_1=5.5)
    g0 = agg["by_gamma2"][0.0]
    assert g0["ir"]["n"] == 0
    assert math.isnan(g0["ir"]["mean"])
    assert g0["ir"]["win_rate"] is None


def test_merge_score_rows_joins_ir_by_image_path():
    main = [{"image_path": "/a.png", "prompt_id": "p0", "seed": 1, "method": "standard",
             "gamma_2": None, "hps": 0.2, "aes": 5.0, "pick": 21.0, "ir": float("nan")}]
    ir = [{"image_path": "/a.png", "ir": 0.7}]
    merged = merge_score_rows(main, ir)
    assert merged[0]["ir"] == 0.7
    assert merged[0]["hps"] == 0.2
