from zigzag_project.metrics import make_metric_score_rows


def test_make_metric_score_rows_scores_each_image_per_metric():
    manifest = [
        {"prompt_id": "p0", "seed": 42, "method": "standard", "gamma_2": None,
         "prompt": "a cat", "image_path": "/img/p0_std.png"},
        {"prompt_id": "p0", "seed": 42, "method": "zigzag", "gamma_2": 0.0,
         "prompt": "a cat", "image_path": "/img/p0_g0.png"},
    ]
    # Fake scorers: HPS/Pick depend on (path, prompt); AES on path only.
    scorers = {
        "hps": lambda path, prompt: 0.3,
        "pick": lambda path, prompt: 22.0,
        "aes": lambda path, prompt: 6.1,
    }
    rows = make_metric_score_rows(manifest, scorers)
    assert len(rows) == 2
    assert rows[0]["hps"] == 0.3 and rows[0]["pick"] == 22.0 and rows[0]["aes"] == 6.1
    # passthrough identity fields preserved for later aggregation/merge
    assert rows[1]["method"] == "zigzag" and rows[1]["gamma_2"] == 0.0
    assert rows[1]["image_path"] == "/img/p0_g0.png"


def test_make_metric_score_rows_defaults_missing_metrics_to_nan():
    import math
    manifest = [{"prompt_id": "p0", "seed": 1, "method": "standard", "gamma_2": None,
                 "prompt": "x", "image_path": "/i.png"}]
    rows = make_metric_score_rows(manifest, {"hps": lambda p, t: 0.1}, all_metrics=("hps", "ir"))
    assert rows[0]["hps"] == 0.1
    assert math.isnan(rows[0]["ir"])
