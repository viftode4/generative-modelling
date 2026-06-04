"""Smoke tests for the GenEval detector runner without loading mmdet/CLIP.

The real detector + CLIP color classifier are heavy and slow; the test
exercises the lookup + aggregation + IO helpers that the runner stitches
together, plus a mocked variant of the script's per-row evaluator.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zigzag_project.experiment_io import read_jsonl, write_jsonl
from zigzag_project.geneval_io import (
    aggregate_geneval_pairs,
    load_geneval_metadata,
)


def test_geneval_metadata_lookup_matches_by_source_index(tmp_path: Path) -> None:
    meta_path = tmp_path / "evaluation_metadata.jsonl"
    rows = [
        {"tag": "single_object", "include": [{"class": "bench", "count": 1}], "prompt": "p0"},
        {"tag": "single_object", "include": [{"class": "cow", "count": 1}], "prompt": "p1"},
    ]
    meta_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    index = load_geneval_metadata(meta_path)
    assert index.lookup(source_index=1, prompt="p1")["prompt"] == "p1"


def test_geneval_metadata_lookup_falls_back_to_prompt_string(tmp_path: Path) -> None:
    meta_path = tmp_path / "evaluation_metadata.jsonl"
    rows = [
        {"tag": "single_object", "include": [{"class": "bench", "count": 1}], "prompt": "p0"},
    ]
    meta_path.write_text(json.dumps(rows[0]) + "\n")
    index = load_geneval_metadata(meta_path)
    assert index.lookup(source_index=42, prompt="p0")["prompt"] == "p0"


def test_geneval_metadata_lookup_raises_when_missing(tmp_path: Path) -> None:
    meta_path = tmp_path / "evaluation_metadata.jsonl"
    meta_path.write_text(json.dumps({"tag": "x", "prompt": "p0"}) + "\n")
    index = load_geneval_metadata(meta_path)
    with pytest.raises(KeyError):
        index.lookup(source_index=99, prompt="nope")


def test_aggregate_geneval_pairs_groups_by_tag() -> None:
    rows = [
        {"tag": "single_object", "standard_pass": True, "zigzag_pass": True},
        {"tag": "single_object", "standard_pass": False, "zigzag_pass": True},
        {"tag": "counting", "standard_pass": True, "zigzag_pass": False},
        {"tag": "counting", "standard_pass": False, "zigzag_pass": False},
    ]
    summary = aggregate_geneval_pairs(rows)
    overall = summary["overall"]
    assert overall["pairs"] == 4
    assert overall["standard_pass"] == 2
    assert overall["zigzag_pass"] == 2
    assert overall["both_pass"] == 1
    assert overall["zigzag_only"] == 1
    assert overall["standard_only"] == 1
    assert overall["neither_pass"] == 1
    by_tag = summary["by_tag"]
    assert by_tag["single_object"]["zigzag_pass"] == 2
    assert by_tag["counting"]["zigzag_pass"] == 0


def test_runner_score_rows_roundtrip_through_write_jsonl(tmp_path: Path) -> None:
    """Mock the detector by injecting a deterministic pass/fail function."""
    manifest_path = tmp_path / "manifest.jsonl"
    metadata_path = tmp_path / "evaluation_metadata.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "tag": "single_object",
                "include": [{"class": "bench", "count": 1}],
                "prompt": "p0",
            }
        )
        + "\n"
    )
    manifest_rows = [
        {
            "prompt_id": "geneval_000_single_object",
            "source_index": 0,
            "tag": "single_object",
            "prompt": "p0",
            "seed": 42,
            "standard_path": "stub_std.png",
            "zigzag_path": "stub_zig.png",
        }
    ]
    write_jsonl(manifest_path, manifest_rows)

    index = load_geneval_metadata(metadata_path)

    def fake_eval(path: str, meta: dict, *_args) -> dict:
        # Z 'passes', standard 'fails' to mimic a non-trivial outcome.
        passes = path.endswith("zig.png")
        return {
            "filename": path,
            "tag": meta["tag"],
            "prompt": meta["prompt"],
            "correct": passes,
            "reason": "" if passes else "expected bench>=1, found 0",
            "detected_counts": {"bench": int(passes)},
            "matched_groups": [],
        }

    score_rows = []
    for row in read_jsonl(manifest_path):
        meta = index.lookup(source_index=int(row["source_index"]), prompt=row["prompt"])
        std = fake_eval(row["standard_path"], meta)
        zig = fake_eval(row["zigzag_path"], meta)
        score_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "source_index": int(row["source_index"]),
                "tag": row["tag"],
                "prompt": row["prompt"],
                "seed": int(row["seed"]),
                "standard_path": row["standard_path"],
                "zigzag_path": row["zigzag_path"],
                "standard_pass": bool(std["correct"]),
                "zigzag_pass": bool(zig["correct"]),
                "standard_details": std,
                "zigzag_details": zig,
            }
        )

    out_path = tmp_path / "geneval_scores.jsonl"
    write_jsonl(out_path, score_rows)
    reloaded = read_jsonl(out_path)
    assert len(reloaded) == 1
    assert reloaded[0]["zigzag_pass"] is True
    assert reloaded[0]["standard_pass"] is False
    summary = aggregate_geneval_pairs(reloaded)
    assert summary["overall"]["zigzag_only"] == 1
