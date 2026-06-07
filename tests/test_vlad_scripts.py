import json
import subprocess
import sys
from pathlib import Path


def test_run_vlad_geneval_dry_run_writes_manifest(tmp_path: Path):
    prompts = tmp_path / "prompts.md"
    prompts.write_text(
        '# Prompts\n\n```jsonl\n'
        '{"source_index": 0, "tag": "single_object", "prompt": "a photo of a bench"}\n'
        '```\n'
    )
    out_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_vlad_geneval.py",
            "--prompts",
            str(prompts),
            "--seeds",
            "42",
            "--out",
            str(out_dir),
            "--dry-run",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in (out_dir / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt_id"] == "geneval_000_single_object"
    assert rows[0]["standard_path"].endswith("geneval_000_single_object_seed42_standard.png")
    assert rows[0]["zigzag_path"].endswith("geneval_000_single_object_seed42_zigzag.png")
    assert rows[0]["dry_run"] is True


def test_run_vlad_compute_matched_standard_dry_run_writes_manifest(tmp_path: Path):
    source_manifest = tmp_path / "source_manifest.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "prompt_id": "geneval_000_single_object",
                "source_index": 0,
                "tag": "single_object",
                "prompt": "a photo of a bench",
                "seed": 42,
                "standard_path": "outputs/original_standard.png",
                "zigzag_path": "outputs/original_zigzag.png",
                "model_id": "Lykon/dreamshaper-xl-v2-turbo",
                "image_size": 768,
                "steps": 4,
                "gamma_1": 3.5,
                "gamma_2": 0.0,
                "lambda_step": 3,
                "t_max": 1,
                "zigzag_runtime_s": 13.8,
            }
        )
        + "\n"
    )
    out_dir = tmp_path / "compute_matched"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_vlad_compute_matched_standard.py",
            "--source-manifest",
            str(source_manifest),
            "--out",
            str(out_dir),
            "--standard-steps",
            "9",
            "--dry-run",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in (out_dir / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["comparison"] == "compute_matched_standard_vs_existing_zigzag"
    assert rows[0]["standard_steps"] == 9
    assert rows[0]["zigzag_steps"] == 4
    assert rows[0]["standard_path"].endswith("geneval_000_single_object_seed42_standard9.png")
    assert rows[0]["zigzag_path"] == "outputs/original_zigzag.png"
    assert rows[0]["source_standard_path"] == "outputs/original_standard.png"
    assert rows[0]["dry_run"] is True


def test_summarize_vlad_results_writes_markdown(tmp_path: Path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "tag": "single_object",
                "prompt": "a photo of a bench",
                "seed": 42,
                "standard_path": "standard.png",
                "zigzag_path": "zigzag.png",
                "standard_score": 0.1,
                "zigzag_score": 0.2,
                "delta": 0.1,
            }
        )
        + "\n"
    )
    out = tmp_path / "summary.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_vlad_results.py",
            "--scores",
            str(scores),
            "--out",
            str(out),
        ],
        check=True,
    )

    text = out.read_text()
    assert "# Vlad GenEval HPSv2 Summary" in text
    assert "Overall Z win rate" in text
    assert "Top Z-Sampling wins" in text
