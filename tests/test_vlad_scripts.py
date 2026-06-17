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


def test_run_bogdan_hunyuan_pickapic_dry_run_writes_manifest(tmp_path: Path):
    prompts = tmp_path / "prompts.md"
    prompts.write_text(
        '# Prompts\n\n```jsonl\n'
        '{"source_index": 0, "tag": "official", "prompt": "a robot on a bench"}\n'
        '```\n'
    )
    out_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_bogdan_hunyuan_pickapic.py",
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
    assert rows[0]["prompt_id"] == "pickapic_000_official"
    assert rows[0]["model_id"] == "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
    assert rows[0]["standard_path"].endswith("pickapic_000_official_seed42_standard.png")
    assert rows[0]["zigzag_path"].endswith("pickapic_000_official_seed42_zigzag.png")
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


def test_run_sdxl_pickapic_lambda_sweep_dry_run_writes_manifest(tmp_path: Path):
    prompts = tmp_path / "prompts.md"
    prompts.write_text(
        '# Prompts\n\n```jsonl\n'
        '{"source_index": 0, "tag": "official", "prompt": "a robot on a bench"}\n'
        '```\n'
    )
    out_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_sdxl_pickapic_lambda_sweep.py",
            "--prompts",
            str(prompts),
            "--seeds",
            "42",
            "--lambda-steps",
            "0",
            "3",
            "--steps",
            "4",
            "--out",
            str(out_dir),
            "--dry-run",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in (out_dir / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["prompt_id"] == "pickapic_000_official"
    assert rows[0]["model_id"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert rows[0]["lambda_step"] == 0
    assert rows[1]["lambda_step"] == 3
    assert rows[1]["zigzag_path"].endswith("pickapic_000_official_seed42_lambda03_zigzag.png")
    assert rows[1]["dry_run"] is True


def test_run_sdxl_pickapic_lambda_sweep_start_skips_prompts(tmp_path: Path):
    prompts = tmp_path / "prompts.md"
    prompts.write_text(
        '# Prompts\n\n```jsonl\n'
        '{"source_index": 0, "tag": "official", "prompt": "first"}\n'
        '{"source_index": 1, "tag": "official", "prompt": "second"}\n'
        '{"source_index": 2, "tag": "official", "prompt": "third"}\n'
        '```\n'
    )
    out_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_sdxl_pickapic_lambda_sweep.py",
            "--prompts",
            str(prompts),
            "--seeds",
            "42",
            "--lambda-steps",
            "0",
            "--steps",
            "4",
            "--start",
            "1",
            "--limit",
            "1",
            "--out",
            str(out_dir),
            "--dry-run",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in (out_dir / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt_id"] == "pickapic_001_official"
    assert rows[0]["prompt"] == "second"


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


def test_summarize_sdxl_lambda_sweep_writes_markdown_and_plot(tmp_path: Path):
    scores = tmp_path / "scores.jsonl"
    rows = [
        {
            "prompt_id": "p1",
            "tag": "official",
            "prompt": "a photo of a bench",
            "seed": 42,
            "standard_path": "standard.png",
            "zigzag_path": "zigzag_l0.png",
            "standard_score": 0.2,
            "zigzag_score": 0.3,
            "delta": 0.1,
            "lambda_step": 0,
        },
        {
            "prompt_id": "p1",
            "tag": "official",
            "prompt": "a photo of a bench",
            "seed": 42,
            "standard_path": "standard.png",
            "zigzag_path": "zigzag_l3.png",
            "standard_score": 0.4,
            "zigzag_score": 0.1,
            "delta": -0.3,
            "lambda_step": 3,
        },
    ]
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "summary.md"
    plot = tmp_path / "lambda_winrate.svg"

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_sdxl_lambda_sweep.py",
            "--scores",
            str(scores),
            "--out",
            str(out),
            "--plot-out",
            str(plot),
        ],
        check=True,
    )

    assert "# SDXL Pick-a-Pic Lambda-Step Sweep" in out.read_text()
    assert "| 0 | 1 | 1 | 0 | 0 | 100.0% | 0.1000 |" in out.read_text()
    assert "<svg" in plot.read_text()
