import json
from pathlib import Path

from zigzag_project.experiment_io import (
    GenerationPair,
    PromptSpec,
    aggregate_hps_pairs,
    load_prompt_specs,
    write_jsonl,
)


def test_load_prompt_specs_reads_jsonl_block_from_markdown(tmp_path: Path):
    prompt_file = tmp_path / "prompts.md"
    prompt_file.write_text(
        '# Prompts\n\n```jsonl\n'
        '{"source_index": 0, "tag": "single_object", "prompt": "a photo of a bench"}\n'
        '{"source_index": 80, "tag": "two_object", "prompt": '
        '"a photo of a bench and a sports ball"}\n'
        '```\n'
    )

    specs = load_prompt_specs(prompt_file)

    assert specs == [
        PromptSpec(source_index=0, tag="single_object", prompt="a photo of a bench"),
        PromptSpec(
            source_index=80,
            tag="two_object",
            prompt="a photo of a bench and a sports ball",
        ),
    ]
    assert specs[0].prompt_id == "geneval_000_single_object"


def test_write_jsonl_serializes_generation_pairs(tmp_path: Path):
    pair = GenerationPair(
        prompt_id="geneval_000_single_object",
        source_index=0,
        tag="single_object",
        prompt="a photo of a bench",
        seed=42,
        standard_path="standard.png",
        zigzag_path="zigzag.png",
        model_id="Lykon/dreamshaper-xl-v2-turbo",
        image_size=768,
        steps=4,
        gamma_1=3.5,
        gamma_2=0.0,
        lambda_step=3,
        t_max=1,
        dry_run=True,
    )

    out = tmp_path / "manifest.jsonl"
    write_jsonl(out, [pair.to_json_dict()])

    assert json.loads(out.read_text())["dry_run"] is True


def test_aggregate_hps_pairs_reports_overall_and_per_tag_win_rates():
    rows = [
        {
            "prompt_id": "p1",
            "tag": "single_object",
            "seed": 1,
            "standard_score": 0.1,
            "zigzag_score": 0.2,
        },
        {
            "prompt_id": "p2",
            "tag": "single_object",
            "seed": 1,
            "standard_score": 0.3,
            "zigzag_score": 0.2,
        },
        {
            "prompt_id": "p3",
            "tag": "counting",
            "seed": 1,
            "standard_score": 0.4,
            "zigzag_score": 0.4,
        },
    ]

    summary = aggregate_hps_pairs(rows)

    assert summary["overall"]["pairs"] == 3
    assert summary["overall"]["zigzag_wins"] == 1
    assert summary["overall"]["standard_wins"] == 1
    assert summary["overall"]["ties"] == 1
    assert summary["overall"]["zigzag_win_rate"] == 1 / 3
    assert summary["by_tag"]["single_object"]["zigzag_win_rate"] == 0.5
    assert summary["by_tag"]["counting"]["ties"] == 1


def test_make_hps_score_rows_scores_standard_and_zigzag_paths():
    from zigzag_project.experiment_io import make_hps_score_rows

    manifest_rows = [
        {
            "prompt_id": "p1",
            "source_index": 0,
            "tag": "single_object",
            "prompt": "a photo of a bench",
            "seed": 42,
            "standard_path": "standard.png",
            "zigzag_path": "zigzag.png",
        }
    ]

    def fake_scorer(paths, prompt):
        assert paths == ["standard.png", "zigzag.png"]
        assert prompt == "a photo of a bench"
        return [0.1, 0.25]

    rows = make_hps_score_rows(manifest_rows, fake_scorer)

    assert rows == [
        {
            "prompt_id": "p1",
            "source_index": 0,
            "tag": "single_object",
            "prompt": "a photo of a bench",
            "seed": 42,
            "standard_path": "standard.png",
            "zigzag_path": "zigzag.png",
            "standard_score": 0.1,
            "zigzag_score": 0.25,
            "delta": 0.15,
        }
    ]
