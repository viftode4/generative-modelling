from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptSpec:
    source_index: int
    tag: str
    prompt: str

    @property
    def prompt_id(self) -> str:
        return f"geneval_{self.source_index:03d}_{self.tag}"


@dataclass(frozen=True)
class GenerationPair:
    prompt_id: str
    source_index: int
    tag: str
    prompt: str
    seed: int
    standard_path: str
    zigzag_path: str
    model_id: str
    image_size: int
    steps: int
    gamma_1: float
    gamma_2: float
    lambda_step: int
    t_max: int
    dry_run: bool = False
    standard_runtime_s: float | None = None
    zigzag_runtime_s: float | None = None
    source_commit: str | None = None
    upstream_commit: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_prompt_specs(path: str | Path) -> list[PromptSpec]:
    """Load prompt specs from the first fenced jsonl block in a Markdown file."""
    text = Path(path).read_text()
    in_jsonl = False
    specs: list[PromptSpec] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "```jsonl":
            in_jsonl = True
            continue
        if in_jsonl and stripped == "```":
            break
        if not in_jsonl or not stripped:
            continue
        payload = json.loads(stripped)
        specs.append(
            PromptSpec(
                source_index=int(payload["source_index"]),
                tag=str(payload["tag"]),
                prompt=str(payload["prompt"]),
            )
        )
    if not specs:
        raise ValueError(f"No prompt specs found in jsonl block: {path}")
    return specs


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate_hps_pairs(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    return {
        "overall": _aggregate_group(materialized),
        "by_tag": {
            tag: _aggregate_group(tag_rows)
            for tag, tag_rows in sorted(_group_by_tag(materialized).items())
        },
    }


def _group_by_tag(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["tag"])].append(row)
    return grouped


def _aggregate_group(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    rows = list(rows)
    pairs = len(rows)
    zigzag_wins = 0
    standard_wins = 0
    ties = 0
    total_delta = 0.0
    for row in rows:
        standard_score = float(row["standard_score"])
        zigzag_score = float(row["zigzag_score"])
        delta = zigzag_score - standard_score
        total_delta += delta
        if delta > 0:
            zigzag_wins += 1
        elif delta < 0:
            standard_wins += 1
        else:
            ties += 1
    return {
        "pairs": pairs,
        "zigzag_wins": zigzag_wins,
        "standard_wins": standard_wins,
        "ties": ties,
        "zigzag_win_rate": zigzag_wins / pairs if pairs else 0.0,
        "standard_win_rate": standard_wins / pairs if pairs else 0.0,
        "tie_rate": ties / pairs if pairs else 0.0,
        "mean_hps_delta": total_delta / pairs if pairs else 0.0,
    }


def make_hps_score_rows(
    manifest_rows: Iterable[Mapping[str, Any]],
    scorer: Any,
) -> list[dict[str, Any]]:
    """Score standard/Z image pairs with an HPS-like callable.

    ``scorer`` must accept ``([standard_path, zigzag_path], prompt)`` and return
    two numeric scores in the same order.
    """
    score_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        standard_path = str(row["standard_path"])
        zigzag_path = str(row["zigzag_path"])
        scores = scorer([standard_path, zigzag_path], str(row["prompt"]))
        standard_score = float(scores[0])
        zigzag_score = float(scores[1])
        score_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "source_index": row["source_index"],
                "tag": row["tag"],
                "prompt": row["prompt"],
                "seed": row["seed"],
                "standard_path": standard_path,
                "zigzag_path": zigzag_path,
                "standard_score": standard_score,
                "zigzag_score": zigzag_score,
                "delta": zigzag_score - standard_score,
            }
        )
    return score_rows
