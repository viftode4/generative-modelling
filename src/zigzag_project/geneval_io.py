"""IO helpers for the GenEval object-detection evaluation.

This module shares prompt-metadata lookup logic between the GenEval detector
runner script and any future GenEval-style summary scripts. The goal is to
read the upstream GenEval ``evaluation_metadata.jsonl`` once, and serve both
prompt-string matching and ``source_index`` matching against the rows of
Vlad's standard-vs-Z manifest.

The upstream metadata file is line-aligned with the ``source_index`` field
in Vlad's manifest (source_index N -> line N, 0-indexed). We still provide
prompt-string fallback for robustness.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenEvalMetadataIndex:
    """Quick lookup over the upstream GenEval ``evaluation_metadata.jsonl``."""

    rows: list[dict[str, Any]]
    by_index: dict[int, dict[str, Any]]
    by_prompt: dict[str, dict[str, Any]]

    def lookup(self, *, source_index: int | None, prompt: str | None) -> dict[str, Any]:
        """Resolve a manifest row to its upstream metadata entry.

        Prefers ``source_index`` because Vlad's manifest records it explicitly
        and the upstream file is line-aligned. Falls back to prompt-string
        equality when the index lookup is unavailable or inconsistent.
        """
        if source_index is not None and source_index in self.by_index:
            row = self.by_index[source_index]
            if prompt is None or row["prompt"] == prompt:
                return row
        if prompt is not None and prompt in self.by_prompt:
            return self.by_prompt[prompt]
        raise KeyError(
            f"No upstream GenEval metadata for source_index={source_index} prompt={prompt!r}"
        )


def load_geneval_metadata(path: str | Path) -> GenEvalMetadataIndex:
    """Read upstream ``evaluation_metadata.jsonl`` into a lookup index.

    ``source_index`` is assigned from the row's position (0-indexed) which
    matches Vlad's manifest convention.
    """
    rows: list[dict[str, Any]] = []
    by_index: dict[int, dict[str, Any]] = {}
    by_prompt: dict[str, dict[str, Any]] = {}
    with Path(path).open() as handle:
        for idx, raw in enumerate(handle):
            stripped = raw.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            payload["source_index"] = idx
            rows.append(payload)
            by_index[idx] = payload
            by_prompt[str(payload["prompt"])] = payload
    if not rows:
        raise ValueError(f"No GenEval metadata rows in {path}")
    return GenEvalMetadataIndex(rows=rows, by_index=by_index, by_prompt=by_prompt)


def aggregate_geneval_pairs(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-pair GenEval pass/fail rows into overall + per-tag summaries."""
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
    standard_pass = 0
    zigzag_pass = 0
    both_pass = 0
    neither_pass = 0
    z_only = 0
    s_only = 0
    for row in rows:
        s = bool(row["standard_pass"])
        z = bool(row["zigzag_pass"])
        standard_pass += int(s)
        zigzag_pass += int(z)
        if s and z:
            both_pass += 1
        elif not s and not z:
            neither_pass += 1
        elif z and not s:
            z_only += 1
        elif s and not z:
            s_only += 1
    return {
        "pairs": pairs,
        "standard_pass": standard_pass,
        "zigzag_pass": zigzag_pass,
        "standard_pass_rate": standard_pass / pairs if pairs else 0.0,
        "zigzag_pass_rate": zigzag_pass / pairs if pairs else 0.0,
        "delta_pass_rate": (zigzag_pass - standard_pass) / pairs if pairs else 0.0,
        "both_pass": both_pass,
        "neither_pass": neither_pass,
        "zigzag_only": z_only,
        "standard_only": s_only,
    }
