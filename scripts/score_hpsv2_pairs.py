"""Score standard-vs-Z image pairs from a Vlad manifest with HPSv2.

Install the optional scorer before running the real metric step:

    uv pip install hpsv2
"""
from __future__ import annotations

import argparse

from zigzag_project.experiment_io import make_hps_score_rows, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hps-version", default="v2.1")
    args = parser.parse_args()

    try:
        import hpsv2
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency hpsv2. Install with: uv pip install hpsv2"
        ) from exc

    def scorer(paths: list[str], prompt: str) -> list[float]:
        return hpsv2.score(paths, prompt, hps_version=args.hps_version)

    rows = make_hps_score_rows(read_jsonl(args.manifest), scorer)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} HPSv2 score rows -> {args.out}")


if __name__ == "__main__":
    main()
