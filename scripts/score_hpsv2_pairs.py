"""Score standard-vs-Z image pairs from a Vlad manifest with HPSv2.

Install the optional scorer before running the real metric step:

    .venv/bin/pip install hpsv2==1.2.0

hpsv2==1.2.0 ships without OpenCLIP's BPE vocab file; download it manually:

    PKG=$(.venv/bin/python -c 'import hpsv2, os; print(os.path.dirname(hpsv2.__file__))')
    curl -L https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
        -o "$PKG/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
"""
from __future__ import annotations

import argparse

import numpy as np

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
            "Missing optional dependency hpsv2. See module docstring for install."
        ) from exc

    def scorer(paths: list[str], prompt: str) -> list[float]:
        raw = hpsv2.score(paths, prompt, hps_version=args.hps_version)
        # hpsv2 has returned both list[float] and list[np.ndarray[1]] across
        # point releases; coerce to a flat list of two scalars so downstream
        # `float(scores[i])` indexing is stable.
        flat = np.asarray(raw, dtype=float).reshape(-1).tolist()
        if len(flat) != len(paths):
            raise RuntimeError(
                f"hpsv2.score returned {len(flat)} values for {len(paths)} paths"
            )
        return flat

    rows = make_hps_score_rows(read_jsonl(args.manifest), scorer)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} HPSv2 score rows -> {args.out}")


if __name__ == "__main__":
    main()
