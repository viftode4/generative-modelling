from __future__ import annotations

from pathlib import Path

import pandas as pd


def limit_rows(df, n=None):
    """Returns the first n rows, preserving all rows when n is None."""
    if n is None:
        return df.reset_index(drop=True)
    return df.head(n).reset_index(drop=True)


def clean_prompts(df):
    """Removes empty prompts and normalize prompt strings."""
    df = df.copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"].ne("")]
    return df.reset_index(drop=True)


def load_pickapic_prompts(csv_path, limit=100):
    """
    Loads stored Pick-a-Pic prompts from a CSV file.

    Takes only the first 100 prompts by default, matching the paper's test-set
    COUNT (Appendix A.1: "we use only the first 100 prompts as the test set").
    NOTE: the CSV is the author's PickaPic-selected-prompts substitute (the
    original v1/v2 datasets are no longer accessible), so the exact prompts are
    not guaranteed to match the paper. See experiments/prompts/pickapic_full.md.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Pick-a-Pic CSV not found: {csv_path}")

    prompts = []
    with csv_path.open("r", encoding="utf-8") as file:
        for line in file:
            prompt = line.strip()
            if not prompt:
                continue
            if prompt.lower() == "prompt":
                continue
            prompts.append(prompt)

    df = pd.DataFrame({"prompt": prompts})
    df = clean_prompts(df)
    df.insert(0, "prompt_id", range(len(df)))
    return limit_rows(df, limit)


def load_drawbench_prompts(csv_path, limit=None):
    """Load stored DrawBench prompts from a CSV file."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"DrawBench CSV not found: {csv_path}")

    file = pd.read_csv(csv_path, header=None)
    file = file.iloc[1:, :2]
    file.columns = ["prompt", "category"]

    df = clean_prompts(file)
    df["category"] = df["category"].astype("string").str.strip()
    df.insert(0, "prompt_id", range(len(df)))
    return limit_rows(df, limit)

def load_geneval_prompts(jsonl_path, limit=None):
    """Load GenEval prompts from a JSONL file into a pandas DataFrame."""
    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find GenEval file at: {path}")

    # Read the JSONL file (lines=True is required for .jsonl files)
    df = pd.read_json(path, lines=True)

    # If we only want to test a few prompts, limit the dataframe
    if limit is not None:
        df = df.head(limit)

    return df
