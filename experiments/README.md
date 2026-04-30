# Experiments

This folder stores reproducible experiment definitions, prompts, and result indexes. Large generated outputs should go to ignored directories such as `outputs/`, `runs/`, or `results/`.

## Structure

- `configs/` - parameter files or command snippets.
- `prompts/` - prompt sets used for generation.
- `results-index/` - lightweight summaries pointing to output directories and experiment logs.

## Minimum information per experiment

- Goal / hypothesis.
- Code commit and upstream submodule commit.
- Environment and hardware.
- Prompt set and seed(s).
- Command.
- Output directory.
- Observations and interpretation.

Use `knowledgebase/templates/experiment-log-template.md` for detailed logs.
