# 2026-06-03 - Bogdan Hunyuan-DiT Pick-a-Pic smoke attempt

## Metadata

- **Owner:** Bogdan / AI assistant
- **Goal:** Run a 1-prompt, 1-seed, 2-step smoke comparison for Hunyuan-DiT standard sampling vs. Z-Sampling on Pick-a-Pic before attempting the full 100x3 run.
- **Model:** `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers`
- **Prompt file:** `experiments/prompts/pickapic_full.md`
- **Prompt:** first Pick-a-Pic prompt, source index `0`
- **Seed:** `42`
- **Hardware detected:** `torch.cuda.is_available() == False`; `torch.backends.mps.is_available() == False`

## Command

```powershell
.venv\Scripts\python.exe scripts\run_bogdan_hunyuan_pickapic.py `
  --prompts experiments\prompts\pickapic_full.md `
  --seeds 42 `
  --steps 2 `
  --cfg 5.0 `
  --inv-cfg 0 `
  --lambda-step 1 `
  --t-max 1 `
  --size 1024 `
  --limit 1 `
  --out outputs\bogdan_hunyuan_pickapic\smoke_1x1 `
  --force
```

## Outcome

- First run downloaded the Hunyuan-DiT snapshot from Hugging Face, then failed during tokenizer loading because `sentencepiece` and `tiktoken` were missing.
- Installed required tokenizer packages:

```powershell
uv pip install --python .venv\Scripts\python.exe sentencepiece tiktoken
uv pip install --python .venv\Scripts\python.exe protobuf
```

- Second run loaded further, then continued on CPU for more than 15 minutes without writing images. The Python process reached roughly 18 GB resident memory and was manually stopped.
- No standard/Z image pair was produced.
- No HPSv2 scores were produced.

## Interpretation

This local Windows environment is not suitable for Hunyuan-DiT generation because it has no CUDA or MPS accelerator. Hunyuan-DiT is much heavier than the SDXL/DreamShaper runs already completed in this repo, and a CPU-only smoke run is not practical at 1024px.

## Follow-up

- Run the smoke command on a CUDA machine before attempting the full run.
- The runner now refuses CPU generation by default unless `--allow-cpu` is passed intentionally.
- After a CUDA smoke produces `manifest.jsonl` and images, score with `scripts/score_hpsv2_pairs.py` and summarize with `scripts/summarize_vlad_results.py`.
