# Zigzag Diffusion Sampling: Paper Notes

## Source metadata

- **Title:** Zigzag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection
- **OpenReview:** https://openreview.net/forum?id=MKvQH1ekeY
- **PDF:** https://openreview.net/pdf?id=MKvQH1ekeY
- **Reference code:** https://github.com/xie-lab-ml/zigzag-diffusion-sampling
- **Venue status on OpenReview:** ICLR 2025 Poster
- **Authors listed on OpenReview:** Bai LiChen, Shitong Shao, zikai zhou, Zipeng Qi, zhiqiang xu, Haoyi Xiong, Zeke Xie

## TL;DR from source

The authors propose a guidance mechanism that uses a guidance gap to inject semantic information step-by-step throughout the generation process.

## Initial understanding

Z-Sampling alternates denoising and inversion steps. The paper argues that the gap between these guided processes can accumulate prompt-related semantic information along the sampling path, improving alignment and quality for challenging text-to-image prompts.

## Key concepts to unpack

- Diffusion self-reflection
- Denoising vs. inversion
- Guidance gap
- Classifier-free guidance scales `gamma_1` and `gamma_2`
- Zigzag iteration count / `lambda_step`
- Plug-and-play sampling method

## Implementation anchors from the reference repository

The GitHub README lists:

- Python 3.10.14
- PyTorch with CUDA
- `diffusers == 0.22.3`
- Stable Diffusion XL inference support
- Main script: `infer.py`
- Key arguments: `--gamma_1`, `--gamma_2`, `--infer_step`, `--lambda_step`, `--image_size`, `--T_max`, `--seed`, `--device`, `--save_dir`

## Reading questions

1. What equation defines the denoising/inversion guidance gap?
2. Which parts of the method are theoretical vs. empirical?
3. Which benchmark datasets and metrics are used?
4. What does the method cost in wall-clock time and memory?
5. Does the method require model retraining, fine-tuning, or only sampling-time changes?
6. What limitations or failure cases do the authors report?

## Presentation notes draft

- Problem: strong diffusion models can still fail on challenging prompts, especially prompt-image alignment.
- Idea: use the model's own denoising/inversion discrepancy as a signal during generation.
- Method: perform a zigzag path that alternates denoising and inversion rather than a one-way denoising trajectory.
- Claim: this accumulates semantic information with limited code and compute overhead.
- Evidence to verify: reported winning rates, benchmark metrics, qualitative examples, and ablations.

## TODO

- Read the PDF closely and extract the main equations.
- Create a method diagram.
- Clone upstream code locally with `scripts/fetch-upstream.ps1` and inspect `infer.py`.
- Record the upstream commit hash used for any reproduction.
