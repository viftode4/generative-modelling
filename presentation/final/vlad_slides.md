# Vlad — Opening Slides (1–3): Paper + Background

Speaker: Vlad. ~2 minutes total (45 / 35 / 35 s). Sets up the whole talk; hands off to Radu.
Source anchors: paper §1 (intro/CFG trade-off), Fig 1–3; `experiments/configs/sdxl_pickapic_exp8.md`;
`experiments/results-index/coverage.md`.

---

## Slide 1 — The idea  (~40 s)

**Headline:** Better images from a pretrained model — no retraining.

- **Problem:** following the prompt vs. image quality is a trade-off — you can't max both.
- **Z-Sampling:** zig-zag denoise ↓ + invert ↑ → each loop adds prompt meaning → get both.

*Visual:* zig-zag arrows + one before/after pair (the "Diffusion" text example).

*Say it:* "Normally it's prompt-accuracy *or* quality. Z-Sampling gets both by zig-zagging as it generates — no retraining."

---

## Slide 2 — Models  (~25 s)

**Headline:** Three pretrained models — no fine-tuning, just a new sampler.

| Model | Type |
|---|---|
| Stable Diffusion 2.1 | UNet |
| SDXL base 1.0 | large UNet |
| DreamShaper-xl-turbo | distilled |

*Say it:* "Different backbones, all off-the-shelf — Z-Sampling is plug-and-play. (Hunyuan was planned, descoped for time.)"

---

## Slide 3 — Datasets  (~25 s)

**Headline:** Three benchmarks — taste, hard prompts, and correctness.

| Dataset | Tests | Metric |
|---|---|---|
| Pick-a-Pic | general taste | HPS v2 |
| DrawBench | hard prompts | HPS v2, AES, PickScore |
| GenEval | correct objects (count, color) | detector |

*Say it:* "HPS v2 scores human preference; GenEval objectively checks the right things got drawn. Over to Radu for the methods."

---

## Zig-zag diagram (for Canva — rebuild as arrows)

```
        prompt: "two cats"
   noise ●
          \  ↓ denoise (STRONG guidance — inject prompt)
           ●
          /  ↑ invert  (WEAK guidance — keep what was injected)
           ●
            \  ↓ denoise
             ●
             ...  (repeat: each zig-zag = +1 dose of prompt meaning)
              \  ↓
               ◆  final image — on-prompt AND high fidelity

   guidance gap = (strong scale on ↓) − (weak scale on ↑)
                = the thing that loads meaning into the latent
```
