# Project Overview

## Identity

- **Course:** Generative Modelling
- **Project group:** Group 14
- **Group ID:** Group_774744_62489_16
- **Topic:** ZigZag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection

## Primary sources

- OpenReview forum: https://openreview.net/forum?id=MKvQH1ekeY
- OpenReview PDF: https://openreview.net/pdf?id=MKvQH1ekeY
- Reference implementation: https://github.com/xie-lab-ml/zigzag-diffusion-sampling

## One-sentence project framing

Study and explain Zigzag Diffusion Sampling (Z-Sampling), a sampling method that uses diffusion self-reflection through alternating denoising and inversion to improve prompt-image alignment and generation quality.

## Current known facts from sources

- The OpenReview page lists the paper as an ICLR 2025 poster, published 2025-01-22 and last modified 2025-03-06.
- OpenReview keywords include diffusion model, semantic information, and classifier guidance gap.
- The authors describe the method as leveraging a guidance gap to inject semantic information step-by-step during generation.
- The reference repository says the code supports Stable Diffusion XL inference and is licensed Apache-2.0.

## Open questions

- What is the exact mathematical definition of the guidance gap?
- How does alternating denoising/inversion differ from standard classifier-free guidance in implementation?
- What are the added computational costs of Z-Sampling in practice?
- Which baselines and metrics are most relevant for the course presentation?
- Can we reproduce a small qualitative example on available hardware?

## Deliverable ideas

- Paper summary with method diagram.
- Concept glossary for diffusion sampling, inversion, denoising, and guidance.
- Minimal reproduction notes using upstream code.
- Comparison table: standard sampling vs. Z-Sampling.
- Presentation-ready explanation of why self-reflection can improve diffusion outputs.

## Assessment objective

The official course documents require a 3-page double-column individual report. The repo structure is organized to support the rubric:

- Motivation/work plan: `planning/`, TA questions, task board, meeting notes.
- Content: reproducible setup, experiments, troubleshooting logs, results.
- Writing: report outline, figures/tables, critical interpretation notes.

See `knowledgebase/course/requirements-summary.md` for the distilled rubric and constraints.
