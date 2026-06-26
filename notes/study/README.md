# DSAIT4030 Generative Modelling study hub

Use this folder as the exam workspace.

## Start here

1. `01_exam_study_plan.md` — time-boxed plan for the exam.
2. `02_concept_map.md` — how the course topics fit together.
3. `05_topic_checklist.md` — what you must be able to explain/derive per topic.
4. `06_formula_derivation_sheet.md` — compact formula and derivation targets.
5. `03_practice_questions.md` — active-recall questions with short answers.
6. `07_mock_exam.md` — mixed mock exam; do this closed-book.
7. `04_flashcards.tsv` — TSV flashcards for Anki/Quizlet-style import.
8. `00_lecture_inventory.md` — map back to every lecture PDF and extracted text.

## Best structure for studying this course

Do **not** study lecture-by-lecture only. The course is cumulative, so study by model family and by the recurring exam questions:

1. **What distribution is modeled?**
   - `p(x)`, `p(x,z)`, `p(z|x)`, unnormalized `exp(-E(x))`, score `∇ log p(x)`, ODE/SDE dynamics.

2. **What is the training objective?**
   - MLE/KL, ELBO, EM bound, EBM likelihood gradient, score matching, denoising score matching, diffusion noise prediction, flow matching, evaluation metrics.

3. **What is tractable/intractable?**
   - Likelihood, posterior inference, partition function, Jacobian determinant, sampling, evaluation.

4. **How do we sample?**
   - Decoder sampling, ancestral latent sampling, MCMC/Langevin, invertible transform, reverse diffusion, ODE/CNF solve.

5. **How do we evaluate/control it?**
   - Likelihood vs sample metrics, IS/FID/MMD/KID/precision/recall, conditioning/guidance, and domain constraints.

## Daily loop

For each block:

1. Read/skim the relevant slides for 20-40 min.
2. Close the slides and write the model/objective/sampling story from memory.
3. Answer the matching questions in `03_practice_questions.md`.
4. Add weak points to a small “redo tomorrow” list.
5. End with one comparison: e.g. VAE vs flow, EBM vs score model, diffusion vs flow matching.

## Highest-yield exam themes

- KL divergence to maximum likelihood.
- Latent variable marginal likelihood and ELBO.
- EM as coordinate ascent on a bound.
- VAE: encoder/decoder/reparameterization/KL term.
- EBM: partition function, positive/negative phase, MCMC.
- Normalizing flow: invertibility and change of variables.
- Score matching and denoising score matching.
- Diffusion: forward noising, reverse denoising, DDPM objective, SDE/ODE view.
- Flow matching and continuous normalizing flows.
- Evaluation metrics and their failure modes.
- Conditioning vs guidance, especially classifier-free guidance.
- Causal generative modeling is kept only as optional context; it is not part of the required exam path.
