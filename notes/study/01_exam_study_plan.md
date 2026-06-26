# DSAIT4030 Generative Modelling exam study plan

Grounding from local notes:
- Course export: `notes/DSAIT4030 Generative Modelling (2025_26 Q4)/`
- Schedule PDF says exam: **2026-06-22, 09:00-12:00**.
- Intro slide says assessment: **Digital Exam 60%**, **Project Group Report 40%**.
- Lecture material present: 13 PDFs, but Week 1 Lecture 2 and Week 2 Lecture 1 appear to be the same 40-page MLL/LVM deck, so treat them as one concept block.

## What is actually in the notes folder

1. `Content/Course Information/`
   - `Schedule and Lectures.pdf`: one-page course schedule + exam/resit dates.
2. `Content/Week 1` to `Content/Week 7`
   - Main lecture slides for the exam.
3. `Content/Project/`
   - Project description, rubric, template; useful for report/project, probably not the core exam cram target unless the exam asks general project/evaluation concepts.
4. `Assignments/`
   - Brightspace HTML placeholders only: each says “No instructions provided.”
5. Generated study workspace:
   - `notes/study/00_lecture_inventory.md`
   - `notes/study/extracted_text/*.txt`
   - this plan and the companion practice files.

## High-yield lecture order

Study in conceptual dependency order, not folder order:

1. **Foundations + objective design**
   - Intro, probability view, discriminative vs generative, conditional generation.
   - KL divergence -> maximum likelihood -> log-likelihood objective.
2. **Latent variable models**
   - GMMs, EM, ELBO, PPCA, VAEs.
3. **Unnormalized and exact-likelihood models**
   - Energy-based models, partition function, positive/negative phase, MCMC/Langevin.
   - Normalizing flows, invertibility, Jacobians, change of variables.
4. **Score/diffusion/continuous-time models**
   - Score matching, denoising score matching, Langevin sampling.
   - DDPM/DDIM, SDE/ODE view, probability flow ODE.
   - Continuous normalizing flows and flow matching.
5. **Evaluation**
   - Likelihood, importance sampling, IS/FID/MMD/KID, precision/recall, overfitting and classifier-metric caveats.
6. **Applications and guidance**
   - Conditioning vs guidance, classifier guidance, classifier-free guidance, protein design constraints.
   - Causal generative modelling is optional/non-exam for this pass.

## Two-day cram plan

### Saturday, 2026-06-20

**Block 1: Foundations and likelihood (90 min)**
- Read: Intro + MLL/LVM deck.
- Output: explain from memory why minimizing KL from data to model becomes maximizing log-likelihood.
- Drill: define generative vs discriminative model; define `pθ(x)`, `pθ(x,z)`, `pθ(z|x)`.

**Block 2: EM, PPCA, VAE (120 min)**
- Read: EM/PPCA + VAE decks.
- Output: write the ELBO and identify reconstruction vs KL terms.
- Drill: E-step vs M-step; why VAEs need amortized inference and reparameterization.

**Block 3: EBM + flows (120 min)**
- Read: Energy-Based Models + Normalising Flow.
- Output: compare EBM, VAE, GAN/implicit, normalizing flow by likelihood tractability and sampling.
- Drill: derive change-of-variables likelihood and EBM gradient intuition.

**Block 4: quick self-test (45 min)**
- Use `03_practice_questions.md`, sections A-C.
- Mark weak spots; do not reread passively.

### Sunday, 2026-06-21

**Block 1: score-based + diffusion (150 min)**
- Read: Score-based + Diffusion Models.
- Output: explain how denoising score matching leads into diffusion sampling.
- Drill: score function, Langevin dynamics, forward noising, reverse denoising, DDPM vs DDIM/SDE/ODE view.

**Block 2: continuous flows + flow matching (90 min)**
- Read: Continuous Flows and Flow Matching.
- Output: compare NF, CNF, diffusion, flow matching.
- Drill: continuity equation, divergence term, vector-field training intuition.

**Block 3: evaluation (120 min)**
- Read: Evaluation of Generative Models.
- Output: one table: metric -> measures -> assumptions -> failure mode.
- Drill: why likelihood is not sample quality; FID vs IS; precision vs recall; overfitting checks.

**Block 4: applications/guidance (60-75 min)**
- Read: Life Sciences. Keep Causal Generative Modeling optional/non-exam.
- Output: explain conditioning vs guidance and why biological constraints matter.
- Drill: classifier guidance vs classifier-free guidance; protein constraints; motif scaffolding.

**Block 5: full retrieval pass (90 min)**
- Answer all practice questions without notes.
- Make a one-page formula/idea sheet from memory.

### Monday, 2026-06-22 before exam

- 30-45 min only: formula sheet + weak spots.
- No new concepts.
- Focus on: ELBO, EM, EBM gradient, flow likelihood, score/diffusion pipeline, evaluation metrics.

## If time collapses: minimum viable exam prep

1. ELBO/EM/VAE.
2. Score matching -> diffusion.
3. Normalizing flow change of variables.
4. EBM partition function/training gradient/MCMC.
5. Evaluation metrics and caveats.
6. Conditioning/guidance and domain-specific validity constraints.

## How we can use the lecture PDFs from here

- `00_lecture_inventory.md`: quick map of every file and extracted text.
- `02_concept_map.md`: compressed conceptual explanation of the course.
- `03_practice_questions.md`: active recall practice.
- Next useful step: generate an Anki-style card deck or run an interactive oral exam over one topic at a time.
