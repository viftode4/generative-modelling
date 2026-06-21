# Mock exam: DSAIT4030 Generative Modelling

Suggested use: 90 minutes, closed-book. Then check against `02_concept_map.md`, `03_practice_questions.md`, and `06_formula_derivation_sheet.md`.

## Section A — Short answers

1. Explain the difference between a generative and discriminative model. Give one example of each.
2. Starting from `KL(p_data || pθ)`, explain why maximum likelihood is a reasonable training objective.
3. What is the marginal likelihood in a latent variable model, and why is it often hard to compute?
4. In EM for a GMM, what is updated in the E-step and what is updated in the M-step?
5. Explain the VAE ELBO in words. What does each term encourage?
6. Why is the VAE reparameterization trick needed?
7. What is the partition function in an EBM, and why does it cause difficulty?
8. Explain the positive phase and negative phase of EBM training.
9. What properties must a normalizing flow transformation have to compute exact likelihood?
10. What is the score function, and why is it useful for models with unknown normalizing constants?
11. Why does denoising score matching add noise to data?
12. Describe the forward and reverse processes of a diffusion model.
13. What is the difference between a stochastic reverse diffusion sampler and a probability-flow ODE sampler?
14. What is learned in flow matching?
15. Why is generative model evaluation not solved by one metric?
16. Compare FID and precision/recall as evaluation tools.
17. Explain conditioning vs guidance in diffusion models.
18. Why can domain constraints make a generated sample invalid even if it looks plausible?

## Section B — Derivations / formula explanations

19. Derive the ELBO by inserting `q(z|x)` into `log pθ(x)` and applying Jensen’s inequality.
20. Show the normalizing-flow change-of-variables formula and explain the Jacobian determinant term.
21. Starting from `pθ(x)=exp(-Eθ(x))/Zθ`, explain the two terms in the log-likelihood gradient.
22. Write the DDPM closed-form noising equation `x_t` in terms of `x_0` and `ε`, and explain the usual noise-prediction objective.
23. Write the CNF log-density evolution equation and explain the role of divergence.
24. Show how classifier guidance follows from `log p(x|y) = log p(y|x) + log p(x) - log p(y)`.

## Section C — Comparisons

25. Compare VAE vs normalizing flow:
    - likelihood tractability
    - latent dimensionality constraints
    - sampling
    - main weakness

26. Compare EBM vs score-based model:
    - what is parameterized
    - how normalization is handled
    - how sampling works
    - training difficulty

27. Compare score-based models vs diffusion models:
    - score learning
    - noise schedule
    - sampling process
    - why they are closely related

28. Compare normalizing flows, continuous normalizing flows, and flow matching.

29. Compare likelihood, FID, and precision/recall for evaluating a generated image model.

30. Compare classifier guidance and classifier-free guidance.

## Section D — Applied reasoning

31. A model gets excellent likelihood but blurry samples. Explain how this can happen and which extra metrics/checks you would use.
32. A model generates sharp images but only a few modes. Which metric family diagnoses this and why?
33. You train an EBM and samples look poor because the negative samples are unrealistic. What part of training is likely failing?
34. A normalizing flow has high memory/computation from Jacobian determinants. What architectural constraints are causing this?
35. A diffusion model follows prompts too weakly. Explain how guidance could help and what tradeoff may appear.
36. A biological generative model produces geometrically impossible proteins. Why might image-generation metrics miss the problem?
37. A generator produces samples that are near-duplicates of training examples. Which overfitting checks would you use and why?
38. You need to choose between VAE, flow, EBM, and diffusion for a task. What questions determine the model choice?

## Answer key: compact targets

1. Generative models model data distributions and can sample; discriminative models model labels/decisions.
2. KL equals constant data entropy minus expected model log-likelihood.
3. `pθ(x)=∫pθ(x,z)dz`; hard due to integration/summation over latent variables.
4. E-step: latent posterior/responsibilities. M-step: parameters maximizing expected complete log-likelihood.
5. Reconstruction/data-fit term plus KL regularization toward prior.
6. To make stochastic latent sampling differentiable via external noise.
7. `Zθ=∫exp(-Eθ(x))dx`; high-dimensional integral is intractable.
8. Lower energy for data; raise energy for model negatives.
9. Invertible, differentiable, tractable inverse and Jacobian determinant.
10. `∇x log p(x)`; ignores constants like partition functions and guides sampling.
11. Makes learning well behaved near data manifolds and across density levels.
12. Forward adds noise; reverse learns denoising from noise to data.
13. SDE is stochastic; probability-flow ODE is deterministic but matches marginals conceptually.
14. A time-dependent vector field transporting noise to data.
15. Different goals require different properties: likelihood, fidelity, diversity, utility, robustness.
16. FID compares feature moments; precision/recall separates fidelity from coverage.
17. Conditioning is model input/training; guidance steers sampling with condition/objective signal.
18. Because the task may require geometric, physical, discrete, functional, or downstream validity that generic visual metrics do not measure.
19-24. See `06_formula_derivation_sheet.md`.
25-30. Good answers identify objective, tractability, sampling path, and tradeoffs.
31-38. Good answers justify metric/model choice from the target distributional property and constraints.
