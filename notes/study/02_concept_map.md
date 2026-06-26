# DSAIT4030 Generative Modelling concept map

## 1. The core problem

A generative model tries to model a data distribution and/or generate new samples from it. The recurring course question is:

> How do we choose a model family, train it, sample from it, and evaluate whether it captured the distributional property we care about?

Key axes:
- **Can we compute likelihood?** exact, approximate, unnormalized, or implicit.
- **Can we sample efficiently?** direct decoder, invertible map, MCMC, reverse diffusion, ODE/SDE solver.
- **Can we control generation?** conditioning, guidance, and domain constraints.
- **Can we evaluate it?** likelihood, sample metrics, downstream utility, overfitting checks.

## 2. Objective design: KL -> maximum likelihood

The course starts from the idea that we need a distance/divergence between the empirical data distribution and model distribution. Minimizing KL from data to model gives maximum likelihood because the data entropy term does not depend on model parameters.

Study target:
- Be able to explain why maximizing `sum_i log pθ(x_i)` is a distribution-matching objective.
- Know that KL is nonnegative and asymmetric.
- Know that “good likelihood” and “good samples” can diverge later in the evaluation lecture.

## 3. Latent variable models

Latent variable model form:
- Prior: `p(z)`
- Decoder/generative likelihood: `pθ(x|z)`
- Joint: `pθ(x,z)=pθ(x|z)p(z)`
- Marginal likelihood: `pθ(x)=∫ pθ(x,z) dz` or `sum_z pθ(x,z)`

The main difficulty is inference over `z` and marginalization.

### GMM and EM

Discrete latent assignment variable tells which component generated a point.

- **E-step:** infer/update posterior responsibilities over latent assignments.
- **M-step:** update parameters using expected complete-data log-likelihood.
- EM can be seen as coordinate ascent on an ELBO/lower bound.

### PPCA

PPCA gives a probabilistic latent-variable view of PCA:
- low-dimensional latent `z`
- linear map to data space
- Gaussian noise
- covariance structure `WWᵀ + σ²I`

Study target:
- Explain why latent variables help with high-dimensional data/manifold structure.

### VAE

VAE handles nonlinear latent-variable models where exact posterior/marginal likelihood is intractable.

Core pieces:
- Encoder / inference network: `qφ(z|x)`
- Decoder / generative network: `pθ(x|z)`
- ELBO: expected reconstruction/log-likelihood minus KL to prior.
- Reparameterization trick: sample noise independently, transform by encoder outputs, backpropagate through the transform.

Study target:
- Derive/identify the ELBO.
- Explain amortized inference.
- Explain why the reparameterization trick is needed.

## 4. Energy-based models

EBM form:
- `pθ(x) = exp(-Eθ(x)) / Zθ`
- `Zθ` is the partition function.

Strength:
- Very flexible unnormalized density model.

Difficulty:
- Normalizer and likelihood gradient require expectations under the model distribution.
- Training often needs approximate sampling.

Gradient intuition:
- Lower energy for data samples.
- Raise energy for model-generated/negative samples.
- Approximate negative samples using MCMC such as Metropolis-Hastings or Langevin dynamics.

Study target:
- Explain why partition function makes training hard.
- Explain positive phase vs negative phase.

## 5. Normalizing flows

A normalizing flow uses an invertible differentiable transformation between base noise and data.

Change of variables:
- Map base `z` to data `x=g(z)` or inverse `z=f(x)`.
- Exact likelihood uses the Jacobian determinant.
- Log-likelihood is base log-density plus log absolute determinant of inverse Jacobian.

Strength:
- Exact likelihood and exact sampling if transform is tractable.

Cost/constraint:
- Need invertibility and tractable determinant, often same dimensionality between latent and data.

Study target:
- Write the change-of-variables formula.
- Compare flows to VAEs and EBMs.

## 6. Score-based models

The score is the gradient of log density with respect to data:
- `s(x)=∇x log p(x)`

Why useful:
- The score ignores unknown normalizing constants.
- Can guide samples toward high-density regions.

Training ideas:
- Score matching learns the score without directly knowing normalized density.
- Denoising score matching perturbs data with noise and learns to denoise/estimate noisy scores.
- Noise-conditional score networks learn across multiple noise levels.

Sampling:
- Langevin dynamics moves samples using score gradients plus noise.
- Annealed Langevin dynamics starts at high noise and gradually reduces it.

Study target:
- Explain why low-density regions are hard.
- Explain why adding noise helps.

## 7. Diffusion models

Diffusion models turn data into noise through a forward noising process and learn the reverse denoising process.

Discrete view:
- Forward process: gradually add Gaussian noise.
- Reverse model: learn to denoise step by step.
- DDPM objective often becomes predicting the added noise.
- Can be interpreted as a hierarchical VAE / ELBO training setup.

Continuous view:
- Forward SDE perturbs data to noise.
- Reverse SDE or probability-flow ODE generates samples.
- DDIM/ODE-style samplers can be deterministic variants.

Study target:
- Connect score-based models to diffusion.
- Explain forward noising, reverse denoising, and noise prediction.
- Compare stochastic SDE sampling with ODE/probability-flow sampling.

## 8. Continuous flows and flow matching

Continuous normalizing flows define a time-dependent ODE:
- `dx/dt = vθ(t,x)`

Density changes according to the continuity equation / divergence of the vector field.

Flow matching trains a vector field to transport simple base distribution samples toward data without needing expensive simulation inside the training objective.

Study target:
- Explain NF vs CNF vs flow matching.
- Explain the vector-field view: learn “how particles should move from noise to data.”

## 9. Evaluation

No universal metric exists. Match metric to goal.

Likelihood-based:
- Exact likelihood when tractable.
- Approximate likelihood with importance sampling/KDE/ELBO-style bounds when needed.
- Caveat: high likelihood does not guarantee high perceptual sample quality.

Sample-based:
- IS: sample quality/classifiability and diversity, but can be gamed and uses pretrained classifier assumptions.
- FID: compares feature-space Gaussian statistics between real and generated samples.
- MMD/KID: kernel two-sample distances; KID is an unbiased MMD-style estimator in Inception space.
- Precision/recall: quality/fidelity vs coverage/diversity.

Other concerns:
- Overfitting/memorization.
- Domain mismatch of pretrained classifiers/features.
- Downstream task utility.

Study target:
- For every metric, know what it measures and one failure mode.

## 10. Controlled generation and life sciences

Controlled generation asks for samples satisfying conditions or objectives.

- **Conditioning:** feed condition into the model.
- **Classifier guidance:** use gradient from a classifier/objective to steer generation.
- **Classifier-free guidance:** train conditional and unconditional behavior and combine guidance without a separate classifier.

Life sciences/protein generation adds constraints:
- data can be structured, geometric, discrete, and multimodal.
- valid generation may require physical/geometric constraints.
- motif scaffolding and protein design are domain-specific examples.

Study target:
- Explain conditioning vs guidance.
- Explain why protein design is harder than ordinary image generation.

## 11. Causal generative modeling (optional / non-exam)

This section is kept as course context only. Do not spend required exam-review time here unless your instructor re-adds it to scope.

Why causal?
- Generative models can learn spurious correlations and fail under interventions/distribution shift.

Core concepts:
- Disentangled/causal representations.
- Identifiability: when latent causes can be recovered from observations under assumptions.
- Interventions and counterfactuals.
- Causal world models for planning.

Optional study target:
- Explain correlation vs causation with a generative-model failure case.
- Define intervention/counterfactual at a high level.
- Explain why causal structure helps robustness and planning.
