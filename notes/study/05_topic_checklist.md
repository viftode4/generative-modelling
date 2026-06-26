# Topic checklist

Use this as the “can I explain it without notes?” checklist.

## 1. Foundations: probability, KL, MLE

You should be able to:
- [ ] Define generative vs discriminative modeling.
- [ ] Explain why a generative model describes a distribution over data.
- [ ] Explain why minimizing `KL(p_data || pθ)` becomes maximizing log-likelihood.
- [ ] State why KL is asymmetric and nonnegative.
- [ ] Give one reason likelihood can disagree with perceptual/sample quality.

Common exam prompts:
- “Show that maximum likelihood minimizes a KL divergence.”
- “Why is a generative model not just a classifier run backwards?”
- “Give an example where high likelihood does not imply good samples.”

## 2. Latent variable models and EM

You should be able to:
- [ ] Write `pθ(x,z)=pθ(x|z)p(z)` and `pθ(x)=∫pθ(x,z)dz`.
- [ ] Explain why marginal likelihood can be intractable.
- [ ] Describe GMM latent assignment variables.
- [ ] Explain E-step and M-step.
- [ ] Explain EM as optimizing a lower bound / ELBO-style objective.
- [ ] Explain the manifold motivation for latent variables.

Common exam prompts:
- “Why do latent variable models need inference?”
- “What exactly is updated in the E-step and M-step?”
- “Why does EM monotonically improve a bound?”

## 3. PPCA

You should be able to:
- [ ] State the PPCA generative process.
- [ ] Explain how PPCA relates to PCA.
- [ ] Explain the role of Gaussian latent variables and Gaussian observation noise.
- [ ] Connect PPCA to dimensionality reduction/manifold assumptions.

Common exam prompts:
- “What probabilistic assumptions turn PCA into PPCA?”
- “Why is PPCA a latent variable model?”

## 4. VAE

You should be able to:
- [ ] Identify encoder `qφ(z|x)`, decoder `pθ(x|z)`, and prior `p(z)`.
- [ ] Write the ELBO in words and symbols.
- [ ] Explain reconstruction term vs KL regularizer.
- [ ] Explain amortized inference.
- [ ] Explain the reparameterization trick.
- [ ] Explain why VAE likelihood is usually optimized via a lower bound, not exact MLE.

Common exam prompts:
- “Derive the ELBO from log pθ(x).”
- “Why can’t we backpropagate through naive sampling?”
- “What happens if the KL term is too strong/weak?”

## 5. Energy-based models

You should be able to:
- [ ] Write `pθ(x)=exp(-Eθ(x))/Zθ`.
- [ ] Define partition function `Zθ`.
- [ ] Explain why `Zθ` makes likelihood and gradients hard.
- [ ] Explain positive phase and negative phase.
- [ ] Explain why MCMC or Langevin dynamics is used.
- [ ] Compare EBMs to normalized likelihood models.

Common exam prompts:
- “Why are EBMs flexible but hard to train?”
- “What is the push-pull interpretation of the EBM gradient?”
- “Why does Langevin dynamics help sampling?”

## 6. Normalizing flows

You should be able to:
- [ ] State the change-of-variables formula.
- [ ] Explain why invertibility matters.
- [ ] Explain why the Jacobian determinant matters.
- [ ] Compare flows to VAEs and EBMs on likelihood/sampling/inference.
- [ ] Explain architectural constraints caused by tractable determinants.

Common exam prompts:
- “How does a flow compute exact likelihood?”
- “What is the tradeoff of normalizing flows?”
- “Why can a VAE have a lower-dimensional latent but a standard flow often cannot?”

## 7. Score-based models

You should be able to:
- [ ] Define score `∇x log p(x)`.
- [ ] Explain why the score removes normalizing constants.
- [ ] Explain score matching at a high level.
- [ ] Explain denoising score matching.
- [ ] Explain the low-density/manifold problem.
- [ ] Explain Langevin and annealed Langevin sampling.

Common exam prompts:
- “Why learn the score rather than the density?”
- “Why add noise levels?”
- “How does Langevin dynamics generate samples?”

## 8. Diffusion models

You should be able to:
- [ ] Describe the forward noising process.
- [ ] Describe the reverse denoising process.
- [ ] Explain why predicting noise is a training target.
- [ ] Explain the connection to VAEs/ELBO.
- [ ] Explain discrete vs continuous diffusion.
- [ ] Explain SDE vs probability-flow ODE at a high level.

Common exam prompts:
- “How does diffusion turn generation into iterative denoising?”
- “What is learned in DDPM training?”
- “How are score-based models and diffusion connected?”

## 9. Continuous flows and flow matching

You should be able to:
- [ ] Define a continuous normalizing flow as an ODE.
- [ ] Explain continuity equation / density transport intuition.
- [ ] Explain why divergence controls density change.
- [ ] Explain flow matching as vector-field supervision.
- [ ] Compare NF, CNF, diffusion, and flow matching.

Common exam prompts:
- “What is learned in flow matching?”
- “Why is a vector field a generative model?”
- “How is flow matching related to diffusion?”

## 10. Evaluation

You should be able to:
- [ ] Explain why evaluation is multi-objective.
- [ ] Separate density evaluation from sample evaluation.
- [ ] Define IS, FID, MMD/KID, precision/recall at a high level.
- [ ] Give one failure mode for each metric.
- [ ] Explain overfitting/memorization risk.
- [ ] Explain why pretrained classifier metrics can fail out-of-domain.

Common exam prompts:
- “Which metric would you choose and why?”
- “Why is FID not a universal measure of quality?”
- “How do precision and recall diagnose different failures?”

## 11. Conditioning, guidance, life-science applications

You should be able to:
- [ ] Distinguish conditioning from guidance.
- [ ] Explain classifier guidance.
- [ ] Explain classifier-free guidance.
- [ ] Explain why controlled generation matters.
- [ ] Explain why protein generation has geometric/discrete/functional constraints.
- [ ] Explain motif scaffolding at a high level.

Common exam prompts:
- “Conditioning vs guidance: what changes during training and sampling?”
- “Why does classifier-free guidance avoid a separate classifier?”
- “Why are biological generative models not just image generators?”

## 12. Causal generative modeling (optional / non-exam)

Use this only as context, not as required exam checklist. If you read it, you should be able to:
- [ ] Explain why correlations can fail under distribution shift.
- [ ] Define intervention and counterfactual.
- [ ] Explain identifiability.
- [ ] Explain why causal representations can help robustness and planning.
- [ ] Give an example of shortcut learning in a generative setting.

Optional prompts:
- “Why is causality relevant to generative modeling?”
- “What does it mean for a latent representation to be identifiable?”
- “How do interventions differ from observations?”
