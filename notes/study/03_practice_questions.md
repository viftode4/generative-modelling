# DSAIT4030 practice questions

Use this as active recall. First answer without notes, then check the compact answer.

## A. Foundations and objectives

1. **What is the difference between a discriminative and a generative model?**
   - Discriminative models learn decision/conditional boundaries such as `p(y|x)` or direct predictors. Generative models model how data are produced, often `p(x)` or `p(x,y)`, and can sample new data.

2. **Why does minimizing `KL(p_data || pθ)` lead to maximum likelihood?**
   - The data entropy term is constant w.r.t. θ, so minimizing the KL is equivalent to maximizing expected `log pθ(x)` over data.

3. **Name two important properties of KL divergence.**
   - Nonnegative; zero only when distributions match almost everywhere; asymmetric.

4. **What is the universal recipe for a generative learning objective?**
   - Choose a model family, define a distributional closeness/objective, then optimize parameters from data.

5. **Why can likelihood and sample quality disagree?**
   - Likelihood rewards density assignment, which may be high for broad/blurred or perceptually poor models; visual/sample quality depends on distributional properties not fully captured by scalar likelihood.

## B. Latent variable models, EM, PPCA, VAE

6. **Write the joint distribution for a standard latent variable model.**
   - `pθ(x,z)=pθ(x|z)p(z)`.

7. **Why is marginal likelihood hard in latent-variable models?**
   - It requires summing/integrating over latent variables, which is often intractable for nonlinear/high-dimensional models.

8. **What does the E-step do in EM?**
   - Computes/updates posterior beliefs over latent variables under current parameters.

9. **What does the M-step do in EM?**
   - Updates model parameters to maximize expected complete-data log-likelihood / ELBO using current posterior beliefs.

10. **How does EM relate to the ELBO?**
    - EM performs coordinate ascent on a lower bound: update the variational/posterior distribution, then update parameters.

11. **What is PPCA’s generative story?**
    - Sample low-dimensional Gaussian latent `z`, map linearly to data with `Wz + μ`, then add isotropic Gaussian noise.

12. **Why does PPCA connect to the manifold hypothesis?**
    - It models high-dimensional observations as noisy versions of lower-dimensional latent structure.

13. **What is the VAE ELBO in words?**
    - Reconstruction/log-likelihood term encouraging decoded samples to match data, minus a KL regularizer keeping approximate posterior close to the prior.

14. **Why do VAEs use amortized inference?**
    - Instead of optimizing separate latent posterior parameters for every data point, an encoder network predicts them.

15. **What is the reparameterization trick?**
    - Express latent sampling as deterministic transformation of parameters and external noise, e.g. `z=μ+σ⊙ε`, enabling backpropagation.

## C. EBMs and normalizing flows

16. **Write the EBM density form.**
    - `pθ(x)=exp(-Eθ(x))/Zθ`.

17. **What is the partition function and why is it hard?**
    - `Zθ=∫ exp(-Eθ(x)) dx`; it is hard because the integral over data space is usually intractable.

18. **What is the EBM training gradient intuition?**
    - Push data energies down and model/negative-sample energies up.

19. **Why do EBMs use MCMC/Langevin dynamics?**
    - To approximate samples from the current model distribution needed for the negative phase.

20. **What makes normalizing flows different from VAEs?**
    - Flows use invertible maps and exact change-of-variables likelihood; VAEs use approximate inference and an ELBO.

21. **Write the change-of-variables idea for flows.**
    - Density in data space equals base density at the inverse point times the absolute determinant of the inverse Jacobian.

22. **What constraints do normalizing flows impose?**
    - Transformations must be invertible and have tractable Jacobian determinants, often with equal latent/data dimensionality.

23. **When might a flow be preferable?**
    - When exact likelihood and efficient exact sampling/inference are important and invertible architecture constraints are acceptable.

## D. Score-based and diffusion models

24. **What is the score function?**
    - `∇x log p(x)`, the gradient pointing toward increasing log-density.

25. **Why does score modeling avoid the partition function problem?**
    - Gradients of log density remove additive constants such as `log Z`.

26. **What is score matching trying to learn?**
    - A vector field matching the data score without directly estimating normalized density.

27. **Why add noise for denoising score matching?**
    - It makes score estimation better behaved around data manifolds and teaches scores at multiple noise levels.

28. **What is Langevin dynamics used for?**
    - Sampling by iteratively moving in the score direction plus injecting noise.

29. **What is annealed Langevin dynamics?**
    - Sampling through a schedule from high noise to low noise, using scores at each level.

30. **Describe the diffusion forward process.**
    - Gradually corrupt data with noise until it approaches a simple noise distribution.

31. **Describe the diffusion reverse process.**
    - Learn a denoising model that reverses the noising steps to generate data from noise.

32. **Why is DDPM connected to VAEs?**
    - It can be viewed as a hierarchical latent-variable model trained with an ELBO-like objective over noising/denoising steps.

33. **What does noise prediction mean in DDPM training?**
    - The model predicts the Gaussian noise added to a clean sample at a timestep.

34. **What is the probability-flow ODE idea?**
    - A deterministic ODE can share marginal distributions with the stochastic reverse diffusion process, enabling ODE-based sampling/likelihood views.

## E. Continuous flows and flow matching

35. **What is a continuous normalizing flow?**
    - A generative model where samples evolve through an ODE defined by a learned vector field.

36. **What does the continuity equation track?**
    - How probability density changes as particles move under a vector field.

37. **Why does divergence appear in CNF likelihoods?**
    - Divergence measures local expansion/contraction of volume, changing log-density along the ODE path.

38. **What is the intuition of flow matching?**
    - Train a vector field directly from examples of how points should move from noise to data, avoiding expensive simulation in the training loss.

39. **Compare NF, CNF, and flow matching.**
    - NF: discrete invertible layers with tractable Jacobians. CNF: continuous ODE transform with divergence-based density. Flow matching: trains the ODE vector field via a supervised matching objective.

## F. Evaluation

40. **Why is generative model evaluation multi-objective?**
    - Different uses care about likelihood, fidelity, diversity, coverage, representations, downstream utility, or controllability.

41. **What does Inception Score try to measure?**
    - Classifiability/quality and diversity via a pretrained classifier’s label distribution.

42. **What is a key weakness of Inception Score?**
    - It ignores real data statistics and depends heavily on pretrained classifier/domain assumptions.

43. **What does FID compare?**
    - Mean and covariance of real vs generated samples in pretrained feature space, usually Inception features.

44. **What are MMD/KID used for?**
    - Kernel two-sample comparison between real and generated distributions; KID is an unbiased MMD-style metric in feature space.

45. **What do precision and recall mean for generative models?**
    - Precision: generated samples are realistic/fidelitous. Recall: generated samples cover the real data distribution/diversity.

46. **How can overfitting appear in generative modeling?**
    - The model memorizes or reproduces training examples while appearing high-quality under some metrics.

47. **What is the safest evaluation principle from the lecture?**
    - There is no universal best metric; evaluate the property and downstream goal you care about.

## G. Applications and guidance

48. **Conditioning vs guidance: what is the difference?**
    - Conditioning is built into the model input/training; guidance steers generation during sampling using an external or implicit objective/gradient.

49. **What is classifier guidance?**
    - Use gradients from a classifier/objective to steer diffusion samples toward a desired condition.

50. **What is classifier-free guidance?**
    - Combine conditional and unconditional model predictions to guide generation without a separate classifier.

51. **Why is protein generation structurally harder than image generation?**
    - Proteins involve geometric, physical, discrete, and functional constraints, not just perceptual plausibility.

52. **What is motif scaffolding?**
    - Designing a larger protein structure around a functional motif that must be preserved.


> Causal prompts were removed from the required drill because the causal lecture is optional/non-exam for this pass.
