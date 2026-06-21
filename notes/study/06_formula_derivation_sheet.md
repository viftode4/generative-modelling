# Formula and derivation sheet

This is not a substitute for the slides; it is the compact list of formulas/derivations to be able to reproduce.

## 1. KL divergence and maximum likelihood

KL divergence:

```text
KL(p_data || pθ) = E_{x~p_data}[log p_data(x) - log pθ(x)]
```

Since `E[log p_data(x)]` does not depend on `θ`:

```text
argmin_θ KL(p_data || pθ)
= argmax_θ E_{x~p_data}[log pθ(x)]
≈ argmax_θ (1/N) Σ_i log pθ(x_i)
```

Derivation target:
- Start from KL.
- Drop the data entropy term.
- Arrive at MLE.

## 2. Latent variable marginal likelihood

Joint:

```text
pθ(x,z) = pθ(x|z)p(z)
```

Marginal:

```text
pθ(x) = ∫ pθ(x,z) dz
```

or for discrete latents:

```text
pθ(x) = Σ_z pθ(x,z)
```

Problem:
- The integral/sum can be hard for complex/nonlinear models.

## 3. ELBO

Introduce approximate posterior `q(z|x)`:

```text
log pθ(x)
= log ∫ pθ(x,z) dz
= log ∫ q(z|x) pθ(x,z)/q(z|x) dz
≥ E_q[log pθ(x,z) - log q(z|x)]
```

ELBO:

```text
L(θ,φ;x) = E_{qφ(z|x)}[log pθ(x|z)] - KL(qφ(z|x) || p(z))
```

Relation:

```text
log pθ(x) = ELBO + KL(qφ(z|x) || pθ(z|x))
```

Derivation target:
- Insert `q(z|x)`.
- Apply Jensen.
- Split into reconstruction and KL-to-prior terms.

## 4. EM

ELBO form for EM:

```text
L(q,θ) = E_{q(z)}[log pθ(x,z)] + H(q)
```

E-step:

```text
q(z) ← p_{θ_old}(z|x)
```

M-step:

```text
θ ← argmax_θ E_{q(z)}[log pθ(x,z)]
```

Derivation target:
- Explain EM as coordinate ascent over `q` and `θ`.

## 5. PPCA

Generative model:

```text
z ~ N(0, I)
x | z ~ N(Wz + μ, σ²I)
```

Marginal:

```text
x ~ N(μ, WWᵀ + σ²I)
```

Derivation target:
- Explain covariance contribution from latent projection plus isotropic noise.

## 6. VAE reparameterization

For Gaussian encoder:

```text
qφ(z|x) = N(z; μφ(x), diag(σφ(x)^2))
ε ~ N(0, I)
z = μφ(x) + σφ(x) ⊙ ε
```

Purpose:
- Sampling randomness is isolated in `ε`, so gradients can flow through `μ` and `σ`.

## 7. Energy-based model

Density:

```text
pθ(x) = exp(-Eθ(x)) / Zθ
Zθ = ∫ exp(-Eθ(x)) dx
```

Log-likelihood:

```text
log pθ(x) = -Eθ(x) - log Zθ
```

Gradient intuition:

```text
∇θ log pθ(x)
= -∇θ Eθ(x) + E_{x'~pθ}[∇θ Eθ(x')]
```

Interpretation:
- Data term lowers data energy.
- Model expectation raises negative-sample energy.

## 8. Langevin dynamics

Basic sampling update:

```text
x_{t+1} = x_t + η ∇x log p(x_t) + sqrt(2η) ε_t
```

Score-model version:

```text
x_{t+1} = x_t + η sθ(x_t) + sqrt(2η) ε_t
```

Purpose:
- Move toward high-density regions while preserving stochastic exploration.

## 9. Normalizing flow change of variables

If `x = gθ(z)` and `z = fθ(x) = gθ^{-1}(x)`:

```text
p_X(x) = p_Z(fθ(x)) |det J_fθ(x)|
```

Log form:

```text
log p_X(x) = log p_Z(fθ(x)) + log |det J_fθ(x)|
```

Derivation target:
- Explain volume correction: stretching lowers density; compression raises density.

## 10. Score function

```text
s(x) = ∇x log p(x)
```

If `p(x)=exp(-E(x))/Z`:

```text
∇x log p(x) = -∇x E(x)
```

because `Z` does not depend on `x`.

## 11. Score matching objective, high-level

Goal:

```text
min_θ E_{p_data} ||sθ(x) - ∇x log p_data(x)||²
```

Denoising/noise-conditional version learns scores of noisy data distributions:

```text
sθ(x_t, t) ≈ ∇_{x_t} log p_t(x_t)
```

Study target:
- Know the intuition more than every algebraic variant unless the lecture emphasized a derivation.

## 12. Diffusion forward noising

Discrete DDPM-style process:

```text
q(x_t | x_{t-1}) = N(sqrt(1-β_t) x_{t-1}, β_t I)
```

Closed form:

```text
q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t)I)
```

Noise parameterization:

```text
x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
```

Typical objective:

```text
E_{x0,ε,t} ||ε - εθ(x_t,t)||²
```

## 13. Reverse diffusion idea

Generation starts from noise:

```text
x_T ~ N(0,I)
```

Then repeatedly apply learned reverse denoising transitions:

```text
pθ(x_{t-1}|x_t)
```

Study target:
- Explain the sampling story without needing all implementation constants.

## 14. SDE / ODE view

SDE view:

```text
dx = f(x,t)dt + g(t)dw
```

Reverse process uses learned score.

Probability-flow ODE idea:
- There exists a deterministic ODE whose time marginals match the stochastic diffusion process.

Study target:
- Understand stochastic vs deterministic samplers at a conceptual level.

## 15. Continuous normalizing flow

ODE:

```text
dx/dt = vθ(t,x)
```

Density evolution:

```text
d log p_t(x_t)/dt = - div vθ(t,x_t)
```

Study target:
- Divergence measures volume expansion/contraction along flow.

## 16. Flow matching

Train vector field directly:

```text
min_θ E_{t,x_t}[ ||vθ(t,x_t) - u_t(x_t)||² ]
```

where `u_t` is the target velocity field induced by a chosen path from noise to data.

Study target:
- The model learns how points should move from base distribution to data distribution.

## 17. Classifier guidance

For conditional generation, adjust score using a classifier gradient:

```text
∇x log p(x|y) = ∇x log p(x) + ∇x log p(y|x)
```

Study target:
- Guidance adds a condition-seeking gradient during sampling.

## 18. Classifier-free guidance

Use conditional and unconditional predictions:

```text
ε_guided = ε_uncond + w(ε_cond - ε_uncond)
```

Equivalent forms vary by notation.

Study target:
- No separate classifier; guidance comes from combining model predictions.

## 19. Evaluation metrics: compact definitions

Inception Score:

```text
IS = exp(E_x KL(p(y|x) || p(y)))
```

FID:

```text
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_rΣ_g)^{1/2})
```

MMD:

```text
MMD²(P,Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
```

Precision/recall:
- Precision: generated samples lie on/near data manifold.
- Recall: generated distribution covers the data manifold.

Study target:
- For each metric: what it measures, assumption, failure mode.
