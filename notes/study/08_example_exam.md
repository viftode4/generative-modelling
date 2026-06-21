# Example exam: 30 June digital test

Structured from the pasted digital-test example. Personal identifiers from the screenshot are intentionally omitted from this study source.

## Section 1 — VAEs

### 1a — 1.0p
**Prompt**

Variational Autoencoders (VAEs) are continuous latent variable models. We define the generative process as $z \sim p(z)$ and $x \sim p(x\mid z)$. To fit the model to data with parameters $w$, the likelihood is

$$p(x\mid w)=\int p(x\mid z,w)p(z)\,dz.$$

Can we directly minimize the log-likelihood $\log p(x\mid w)$? Explain.

**Answer target**

In practice we maximize log-likelihood, or equivalently minimize negative log-likelihood. For VAEs the exact marginal likelihood is generally not directly tractable because it requires integrating over all latent variables $z$ and depends on an intractable posterior $p(z\mid x)$. This motivates optimizing a tractable lower bound instead.

**Explanation**

Say explicitly that the obstacle is the latent-variable integral/posterior, not that likelihood is conceptually wrong.

**Concept hints:** w3-vae-concept-nonlinear-latent-variable-model, w1-mle-lvm-concept-marginal-likelihood

### 1b — 2.0p
**Prompt**

To train the VAE, we instead approximate the Evidence Lower Bound (ELBO). Explain why the ELBO is called "evidence lower bound" and give an interpretation of the two terms that compose the ELBO.

**Answer target**

The evidence is $\log p_w(x)$, the log marginal likelihood of the observed data. The ELBO is a lower bound because Jensen's inequality gives $\log p_w(x) \ge \mathcal L(w,\phi;x)$. In the common VAE form,

$$\mathcal L=\mathbb E_{q_\phi(z\mid x)}[\log p_w(x\mid z)]-D_{KL}(q_\phi(z\mid x)\Vert p(z)).$$

The first term is reconstruction/data fit; the second regularizes the approximate posterior toward the prior.

**Explanation**

A complete answer names both the bound relation and the reconstruction-versus-KL interpretation.

**Concept hints:** w3-vae-concept-vae-elbo

### 1c — 3.0p
**Prompt**

Derive the ELBO for VAE.

**Answer target**

Insert an approximate posterior $q_\phi(z\mid x)$ into the marginal likelihood and apply Jensen:

$$\log p_w(x)=\log\int q_\phi(z\mid x)\frac{p_w(x,z)}{q_\phi(z\mid x)}\,dz$$

$$\ge \mathbb E_{q_\phi(z\mid x)}[\log p_w(x,z)-\log q_\phi(z\mid x)].$$

Using $p_w(x,z)=p_w(x\mid z)p(z)$ gives

$$\mathcal L=\mathbb E_{q_\phi(z\mid x)}[\log p_w(x\mid z)]-D_{KL}(q_\phi(z\mid x)\Vert p(z)).$$

**Explanation**

The key steps are multiply/divide by $q$, convert the integral into an expectation, apply Jensen, then split the joint.

**Concept hints:** w3-vae-concept-vae-elbo, w2-em-ppca-concept-elbo-and-gap

### 1d — 2.0p
**Prompt**

There is yet another problem: the lower bound is still intractable. Why?

**Answer target**

The expectation over $q_\phi(z\mid x)$ usually has no closed form for neural decoders, and direct sampling of $z$ creates a stochastic node through which gradients must pass. The true posterior is also unavailable, so we need an approximate posterior and practical gradient estimator.

**Explanation**

Do not stop at "it has an integral"; mention the neural expectation / gradient-through-sampling issue.

**Concept hints:** w3-vae-concept-amortized-inference, w3-vae-concept-reparameterization-trick

### 1e — 1.0p
**Prompt**

What is amortized variational inference?

**Answer target**

Amortized variational inference learns a shared inference network $q_\phi(z\mid x)$ that outputs variational posterior parameters for each input $x$, instead of optimizing separate variational parameters independently for every datapoint.

**Explanation**

The word amortized means the cost of inference is shared through learned parameters $\phi$.

**Concept hints:** w3-vae-concept-amortized-inference

### 1f — 1.0p
**Prompt**

We can address the intractability of the lower bound using the reparameterization trick. Explain the reparameterization trick and provide a mathematical formula that illustrates it.

**Answer target**

Move the randomness to an external noise variable so the sampled latent is a differentiable function of network outputs. For a diagonal Gaussian posterior,

$$z=\mu_\phi(x)+\sigma_\phi(x)\odot \epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

Gradients can then flow through $\mu_\phi$ and $\sigma_\phi$.

**Explanation**

The trick does not remove randomness; it makes the randomness independent of the learned parameters.

**Concept hints:** w3-vae-concept-reparameterization-trick

## Section 2 — Energy-based models

### 2a — 1.0p
**Prompt**

Energy-based models have density

$$p_\theta(x)=\frac{\exp(-E_\theta(x))}{Z_\theta}.$$

Write down the expression of $Z_\theta$.

**Answer target**

For continuous data,

$$Z_\theta=\int \exp(-E_\theta(x))\,dx.$$

For discrete data, replace the integral by a sum over states.

**Explanation**

$Z_\theta$ is the partition function / normalizing constant.

**Concept hints:** w5-ebm-concept-partition-function

### 2b — 1.0p
**Prompt**

Prove that $p_\theta(x)$ satisfies the PDF requirements: non-negativity and normalization.

**Answer target**

Because $\exp(-E_\theta(x))\ge 0$ and $Z_\theta>0$ when finite, $p_\theta(x)\ge0$. Also,

$$\int p_\theta(x)\,dx=\frac{1}{Z_\theta}\int\exp(-E_\theta(x))\,dx=1.$$

**Explanation**

The proof relies on the definition of $Z_\theta$.

**Concept hints:** w5-ebm-concept-energy-parameterization, w5-ebm-concept-partition-function

### 2c — 1.0p
**Prompt**

What part makes this expression $p_\theta$ so flexible?

**Answer target**

The energy function $E_\theta(x)$ is highly flexible, often represented by a neural network. The model only needs an unnormalized energy, so it can define rich density shapes without requiring a tractable normalized form up front.

**Explanation**

The flexibility comes with the cost of normalization and sampling.

**Concept hints:** w5-ebm-concept-energy-parameterization

### 2d — 1.0p
**Prompt**

List at least two typical shortcomings of EBMs.

**Answer target**

Typical shortcomings include an intractable partition function, slow or difficult sampling, reliance on MCMC/negative samples, unstable training, and poor mixing in high-dimensional spaces.

**Explanation**

Name at least two and connect them to normalization or sampling.

**Concept hints:** w5-ebm-concept-partition-function, w5-ebm-concept-contrastive-divergence-and-persistent-chains

### 2e — 1.0p
**Prompt**

What is the log-likelihood $\log p_\theta(x)$?

**Answer target**

$$\log p_\theta(x)=-E_\theta(x)-\log Z_\theta.$$

**Explanation**

Take the log of the exponential numerator divided by $Z_\theta$.

**Concept hints:** w5-ebm-concept-energy-parameterization, w5-ebm-concept-partition-function

### 2f — 1.0p
**Prompt**

Show that the gradient of log-likelihood is

$$\nabla_\theta \log p_\theta(x)=-\nabla_\theta E_\theta(x)+\mathbb E_{x'\sim p_\theta}[\nabla_\theta E_\theta(x')].$$

Provide the process of your deduction.

**Answer target**

Start with $\log p_\theta(x)=-E_\theta(x)-\log Z_\theta$. Then

$$\nabla_\theta\log Z_\theta=\frac{1}{Z_\theta}\nabla_\theta\int e^{-E_\theta(x')}dx'$$

$$=\frac{1}{Z_\theta}\int -\nabla_\theta E_\theta(x')e^{-E_\theta(x')}dx'=-\mathbb E_{x'\sim p_\theta}[\nabla_\theta E_\theta(x')].$$

Therefore,

$$\nabla_\theta \log p_\theta(x)=-\nabla_\theta E_\theta(x)+\mathbb E_{x'\sim p_\theta}[\nabla_\theta E_\theta(x')].$$

**Explanation**

The sign is the common trap: differentiating $e^{-E}$ introduces a minus sign that cancels with $-\nabla\log Z$.

**Concept hints:** w5-ebm-concept-positive-phase-and-negative-phase

### 2g — 1.0p
**Prompt**

Briefly describe what the two terms in the EBM likelihood gradient are doing to the energy function qualitatively.

**Answer target**

Under gradient ascent, the data term lowers energy on observed data. The model-expectation term raises energy on samples from the current model distribution, often called negative samples. This creates the positive-phase / negative-phase push-pull.

**Explanation**

A good answer uses energy language, not only probability language.

**Concept hints:** w5-ebm-concept-positive-phase-and-negative-phase

### 2h — 1.0p
**Prompt**

In practice, why do we often use MCMC to estimate the second term? Describe the MCMC process given $p_\theta(x)$.

**Answer target**

The second term is an expectation under $p_\theta$, but direct sampling is hard because $Z_\theta$ is intractable. MCMC can use the unnormalized density $\exp(-E_\theta(x))$: initialize a chain, propose a new state, accept/reject based on the energy ratio or use Langevin updates, discard burn-in, then average $\nabla_\theta E_\theta(x')$ over the samples.

**Explanation**

The key is that MCMC can often use ratios of unnormalized probabilities where $Z_\theta$ cancels.

**Concept hints:** w5-ebm-concept-langevin-dynamics, w5-ebm-concept-contrastive-divergence-and-persistent-chains

### 2i — 2.0p
**Prompt**

Choose all correct statements.

- EBMs can be combined with latent variables, for example Restricted Boltzmann Machines (RBMs).
- EBMs are hard to train because the partition function $Z_\theta$ is generally intractable.
- The goal of MCMC using Metropolis-Hastings is to simulate a Markov chain with stationary distribution $p_\theta(x)$ and sample from it.
- In EBMs, higher energy means higher likelihood.

**Answer target**

Correct: the first three statements. Incorrect: higher energy means higher likelihood. In an EBM, probability is proportional to $\exp(-E_\theta(x))$, so lower energy corresponds to higher probability.

**Explanation**

The last option reverses the sign of the energy-probability relationship.

**Concept hints:** w5-ebm-concept-energy-parameterization, w5-ebm-concept-partition-function, w5-ebm-concept-langevin-dynamics

## Section 3 — Score-based Models

### 3a — 1.0p
**Prompt**

Score-based models start by directly modelling the PDF. How can we directly model the probability distribution?

**Answer target**

A flexible direct model can define an unnormalized positive density and normalize it:

$$p_\theta(x)=\frac{\tilde p_\theta(x)}{Z_\theta}=\frac{\exp(f_\theta(x))}{Z_\theta},$$

or equivalently through an energy. This is valid in principle but hard to train directly.

**Explanation**

This sets up why scores are useful: they avoid needing the normalizer in the data gradient.

**Concept hints:** w5-score-concept-score-function, w5-score-concept-normalizer-disappears-in-the-score

### 3b — 1.0p
**Prompt**

Explain why your answer is a valid way of modelling.

**Answer target**

The numerator is non-negative if parameterized by an exponential or positive function, and dividing by $Z_\theta=\int \tilde p_\theta(x)dx$ makes the total probability integrate to one, assuming $Z_\theta$ is finite.

**Explanation**

This mirrors the normalizing argument for EBMs.

**Concept hints:** w5-score-concept-normalizer-disappears-in-the-score, w5-ebm-concept-partition-function

### 3c — 1.0p
**Prompt**

Is there a problem in directly training your model?

**Answer target**

Yes. The normalizing constant $Z_\theta$ is generally intractable, making likelihood training and exact density evaluation difficult. This motivates learning the score instead of the normalized density.

**Explanation**

Mention normalization difficulty explicitly.

**Concept hints:** w5-score-concept-normalizer-disappears-in-the-score

### 3d — 1.0p
**Prompt**

Define the score function. What is the benefit of using it?

**Answer target**

The score is

$$s(x)=\nabla_x\log p(x).$$

It points toward locally higher density and removes constants such as $\log Z_\theta$ because gradients with respect to $x$ of constants are zero.

**Explanation**

The benefit is both normalization cancellation and a sampling direction.

**Concept hints:** w5-score-concept-score-function, w5-score-concept-normalizer-disappears-in-the-score

### 3e — 1.0p
**Prompt**

The score matching loss is

$$L(x)=\frac12\mathbb E_{x\sim p(x)}[s_\theta(x)^2]+\mathbb E_{x\sim p(x)}[\nabla_x s_\theta(x)].$$

Give an interpretation of the two terms.

**Answer target**

The squared-score term penalizes unnecessarily large score magnitudes. The derivative/divergence term shapes the vector field so it matches how probability mass changes around the data distribution. Together they are the integration-by-parts form of matching the model score to the data score without knowing the data score directly.

**Explanation**

For multidimensional data, use divergence or trace-of-Jacobian language.

**Concept hints:** w5-score-concept-score-matching

### 3f — 1.0p
**Prompt**

What is the problem when training with the score-matching loss?

**Answer target**

Classical score matching can require expensive derivatives of the score network, such as divergence or trace of a Jacobian. It can also be ill-behaved near data manifolds where the empirical data distribution has little support.

**Explanation**

This motivates denoising score matching and noise levels.

**Concept hints:** w5-score-concept-score-matching, w5-score-concept-denoising-score-matching

### 3g — 1.0p
**Prompt**

In practice, we have access to only finite data points for training. What is the problem with having a finite dataset $D$?

**Answer target**

The empirical distribution is a collection of discrete points / delta masses. Its score is not a useful smooth vector field in the surrounding space, so it gives poor guidance between data points and in low-density regions.

**Explanation**

This is why directly learning the clean-data score is unstable.

**Concept hints:** w5-score-concept-denoising-score-matching, w5-score-concept-noise-conditional-score-networks

### 3h — 1.0p
**Prompt**

How do we overcome this in practice?

**Answer target**

Perturb the data with Gaussian noise and learn scores of the smoothed noisy distributions, often at multiple noise levels. This is denoising score matching / noise-conditional score modelling.

**Explanation**

Noise turns isolated data points into smoother distributions with meaningful gradients.

**Concept hints:** w5-score-concept-denoising-score-matching, w5-score-concept-noise-conditional-score-networks

### 3i — 1.0p
**Prompt**

For sampling, we follow the learned scores. However, instead of directly following the score, we further add noise in each step. Why?

**Answer target**

Adding noise makes the process a sampler rather than pure gradient ascent to a mode. It helps exploration, avoids collapsing to local maxima, and gives Langevin dynamics the desired stochastic behavior for sampling from the target distribution.

**Explanation**

The score term moves toward high density; the noise term maintains stochastic exploration.

**Concept hints:** w5-score-concept-langevin-score-sampling

## Section 4 — LLMs

### 4a — 3.0p
**Prompt**

Large language models are based on the transformer architecture. What is the attention formula in transformers? Explain the attention operation in relation to $Q$, $K$, and $V$.

**Answer target**

Scaled dot-product attention is

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.$$

Queries $Q$ represent what each position is looking for, keys $K$ represent what each position offers for matching, and values $V$ are the information that gets mixed. Dot products score query-key compatibility, softmax turns scores into attention weights, and multiplying by $V$ returns a weighted combination of value vectors.

**Explanation**

Mention the $\sqrt{d_k}$ scaling and the role of the softmax weights.

**Concept hints:** w1-intro-concept-model-family-map

### 4b — 2.0p
**Prompt**

Which statements are true about the transformer architecture? Select all that apply.

- The decoder uses masked self-attention to prevent attending to future positions during training.
- The Transformer encoder consists of stacked layers, each containing a multi-head self-attention sublayer followed by a feedforward sublayer.
- Transformer models have a fixed-length context window determined by the number of layers in the network.
- A Transformer model must always include both an encoder and a decoder.
- Residual connections are used in Transformer layers to help stabilize training and mitigate vanishing gradients.

**Answer target**

Correct: masked decoder self-attention; stacked encoder self-attention plus feedforward layers; residual connections. Incorrect: context length is not determined simply by number of layers, and transformer models do not always require both encoder and decoder.

**Explanation**

Decoder-only LLMs are transformers without a separate encoder.

**Concept hints:** w1-intro-concept-model-family-map

### 4c — 1.0p
**Prompt**

Transitioning into LLMs: briefly describe the typical three phases of developing a large language model: pretraining, supervised finetuning, and preference tuning.

**Answer target**

Pretraining learns broad language/statistical structure, usually by next-token prediction on massive text. Supervised finetuning trains the model on curated instruction-response examples. Preference tuning, such as RLHF or DPO-style training, adjusts behavior toward preferred outputs using human or model preference comparisons.

**Explanation**

Keep the phases in order: broad prediction, instruction following, preference alignment.

**Concept hints:** w1-intro-concept-model-family-map

### 4d — 1.0p
**Prompt**

Evaluation of LLMs: what is the effect of temperature scaling in the generated response?

**Answer target**

Temperature rescales logits before sampling. Lower temperature sharpens the distribution and makes outputs more deterministic/conservative. Higher temperature flattens the distribution and increases diversity and randomness, but can reduce coherence.

**Explanation**

Temperature changes sampling behavior, not the trained model weights.

**Concept hints:** w6-evaluation-concept-domain-specific-validity

### 4e — 1.0p
**Prompt**

What is the impact of beam search on the quality of generated responses?

**Answer target**

Beam search keeps several high-probability partial generations and can improve likelihood or structured decoding. For open-ended language responses, it can also make outputs generic, repetitive, or less diverse, so higher beam count is not automatically better.

**Explanation**

The exam point is the quality-diversity tradeoff.

**Concept hints:** w6-evaluation-concept-likelihood-evaluation

### 4f — 1.0p
**Prompt**

The formulation of perplexity is

$$\operatorname{Perplexity}_\theta(X)=\exp\left[-\frac{1}{n}\sum_{i=1}^{n}\log P_\theta(x_i\mid x_1,\ldots,x_{i-1})\right].$$

What does perplexity measure? Is higher better or lower better? Explain your reasoning.

**Answer target**

Perplexity is the exponentiated average negative log-likelihood per token. It measures how surprised the model is by the sequence, or roughly the effective branching factor. Lower perplexity is better because it means the model assigns higher probability to the observed tokens.

**Explanation**

Connect the sign: lower negative log-likelihood means lower perplexity and better predictive fit.

**Concept hints:** w6-evaluation-concept-likelihood-evaluation
