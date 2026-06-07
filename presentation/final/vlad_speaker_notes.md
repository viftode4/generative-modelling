# Vlad Speaker Notes — End-term Presentation

Use this if Vlad owns the implementation/results section or has to present the whole deck quickly.

## 60-second Vlad slot

> My lane was DreamShaper-xl-v2-turbo on a balanced GenEval subset. I tested standard 4-step DDIM against Z-Sampling with the paper's turbo settings: `T=4`, `gamma_1=3.5`, `gamma_2=0`, `lambda_step=3`, `t_max=1`. The important part is that every standard/Z pair shares the same initial latent, prompt, and seed.
>
> The strongest result is the detector score. Standard sampling passes 17 out of 54 pairs, or 31.5%. Z-Sampling passes 27 out of 54, or 50.0%. There are 10 cases where only Z passes and zero where only standard passes. HPSv2 agrees directionally: Z wins 53 out of 54 pairwise comparisons, with mean delta +0.0358.
>
> I also checked the runtime criticism. Z is about 2.24 times slower than 4-step standard, so it is not free. But compared against a slower 9-step standard baseline, Z still wins 42 out of 54 HPS pairs. So the cautious conclusion is: Z-Sampling helps this turbo model on this subset, especially colors/counting/two-object prompts, but it has runtime cost and does not solve every compositional category.

## 2-minute Vlad slot

1. **Setup:** DreamShaper-xl-v2-turbo + GenEval subset, 18 prompts, 3 seeds, 54 pairs.
2. **Implementation:** repo-local `zsampling.py`, DDIM + DDIM inverse scheduler, shared initial latents.
3. **Detector result:** standard 31.5%, Z 50.0%, +18.5 percentage points.
4. **HPS result:** Z 53/54 wins, mean delta +0.0358.
5. **Cost result:** standard 6.16s/image, Z 13.80s/image, 9-step standard 18.14s/image; Z still wins 42/54 against 9-step standard.
6. **Caveat:** subset is small; HPS is not GenEval; MPS/CPU stack differs from paper.
7. **Conclusion:** useful but not magic; report with limitations.

## If asked: why not full GenEval?

> We did run detector-style GenEval scoring for Vlad's existing 54 pairs, but not the full benchmark. The local path used `mmdet==2.28.2`, `mmcv-full==1.7.2`, and CPU execution on Apple Silicon. That is enough to validate the subset, but it is not a full paper-scale GenEval reproduction.

## If asked: why compare to 9-step standard?

> Z-Sampling does extra model calls. A fair criticism is that the improvement might only come from extra compute. So I regenerated a stronger standard baseline with 9 DDIM steps. It is slower than the 4-step Z run on this machine, but Z still wins 42/54 HPS pairs. That narrows the claim but makes it more credible.

## If asked: what is the honest final claim?

> Z-Sampling improves our verified local DreamShaper/GenEval subset and a separate SD2.1/Pick-a-Pic run, but we should not claim full benchmark reproduction. The correct framing is controlled local reproduction with positive evidence, runtime cost, and metric/environment caveats.
