# Vlad Speaker Notes — Short Version

## If Vlad presents Slide 4 results

- My lane: DreamShaper-xl-v2-turbo + GenEval.
- 18 prompts, 3 seeds, 54 paired comparisons.
- Same prompt, seed, and initial latent for standard and Z.
- Detector result:
  - Standard: 17/54 = 31.5%.
  - Z-Sampling: 27/54 = 50.0%.
  - Z-only wins: 10.
  - Standard-only wins: 0.
- HPS result:
  - Z wins 53/54 pairs.
  - Mean delta: +0.0358.
- Main interpretation:
  - Z helped this subset.
  - Best gains were color/count/two-object prompts.
  - Position and color attribution remain hard.

## If Vlad presents Slide 5 challenges

- Z-Sampling is slower.
- In my run:
  - 4-step standard: 6.16s/image.
  - 4-step Z-Sampling: 13.80s/image.
- I tested the obvious criticism:
  - Maybe Z only wins because it does more work.
- So I ran 9-step standard.
  - 9-step standard: 18.14s/image.
  - Z still wins 42/54 HPS pairs.
- Conclusion:
  - Gain is not just extra steps.
  - But runtime cost is real.

## 30-second version

> My lane was DreamShaper-xl-v2-turbo on GenEval. On 54 paired images, Z-Sampling improved detector pass rate from 31.5% to 50.0%, and HPS favored Z in 53 out of 54 pairs. It is not free: runtime was about 2.24 times slower than 4-step standard. But against a slower 9-step standard baseline, Z still won 42 out of 54 HPS pairs. So my conclusion is: useful local gain, especially on color/count/two-object prompts, but not a universal or free fix.
