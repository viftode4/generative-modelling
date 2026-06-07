# Group Report Content Plan — What Can Be Written Now

This file maps the current repo evidence to a group scientific-paper-style report. It intentionally separates **safe to write now** from **needs teammate evidence**.

## Safe thesis right now

> We implemented the core inference-time Z-Sampling idea in a shared course repo and verified positive, controlled paired results in the committed Eugen and Vlad lanes. The reproduction is useful but bounded: it is smaller than the paper's full benchmark matrix, uses mixed local/Colab environments, and shows real runtime/metric trade-offs.

## Current evidence inventory

| Evidence | What it supports | Source |
|---|---|---|
| Course clarification | Group report + 7-minute graded presentation requirements | `knowledgebase/course/2026-06-04_presentation-report-clarification.md` |
| Midterm plan framing | Planned sampling methods, model/dataset split, metrics/ablations ambition | `knowledgebase/course/2026-06-07_midterm-plan-to-final-framing.md`, `planning/experiment-division.md` |
| Paper method | Z-Sampling mechanism, guidance gap, deterministic inversion caveat, paper settings | `knowledgebase/papers/zigzag-diffusion-sampling.md` |
| Shared implementation | Repo-local implementation, upstream provenance, sampler guardrails | `src/zigzag_project/`, `knowledgebase/decision-log.md` |
| Eugen result | SD2.1 + Pick-a-Pic HPS result, 300 pairs | `report/tables/eugen_results.md` |
| Vlad result | DreamShaper + GenEval HPS + detector result, 54 pairs | `report/tables/vlad_results.md`, `report/tables/vlad_geneval_detector.md` |
| Vlad compute control | Runtime cost and 9-step standard baseline | `report/tables/vlad_compute_matched.md` |
| Qualitative examples | Report figures for visual comparison | `report/figures/eugen_sd2.1_pickapic/`, `report/figures/vlad_geneval/` |

## Sections we can write now

### Abstract

Can write now, but keep it modest:

- paper/method in one sentence;
- our scoped implementation;
- two verified result lanes;
- bounded conclusion.

Avoid:

- claiming full reproduction;
- claiming all group lanes are done.

### 1. Introduction

Can write now:

- prompt alignment problem;
- count/color/spatial failures;
- inference-time appeal of Z-Sampling;
- course project scope.

Needs no extra teammate data.

### 2. Background / Method

Can write now:

- standard sampling;
- resampling at high level;
- Z-Sampling denoise → invert → re-denoise;
- `gamma_1`, `gamma_2`, guidance gap;
- deterministic DDIM inversion requirement.

Needs:

- if resampling final implementation differs, use only high-level wording unless Lenka/Gouri provide verified details.

### 3. Project Plan and Division

Can write now:

- midterm plan: Standard / Resampling / Z-Sampling;
- model/dataset/method split;
- TA guidance: implement inference-time behaviour, no retraining, reasonable subset;
- final reality: scoped verified evidence instead of full matrix.

Needs:

- teammate confirmation if we name exact final owners/results beyond committed files.

### 4. Implementation

Can write now:

- upstream code as reference submodule;
- repo-local implementation under `src/zigzag_project/`;
- reproducible scripts/manifests/tables;
- sampler integration cleanup and config guardrails;
- environment divergence and compatibility challenges.

Needs:

- short final check that branch is current before submission.

### 5. Results

Can write now with these exact claims:

- **Eugen / SD2.1 + Pick-a-Pic:** 300 pairs, Z wins 213/300 = 71.0%, mean HPS delta +0.0110, CI [+0.0087, +0.0132].
- **Vlad / DreamShaper + GenEval subset:** 54 pairs, detector pass rate 31.5% → 50.0%, HPS Z wins 53/54, mean HPS delta +0.0358.
- **Runtime control:** 4-step Z slower than 4-step standard, but 9-step standard is slower than Z and Z still wins 42/54 HPS pairs.

Should not write yet:

- Daniel/Radu/Bogdan/Gouri/Lenka quantitative claims unless committed tables/logs are added.

### 6. Discussion

Can write now:

- HPS vs detector mismatch;
- positive but uneven gains;
- runtime cost;
- environment differences;
- small benchmark scale;
- integration complexity.

Good framing:

- “controlled positive reproduction”;
- “not paper-scale reproduction”.

### 7. Conclusion

Can write now:

- core method implemented;
- verified positive evidence in two lanes;
- report should be honest about limits;
- future work: fill missing lanes, full detector scoring, ablations.

## Recommended final group-report structure

1. Abstract
2. Introduction and Motivation
3. Background and Method
   - Standard sampling
   - Resampling
   - Z-Sampling
4. Project Plan and Task Division
5. Implementation and Experimental Setup
6. Results
   - SD2.1 + Pick-a-Pic
   - DreamShaper + GenEval
   - Runtime/compute control
7. Discussion and Critical Appraisal
8. Conclusion
9. References / appendix as required

## Recommended presentation structure from the same content

For the 7-minute end-term deck:

1. Paper question + midterm plan.
2. Sampling methods.
3. What we implemented.
4. Verified results.
5. Challenges / plan vs actual.
6. Bounded conclusion.

## Missing inputs to ask teammates for

Only ask for these if they can provide committed evidence quickly:

- One table per additional lane: model, dataset, prompts/seeds, metric, result.
- One curated image pair per additional lane.
- Clear status for resampling: final implementation result or “implemented but not fully evaluated”.
- Any failed experiment worth mentioning as a challenge, with command/log/provenance.

## If no more teammate evidence arrives

The report can still be coherent:

- Treat Eugen + Vlad as the verified quantitative case studies.
- Treat the rest of the matrix as planned division / implementation scope / challenges.
- Do not pad with unsupported numbers.
