# TA / Project Questions

Use this file to prepare discussion points for project feedback.
Print or read these out at the TA meeting.

## Group's questions (priority order)

1. **How many datasets should we implement the paper on?**
   The paper uses Pick-a-Pic, DrawBench, and GenEval. Running Z-Sampling across all three for all four models is a large compute commitment. Is one or two enough for the rubric?

2. **What is the best way to divide tasks among the group? How can we be non-redundant? What is a fair task assignment?**
   We are 7 people writing 7 individual reports. Each report needs a distinct angle. Our proposed split is in `planning/role-assignments.md` — happy to walk through it for feedback.

3. **Can we focus on one sampling method (Z-Sampling vs. Standard) and ignore the other baselines?**
   The paper compares against Semantic-aware CFG, Diffusion-DPO, AYS-Sampling, SEG, and CFG++. Each is a separate paper. Implementing all five is a major scope expansion. Is comparing only against standard sampling acceptable?

4. **What should we focus on in the final presentation? Implementation and experimentation only?**
   Or should we also cover paper theory, related work, critical appraisal?

5. **How important is the theoretical foundation of Z-Sampling for the end-term presentation? How familiar are the people grading us with the paper? Is the TA the only grader?**

## Additional scoping questions

6. **Multi-model breadth vs. single-model depth?**
   We can either run all 4 models (SD2.1, SDXL, Hunyuan-DiT, DreamShaper-xl-v2-turbo) superficially or 1–2 models with deeper analysis. Which scores better on the rubric?

7. **Reproducibility expectations under hardware variance?**
   Different GPUs produce slightly different outputs for the same seed. Do you expect us to reproduce the paper's exact numbers, or is directional/qualitative reproduction acceptable?

## Reproduction scope

- Which paper result is most important to reproduce for the course report?
- Is qualitative reproduction acceptable if compute is limited?
- Are smaller image sizes / fewer inference steps acceptable for experimentation, if clearly documented?

## Implementation scope

- Is the upstream codebase acceptable as a reference (read-only) while we reimplement from scratch?
- How much adaptation/extension is expected for a high content score?
- Should any components be implemented from scratch for learning value?

## Evaluation

- Which metrics are feasible for this course project? (Paper uses HPSv2, PickScore, ImageReward, AES.)
- Is human preference / visual comparison acceptable as a fallback if metrics fail to install?
- How should non-reproducibility due to compute or dependency constraints be discussed?

## Report and presentation

- What should be emphasized in the 3-page individual report?
- How should implementation effort and limitations be documented?
- Is it acceptable for all 7 reports to share an Introduction/Method section (paraphrased), as long as Results/Analysis are distinct?
