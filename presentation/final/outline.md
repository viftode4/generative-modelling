# End-term Presentation Outline

Purpose: show implementation/reproduction effort, results, interpretation, blockers, and what remains before final reports. The course requirement says the end-term presentation is for implementation feedback; complete final results from every lane are useful but not required.

## Recommended story

> We reproduced the core inference-time idea of Zigzag Diffusion Sampling in a shared `diffusers` codebase, split evaluation across model/dataset cells, and found that Z-Sampling gives positive local gains in the verified lanes, but the gains are metric- and prompt-type dependent and come with a real runtime cost.

## Slide plan

| # | Owner | Title | Key claim | Visual | Source |
|---:|---|---|---|---|---|
| 1 | TBD | Project recap | We study Z-Sampling as an inference-time self-reflection method for text-to-image diffusion. | Paper title + one-line method diagram | `knowledgebase/project-overview.md`; OpenReview `MKvQH1ekeY` |
| 2 | TBD | What Z-Sampling changes | Instead of only denoising, Z-Sampling denoises, partially inverts with low guidance, then re-denoises. | Standard loop vs zigzag loop | `knowledgebase/papers/zigzag-diffusion-sampling.md`; upstream commit `513eb51` |
| 3 | TBD | Group experiment matrix | Seven students covered different model/dataset/method cells rather than duplicating one run. | Matrix table | `planning/role-assignments.md`; `planning/experiment-division.md` |
| 4 | Vlad or implementation lead | Implementation progress | Shared repo-local code now supports standard, Z-Sampling, and resampling guardrails while keeping old imports compatible. | Code architecture bullets | `src/zigzag_project/zsampling.py`; `src/zigzag_project/samplers/`; tests |
| 5 | Eugen / SD2.1 owner | Verified result 1: SD2.1 + Pick-a-Pic | On 300 Pick-a-Pic pairs, Z-Sampling wins 71.0% by HPSv2.1, mean delta +0.0110. | Small table | `report/tables/eugen_results.md` |
| 6 | Vlad | Verified result 2: DreamShaper + GenEval | On 54 GenEval pairs, Z improves detector pass rate 31.5% to 50.0%; HPSv2 also favors Z. | Detector table + one image pair | `report/tables/vlad_results.md`; `report/figures/vlad_geneval/` |
| 7 | Vlad | Cost and control | Z is 2.24x slower than 4-step standard, but still beats a slower 9-step standard baseline on 42/54 HPS pairs. | Runtime/control table | `report/tables/vlad_compute_matched.md` |
| 8 | TBD / each owner | What remains | Other lanes need final result consolidation; metric caveats and environment divergence must be explicit. | Checklist | `knowledgebase/research-log.md`; per-owner notes |
| 9 | TBD | Feedback request | Ask whether the balance of implementation evidence, small-subset evaluation, and limitations is sufficient for final reports. | 3 questions | `knowledgebase/course/requirements-summary.md` |

## If time is very short: 6-slide version

1. Problem and paper idea.
2. Z-Sampling method diagram.
3. Group experiment matrix and implementation status.
4. Verified results snapshot: Eugen + Vlad.
5. Vlad deep dive: detector/HPS + compute control.
6. Limitations, remaining work, and feedback request.

## Do not overclaim

- Do **not** say the whole paper was reproduced at full scale.
- Do **not** present HPSv2 as the same thing as GenEval detector correctness.
- Do **not** hide runtime cost: Z-Sampling is slower.
- Do **not** imply all seven lanes are finished unless each owner has pushed verified results.
