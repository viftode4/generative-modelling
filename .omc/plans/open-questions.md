# Open Questions

## zigzag-ralplan - 2026-05-03

- [ ] Which diffusers version should we target for our reimplementation (latest stable vs. 0.22.3)? -- Affects scheduler API compatibility and parity testing strategy.
- [ ] Is HPSv2 installable and functional in our environment? -- Primary evaluation metric from the paper; need fallback plan if it fails.
- [ ] What is the university cluster's GPU queue policy and availability during exam period? -- Determines whether we can complete the 500-image experiment budget.
- [ ] Does the TA expect us to reproduce the paper's exact winning-rate numbers, or is a qualitative/directional reproduction sufficient? -- Scoping question that affects how much compute we allocate to the core matrix.
- [ ] Should the 7 individual reports share an identical Introduction/Method section (paraphrased) or must every word be independently written? -- Affects writing workload distribution.
- [ ] What is the exact submission format (LaTeX template, PDF upload, etc.)? -- The rubric says "double-column 3 pages" but does not specify a template.
- [ ] How should non-determinism across GPU architectures be disclosed in the report? -- Different GPUs yield slightly different outputs for the same seed; need a standard disclaimer.
- [ ] Who among the 7 students is P2 (Implementation Lead)? -- This is the only single-point-of-failure role; needs a designated backup (P1).
- [ ] Is Google Colab an acceptable compute platform for final results, or does the TA expect university-cluster provenance? -- Affects our GPU fallback strategy.
