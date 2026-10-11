# GM lecture notes reader

This is a quiet notes-first study site for DSAIT4030 Generative Modelling. It now has one academic note page per course lecture, plus the friend-note notebook as a separate reference.

Built from local sources:

```text
notes/study/lecture_inventory.json
notes/study/extracted_text/*.txt
notes/study/08_example_exam.md
DSAIT_Notes/Q4/GM friend-note checkout
```

The default route opens directly into the lecture notes. There is no splash screen, landing page, or admin-style overview. The study tools rail is collapsed by default so the main reading pane gets the screen space; use **Show Tools** or **Hide Contents** in the top bar when you want the side panels.

## Local preview

```bash
/usr/bin/python3 -m http.server 4173 --bind 127.0.0.1 --directory notes/study/interactive
```

Open `http://127.0.0.1:4173/`.

## Main routes

- `#/` — lecture notes reader.
- `#/concepts` — all 74 structured concepts, grouped into exam-ready cards.
- `#/concepts/week/Week%206` — filter the concept index to one course week.
- `#/concepts/lecture/w6-diffusion` — filter the concept index to one lecture.
- `#/skill-tree` — prerequisite path with progress checks, lecture links, and unlock order.
- `#/exam` — 534 exam-style questions from the local example paper, exercises, and concept bank.
- `#/exam/example` — the cleaned 30-subpart 30 June digital-test example exam.
- `#/exam/generated` — generated full-course exam sections that reuse the example-exam style.
- `#/flashcards` — 432 flippable concept cards with explanations, citations, and known/review state on the back.
- `#/speedrun` — compact goal loop through the required exam scope, linked to practice and generated sections.
- `#/a4-master-sheet` — double-sided A4 printable master sheet: formulas on Side A, answer skeletons/traps on Side B.
- `#/coverage` — concept coverage and gap loop.
- `#/lectures/w1-intro` — Lecture 1: Introduction.
- `#/lectures/w1-mle-lvm` — Lecture 2: MLE and latent variable models.
- `#/lectures/w2-mle-lvm` — Lecture 3: repeated/reinforcement MLE and LVM deck.
- `#/lectures/w2-em-ppca` — Lecture 4: EM and PPCA.
- `#/lectures/w3-vae` — Lecture 5: nonlinear LVMs and VAEs.
- `#/lectures/w5-ebm` — Lecture 6: Energy-Based Models.
- `#/lectures/w5-flow` — Lecture 7: Normalising Flows.
- `#/lectures/w5-score` — Lecture 8: Score-based Models.
- `#/lectures/w6-diffusion` — Lecture 9: Diffusion Models.
- `#/lectures/w6-flow-matching` — Lecture 10: Continuous Flows and Flow Matching.
- `#/lectures/w6-evaluation` — Lecture 11: Evaluation of Generative Models.
- `#/lectures/w7-life` — Lecture 12: Generative Models for Life Sciences.
- `#/lectures/w7-causal` — Lecture 13: Causal Generative Modeling (optional/non-exam).
- `#/friend-notes` — 11 exam-relevant friend-note chapters, excluding causal.
- `#/review` — closed-book prompts per lecture.
- `#/images` — copied PNG figures from the friend notes.
- `#/sources` — local lecture, extracted text, slide, and friend-note paths.

## Source boundary

The deploy folder includes generated lecture-note text, the 11 Markdown-derived exam friend-note routes, the cleaned example exam, generated exam/flashcard data, and the 170 PNG images referenced by those friend notes. `10_Evaluation.md` and `11_GenerativeForScience.md` currently exist but are empty in the pulled friend-note checkout, so those two routes show labelled lecture-note fallback content instead of a blank page. It does **not** include lecture PDFs, PPTX, KEY, ZIP, package files, or lockfiles.

If this content should remain private, use it locally and do not deploy it publicly.

## Regenerate note data

From the repository root:

```bash
/usr/bin/python3 scripts/study/build_friend_notes_site.py
```

That script reads the lecture inventory, extracted lecture text, cleaned example exam, and friend notes; converts Markdown/Obsidian image embeds into static HTML data; renders LaTeX formulas to MathML; builds the flashcard/exam/speedrun banks; and copies only referenced PNG attachments. If the friend-note checkout is somewhere else, run it with `GM_FRIEND_NOTES_ROOT=/path/to/Q4/GM`.

If the MathML module is missing:

```bash
python3 -m pip install --user latex2mathml
```

## Gap loop

After changing lecture notes or concept blueprints, regenerate the data and run the concept audit:

```bash
/usr/bin/python3 scripts/study/build_friend_notes_site.py
/usr/bin/python3 scripts/study/validate_study_bank.py
/usr/bin/python3 scripts/study/audit_genmod_concept_gaps.py
/usr/bin/python3 scripts/study/validate_genmod_study_site.py
node scripts/study/smoke_interactive_routes.js
/usr/bin/python3 scripts/study/verify_a4_print_layout.py
```

The audit fails if any lecture has too few concept cards, missing required terms, missing formulas/rules, missing traps, missing exam use, or missing recall checks.
The study-bank validator also checks that exam practice, flashcards, generated exam sections, and the speedrun loop exclude the optional causal lecture from required coverage while keeping the cleaned example exam available as a separate practice route.
