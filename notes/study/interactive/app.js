"use strict";

const DATA = window.GM_NOTES;
const lectures = DATA.lectures || [];
const chapters = DATA.chapters || [];
const studyBank = DATA.studyBank || { examScope: {}, flashcards: [], examQuestions: [], speedrunGoals: [] };

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const stripTags = (html) => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
const escapeHtml = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

function railDatasetKey(side) {
  return side === "left" ? "leftCollapsed" : "rightCollapsed";
}

function railElement(side) {
  return side === "left" ? $("#leftRail") : $("#rightRail");
}

function railToggle(side) {
  return side === "left" ? $("#toggleLeft") : $("#toggleRight");
}

function storedRailState(side) {
  const saved = localStorage.getItem(`gm:rail:${side}:collapsed`);
  if (saved === "true" || saved === "false") return saved === "true";
  return document.body.dataset[railDatasetKey(side)] === "true";
}

function syncRailToggles() {
  ["left", "right"].forEach((side) => {
    const collapsed = document.body.dataset[railDatasetKey(side)] === "true";
    const toggle = railToggle(side);
    const rail = railElement(side);
    if (!toggle || !rail) return;
    toggle.setAttribute("aria-expanded", String(!collapsed));
    rail.setAttribute("aria-hidden", String(collapsed));
    if (side === "left") toggle.textContent = collapsed ? "Show Contents" : "Hide Contents";
    else toggle.textContent = collapsed ? "Show Tools" : "Hide Tools";
  });
}

function setRailState(side, collapsed, persist = true) {
  document.body.dataset[railDatasetKey(side)] = String(collapsed);
  if (persist) localStorage.setItem(`gm:rail:${side}:collapsed`, String(collapsed));
  syncRailToggles();
}

function initRailToggles() {
  setRailState("left", storedRailState("left"), false);
  setRailState("right", storedRailState("right"), false);
  $("#toggleLeft").addEventListener("click", () => setRailState("left", document.body.dataset.leftCollapsed !== "true"));
  $("#toggleRight").addEventListener("click", () => setRailState("right", document.body.dataset.rightCollapsed !== "true"));
}

function routeParts() {
  return location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
}

function activeLectureSlug() {
  const parts = routeParts();
  if (parts[0] === "lectures" && lectures.some((lecture) => lecture.slug === parts[1])) return parts[1];
  const saved = localStorage.getItem("gm:lastLecture");
  return lectures.some((lecture) => lecture.slug === saved) ? saved : lectures[0].slug;
}

function activeFriendSlug() {
  const parts = routeParts();
  if (parts[0] === "friend-notes" && chapters.some((chapter) => chapter.slug === parts[1])) return parts[1];
  const saved = localStorage.getItem("gm:lastFriendNote");
  return chapters.some((chapter) => chapter.slug === saved) ? saved : chapters[0].slug;
}

function activeSectionId() {
  const parts = routeParts();
  return parts[2] === "sections" ? decodeURIComponent(parts[3] || "") : "";
}

function scrollActiveSection() {
  const id = activeSectionId();
  if (!id) return;
  requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: "start" }));
}

function lectureBySlug(slug) {
  return lectures.find((lecture) => lecture.slug === slug) || lectures[0];
}

function friendBySlug(slug) {
  return chapters.find((chapter) => chapter.slug === slug) || chapters[0];
}

function renderRouteNav(page) {
  $$(".route-nav a, .quick-nav a").forEach((link) => {
    const href = link.getAttribute("href").replace("#/", "") || "lectures";
    const route = href.split("/")[0] || "lectures";
    const active = page === route || (page === "lectures" && route === "lectures");
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

function searchableLecture(lecture) {
  return [
    lecture.title,
    lecture.shortTitle,
    lecture.week,
    lecture.focus,
    lecture.sourceTitle,
    lecture.sourcePath,
    lecture.textPath,
    (lecture.concepts || []).map((concept) => [concept.title, concept.mustKnow, concept.coreIdea, concept.formulaOrRule, concept.commonTrap, concept.examUse, (concept.checks || []).join(" ")].join(" ")).join(" "),
    stripTags(lecture.html),
    lecture.sections.map((section) => section.title).join(" "),
  ].join(" ").toLowerCase();
}

function searchableFriendNote(chapter) {
  return [chapter.title, chapter.shortTitle, chapter.sourceFile, stripTags(chapter.html), chapter.sections.map((s) => s.title).join(" ")].join(" ").toLowerCase();
}

function allConcepts() {
  return lectures.flatMap((lecture) => (lecture.concepts || []).map((concept) => ({ ...concept, lecture })));
}

function weekName(lecture) {
  return String(lecture.week || "Week").split(" · ")[0] || "Week";
}

function courseWeekSpanCount() {
  const weekNumbers = lectures.map((lecture) => {
    const match = weekName(lecture).match(/\d+/);
    return match ? Number(match[0]) : 0;
  }).filter(Boolean);
  return weekNumbers.length ? Math.max(...weekNumbers) : 0;
}

function uniqueWeeks() {
  return [...new Set(lectures.map(weekName))];
}

function examScopeLectureSlugs() {
  const excluded = new Set((studyBank.examScope || {}).excludedLectureSlugs || []);
  return new Set(lectures.filter((lecture) => !excluded.has(lecture.slug)).map((lecture) => lecture.slug));
}

function conceptById(conceptId) {
  return allConcepts().find((concept) => concept.id === conceptId);
}

function lectureTitleFor(item) {
  if (item.lectureTitle) return item.lectureTitle;
  const lecture = lectures.find((entry) => entry.slug === item.lectureSlug);
  return lecture ? lecture.title : "Course notes";
}

function compactLectureTitle(title) {
  return String(title || "").replace(/^Lecture\s+\d+\s+[—-]\s+/, "");
}

function countBy(items, field) {
  return items.reduce((counts, item) => {
    const key = item[field] || "other";
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function activeConceptFilter() {
  const parts = routeParts();
  if (parts[1] === "week") return { kind: "week", value: decodeURIComponent(parts[2] || "") };
  if (parts[1] === "lecture") return { kind: "lecture", value: parts[2] || "" };
  if (parts[1] === "exam-scope") return { kind: "exam-scope", value: "exam-scope" };
  return { kind: "all", value: "" };
}

function conceptMatchesFilter(item, filter) {
  if (filter.kind === "week") return weekName(item.lecture) === filter.value;
  if (filter.kind === "lecture") return item.lecture.slug === filter.value;
  if (filter.kind === "exam-scope") return examScopeLectureSlugs().has(item.lecture.slug);
  return true;
}

function conceptFilterTitle(filter) {
  if (filter.kind === "week") return `${filter.value} concepts`;
  if (filter.kind === "lecture") return compactLectureTitle(lectureBySlug(filter.value).title);
  if (filter.kind === "exam-scope") return "Required exam-scope concepts";
  return `All ${DATA.conceptCoverage.totalConcepts} concepts`;
}

function conceptFilterSummary(filter, shownCount) {
  if (filter.kind === "all") {
    return `All ${DATA.conceptCoverage.totalConcepts} concepts · Filter by course week, lecture, or required exam scope · Lecture ${DATA.conceptCoverage.totalLectures} pages across ${courseWeekSpanCount()} course weeks`;
  }
  return `${shownCount} concepts shown`;
}

function conceptFilterAria(filter, kind, value) {
  const active = filter.kind === kind && String(filter.value) === String(value);
  return active ? ' aria-current="page"' : "";
}

function renderConceptFilters(filter) {
  const examCount = allConcepts().filter((item) => examScopeLectureSlugs().has(item.lecture.slug)).length;
  return `
    <section class="filter-panel" aria-label="Concept filters">
      <div class="filter-group">
        <h2>Filter by course week</h2>
        <div class="filter-chips">
          <a class="filter-chip" href="#/concepts"${conceptFilterAria(filter, "all", "")}>All ${DATA.conceptCoverage.totalConcepts}</a>
          <a class="filter-chip" href="#/concepts/exam-scope"${conceptFilterAria(filter, "exam-scope", "exam-scope")}>Exam scope ${examCount}</a>
          ${uniqueWeeks().map((week) => {
            const count = allConcepts().filter((item) => weekName(item.lecture) === week).length;
            return `<a class="filter-chip" href="#/concepts/week/${encodeURIComponent(week)}"${conceptFilterAria(filter, "week", week)}>${escapeHtml(week)} · ${count}</a>`;
          }).join("")}
        </div>
      </div>
      <details class="filter-group">
        <summary>Filter by lecture</summary>
        <div class="filter-chips filter-chips-lectures">
          ${lectures.map((lecture) => `
            <a class="filter-chip" href="#/concepts/lecture/${lecture.slug}"${conceptFilterAria(filter, "lecture", lecture.slug)}>
              ${escapeHtml(lecture.shortTitle)} · ${(lecture.concepts || []).length}
            </a>`).join("")}
        </div>
      </details>
    </section>`;
}

const SKILL_TREE_GROUPS = [
  {
    id: "density-foundations",
    title: "Density modelling foundations",
    badge: "Start here",
    lectureSlugs: ["w1-intro", "w1-mle-lvm", "w2-mle-lvm"],
    prerequisites: ["Probability notation", "log likelihood", "latent variable notation"],
    start: "Understand what a generative model estimates, why likelihood is the training objective, and why latent variables create intractable marginals.",
    checkpoint: "Can write pθ(x), pθ(x, z), the log-likelihood objective, and explain why exact marginal likelihood may be hard.",
    unlocks: ["EM and PPCA", "Variational autoencoders", "all later model families"],
  },
  {
    id: "em-ppca",
    title: "EM and probabilistic PCA",
    lectureSlugs: ["w2-em-ppca"],
    prerequisites: ["Density modelling foundations"],
    start: "Learn the E-step/M-step split as a lower-bound optimization pattern for latent variable models.",
    checkpoint: "Can explain responsibilities, expected complete-data log likelihood, and why PPCA is a linear-Gaussian latent variable model.",
    unlocks: ["Variational inference", "ELBO reasoning", "VAE training"],
  },
  {
    id: "vae",
    title: "Variational inference and VAEs",
    lectureSlugs: ["w3-vae"],
    prerequisites: ["EM lower-bound thinking", "encoder/decoder notation"],
    start: "Treat the ELBO as reconstruction plus regularization, then connect the reparameterization trick to gradient-based learning.",
    checkpoint: "Can derive the ELBO terms, name the approximate posterior, and state the common posterior-collapse trap.",
    unlocks: ["energy models", "flows", "score and diffusion objectives"],
  },
  {
    id: "energy-score",
    title: "Energy and score view",
    lectureSlugs: ["w5-ebm", "w5-score"],
    prerequisites: ["log densities", "normalization constants", "gradients"],
    start: "Separate unnormalized energy modelling from score matching: one learns energies, the other learns gradients of log density.",
    checkpoint: "Can state why the partition function is hard, what the score is, and how denoising score matching avoids direct density normalization.",
    unlocks: ["diffusion reverse dynamics", "classifier-free guidance"],
  },
  {
    id: "normalising-flows",
    title: "Normalising flows",
    lectureSlugs: ["w5-flow"],
    prerequisites: ["change of variables", "Jacobian determinant", "base distribution"],
    start: "Learn exact likelihood through invertible transformations before moving to continuous flows.",
    checkpoint: "Can write the change-of-variables formula and explain why invertibility plus tractable log determinant matters.",
    unlocks: ["continuous normalizing flows", "flow matching"],
  },
  {
    id: "diffusion-paths",
    title: "Diffusion and continuous paths",
    lectureSlugs: ["w6-diffusion", "w6-flow-matching"],
    prerequisites: ["score view", "noise schedules", "normalising flows"],
    start: "Connect forward noising, learned reverse denoising, ODE/SDE views, and flow matching as ways to move from noise to data.",
    checkpoint: "Can distinguish DDPM forward noising, reverse sampling, score parameterization, probability-flow ODEs, and flow-matching targets.",
    unlocks: ["guidance", "evaluation tradeoffs", "life-science generation"],
  },
  {
    id: "evaluation",
    title: "Evaluation and exam judgement",
    lectureSlugs: ["w6-evaluation"],
    prerequisites: ["all model families", "sample quality vs likelihood"],
    start: "Learn what each metric can and cannot prove so exam answers avoid overclaiming.",
    checkpoint: "Can compare FID, precision/recall, likelihood, memorization, and human evaluation limits.",
    unlocks: ["model selection", "scientific applications", "mixed exam questions"],
  },
  {
    id: "life-sciences",
    title: "Generative models for life sciences",
    lectureSlugs: ["w7-life"],
    prerequisites: ["diffusion", "guidance", "evaluation caveats"],
    start: "Apply the core model families to molecules/proteins and keep the biological validity constraints explicit.",
    checkpoint: "Can explain conditioning, constraints, validity, diversity, and why evaluation is domain-specific.",
    unlocks: ["final mixed review"],
  },
  {
    id: "causal-optional",
    title: "Optional causal context",
    optional: true,
    lectureSlugs: ["w7-causal"],
    prerequisites: ["graphical-model language"],
    start: "Keep this separate: it is useful context but outside the required exam loop for this build.",
    checkpoint: "Can recognize intervention/counterfactual vocabulary without spending exam-prep time here.",
    unlocks: ["optional context only"],
  },
];

function conceptLabelList(conceptIds = []) {
  const concepts = conceptIds.map(conceptById).filter(Boolean);
  if (!concepts.length) return "";
  return `
    <ul class="concept-link-list">
      ${concepts.map((concept) => `<li><a href="#/lectures/${concept.lecture.slug}">${escapeHtml(concept.lecture.shortTitle)}</a> · ${escapeHtml(concept.title)}</li>`).join("")}
    </ul>`;
}

function citationList(citations = []) {
  if (!citations.length) return "";
  return `
    <ul class="citation-list">
      ${citations.map((citation) => {
        const href = citation.href || "#/";
        const label = citation.label || href;
        const source = citation.sourcePath ? `<small>${escapeHtml(citation.sourcePath)}</small>` : "";
        return `<li><a href="${escapeHtml(href)}">${escapeHtml(label)}</a>${source}</li>`;
      }).join("")}
    </ul>`;
}

function studySearchText(item) {
  const conceptText = (item.conceptIds || []).map((conceptId) => {
    const concept = conceptById(conceptId);
    return concept ? `${concept.title} ${concept.mustKnow} ${concept.coreIdea} ${concept.formulaOrRule} ${concept.examUse}` : "";
  }).join(" ");
  return [
    item.id,
    item.type,
    item.difficulty,
    item.lectureTitle,
    item.front,
    item.prompt,
    item.answer,
    item.explanation,
    conceptText,
  ].join(" ").toLowerCase();
}

function filteredStudyItems(items) {
  const query = $("#noteSearch").value.trim().toLowerCase();
  return items.filter((item) => !query || studySearchText(item).includes(query));
}

function renderStudyStats(items, label) {
  const byType = countBy(items, "type");
  const typeSummary = Object.entries(byType).map(([type, count]) => `${escapeHtml(type)}: ${count}`).join(" · ");
  return `
    <dl class="source-meta study-stats">
      <div><dt>Total</dt><dd>${items.length} ${escapeHtml(label)}</dd></div>
      <div><dt>Scope</dt><dd>${escapeHtml((studyBank.examScope || {}).excludedReason || "Course exam scope")}</dd></div>
      <div><dt>Mix</dt><dd>${typeSummary || "concept recall"}</dd></div>
    </dl>`;
}

function richOrEscaped(value, htmlValue = "") {
  return htmlValue || `<p>${escapeHtml(value || "")}</p>`;
}

function draftKey(id) {
  return `gm:draft:${id}`;
}

function bindAnswerDrafts() {
  $$("[data-answer-draft]").forEach((field) => {
    const key = draftKey(field.getAttribute("data-answer-draft"));
    field.value = localStorage.getItem(key) || "";
    field.addEventListener("input", () => localStorage.setItem(key, field.value));
  });
}

function renderExamQuestion(question, index) {
  const formula = question.formulaHtml ? `<div class="answer-formula">${question.formulaHtml}</div>` : "";
  const prompt = question.promptHtml ? `<div class="question-prompt-html">${question.promptHtml}</div>` : `<h2>${index + 1}. ${escapeHtml(question.prompt)}</h2>`;
  const answer = question.answerHtml || `<p>${escapeHtml(question.answer)}</p>`;
  return `
    <article class="study-card exam-question-card">
      <p class="study-kicker">${escapeHtml(question.type || "question")} · ${escapeHtml(question.difficulty || "practice")} · ${escapeHtml(compactLectureTitle(lectureTitleFor(question)))}</p>
      ${question.promptHtml ? `<h2>${index + 1}. ${escapeHtml(question.type || "Question")}</h2>` : ""}
      ${prompt}
      ${conceptLabelList(question.conceptIds)}
      <label class="draft-label" for="draft-${escapeHtml(question.id)}">Write your answer first</label>
      <textarea id="draft-${escapeHtml(question.id)}" class="answer-draft" data-answer-draft="${escapeHtml(question.id)}" rows="4" placeholder="Closed-book answer before opening the solution…"></textarea>
      <details>
        <summary>Show answer, explanation, and citations</summary>
        <div class="answer-panel">
          <h3>Answer</h3>
          ${answer}
          ${formula}
          <h3>Why this is enough</h3>
          <p>${escapeHtml(question.explanation)}</p>
          <h3>Where to review</h3>
          ${citationList(question.citations)}
        </div>
      </details>
    </article>`;
}

function renderFlashcard(card, index) {
  const formula = card.formulaHtml ? `<div class="answer-formula">${card.formulaHtml}</div>` : "";
  const answer = card.answerHtml || `<p>${escapeHtml(card.answer)}</p>`;
  return `
    <article class="study-card flashcard-card" data-card-id="${escapeHtml(card.id)}">
      <div class="flashcard-stage" id="flashcard-panel-${escapeHtml(card.id)}">
        <section class="flashcard-face flashcard-front">
          <p class="study-kicker">${escapeHtml(card.type || card.cardType || "card")} · ${escapeHtml(compactLectureTitle(lectureTitleFor(card)))}</p>
          <h2>${index + 1}. ${escapeHtml(card.front)}</h2>
          ${conceptLabelList(card.conceptIds)}
          <p class="flashcard-hint">Say it out loud, then Flip card.</p>
        </section>
        <section class="flashcard-face flashcard-back" aria-hidden="true">
          <p class="study-kicker">Back of card</p>
          ${answer}
          ${formula}
          <h3>Explanation</h3>
          <p>${escapeHtml(card.explanation)}</p>
          <h3>Review source</h3>
          ${citationList(card.citations)}
        </section>
      </div>
      <div class="card-actions">
        <button type="button" class="small-action" data-flashcard-flip="${escapeHtml(card.id)}" aria-controls="flashcard-panel-${escapeHtml(card.id)}" aria-expanded="false">Flip card</button>
        <button type="button" class="small-action" data-flashcard-grade="${escapeHtml(card.id)}" data-grade="review">Review again</button>
        <button type="button" class="small-action is-positive" data-flashcard-grade="${escapeHtml(card.id)}" data-grade="known">I knew it</button>
        <span class="card-status" data-flashcard-status="${escapeHtml(card.id)}">new</span>
      </div>
    </article>`;
}

function flashcardKey(cardId) {
  return `gm:flashcard:${cardId}`;
}

function setFlashcardStatus(cardId, grade) {
  if (grade) localStorage.setItem(flashcardKey(cardId), grade);
  const status = $(`[data-flashcard-status="${cardId}"]`);
  if (status) status.textContent = grade === "known" ? "known" : grade === "review" ? "review again" : "new";
}

function bindFlashcards() {
  $$("[data-flashcard-flip]").forEach((button) => {
    const cardId = button.getAttribute("data-flashcard-flip");
    const card = button.closest ? button.closest("[data-card-id]") : null;
    button.addEventListener("click", () => {
      if (!card) return;
      const flipped = card.classList.toggle("is-flipped");
      button.setAttribute("aria-expanded", String(flipped));
      button.textContent = flipped ? "Show front" : "Flip card";
      const back = $(".flashcard-back", card);
      if (back) back.setAttribute("aria-hidden", String(!flipped));
    });
  });
  $$("[data-flashcard-grade]").forEach((button) => {
    const cardId = button.getAttribute("data-flashcard-grade");
    button.addEventListener("click", () => setFlashcardStatus(cardId, button.getAttribute("data-grade")));
  });
  $$("[data-card-id]").forEach((card) => {
    const cardId = card.getAttribute("data-card-id");
    setFlashcardStatus(cardId, localStorage.getItem(flashcardKey(cardId)) || "");
  });
}

function speedrunConceptRows(conceptIds = []) {
  const concepts = conceptIds.map(conceptById).filter(Boolean);
  return concepts.map((concept) => {
    const cardCount = (studyBank.flashcards || []).filter((card) => (card.conceptIds || []).includes(concept.id)).length;
    const questionCount = (studyBank.examQuestions || []).filter((question) => (question.conceptIds || []).includes(concept.id)).length;
    return `
      <li>
        <a href="#/lectures/${concept.lecture.slug}">${escapeHtml(concept.title)}</a>
        <small>${escapeHtml(concept.lecture.shortTitle)} · ${cardCount} cards · ${questionCount} questions</small>
      </li>`;
  }).join("");
}

function bindSpeedrunChecks() {
  $$("[data-speedrun-check]").forEach((input) => {
    const key = `gm:speedrun:${input.getAttribute("data-speedrun-check")}`;
    input.checked = localStorage.getItem(key) === "done";
    input.addEventListener("change", () => {
      if (input.checked) localStorage.setItem(key, "done");
      else localStorage.removeItem(key);
      const done = $$("[data-speedrun-check]").filter((item) => item.checked).length;
      const progress = $("#speedrunProgress");
      if (progress) progress.textContent = `${done}/${(studyBank.speedrunGoals || []).length} goals complete`;
    });
  });
}

function renderSideNav(mode, activeSlug) {
  const query = $("#noteSearch").value.trim().toLowerCase();
  const nav = $("#lectureNav");
  const title = $("#navTitle");
  if (mode === "friend") {
    title.textContent = "Friend notes";
    const visible = chapters.filter((chapter) => !query || searchableFriendNote(chapter).includes(query));
    nav.setAttribute("aria-label", "Friend-note chapters");
    nav.innerHTML = visible.map((chapter, index) => `
      <a href="#/friend-notes/${chapter.slug}" ${chapter.slug === activeSlug ? 'aria-current="page"' : ""}>
        ${index + 1}. ${escapeHtml(chapter.shortTitle)}
        <small>${chapter.formulaCount} formulas · ${chapter.images.length} images</small>
      </a>`).join("") || `<p class="empty">No matching friend-note chapter.</p>`;
    return;
  }

  title.textContent = "Lectures";
  const visible = lectures.filter((lecture) => !query || searchableLecture(lecture).includes(query));
  nav.setAttribute("aria-label", "Course lecture notes");
  nav.innerHTML = visible.map((lecture) => `
    <a href="#/lectures/${lecture.slug}" ${lecture.slug === activeSlug ? 'aria-current="page"' : ""}>
      ${lecture.order}. ${escapeHtml(lecture.shortTitle)}
      <small>${escapeHtml(lecture.week)} · ${(lecture.concepts || []).length} concepts · ${lecture.formulaCount} formulas</small>
    </a>`).join("") || `<p class="empty">No matching lecture.</p>`;
}

function relatedFriendLink(lecture) {
  if (!lecture.relatedFriendChapter) return "";
  const chapter = chapters.find((item) => item.slug === lecture.relatedFriendChapter);
  if (!chapter) return "";
  return `<p><a class="quiet-link" href="#/friend-notes/${chapter.slug}">Open matching friend-note chapter: ${escapeHtml(chapter.shortTitle)}</a></p>`;
}

function renderLecture(slug) {
  const lecture = lectureBySlug(slug);
  localStorage.setItem("gm:lastLecture", lecture.slug);
  renderSideNav("lecture", lecture.slug);
  $("#readerStatus").textContent = `${lecture.week} · ${lecture.shortTitle}`;
  $("#readerArticle").innerHTML = `
    <article class="note-article">
      <p class="note-kicker">${escapeHtml(lecture.week)} · Course lecture</p>
      <h1 class="note-title">${escapeHtml(lecture.title)}</h1>
      <p class="note-meta">${escapeHtml(lecture.focus)}</p>
      <dl class="source-meta">
        <div><dt>Slide source</dt><dd>${escapeHtml(lecture.sourceTitle)}</dd></div>
        <div><dt>Pages</dt><dd>${escapeHtml(lecture.pages ?? "n/a")}</dd></div>
        <div><dt>Concepts</dt><dd>${escapeHtml((lecture.concepts || []).length)}</dd></div>
      </dl>
      ${lecture.html}
    </article>`;
  renderLectureTools(lecture);
}

function renderLectureTools(lecture) {
  const query = $("#noteSearch").value.trim();
  const hits = query ? searchHits(query) : [];
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>How to use this lecture</h2>
      <p>Read one section. Then close the left nav mentally and answer the self-check before continuing.</p>
      <p><strong>${(lecture.concepts || []).length}</strong> concepts · <strong>${lecture.formulaCount}</strong> formulas · <strong>${lecture.sections.length}</strong> sections</p>
      <p><a class="quiet-link" href="#/concepts">Open the all-concepts index</a></p>
      ${relatedFriendLink(lecture)}
      <details open>
        <summary>Self-check</summary>
        <ol>${lecture.questions.map((q) => `<li>${escapeHtml(q)}</li>`).join("")}</ol>
      </details>
    </section>
    <section class="tool-panel">
      <h2>Source boundary</h2>
      <p>Lecture note page generated from <span class="path-inline">${escapeHtml(lecture.textPath)}</span>. The PDF is referenced, not copied into this web folder.</p>
    </section>
    ${query ? `<section class="tool-panel"><h2>Search results</h2>${hits.length ? hits.map(renderHit).join("") : `<p class="empty">No section title matches. Browser find can still search inside the article.</p>`}</section>` : ""}`;
  renderSectionNav(lecture, `lectures/${lecture.slug}`);
}

function renderFriendNote(slug) {
  const chapter = friendBySlug(slug);
  localStorage.setItem("gm:lastFriendNote", chapter.slug);
  renderSideNav("friend", chapter.slug);
  $("#readerStatus").textContent = `Friend notes · ${chapter.shortTitle}`;
  $("#readerArticle").innerHTML = `
    <article class="note-article">
      <p class="note-kicker">Friend notes · ${escapeHtml(chapter.sourceFile)}</p>
      <h1 class="note-title">${escapeHtml(chapter.title)}</h1>
      <p class="note-meta">Original Markdown-derived notes, with images inline and formulas rendered as MathML.</p>
      ${chapter.html}
    </article>`;
  renderFriendTools(chapter);
}

function renderFriendTools(chapter) {
  const query = $("#noteSearch").value.trim();
  const hits = query ? searchHits(query) : [];
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>Use this chapter</h2>
      <p>Use this as the expanded notebook after the cleaner lecture page.</p>
      <p><strong>${chapter.formulaCount}</strong> formula blocks · <strong>${chapter.images.length}</strong> images</p>
      <details>
        <summary>Self-check</summary>
        <ol>${chapter.questions.map((q) => `<li>${escapeHtml(q)}</li>`).join("")}</ol>
      </details>
    </section>
    ${query ? `<section class="tool-panel"><h2>Search results</h2>${hits.length ? hits.map(renderHit).join("") : `<p class="empty">No section title matches. Browser find can still search inside the article.</p>`}</section>` : ""}`;
  renderSectionNav(chapter, `friend-notes/${chapter.slug}`);
}

function renderSectionNav(item, baseRoute) {
  const active = activeSectionId();
  $("#sectionNav").innerHTML = `<h2>Sections</h2>${item.sections.filter((section) => section.level <= 3).map((section) => {
    const href = `#/${baseRoute}/sections/${encodeURIComponent(section.id)}`;
    const current = active === section.id ? ` aria-current="true"` : "";
    return `<a href="${href}"${current}>${escapeHtml(section.title)}</a>`;
  }).join("")}`;
}

function renderHit(hit) {
  return `<a class="search-hit" href="${hit.href}">${escapeHtml(hit.label)} · ${escapeHtml(hit.section)}</a>`;
}

function searchHits(query) {
  const q = query.toLowerCase();
  const hits = [];
  lectures.forEach((lecture) => {
    lecture.sections.forEach((section) => {
      if (section.title.toLowerCase().includes(q)) hits.push({ href: `#/lectures/${lecture.slug}/sections/${section.id}`, label: lecture.shortTitle, section: section.title });
    });
    if (hits.filter((hit) => hit.href.includes(`/lectures/${lecture.slug}`)).length === 0 && searchableLecture(lecture).includes(q)) {
      hits.push({ href: `#/lectures/${lecture.slug}`, label: lecture.shortTitle, section: "lecture text contains match" });
    }
  });
  chapters.forEach((chapter) => {
    chapter.sections.forEach((section) => {
      if (section.title.toLowerCase().includes(q)) hits.push({ href: `#/friend-notes/${chapter.slug}/sections/${section.id}`, label: `Friend: ${chapter.shortTitle}`, section: section.title });
    });
    if (hits.filter((hit) => hit.href.includes(`/friend-notes/${chapter.slug}`)).length === 0 && searchableFriendNote(chapter).includes(q)) {
      hits.push({ href: `#/friend-notes/${chapter.slug}`, label: `Friend: ${chapter.shortTitle}`, section: "chapter text contains match" });
    }
  });
  return hits.slice(0, 14);
}

function renderConceptCard(item, includeLecture = true) {
  const concept = item.concept || item;
  const lecture = item.lecture;
  return `
    <article id="${escapeHtml(concept.id)}" class="concept-card">
      ${includeLecture && lecture ? `<p class="concept-kicker">${escapeHtml(lecture.week)} · <a href="#/lectures/${lecture.slug}">${escapeHtml(lecture.shortTitle)}</a></p>` : ""}
      <h2>${escapeHtml(concept.title)}</h2>
      <dl>
        <div><dt>Must know</dt><dd>${escapeHtml(concept.mustKnow)}</dd></div>
        <div><dt>Core idea</dt><dd>${escapeHtml(concept.coreIdea)}</dd></div>
        <div><dt>Formula / rule</dt><dd>${escapeHtml(concept.formulaOrRule)}${concept.formulaHtml ? `<div class="mini-formula">${concept.formulaHtml}</div>` : ""}</dd></div>
        <div><dt>Common trap</dt><dd>${escapeHtml(concept.commonTrap)}</dd></div>
        <div><dt>Exam use</dt><dd>${escapeHtml(concept.examUse)}</dd></div>
      </dl>
      <details>
        <summary>Recall checks</summary>
        <ol>${(concept.checks || []).map((check) => `<li>${escapeHtml(check)}</li>`).join("")}</ol>
      </details>
    </article>`;
}

function renderConcepts() {
  renderSideNav("lecture", "");
  const filter = activeConceptFilter();
  const query = $("#noteSearch").value.trim().toLowerCase();
  const items = allConcepts().filter((item) => conceptMatchesFilter(item, filter)).filter((item) => !query || [
    item.lecture.title,
    item.lecture.shortTitle,
    item.lecture.week,
    item.title,
    item.mustKnow,
    item.coreIdea,
    item.formulaOrRule,
    item.commonTrap,
    item.examUse,
    (item.checks || []).join(" "),
  ].join(" ").toLowerCase().includes(query));
  $("#readerStatus").textContent = `${items.length} concepts`;
  $("#readerArticle").innerHTML = `
    <section class="plain-page">
      <p class="note-kicker">Concept index</p>
      <h1 class="note-title">${escapeHtml(conceptFilterTitle(filter))}</h1>
      <p class="note-meta">${escapeHtml(conceptFilterSummary(filter, items.length))}. Each card has the minimum exam-ready explanation: what to know, the core idea, formula or rule, common trap, exam use, and recall checks.</p>
      ${renderConceptFilters(filter)}
      <div class="concept-grid">
        ${items.map((item) => renderConceptCard(item)).join("") || `<p class="empty">No concept matches the search.</p>`}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>How to study concepts</h2>
      <p>Pick one concept. Explain the core idea and formula. Then answer the recall checks without looking.</p>
      <p><strong>${DATA.conceptCoverage.totalConcepts}</strong> total concepts · <strong>${DATA.conceptCoverage.totalLectures}</strong> lectures</p>
      <p><a class="quiet-link" href="#/skill-tree">Open the prerequisite skill tree</a></p>
      <p><a class="quiet-link" href="#/exam">Practice exam questions</a> · <a class="quiet-link" href="#/flashcards">Drill flashcards</a></p>
    </section>
    <section class="tool-panel">
      <h2>Week labels</h2>
      <p>The source material provides ${DATA.conceptCoverage.totalLectures} lecture pages and labels them across course weeks 1–${courseWeekSpanCount()}; missing calendar weeks are not missing lecture pages.</p>
    </section>`;
  $("#sectionNav").innerHTML = "";
}

function skillTreeKey(groupId) {
  return `gm:skill-tree:${groupId}`;
}

function skillGroupConcepts(group) {
  const allowedSlugs = new Set(group.lectureSlugs || []);
  return allConcepts().filter((concept) => allowedSlugs.has(concept.lecture.slug));
}

function skillGroupLectures(group) {
  const allowedSlugs = new Set(group.lectureSlugs || []);
  return lectures.filter((lecture) => allowedSlugs.has(lecture.slug));
}

function skillGroupProgress(group) {
  return localStorage.getItem(skillTreeKey(group.id)) === "done" ? "complete" : "open";
}

function skillTreeOverallProgress() {
  const required = SKILL_TREE_GROUPS.filter((group) => !group.optional);
  const complete = required.filter((group) => skillGroupProgress(group) === "complete").length;
  return `${complete}/${required.length} required nodes complete`;
}

function renderSkillTreeMap() {
  return `
    <ol class="skill-map" aria-label="Learning dependency path">
      ${SKILL_TREE_GROUPS.map((group) => `
        <li>
          <a href="#skill-${escapeHtml(group.id)}">${escapeHtml(group.title)}</a>
          ${group.optional ? `<small>optional</small>` : ""}
        </li>`).join("")}
    </ol>`;
}

function renderSkillTreeCard(group, index) {
  const concepts = skillGroupConcepts(group);
  const groupLectures = skillGroupLectures(group);
  const progress = skillGroupProgress(group);
  return `
    <article id="skill-${escapeHtml(group.id)}" class="skill-tree-card ${group.optional ? "is-optional" : ""}">
      <div class="skill-card-head">
        <div>
          <p class="study-kicker">${group.optional ? "Optional" : `Step ${index + 1}`} ${group.badge ? `· ${escapeHtml(group.badge)}` : ""}</p>
          <h2>${escapeHtml(group.title)}</h2>
        </div>
        <label class="skill-check">
          <input type="checkbox" data-skill-check="${escapeHtml(group.id)}" />
          <span data-skill-progress="${escapeHtml(group.id)}">Progress: ${escapeHtml(progress)}</span>
        </label>
      </div>
      <p>${escapeHtml(group.start)}</p>
      <dl class="skill-links">
        <div><dt>Understand first</dt><dd>${group.prerequisites.map(escapeHtml).join(" · ")}</dd></div>
        <div><dt>Checkpoint</dt><dd>${escapeHtml(group.checkpoint)}</dd></div>
        <div><dt>Unlocks</dt><dd>${group.unlocks.map(escapeHtml).join(" · ")}</dd></div>
      </dl>
      <nav class="goal-actions" aria-label="${escapeHtml(group.title)} study links">
        ${groupLectures.map((lecture) => `<a href="#/lectures/${lecture.slug}">${escapeHtml(lecture.shortTitle)}</a>`).join("")}
        ${groupLectures.map((lecture) => `<a href="#/concepts/lecture/${lecture.slug}">${escapeHtml(lecture.shortTitle)} concepts</a>`).join("")}
      </nav>
      <details>
        <summary>${concepts.length} concepts in this node</summary>
        <ol class="speedrun-concepts">
          ${concepts.map((concept) => {
            const questionCount = (studyBank.examQuestions || []).filter((question) => (question.conceptIds || []).includes(concept.id)).length;
            return `
              <li>
                <a href="#/lectures/${concept.lecture.slug}">${escapeHtml(concept.title)}</a>
                <small>${escapeHtml(concept.lecture.shortTitle)} · ${questionCount} questions</small>
              </li>`;
          }).join("")}
        </ol>
      </details>
    </article>`;
}

function bindSkillTreeChecks() {
  $$("[data-skill-check]").forEach((input) => {
    const groupId = input.getAttribute("data-skill-check");
    const key = skillTreeKey(groupId);
    input.checked = localStorage.getItem(key) === "done";
    input.addEventListener("change", () => {
      if (input.checked) localStorage.setItem(key, "done");
      else localStorage.removeItem(key);
      const status = $(`[data-skill-progress="${groupId}"]`);
      if (status) status.textContent = `Progress: ${input.checked ? "complete" : "open"}`;
      const overall = $("#skillTreeProgress");
      if (overall) overall.textContent = skillTreeOverallProgress();
    });
  });
}

function renderSkillTree() {
  renderSideNav("lecture", "");
  $("#readerStatus").textContent = `Skill tree · ${skillTreeOverallProgress()}`;
  $("#readerArticle").innerHTML = `
    <section class="plain-page study-page">
      <p class="note-kicker">Prerequisite path</p>
      <h1 class="note-title">Skill tree</h1>
      <p class="note-meta">Use this as the order-of-attack map: learn each prerequisite node, prove the checkpoint, then follow the Unlocks line into the next model family.</p>
      <dl class="source-meta">
        <div><dt>Progress</dt><dd id="skillTreeProgress">${escapeHtml(skillTreeOverallProgress())}</dd></div>
        <div><dt>Scope</dt><dd>${escapeHtml((studyBank.examScope || {}).excludedReason || "Causal material is optional.")}</dd></div>
        <div><dt>Resources</dt><dd><a href="#/flashcards">Flashcards</a> · <a href="#/exam">Exam practice</a></dd></div>
      </dl>
      ${renderSkillTreeMap()}
      <div class="skill-tree-grid">
        ${SKILL_TREE_GROUPS.map((group, index) => renderSkillTreeCard(group, index)).join("")}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>How to use the tree</h2>
      <ol>
        <li>Start at the first open node.</li>
        <li>Open its lecture links and read only the listed concepts.</li>
        <li>Do its flashcards and one exam question before checking progress.</li>
      </ol>
    </section>
    <section class="tool-panel">
      <h2>Shortcut</h2>
      <p>If an exam question feels impossible, jump back to the nearest prerequisite node instead of rereading everything.</p>
    </section>`;
  $("#sectionNav").innerHTML = "";
  bindSkillTreeChecks();
}

function renderCoverage() {
  renderSideNav("lecture", "");
  const total = DATA.conceptCoverage.totalConcepts;
  const examQuestions = studyBank.examQuestions || [];
  const flashcards = studyBank.flashcards || [];
  $("#readerStatus").textContent = `Coverage · ${total} concepts`;
  $("#readerArticle").innerHTML = `
    <section class="plain-page">
      <p class="note-kicker">Coverage and repair</p>
      <h1 class="note-title">Coverage loop</h1>
      <p class="note-meta">Use this page to choose what to repair next: every lecture card shows concepts, formulas, flashcards, generated questions, and direct drill links.</p>
      <p class="notice">Audit status: <strong>${escapeHtml(DATA.conceptCoverage.status)}</strong>. Script: <code class="path">${escapeHtml(DATA.conceptCoverage.auditScript)}</code>.</p>
      <div class="coverage-grid">
        ${lectures.map((lecture) => {
          const conceptIds = new Set((lecture.concepts || []).map((concept) => concept.id));
          const questionCount = examQuestions.filter((question) => (question.conceptIds || []).some((id) => conceptIds.has(id))).length;
          const cardCount = flashcards.filter((card) => (card.conceptIds || []).some((id) => conceptIds.has(id))).length;
          return `
          <article class="coverage-card">
            <h2>${lecture.order}. ${escapeHtml(lecture.shortTitle)}</h2>
            <p>${escapeHtml(lecture.week)} · <strong>${(lecture.concepts || []).length}</strong> concepts · <strong>${lecture.formulaCount}</strong> formulas · <strong>${cardCount}</strong> cards · <strong>${questionCount}</strong> questions</p>
            <p>${escapeHtml(lecture.focus)}</p>
            <nav class="goal-actions" aria-label="${escapeHtml(lecture.shortTitle)} repair links">
              <a href="#/lectures/${lecture.slug}">Open lecture</a>
              <a href="#/concepts/lecture/${lecture.slug}">Concepts</a>
              <a href="#/flashcards">Cards</a>
              <a href="#/exam/generated">Generated exam</a>
            </nav>
            <details>
              <summary>Source highlights</summary>
              <ul>${(lecture.sourceHighlights || []).map((line) => `<li>${escapeHtml(line)}</li>`).join("")}</ul>
            </details>
          </article>`;
        }).join("")}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>Loop rule</h2>
      <p>Run the concept audit after any edit. If it fails, fill the missing lecture/concept fields and regenerate notes-data.</p>
      <p><code class="path">python3 scripts/study/audit_genmod_concept_gaps.py</code></p>
    </section>`;
  $("#sectionNav").innerHTML = "";
}

function renderExamPart(part, sectionTitle, index, prefix) {
  const draftId = `${prefix}-${part.partId}`;
  return `
    <article class="study-card exam-question-card exam-part-card">
      <p class="study-kicker">${escapeHtml(part.partId)} · ${escapeHtml(part.points || "1.0p")} · ${escapeHtml(part.type || "question")} · ${escapeHtml(sectionTitle)}</p>
      <div class="question-prompt-html">${part.promptHtml || `<p>${escapeHtml(part.prompt || "")}</p>`}</div>
      ${conceptLabelList(part.conceptIds)}
      <label class="draft-label" for="draft-${escapeHtml(draftId)}">Your closed-book answer</label>
      <textarea id="draft-${escapeHtml(draftId)}" class="answer-draft" data-answer-draft="${escapeHtml(draftId)}" rows="5" placeholder="Write before opening the worked target…"></textarea>
      <details>
        <summary>Show worked target and repair links</summary>
        <div class="answer-panel">
          <h3>Target answer</h3>
          ${part.answerHtml || `<p>${escapeHtml(part.answer || "")}</p>`}
          <h3>Why this is enough</h3>
          <p>${escapeHtml(part.explanation || "Use this to compare your answer against the expected structure.")}</p>
          <h3>Repair links</h3>
          ${citationList(part.citations)}
        </div>
      </details>
    </article>`;
}

function renderExampleExam() {
  renderSideNav("lecture", "");
  const example = studyBank.exampleExam || { sections: [] };
  const sections = example.sections || [];
  const partCount = sections.reduce((total, section) => total + (section.parts || []).length, 0);
  $("#readerStatus").textContent = `${partCount} example subparts`;
  $("#readerArticle").innerHTML = `
    <section class="plain-page study-page">
      <p class="note-kicker">Practice exam source</p>
      <h1 class="note-title">Example exam: 30 June digital test</h1>
      <p class="note-meta">This page keeps the real chained style: one topic, then subparts for definition, proof, interpretation, practical limits, and multiple choice. Personal identifiers are not included.</p>
      <dl class="source-meta">
        <div><dt>Structure</dt><dd>${partCount} subparts</dd></div>
        <div><dt>Sections</dt><dd>${sections.map((section) => escapeHtml(section.title)).join(" · ")}</dd></div>
        <div><dt>Use</dt><dd>Write first, then open target answer</dd></div>
      </dl>
      <div class="exam-section-list">
        ${sections.map((section) => `
          <section class="exam-section-block">
            <h2>${escapeHtml(section.sectionId)}. ${escapeHtml(section.title)}</h2>
            ${(section.parts || []).map((part, index) => renderExamPart(part, section.title, index, `example-${section.sectionId}`)).join("")}
          </section>`).join("")}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>How to use it</h2>
      <ol>
        <li>Answer one whole section without opening targets.</li>
        <li>Compare against the worked target.</li>
        <li>Open only the repair links for missed concepts.</li>
      </ol>
    </section>
    <section class="tool-panel">
      <h2>Next drill</h2>
      <p><a class="quiet-link" href="#/exam/generated">Open generated full-course exam bank</a></p>
      <p><a class="quiet-link" href="#/flashcards">Drill matching flashcards</a></p>
    </section>`;
  $("#sectionNav").innerHTML = "";
  bindAnswerDrafts();
}

function renderGeneratedExams() {
  renderSideNav("lecture", "");
  const sets = studyBank.generatedExamSets || [];
  const partCount = sets.reduce((total, set) => total + (set.parts || []).length, 0);
  $("#readerStatus").textContent = `${sets.length} generated sections · ${partCount} subparts`;
  $("#readerArticle").innerHTML = `
    <section class="plain-page study-page">
      <p class="note-kicker">Generated exam sections</p>
      <h1 class="note-title">Generated full-course exam bank</h1>
      <p class="note-meta">Generated from the example-exam pattern: each lecture becomes a chained section with definition, mechanism, proof, application, and comparison prompts. Use this for full-picture retention.</p>
      <dl class="source-meta">
        <div><dt>Coverage</dt><dd>${sets.length} lecture sections</dd></div>
        <div><dt>Drill size</dt><dd>${partCount} subparts</dd></div>
        <div><dt>Pattern</dt><dd>closed-book answer → target → repair</dd></div>
      </dl>
      <div class="exam-section-list">
        ${sets.map((set) => `
          <section class="exam-section-block">
            <h2>${escapeHtml(set.title)}</h2>
            <p class="note-meta">${(set.parts || []).length} subparts · <a href="#/lectures/${escapeHtml(set.lectureSlug)}">Open lecture notes</a></p>
            ${(set.parts || []).map((part, index) => renderExamPart(part, set.title, index, set.setId)).join("")}
          </section>`).join("")}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>Retention loop</h2>
      <ol>
        <li>Pick the first weak lecture.</li>
        <li>Answer all subparts in order.</li>
        <li>Mark the related skill-tree node only if no target answer surprises you.</li>
      </ol>
    </section>
    <section class="tool-panel">
      <h2>Use with speedrun</h2>
      <p><a class="quiet-link" href="#/speedrun">Open speedrun path</a> after finishing one lecture section.</p>
    </section>`;
  $("#sectionNav").innerHTML = "";
  bindAnswerDrafts();
}

function renderExam() {
  renderSideNav("lecture", "");
  const questions = filteredStudyItems(studyBank.examQuestions || []);
  $("#readerStatus").textContent = `${questions.length} exam-practice questions`;
  $("#readerArticle").innerHTML = `
    <section class="plain-page study-page">
      <p class="note-kicker">Exam practice</p>
      <h1 class="note-title">Question bank from the example exam, exercises, and lecture concepts</h1>
      <p class="note-meta">Answer closed-book, then open the explanation. Each item points back to the lecture page or the local source seed so you can repair the exact gap.</p>
      <nav class="study-mode-nav" aria-label="Exam modes">
        <a href="#/exam/example">Example exam</a>
        <a href="#/exam/generated">Generated full-course exam bank</a>
        <a href="#/flashcards">Flashcards</a>
      </nav>
      ${renderStudyStats(questions, "questions")}
      <div class="study-list">
        ${questions.map((question, index) => renderExamQuestion(question, index)).join("") || `<p class="empty">No exam question matches the search.</p>`}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>Pass loop</h2>
      <ol>
        <li>Write the answer before opening the details.</li>
        <li>Check whether you gave the formula, intuition, and trap.</li>
        <li>Open the cited lecture only for the missing part.</li>
      </ol>
    </section>
    <section class="tool-panel">
      <h2>Question sources</h2>
      <ul>${(studyBank.sourceSeeds || []).map((source) => `<li><span class="path-inline">${escapeHtml(source)}</span></li>`).join("")}</ul>
    </section>`;
  $("#sectionNav").innerHTML = "";
  bindAnswerDrafts();
}

function renderFlashcards() {
  renderSideNav("lecture", "");
  const cards = filteredStudyItems(studyBank.flashcards || []);
  const knownCount = (studyBank.flashcards || []).filter((card) => localStorage.getItem(flashcardKey(card.id)) === "known").length;
  const reviewCount = (studyBank.flashcards || []).filter((card) => localStorage.getItem(flashcardKey(card.id)) === "review").length;
  $("#readerStatus").textContent = `${cards.length} flashcards`;
  $("#readerArticle").innerHTML = `
    <section class="plain-page study-page">
      <p class="note-kicker">Flashcards</p>
      <h1 class="note-title">Fast concept recall, with explanations on the back</h1>
      <p class="note-meta">Use the card front as the prompt. Flip only after saying the answer aloud or writing one sentence.</p>
      <dl class="source-meta">
        <div><dt>Deck</dt><dd>${cards.length} visible flashcards</dd></div>
        <div><dt>Known</dt><dd>${knownCount} saved known</dd></div>
        <div><dt>Review</dt><dd>${reviewCount} saved for review</dd></div>
      </dl>
      <nav class="study-mode-nav" aria-label="Flashcard drill links">
        <a href="#/skill-tree">Use skill tree order</a>
        <a href="#/exam/example">Example exam cards</a>
        <a href="#/exam/generated">Generated exam sections</a>
      </nav>
      ${renderStudyStats(cards, "cards")}
      <div class="study-list flashcard-list">
        ${cards.map((card, index) => renderFlashcard(card, index)).join("") || `<p class="empty">No flashcard matches the search.</p>`}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>Speed method</h2>
      <p>Definition cards check vocabulary. Mechanism cards check formulas. Trap cards prevent the most common wrong exam wording.</p>
      <p>Search a lecture name or formula keyword to drill only that slice.</p>
    </section>`;
  $("#sectionNav").innerHTML = "";
  bindFlashcards();
}

function renderSpeedrun() {
  renderSideNav("lecture", "");
  const goals = studyBank.speedrunGoals || [];
  const done = goals.filter((goal) => localStorage.getItem(`gm:speedrun:${goal.id}`) === "done").length;
  $("#readerStatus").textContent = `${goals.length} speedrun goals`;
  $("#readerArticle").innerHTML = `
    <section class="plain-page study-page">
      <p class="note-kicker">Speedrun</p>
      <h1 class="note-title">A compact route through the exam scope</h1>
      <p class="note-meta">This is the fast path: read only the needed concept, flip its cards, answer one question, then move on. The causal lecture stays optional and outside the required loop.</p>
      <dl class="source-meta">
        <div><dt>Progress</dt><dd id="speedrunProgress">${done}/${goals.length} goals complete</dd></div>
        <div><dt>Loop</dt><dd>Read → recall → answer → repair</dd></div>
        <div><dt>Target</dt><dd>Every concept explained without notes</dd></div>
      </dl>
      <div class="speedrun-grid">
        ${goals.map((goal) => `
          <article class="speedrun-card">
            <label class="goal-check">
              <input type="checkbox" data-speedrun-check="${escapeHtml(goal.id)}" />
              <span>Goal ${goal.order}: ${escapeHtml(goal.title)}</span>
            </label>
            <p>${escapeHtml(goal.instructions)}</p>
            <p class="study-kicker">${(goal.lectureSlugs || []).length} lectures · ${(goal.conceptIds || []).length} concepts</p>
            <nav class="goal-actions" aria-label="${escapeHtml(goal.title)} study links">
              <a href="#/flashcards">Drill cards</a>
              <a href="#/exam">Answer questions</a>
              <a href="#/exam/generated">Generated section</a>
              ${(goal.lectureSlugs || []).slice(0, 4).map((slug) => {
                const lecture = lectureBySlug(slug);
                return `<a href="#/lectures/${lecture.slug}">${escapeHtml(lecture.shortTitle)}</a>`;
              }).join("")}
            </nav>
            <details>
              <summary>Concept checklist</summary>
              <ol class="speedrun-concepts">${speedrunConceptRows(goal.conceptIds)}</ol>
            </details>
          </article>`).join("")}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `
    <section class="tool-panel">
      <h2>Loop rule</h2>
      <p>For each goal: lecture summary → flashcards → exam question → repair. Stop a pass only when every concept can be explained without looking.</p>
    </section>
    <section class="tool-panel">
      <h2>Required scope</h2>
      <p>${escapeHtml((studyBank.examScope || {}).excludedReason || "Causal material is optional for this route.")}</p>
    </section>`;
  $("#sectionNav").innerHTML = "";
  bindSpeedrunChecks();
}

function renderReview() {
  renderSideNav("lecture", "");
  $("#readerStatus").textContent = "Review questions";
  $("#readerArticle").innerHTML = `
    <section class="plain-page">
      <p class="note-kicker">Review</p>
      <h1 class="note-title">Lecture-by-lecture recall prompts</h1>
      <p class="note-meta">Use these after reading each lecture page. Answer closed-book first, then reopen the relevant note.</p>
      <div class="review-grid">
        ${lectures.map((lecture) => `<article class="review-card"><h2>${lecture.order}. ${escapeHtml(lecture.shortTitle)}</h2><ol>${lecture.questions.map((q) => `<li>${escapeHtml(q)}</li>`).join("")}</ol><a href="#/lectures/${lecture.slug}">Open lecture</a></article>`).join("")}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `<section class="tool-panel"><h2>Method</h2><p>For exam prep, write the formula and the intuition. If one is missing, the answer is not ready.</p></section>`;
  $("#sectionNav").innerHTML = "";
}

function renderImages() {
  renderSideNav("friend", "");
  const images = chapters.flatMap((chapter) => chapter.images.map((image) => ({ ...image, chapter: chapter.shortTitle, slug: chapter.slug })));
  $("#readerStatus").textContent = `${images.length} note images`;
  $("#readerArticle").innerHTML = `
    <section class="plain-page">
      <p class="note-kicker">Images</p>
      <h1 class="note-title">Friend-note figures</h1>
      <p class="note-meta">All images referenced by the five Markdown notes, grouped in reading order.</p>
      <div class="image-grid">
        ${images.map((image) => `<figure class="image-card"><a href="#/friend-notes/${image.slug}/sections/${image.id}"><img loading="lazy" src="${image.src}" alt="${escapeHtml(image.alt)}"></a><figcaption>${escapeHtml(image.chapter)} · ${escapeHtml(image.alt)}</figcaption></figure>`).join("")}
      </div>
    </section>`;
  $("#readerTools").innerHTML = `<section class="tool-panel"><h2>Image study</h2><p>Open an image, explain the surrounding concept, then return to the chapter context.</p></section>`;
  $("#sectionNav").innerHTML = "";
}

function renderSources() {
  renderSideNav("lecture", "");
  $("#readerStatus").textContent = "Local source list";
  $("#readerArticle").innerHTML = `
    <section class="plain-page">
      <p class="note-kicker">Sources</p>
      <h1 class="note-title">What this reader uses</h1>
      <p class="notice">This site copies the Markdown-derived text and PNG images needed for local study. It references lecture PDFs and extracted text paths, but does not copy PDFs, PPTX, KEY, or ZIP files into the deploy folder.</p>
      <div class="source-grid">
        <section class="source-block"><h2>Course lectures</h2><ol>${lectures.map((lecture) => `<li><a href="#/lectures/${lecture.slug}">${escapeHtml(lecture.week)} — ${escapeHtml(lecture.shortTitle)}</a><br><code class="path">${escapeHtml(lecture.sourcePath)}</code><br><code class="path">${escapeHtml(lecture.textPath)}</code></li>`).join("")}</ol></section>
        <section class="source-block"><h2>Friend-note Markdown</h2><ul>${chapters.map((chapter) => `<li><a href="#/friend-notes/${chapter.slug}">${escapeHtml(chapter.shortTitle)}</a><br><code class="path">${escapeHtml(chapter.sourcePath)}</code></li>`).join("")}</ul></section>
        <section class="source-block"><h2>Slide sources kept local</h2><ul>${DATA.slides.map((slide) => `<li><code class="path">${escapeHtml(slide.path)}</code></li>`).join("")}</ul></section>
      </div>
    </section>`;
  $("#readerTools").innerHTML = `<section class="tool-panel"><h2>Boundary</h2><p>Publishing this folder publishes the generated lecture notes, friend-note text, and copied PNG images. Keep it local if the notes should stay private.</p></section>`;
  $("#sectionNav").innerHTML = "";
}

function render() {
  const parts = routeParts();
  const page = parts[0] || "lectures";
  const normalizedPage = ["lectures", "concepts", "skill-tree", "exam", "flashcards", "speedrun", "coverage", "friend-notes", "review", "images", "sources"].includes(page) ? page : "lectures";
  renderRouteNav(normalizedPage);
  if (normalizedPage === "concepts") renderConcepts();
  else if (normalizedPage === "skill-tree") renderSkillTree();
  else if (normalizedPage === "exam" && parts[1] === "example") renderExampleExam();
  else if (normalizedPage === "exam" && parts[1] === "generated") renderGeneratedExams();
  else if (normalizedPage === "exam") renderExam();
  else if (normalizedPage === "flashcards") renderFlashcards();
  else if (normalizedPage === "speedrun") renderSpeedrun();
  else if (normalizedPage === "coverage") renderCoverage();
  else if (normalizedPage === "friend-notes") renderFriendNote(activeFriendSlug());
  else if (normalizedPage === "review") renderReview();
  else if (normalizedPage === "images") renderImages();
  else if (normalizedPage === "sources") renderSources();
  else renderLecture(activeLectureSlug());
  scrollActiveSection();
}

initRailToggles();
$("#noteSearch").addEventListener("input", () => render());
window.addEventListener("hashchange", render);
render();
