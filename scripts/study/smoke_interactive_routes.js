#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve("notes/study/interactive");
const routes = {
  "#/concepts": ["All 74 concepts", "Filter by course week", "Lecture 13 pages across 7 course weeks"],
  "#/concepts/week/Week%206": ["Week 6", "18 concepts shown", "Diffusion"],
  "#/concepts/lecture/w6-diffusion": ["Diffusion", "6 concepts shown", "Forward noising process"],
  "#/skill-tree": ["Skill tree", "Start here", "Unlocks", "Progress"],
  "#/exam": ["Question bank from the example exam", "Show answer, explanation, and citations", "exam-practice questions"],
  "#/exam/example": ["Example exam: 30 June digital test", "30 subparts", "Derive the ELBO for VAE", "Energy-based models", "Score-based Models", "attention formula", "Perplexity"],
  "#/exam/generated": ["Generated full-course exam bank", "Generated from the example-exam pattern", "Lecture 1", "Lecture 12", "subparts"],
  "#/flashcards": ["Fast concept recall", "Flip card", "flashcards"],
  "#/speedrun": ["compact route through the exam scope", "Goal 1:", "speedrun goals"],
};

const indexHtml = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
for (const marker of ["quickNav", "#/concepts", "#/skill-tree", "#/exam", "#/exam/example", "#/exam/generated", "#/flashcards", "#/speedrun"]) {
  if (!indexHtml.includes(marker)) {
    console.error(`index.html missing always-visible navigation marker: ${marker}`);
    process.exit(1);
  }
}

class Element {
  constructor(id = "", attributes = {}) {
    this.id = id;
    this.attributes = { ...attributes };
    this.dataset = {};
    this.children = [];
    this.value = "";
    this.textContent = "";
    this.innerHTML = "";
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  getAttribute(name) {
    return this.attributes[name] || "";
  }

  removeAttribute(name) {
    delete this.attributes[name];
  }

  addEventListener() {}

  scrollIntoView() {}
}

function makeDocument() {
  const ids = [
    "leftRail",
    "rightRail",
    "toggleLeft",
    "toggleRight",
    "noteSearch",
    "lectureNav",
    "navTitle",
    "readerStatus",
    "readerArticle",
    "readerTools",
    "sectionNav",
  ];
  const elements = Object.fromEntries(ids.map((id) => [id, new Element(id)]));
  const routeLinks = [
    ["#/"],
    ["#/concepts"],
    ["#/skill-tree"],
    ["#/exam"],
    ["#/exam/example"],
    ["#/exam/generated"],
    ["#/flashcards"],
    ["#/speedrun"],
    ["#/coverage"],
    ["#/friend-notes"],
    ["#/review"],
    ["#/images"],
    ["#/sources"],
  ].map(([href]) => new Element("", { href }));

  const document = {
    body: { dataset: { leftCollapsed: "false", rightCollapsed: "true" } },
    querySelector(selector) {
      if (selector.startsWith("#")) return elements[selector.slice(1)] || null;
      return null;
    },
    querySelectorAll(selector) {
      if (selector === ".route-nav a, .quick-nav a" || selector === ".route-nav a") return routeLinks;
      if (selector === "[data-speedrun-check]" || selector === "[data-skill-check]") return [];
      return [];
    },
    getElementById(id) {
      return elements[id] || null;
    },
  };
  return { document, elements };
}

function runRoute(hash) {
  const { document, elements } = makeDocument();
  const storage = new Map();
  const context = vm.createContext({
    console,
    document,
    location: { hash },
    localStorage: {
      getItem: (key) => (storage.has(key) ? storage.get(key) : null),
      setItem: (key, value) => storage.set(key, String(value)),
      removeItem: (key) => storage.delete(key),
    },
    requestAnimationFrame: (callback) => callback(),
    window: { addEventListener() {} },
  });
  context.window.window = context.window;
  context.window.document = document;
  context.window.location = context.location;
  context.window.localStorage = context.localStorage;
  context.window.requestAnimationFrame = context.requestAnimationFrame;

  vm.runInContext(fs.readFileSync(path.join(ROOT, "notes-data.js"), "utf8"), context, { filename: "notes-data.js" });
  vm.runInContext(fs.readFileSync(path.join(ROOT, "app.js"), "utf8"), context, { filename: "app.js" });

  return `${elements.readerStatus.textContent}\n${elements.readerArticle.innerHTML}\n${elements.readerTools.innerHTML}`;
}

let failed = false;
for (const [hash, checks] of Object.entries(routes)) {
  const output = runRoute(hash);
  const missing = checks.filter((check) => !output.includes(check));
  if (missing.length) {
    failed = true;
    console.error(`${hash} missing: ${missing.join(", ")}`);
  } else {
    console.log(`${hash} smoke passed (${output.length} chars)`);
  }
}

if (failed) process.exit(1);
