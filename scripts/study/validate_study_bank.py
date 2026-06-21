#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re
import sys

ROOT = Path('notes/study/interactive')
DATA = ROOT / 'notes-data.js'
APP = ROOT / 'app.js'
INDEX = ROOT / 'index.html'
README = ROOT / 'README.md'
EXAMPLE_EXAM = Path('notes/study/08_example_exam.md')
MIN_FLASHCARDS_PER_CONCEPT = 6
MIN_QUESTIONS_PER_CONCEPT = 5
CAUSAL_SLUG = 'w7-causal'
FORBIDDEN_RUNTIME_MARKERS = [
    'apiKey',
    'fetch(',
    'XMLHttpRequest',
    'WebSocket',
    'EventSource',
    'sendBeacon',
    'setInterval(timer',
    'countdown',
    'Timed exam',
    'timer',
]
GENERIC_SEED_ANSWERS = [
    'Use the related concept cards to build a complete answer.',
    'Write a structured answer, then compare it against the cited course notes.',
]
errors: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def load_data() -> dict:
    if not DATA.is_file():
        return {}
    raw = DATA.read_text(encoding='utf-8')
    raw = re.sub(r'^window\.GM_NOTES\s*=\s*', '', raw)
    raw = re.sub(r';\s*$', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        require(False, f'invalid notes-data.js JSON payload: {exc}')
        return {}


for path in [DATA, APP, INDEX, README, EXAMPLE_EXAM]:
    require(path.is_file(), f'missing {path}')

data = load_data()
lectures = data.get('lectures') or []
exam_lectures = [lecture for lecture in lectures if lecture.get('slug') != CAUSAL_SLUG]
exam_concepts = [concept for lecture in exam_lectures for concept in (lecture.get('concepts') or [])]
causal_concept_ids = {
    concept.get('id')
    for lecture in lectures
    if lecture.get('slug') == CAUSAL_SLUG
    for concept in (lecture.get('concepts') or [])
}

study_bank = data.get('studyBank') or {}
flashcards = study_bank.get('flashcards') or []
questions = study_bank.get('examQuestions') or []
speedrun = study_bank.get('speedrunGoals') or []
source_seeds = study_bank.get('sourceSeeds') or []
example_exam = study_bank.get('exampleExam') or {}
generated_exam_sets = study_bank.get('generatedExamSets') or []

require(study_bank.get('version') == 1, 'studyBank.version must be 1')
require(CAUSAL_SLUG in ((study_bank.get('examScope') or {}).get('excludedLectureSlugs') or []), 'causal lecture must be excluded from required exam scope')
for seed in ['notes/study/03_practice_questions.md', 'notes/study/07_mock_exam.md', 'notes/study/08_example_exam.md', 'notes/study/04_flashcards.tsv']:
    require(seed in source_seeds, f'missing source seed: {seed}')
require(len(flashcards) >= len(exam_concepts) * MIN_FLASHCARDS_PER_CONCEPT, 'not enough flashcards for exam concepts')
require(len(questions) >= len(exam_concepts) * MIN_QUESTIONS_PER_CONCEPT, 'not enough exam questions for exam concepts')
require(speedrun, 'speedrun goals missing')
require(example_exam.get('sourcePath') == str(EXAMPLE_EXAM), 'example exam source path missing')
example_sections = example_exam.get('sections') or []
example_parts = [part for section in example_sections for part in (section.get('parts') or [])]
require(len(example_sections) == 4, 'example exam should have 4 sections')
require(len(example_parts) == 30, 'example exam should have 30 subparts')
for required_part in ['1a', '1c', '2f', '2i', '3e', '3i', '4a', '4f']:
    require(any(part.get('partId') == required_part for part in example_parts), f'example exam missing part {required_part}')
example_exam_text = EXAMPLE_EXAM.read_text(encoding='utf-8') if EXAMPLE_EXAM.is_file() else ''
require('5539447' not in example_exam_text, 'example exam source must not publish personal student id')
require('Iftode, George' not in example_exam_text, 'example exam source must not publish personal student name')
require(len(generated_exam_sets) >= len(exam_lectures), 'generated full-course exam sets missing lectures')
generated_parts = [part for exam_set in generated_exam_sets for part in (exam_set.get('parts') or [])]
require(len(generated_parts) >= len(exam_concepts), 'generated exam bank should cover every exam concept')

flash_by_concept: dict[str, list[dict]] = {}
question_by_concept: dict[str, list[dict]] = {}
for card in flashcards:
    for concept_id in card.get('conceptIds') or []:
        flash_by_concept.setdefault(concept_id, []).append(card)
for question in questions:
    for concept_id in question.get('conceptIds') or []:
        question_by_concept.setdefault(concept_id, []).append(question)

for concept in exam_concepts:
    concept_id = concept.get('id')
    require(len(flash_by_concept.get(concept_id, [])) >= MIN_FLASHCARDS_PER_CONCEPT, f'{concept_id}: fewer than {MIN_FLASHCARDS_PER_CONCEPT} flashcards')
    require(len(question_by_concept.get(concept_id, [])) >= MIN_QUESTIONS_PER_CONCEPT, f'{concept_id}: fewer than {MIN_QUESTIONS_PER_CONCEPT} exam questions')

for item_type, items in [('flashcard', flashcards), ('exam question', questions)]:
    for item in items:
        item_id = item.get('id', '<missing-id>')
        require(item.get('front') or item.get('prompt'), f'{item_type} {item_id}: prompt/front missing')
        require(item.get('answer'), f'{item_type} {item_id}: answer missing')
        require(item.get('explanation'), f'{item_type} {item_id}: explanation missing')
        citations = item.get('citations') or []
        require(any(str(citation.get('href', '')).startswith('#/lectures/') or str(citation.get('href', '')).startswith('#/concepts') for citation in citations), f'{item_type} {item_id}: site citation missing')
        require(any(citation.get('sourcePath') for citation in citations), f'{item_type} {item_id}: source path citation missing')
        if item_id.startswith('seed-'):
            require(item.get('conceptIds'), f'{item_type} {item_id}: seed item missing topical concept mapping')
            require(item.get('mappingQuality') != 'fallback', f'{item_type} {item_id}: seed item used fallback topical mapping')
            require(item.get('answer') not in GENERIC_SEED_ANSWERS, f'{item_type} {item_id}: generic seed answer fallback remains')
            require(any(citation.get('kind') == 'seed' for citation in citations), f'{item_type} {item_id}: seed source citation missing')
        for concept_id in item.get('conceptIds') or []:
            require(concept_id not in causal_concept_ids, f'{item_type} {item_id}: required item includes causal concept {concept_id}')

question_types = {question.get('type') for question in questions}
for expected_type in ['definition', 'mechanism', 'comparison', 'application', 'proof']:
    require(expected_type in question_types, f'exam question type missing: {expected_type}')

for goal in speedrun:
    require(goal.get('conceptIds'), f'speedrun goal {goal.get("id", "<missing-id>")}: concept ids missing')
    for concept_id in goal.get('conceptIds') or []:
        require(concept_id not in causal_concept_ids, f'speedrun goal {goal.get("id", "<missing-id>")}: includes causal concept {concept_id}')

app = APP.read_text(encoding='utf-8') if APP.is_file() else ''
index = INDEX.read_text(encoding='utf-8') if INDEX.is_file() else ''
combined = '\n'.join([app, index, README.read_text(encoding='utf-8') if README.is_file() else ''])
for marker in ['renderExam', 'renderFlashcards', 'renderSpeedrun', 'renderSkillTree']:
    require(marker in app, f'missing app renderer: {marker}')
for route in ['#/exam', '#/exam/example', '#/exam/generated', '#/flashcards', '#/speedrun', '#/skill-tree']:
    require(route in index or route in app, f'missing route marker: {route}')
for forbidden in FORBIDDEN_RUNTIME_MARKERS:
    require(forbidden not in combined, f'forbidden runtime/timed/backend marker found: {forbidden}')

if errors:
    print('STUDY BANK VALIDATION FAILED')
    for error in errors:
        print(f'- {error}')
    sys.exit(1)

print('STUDY BANK VALIDATION PASSED')
print(f'exam_concepts={len(exam_concepts)} flashcards={len(flashcards)} exam_questions={len(questions)} speedrun_goals={len(speedrun)}')
