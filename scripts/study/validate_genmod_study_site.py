#!/usr/bin/env python3
from html.parser import HTMLParser
from pathlib import Path
import importlib.util
import json
import re
import sys

ROOT = Path('notes/study/interactive')
INDEX = ROOT / 'index.html'
CSS = ROOT / 'styles.css'
APP = ROOT / 'app.js'
DATA = ROOT / 'notes-data.js'
README = ROOT / 'README.md'
VERCEL = ROOT / 'vercel.json'
ASSETS = ROOT / 'assets' / 'friend-notes' / 'attachments'
LECTURE_INVENTORY = Path('notes/study/lecture_inventory.json')
LECTURE_NOTES_BLUEPRINTS = Path('notes/study/lecture_notes_blueprints.json')
CONCEPT_BLUEPRINTS = Path('notes/study/concept_blueprints.json')
CONCEPT_COVERAGE_CONTRACT = Path('notes/study/concept_coverage_contract.json')
FRIEND_NOTES_MANIFEST = Path('notes/study/friend_notes_manifest.json')
SITE_VALIDATION_CONTRACT = Path('notes/study/site_validation_contract.json')
CONCEPT_AUDIT = Path('scripts/study/audit_genmod_concept_gaps.py')
BUILDER = Path('scripts/study/build_friend_notes_site.py')
errors = []


def require(condition, message):
    if not condition:
        errors.append(message)


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.styles = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'script' and attributes.get('src'):
            self.scripts.append(attributes['src'])
        if tag == 'link' and attributes.get('rel') == 'stylesheet':
            self.styles.append(attributes.get('href'))


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        require(False, f'invalid JSON in {path}: {exc}')
        return {} if path.suffix == '.json' else []


def load_notes_data(raw: str) -> dict:
    raw = re.sub(r'^window\.GM_NOTES\s*=\s*', '', raw)
    raw = re.sub(r';\s*$', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        require(False, f'invalid generated notes-data.js: {exc}')
        return {}


def load_builder_module():
    spec = importlib.util.spec_from_file_location('study_site_builder', BUILDER)
    if spec is None or spec.loader is None:
        require(False, f'could not load {BUILDER}')
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


required_paths = [
    INDEX,
    CSS,
    APP,
    DATA,
    README,
    VERCEL,
    LECTURE_INVENTORY,
    LECTURE_NOTES_BLUEPRINTS,
    CONCEPT_BLUEPRINTS,
    CONCEPT_COVERAGE_CONTRACT,
    FRIEND_NOTES_MANIFEST,
    SITE_VALIDATION_CONTRACT,
    CONCEPT_AUDIT,
    BUILDER,
]
for path in required_paths:
    require(path.is_file(), f'missing {path}')
if errors:
    print('VALIDATION FAILED')
    for error in errors:
        print(f'- {error}')
    sys.exit(1)

html = INDEX.read_text(encoding='utf-8')
css = CSS.read_text(encoding='utf-8')
app = APP.read_text(encoding='utf-8')
data_js = DATA.read_text(encoding='utf-8')
readme = README.read_text(encoding='utf-8')
vercel = VERCEL.read_text(encoding='utf-8')
combined = '\n'.join([html, css, app, data_js, readme])
parser = Parser()
parser.feed(html)

site_contract = load_json(SITE_VALIDATION_CONTRACT)
inventory = load_json(LECTURE_INVENTORY)
lecture_blueprints = load_json(LECTURE_NOTES_BLUEPRINTS)
concept_contract = load_json(CONCEPT_COVERAGE_CONTRACT)
friend_manifest = load_json(FRIEND_NOTES_MANIFEST)
notes_data = load_notes_data(data_js)
builder = load_builder_module()

# Static shell and file loading.
static_shell = site_contract.get('staticShell', {})
require(parser.styles == static_shell.get('styles'), f'unexpected styles: {parser.styles}')
require(parser.scripts == static_shell.get('scripts'), f'unexpected scripts: {parser.scripts}')
for marker in static_shell.get('domMarkers', []):
    require(marker in combined, f'missing notes-reader marker: {marker}')
require(all(renderer in app for renderer in static_shell.get('renderers', [])), 'required route renderers missing')
require(static_shell.get('defaultRouteCall', '') in app, 'default route does not open a lecture note')
for route in static_shell.get('navigationRoutes', []):
    require(route in html, f'navigation route missing: {route}')
require(static_shell.get('dynamicLectureRouteMarker', '') in app, 'dynamic lecture route references missing')

# Course lecture source contract: derive course vocabulary from blueprints and inventory, not validator code.
lectures = notes_data.get('lectures') or []
expected_lecture_count = len(lecture_blueprints)
require(len(inventory) == expected_lecture_count, f'expected {expected_lecture_count} source lectures in inventory, got {len(inventory)}')
require(len(lectures) == expected_lecture_count, f'expected {expected_lecture_count} generated lecture notes, got {len(lectures)}')
require(all(lecture.get('kind') == 'lecture' for lecture in lectures), 'generated lecture collection contains a non-lecture item')
require(notes_data.get('courseInventoryPath') == str(LECTURE_INVENTORY), 'course inventory pointer missing or wrong')
require(notes_data.get('lectureNotesBlueprintPath') == str(LECTURE_NOTES_BLUEPRINTS), 'lecture blueprint pointer missing or wrong')
require(notes_data.get('conceptBlueprintPath') == str(CONCEPT_BLUEPRINTS), 'concept blueprint pointer missing or wrong')

for index, (blueprint, source, lecture) in enumerate(zip(lecture_blueprints, inventory, lectures), start=1):
    slug = blueprint.get('slug', f'lecture-{index}')
    for field in ['slug', 'week', 'shortTitle', 'title', 'focus', 'relatedFriendChapter']:
        require(lecture.get(field) == blueprint.get(field), f'{slug}: generated field {field} does not match lecture blueprint')
    require(lecture.get('order') == index, f'{slug}: generated lecture order mismatch')
    require(lecture.get('sourceTitle') == source.get('title'), f'{slug}: source title mismatch')
    require(lecture.get('sourcePath') == source.get('path'), f'{slug}: source path mismatch')
    require(lecture.get('textPath') == source.get('text_path'), f'{slug}: extracted text path mismatch')
    require(lecture.get('sourceHighlightsLabel') == site_contract['course']['sourceHighlightsLabel'], f'{slug}: source highlights label missing')
    require('Concept checklist' in lecture.get('html', ''), f'{slug}: lecture HTML does not include Concept checklist')
    concepts = lecture.get('concepts') or []
    require(len(concepts) >= int(concept_contract.get('minConceptsPerLecture', 0)), f'{slug}: concept list is below contract minimum')
    for concept in concepts:
        title = concept.get('title', '<untitled>')
        require(concept.get('id', '').startswith(f'{slug}-concept-'), f'{slug}/{title}: concept id is not lecture-scoped')
        require(concept.get('tags'), f'{slug}/{title}: coverage tags missing')
        for field in concept_contract.get('requiredFields', []):
            require(str(concept.get(field, '')).strip(), f'{slug}/{title}: missing required concept field {field}')
        require(len(concept.get('checks') or []) >= 2, f'{slug}/{title}: missing recall checks')

require(site_contract['course']['duplicateDeckDisclosure'] in data_js, 'duplicate MLE/LVM lecture should be explicit, not hidden')
coverage = notes_data.get('conceptCoverage') or {}
total_concepts = sum(len(lecture.get('concepts') or []) for lecture in lectures)
require(coverage.get('status') == 'covered', 'concept coverage status missing')
require(coverage.get('contract') == str(CONCEPT_COVERAGE_CONTRACT), 'concept coverage contract pointer missing')
require(coverage.get('totalLectures') == len(lectures), 'concept coverage lecture count mismatch')
require(coverage.get('totalConcepts') == total_concepts, 'concept coverage total mismatch')
require(total_concepts >= int(concept_contract.get('minTotalConcepts', 0)), 'concept explanations are too thin')
for marker in site_contract.get('uiContract', {}).get('requiredPublishedMarkers', ['concept-card', 'coverage-card']):
    require(marker in combined, f'concept/coverage UI marker missing: {marker}')

# Friend-note source contract: derive chapter filenames and slugs from the manifest.
chapters = notes_data.get('chapters') or []
chapter_files = friend_manifest.get('chapterFiles') or []
require(len(chapters) == len(chapter_files), f'expected {len(chapter_files)} main friend-note chapters, got {len(chapters)}')
require(all(chapter.get('kind') == 'friend-note' for chapter in chapters), 'generated friend-note collection contains a non-friend item')
for expected, chapter in zip(chapter_files, chapters):
    slug = expected.get('slug')
    require(chapter.get('slug') == slug, f'{slug}: generated friend-note slug mismatch')
    require(chapter.get('sourceFile') == expected.get('filename'), f'{slug}: generated friend-note source file mismatch')
    require(chapter.get('shortTitle') == expected.get('shortTitle'), f'{slug}: generated friend-note short title mismatch')
    require(chapter.get('sourcePath') == f"{friend_manifest.get('sourceRootLabel')}/{expected.get('filename')}", f'{slug}: generated friend-note public source path mismatch')
    require(str(chapter.get('title', '')).strip(), f'{slug}: generated friend-note title missing')
    require(str(chapter.get('html', '')).strip(), f'{slug}: generated friend-note HTML missing')
    require(len(chapter.get('questions') or []) >= 3, f'{slug}: generated friend-note recall questions missing')
require(notes_data.get('friendNotesManifestPath') == str(FRIEND_NOTES_MANIFEST), 'friend notes manifest pointer missing or wrong')
friend_contract = site_contract.get('friendNotes', {})
image_count = sum(len(chapter.get('images') or []) for chapter in chapters)
require(image_count >= int(friend_contract.get('minCopiedImages', 0)), f'not enough friend-note images in data: {image_count}')
require(data_js.count('math-block') >= int(friend_contract.get('minDisplayMathBlocks', 0)), 'display math blocks not preserved')
require(data_js.count('inline-math') >= int(friend_contract.get('minInlineMath', 0)), 'inline math not preserved')
require(data_js.count('<math') >= int(friend_contract.get('minMathMlNodes', 0)), 'MathML formulas not rendered across friend notes and lectures')
require('<code>' not in data_js and '<code class="inline-math"' not in data_js, 'formulas are still rendered as code')

# Assets: images are allowed because they are part of the friend notes; heavy/source files are not.
require(ASSETS.is_dir(), 'friend-note attachment folder missing')
image_files = list(ASSETS.glob('*.png')) if ASSETS.exists() else []
require(len(image_files) >= int(friend_contract.get('minCopiedImages', 0)), f'not enough copied note images: {len(image_files)}')
deploy_contract = site_contract.get('deployPayload', {})
for path in ROOT.rglob('*'):
    if path.is_file():
        require(path.suffix.lower() not in set(deploy_contract.get('forbiddenExtensions', [])), f'forbidden heavy/source file in deploy root: {path}')
        require(path.name not in set(deploy_contract.get('forbiddenPackageFiles', [])), f'package manager file in deploy root: {path}')
for directory_name in deploy_contract.get('forbiddenDirectories', []):
    require(not any(path.name == directory_name for path in ROOT.rglob('*')), f'{directory_name} found in deploy root')

# Anti-cram / no-launch-page, security, and route safety constraints.
ui_contract = site_contract.get('uiContract', {})
for banned in ui_contract.get('bannedMarkers', []):
    require(banned not in combined, f'old launch/admin/slop marker remains: {banned}')
for marker in ui_contract.get('requiredCssMarkers', []):
    require(marker in css, f'required content-first CSS marker missing: {marker}')
for marker in ui_contract.get('requiredHtmlMarkers', []):
    require(marker in html, f'required content-first HTML marker missing: {marker}')
for marker in ui_contract.get('requiredAppMarkers', []):
    require(marker in app, f'required rail-state app marker missing: {marker}')
require(combined.count('<button') <= int(ui_contract.get('maxButtons', 999)), 'too many controls for a notes reader')

security_contract = site_contract.get('securityContract', {})
for forbidden in security_contract.get('forbiddenPublishedStrings', []):
    require(forbidden not in combined, f'deployable site leaks forbidden string: {forbidden}')
for forbidden in security_contract.get('forbiddenRuntimeMarkers', []):
    require(forbidden not in app, f'app uses forbidden network/backend runtime marker: {forbidden}')
for fragment in security_contract.get('requiredCspFragments', []):
    require(fragment in vercel, f'vercel headers missing CSP fragment: {fragment}')
for marker in security_contract.get('routeSafeSectionMarkers', []):
    require(marker in app, f'route-safe section marker missing: {marker}')
for marker in security_contract.get('forbiddenNestedHashMarkers', []):
    require(marker not in app, f'section links still use nested hash anchors: {marker}')
if builder is not None:
    safe_name = security_contract.get('safeAttachmentName', '')
    require(builder.safe_attachment_name(safe_name) == safe_name, 'safe attachment names should be accepted')
    for unsafe_name in security_contract.get('unsafeAttachmentNames', []):
        try:
            builder.safe_attachment_name(unsafe_name)
            require(False, f'unsafe attachment reference was accepted: {unsafe_name}')
        except ValueError:
            pass

# Accessibility and performance.
for marker in site_contract.get('accessibility', {}).get('requiredMarkers', []):
    require(marker in combined, f'missing accessibility marker: {marker}')
require(not re.search(r'outline:\s*none|transition:\s*all', css), 'bad focus/transition pattern found')
require(not re.search(r'<script[^>]+src=["\']https?://|<link[^>]+href=["\']https?://|@import\s+url|https?://[^\s"\']*(fonts\.googleapis|cdn\.)', combined, re.I), 'external asset/CDN reference found')

try:
    json.loads(vercel)
except json.JSONDecodeError as exc:
    require(False, f'invalid vercel.json: {exc}')

if errors:
    print('VALIDATION FAILED')
    for error in errors:
        print(f'- {error}')
    sys.exit(1)
print('VALIDATION PASSED')
math_blocks = data_js.count('math-block')
file_count = len([path for path in ROOT.rglob('*') if path.is_file()])
mathml = data_js.count('<math')
print(f'chapters={len(chapters)} images={len(image_files)} math_blocks={math_blocks} mathml={mathml} files={file_count}')
