#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re
import sys

DATA = Path('notes/study/interactive/notes-data.js')
CONTRACT = Path('notes/study/concept_coverage_contract.json')


def load_data() -> dict:
    text = DATA.read_text(encoding='utf-8')
    text = re.sub(r'^window\.GM_NOTES\s*=\s*', '', text)
    text = re.sub(r';\s*$', '', text)
    return json.loads(text)


def main() -> int:
    errors: list[str] = []
    data = load_data()
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    min_total = int(contract['minTotalConcepts'])
    min_per_lecture = int(contract['minConceptsPerLecture'])
    required_fields = contract['requiredFields']
    required_tags = contract['requiredTags']

    lectures = data.get('lectures', [])
    if len(lectures) != len(required_tags):
        errors.append(f'expected {len(required_tags)} lectures, got {len(lectures)}')

    total = 0
    for lecture in lectures:
        slug = lecture.get('slug', '<missing>')
        concepts = lecture.get('concepts') or []
        total += len(concepts)
        if len(concepts) < min_per_lecture:
            errors.append(f'{slug}: expected at least {min_per_lecture} concepts, got {len(concepts)}')

        tag_index = {tag for concept in concepts for tag in concept.get('tags', [])}
        for tag in required_tags.get(slug, []):
            if tag not in tag_index:
                errors.append(f'{slug}: missing required coverage tag {tag!r}')

        for concept in concepts:
            title = concept.get('title', '<untitled>')
            if not concept.get('id', '').startswith(f'{slug}-concept-'):
                errors.append(f'{slug}/{title}: concept id is missing or not scoped to lecture')
            if not concept.get('tags'):
                errors.append(f'{slug}/{title}: coverage tags missing')
            for field in required_fields:
                value = str(concept.get(field, '')).strip()
                minimum = 6 if field == 'title' else 24
                if len(value) < minimum:
                    errors.append(f'{slug}/{title}: field {field} is too thin or missing')
            checks = concept.get('checks') or []
            if len(checks) < 2:
                errors.append(f'{slug}/{title}: expected at least two recall checks')
            if concept.get('formula') and '<math' not in str(concept.get('formulaHtml', '')):
                errors.append(f'{slug}/{title}: formula is not rendered to MathML')

        if 'Concept checklist' not in lecture.get('html', ''):
            errors.append(f'{slug}: lecture HTML does not include Concept checklist')

    if total < min_total:
        errors.append(f'expected at least {min_total} concepts, got {total}')
    coverage = data.get('conceptCoverage') or {}
    if coverage.get('status') != 'covered':
        errors.append('conceptCoverage.status must be covered')
    if coverage.get('totalConcepts') != total:
        errors.append('conceptCoverage.totalConcepts does not match lecture concept total')
    if coverage.get('contract') != str(CONTRACT):
        errors.append('conceptCoverage.contract does not point at the coverage contract')

    if errors:
        print('CONCEPT GAP AUDIT FAILED')
        for error in errors:
            print(f'- {error}')
        return 1
    print('CONCEPT GAP AUDIT PASSED')
    print(f'lectures={len(lectures)} concepts={total} min_per_lecture={min_per_lecture}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
