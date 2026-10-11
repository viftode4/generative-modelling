#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import html
import json
import os
import re
import shutil
from latex2mathml.converter import convert as latex_to_mathml

DEFAULT_SRC = Path.home() / 'personal' / 'DSAIT_Notes' / 'Q4' / 'GM'
SRC = Path(os.environ.get('GM_FRIEND_NOTES_ROOT', DEFAULT_SRC)).expanduser()
OUT = Path('notes/study/interactive')
ASSET_OUT = OUT / 'assets' / 'friend-notes' / 'attachments'
LECTURE_INVENTORY = Path('notes/study/lecture_inventory.json')
LECTURE_NOTES_BLUEPRINTS = Path('notes/study/lecture_notes_blueprints.json')
CONCEPT_BLUEPRINTS = Path('notes/study/concept_blueprints.json')
CONCEPT_COVERAGE_CONTRACT = Path('notes/study/concept_coverage_contract.json')
FRIEND_NOTES_MANIFEST = Path('notes/study/friend_notes_manifest.json')
CONCEPT_AUDIT_SCRIPT = Path('scripts/study/audit_genmod_concept_gaps.py')
PRACTICE_QUESTIONS = Path('notes/study/03_practice_questions.md')
FLASHCARDS_TSV = Path('notes/study/04_flashcards.tsv')
MOCK_EXAM = Path('notes/study/07_mock_exam.md')
EXAMPLE_EXAM = Path('notes/study/08_example_exam.md')
FORMULA_SHEET = Path('notes/study/06_formula_derivation_sheet.md')
PUBLIC_SOURCE_ROOT_LABEL = 'DSAIT_Notes/Q4/GM'
EXAM_EXCLUDED_LECTURE_SLUGS = {'w7-causal'}

heading_counts: dict[str, int] = {}
used_images: set[str] = set()


def render_math(expr: str, display: bool) -> str:
    expr = expr.strip()
    return latex_to_mathml(expr, display='block' if display else 'inline')


def slugify(text: str) -> str:
    base = re.sub(r'<[^>]+>', '', text)
    base = re.sub(r'[^a-zA-Z0-9]+', '-', base.lower()).strip('-') or 'section'
    count = heading_counts.get(base, 0)
    heading_counts[base] = count + 1
    return f'{base}-{count + 1}' if count else base


def plain_slugify(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '-', text.lower()).strip('-') or 'concept'


def public_source_path(*parts: str) -> str:
    return '/'.join([PUBLIC_SOURCE_ROOT_LABEL, *[part.strip('/') for part in parts if part]])


def safe_attachment_name(name: str) -> str:
    candidate = Path(name)
    if candidate.name != name or candidate.suffix.lower() != '.png':
        raise ValueError(f'unsafe friend-note attachment reference: {name!r}')
    return candidate.name


def copy_attachment(name: str) -> None:
    safe_name = safe_attachment_name(name)
    src_root = (SRC / 'attachments').resolve()
    dst_root = ASSET_OUT.resolve()
    src = (src_root / safe_name).resolve()
    dst = (dst_root / safe_name).resolve()
    if src.parent != src_root or dst.parent != dst_root:
        raise ValueError(f'attachment path escaped allowed directory: {name!r}')
    shutil.copy2(src, dst)


def inline_format(text: str) -> str:
    text = re.sub(r'\[\[(?:[^\]|]+\|)?([^\]]+)\]\]', r'\1', text)
    inline_math: list[str] = []

    def math_repl(match: re.Match[str]) -> str:
        inline_math.append(f'<span class="inline-math">{render_math(match.group(1), display=False)}</span>')
        return f'@@INLINE_MATH_{len(inline_math) - 1}@@'

    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', math_repl, text)
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<em>\1</em>', escaped)
    for idx, rendered in enumerate(inline_math):
        escaped = escaped.replace(f'@@INLINE_MATH_{idx}@@', rendered)
    return escaped


def flush_paragraph(parts: list[str], out: list[str]) -> None:
    if parts:
        out.append(f'<p>{inline_format(" ".join(p.strip() for p in parts))}</p>')
        parts.clear()


def flush_list(items: list[str], out: list[str], ordered: bool) -> None:
    if items:
        tag = 'ol' if ordered else 'ul'
        out.append(f'<{tag}>' + ''.join(f'<li>{inline_format(item)}</li>' for item in items) + f'</{tag}>')
        items.clear()


def preprocess(text: str) -> tuple[str, list[str]]:
    formulas: list[str] = []

    def image_repl(match: re.Match[str]) -> str:
        return f'\n\n::IMAGE::{match.group(1).strip()}::\n\n'

    text = re.sub(r'!\[\[(.*?)\]\]', image_repl, text)

    def math_repl(match: re.Match[str]) -> str:
        formulas.append(match.group(1).strip())
        return f'\n\n::MATH::{len(formulas)-1}::\n\n'

    text = re.sub(r'\$\$(.*?)\$\$', math_repl, text, flags=re.S)
    return text, formulas


def convert_markdown(text: str, slug: str) -> tuple[str, list[dict], list[dict], list[str]]:
    global heading_counts
    heading_counts = {}
    text, formulas = preprocess(text)
    out: list[str] = []
    para: list[str] = []
    list_items: list[str] = []
    list_ordered = False
    sections: list[dict] = []
    images: list[dict] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph(para, out)
            flush_list(list_items, out, list_ordered)
            continue
        if stripped.startswith('::IMAGE::') and stripped.endswith('::'):
            flush_paragraph(para, out)
            flush_list(list_items, out, list_ordered)
            name = safe_attachment_name(stripped[len('::IMAGE::'):-2])
            used_images.add(name)
            src = f'assets/friend-notes/attachments/{quote(name)}'
            alt = Path(name).stem.replace('Pasted image ', 'Friend note figure ')
            img_id = f'{slug}-image-{len(images)+1}'
            images.append({'id': img_id, 'file': name, 'src': src, 'alt': alt})
            out.append(f'<figure id="{img_id}" class="note-figure"><img class="note-image" loading="lazy" src="{src}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
            continue
        if stripped.startswith('::MATH::') and stripped.endswith('::'):
            flush_paragraph(para, out)
            flush_list(list_items, out, list_ordered)
            idx = int(stripped[len('::MATH::'):-2])
            formula = render_math(formulas[idx], display=True)
            out.append(f'<div class="math-block" role="group" aria-label="Display formula">{formula}</div>')
            continue
        m = re.match(r'^(#{1,4})\s+(.+)$', stripped)
        if m:
            flush_paragraph(para, out)
            flush_list(list_items, out, list_ordered)
            level = min(len(m.group(1)), 4)
            title_html = inline_format(m.group(2))
            title_text = re.sub(r'<[^>]+>', '', title_html)
            hid = slugify(title_text)
            sections.append({'id': hid, 'level': level, 'title': title_text})
            out.append(f'<h{level} id="{hid}">{title_html}</h{level}>')
            continue
        lm = re.match(r'^\s*(\d+\.|[-*])\s+(.+)$', line)
        if lm:
            flush_paragraph(para, out)
            ordered = lm.group(1).endswith('.')
            if list_items and ordered != list_ordered:
                flush_list(list_items, out, list_ordered)
            list_ordered = ordered
            list_items.append(lm.group(2).strip())
            continue
        para.append(stripped)
    flush_paragraph(para, out)
    flush_list(list_items, out, list_ordered)
    return '\n'.join(out), sections, images, formulas


def markdown_fragment_html(text: str, slug: str) -> str:
    body, _, _, _ = convert_markdown(text.strip(), slug)
    return body


def source_highlight_markdown(lines: list[str]) -> str:
    clean = [line.strip() for line in lines if line and line.strip()]
    if not clean:
        return ''
    bullets = '\n'.join(f'- {line}' for line in clean[:8])
    return f'\n\n## Source highlights\n{bullets}\n'


def prepare_concepts(slug: str, raw_concepts: list[dict]) -> list[dict]:
    concepts = []
    for index, raw in enumerate(raw_concepts, start=1):
        formula = raw.get('formula', '').strip()
        concept = {
            'id': f'{slug}-concept-{plain_slugify(raw["title"])}',
            'order': index,
            'title': raw['title'],
            'mustKnow': raw['mustKnow'],
            'coreIdea': raw['coreIdea'],
            'formulaOrRule': raw['formulaOrRule'],
            'formula': formula,
            'formulaHtml': render_math(formula, display=True) if formula else '',
            'commonTrap': raw['commonTrap'],
            'examUse': raw['examUse'],
            'checks': raw['checks'],
            'tags': raw.get('tags') or [plain_slugify(raw['title'])],
        }
        concepts.append(concept)
    return concepts


def concept_markdown(concepts: list[dict]) -> str:
    if not concepts:
        return ''
    parts = ['\n\n## Concept checklist\nThese are the concepts this lecture expects you to know. For each one, learn the idea, the formula/rule, the common trap, and the exam use.\n']
    for concept in concepts:
        parts.append(f'\n### Concept: {concept["title"]}\n')
        parts.append(f'**Must know:** {concept["mustKnow"]}\n')
        parts.append(f'**Core idea:** {concept["coreIdea"]}\n')
        parts.append(f'**Formula/rule:** {concept["formulaOrRule"]}\n')
        if concept.get('formula'):
            parts.append(f'\n$${concept["formula"]}$$\n')
        parts.append(f'**Common trap:** {concept["commonTrap"]}\n')
        parts.append(f'**Exam use:** {concept["examUse"]}\n')
        checks = '\n'.join(f'- {check}' for check in concept['checks'])
        parts.append(f'**Recall checks:**\n{checks}\n')
    return '\n'.join(parts)


def empty_source_fallback(slug: str, filename: str, fallback_lecture: dict | None) -> tuple[str, list[dict], list[dict], int, str | None]:
    title = 'Source note status'
    section_id = f'{slug}-source-status'
    if fallback_lecture is None:
        body = (
            f'<section class="source-gap" id="{section_id}">'
            f'<h2>{title}</h2>'
            f'<p>The friend-note source file {html.escape(filename)} is empty in the pulled DSAIT_Notes checkout. '
            'No lecture fallback was linked for this chapter yet.</p>'
            '</section>'
        )
        return body, [{'id': section_id, 'level': 2, 'title': title}], [], 0, None

    body = (
        f'<section class="source-gap" id="{section_id}">'
        f'<h2>{title}</h2>'
        f'<p>The friend-note source file {html.escape(filename)} is empty in the pulled DSAIT_Notes checkout. '
        f'This exam chapter is kept readable by showing the generated lecture note for {html.escape(fallback_lecture["title"])}.</p>'
        f'<p>Canonical lecture route: #/lectures/{html.escape(fallback_lecture["slug"])}</p>'
        '</section>\n'
        + fallback_lecture['html']
    )
    sections = [{'id': section_id, 'level': 2, 'title': title}] + list(fallback_lecture.get('sections') or [])
    return body, sections, list(fallback_lecture.get('images') or []), int(fallback_lecture.get('formulaCount') or 0), fallback_lecture['slug']


def build_friend_chapters(friend_manifest: dict, lectures: list[dict]) -> list[dict]:
    chapters = []
    questions = friend_manifest['questions']
    lectures_by_friend_slug = {
        lecture['relatedFriendChapter']: lecture
        for lecture in lectures
        if lecture.get('relatedFriendChapter')
    }
    for item in friend_manifest['chapterFiles']:
        slug = item['slug']
        filename = item['filename']
        short = item['shortTitle']
        path = SRC / filename
        text = path.read_text(encoding='utf-8')
        title = next((line.lstrip('#').strip() for line in text.splitlines() if line.startswith('#')), short)
        source_status = 'source-markdown'
        fallback_lecture_slug = None
        if text.strip():
            body, sections, images, formulas = convert_markdown(text, slug)
            formula_count = len(formulas)
        else:
            source_status = 'empty-source-lecture-fallback'
            body, sections, images, formula_count, fallback_lecture_slug = empty_source_fallback(slug, filename, lectures_by_friend_slug.get(slug))
        chapters.append({
            'kind': 'friend-note',
            'slug': slug,
            'shortTitle': short,
            'title': title,
            'sourceFile': filename,
            'sourcePath': public_source_path(filename),
            'sourceStatus': source_status,
            'fallbackLectureSlug': fallback_lecture_slug,
            'html': body,
            'sections': sections,
            'images': images,
            'formulaCount': formula_count,
            'questions': questions[slug],
        })
    return chapters



def study_citations(lecture: dict, concept: dict | None = None, seed_path: str | None = None) -> list[dict]:
    label_suffix = f" · {concept['title']}" if concept else ''
    citations = [
        {
            'label': f"Lecture: {lecture['shortTitle']}{label_suffix}",
            'href': f"#/lectures/{lecture['slug']}",
            'sourcePath': lecture['sourcePath'],
            'kind': 'lecture',
        },
        {
            'label': 'Concept index',
            'href': '#/concepts',
            'sourcePath': lecture['textPath'],
            'kind': 'concept-index',
        },
    ]
    if seed_path:
        citations.append({
            'label': f"Seed material: {Path(seed_path).name}",
            'href': '#/exam',
            'sourcePath': seed_path,
            'kind': 'seed',
        })
    return citations


def concept_flashcards(lecture: dict, concept: dict) -> list[dict]:
    base = {
        'kind': 'flashcard',
        'lectureSlug': lecture['slug'],
        'lectureTitle': lecture['title'],
        'conceptIds': [concept['id']],
        'formulaHtml': concept.get('formulaHtml', ''),
        'citations': study_citations(lecture, concept),
    }
    return [
        {
            **base,
            'id': f"fc-{concept['id']}-definition",
            'type': 'definition',
            'cardType': 'definition',
            'front': f"What must you know about {concept['title']}?",
            'answer': concept['mustKnow'],
            'explanation': concept['coreIdea'],
        },
        {
            **base,
            'id': f"fc-{concept['id']}-mechanism",
            'type': 'mechanism',
            'cardType': 'mechanism',
            'front': f"What is the formula, rule, or mechanism for {concept['title']}?",
            'answer': concept['formulaOrRule'],
            'explanation': concept['coreIdea'] if concept.get('formula') else f"Mechanism to remember: {concept['formulaOrRule']}",
        },
        {
            **base,
            'id': f"fc-{concept['id']}-trap",
            'type': 'trap',
            'cardType': 'common-trap',
            'front': f"What mistake should you avoid for {concept['title']}?",
            'answer': concept['commonTrap'],
            'explanation': f"Exam-safe framing: {concept['examUse']}",
        },
        {
            **base,
            'id': f"fc-{concept['id']}-exam-use",
            'type': 'exam-use',
            'cardType': 'exam-use',
            'front': f"How would you use {concept['title']} in an exam answer?",
            'answer': concept['examUse'],
            'explanation': f"Core idea to include: {concept['coreIdea']}",
        },
        {
            **base,
            'id': f"fc-{concept['id']}-why",
            'type': 'why',
            'cardType': 'why-it-matters',
            'front': f"Why does {concept['title']} matter in the course logic?",
            'answer': concept['coreIdea'],
            'explanation': f"Connect it forward: {concept['examUse']}",
        },
        {
            **base,
            'id': f"fc-{concept['id']}-oral-check",
            'type': 'oral-check',
            'cardType': 'oral-check',
            'front': f"30-second oral check: explain {concept['title']} without notes.",
            'answer': f"Hit these points: {concept['mustKnow']} Formula/rule: {concept['formulaOrRule']}",
            'explanation': f"Then avoid this trap: {concept['commonTrap']}",
        },
    ]


def concept_exam_questions(lecture: dict, concept: dict) -> list[dict]:
    base = {
        'kind': 'exam-question',
        'lectureSlug': lecture['slug'],
        'lectureTitle': lecture['title'],
        'conceptIds': [concept['id']],
        'formulaHtml': concept.get('formulaHtml', ''),
        'citations': study_citations(lecture, concept),
    }
    return [
        {
            **base,
            'id': f"eq-{concept['id']}-definition",
            'type': 'definition',
            'difficulty': 'core',
            'prompt': f"Explain {concept['title']} in exam-ready terms.",
            'answer': concept['coreIdea'],
            'explanation': f"Minimum target: {concept['mustKnow']}",
        },
        {
            **base,
            'id': f"eq-{concept['id']}-mechanism",
            'type': 'mechanism',
            'difficulty': 'core',
            'prompt': f"State the key formula, rule, or mechanism for {concept['title']} and explain why it matters.",
            'answer': concept['formulaOrRule'],
            'explanation': concept['coreIdea'],
        },
        {
            **base,
            'id': f"eq-{concept['id']}-application",
            'type': 'application',
            'difficulty': 'exam',
            'prompt': f"An exam question is testing {concept['title']}. What should your answer include, and what trap should you avoid?",
            'answer': f"Include: {concept['examUse']} Avoid: {concept['commonTrap']}",
            'explanation': f"This connects the concept to the expected exam use while guarding against the common trap. {concept['coreIdea']}",
        },
        {
            **base,
            'id': f"eq-{concept['id']}-proof",
            'type': 'proof',
            'difficulty': 'derivation',
            'prompt': f"Show, derive, or justify the key expression behind {concept['title']}.",
            'answer': concept['formulaOrRule'],
            'explanation': f"Start from the definition, state the needed assumption, and end with the exam use: {concept['examUse']}",
        },
        {
            **base,
            'id': f"eq-{concept['id']}-retention",
            'type': 'mechanism',
            'difficulty': 'closed-book',
            'prompt': f"Closed-book check: explain {concept['title']} using one formula/rule, one intuition, and one common trap.",
            'answer': f"Formula/rule: {concept['formulaOrRule']} Intuition: {concept['coreIdea']} Trap: {concept['commonTrap']}",
            'explanation': f"This mirrors the example-exam pattern: definition, mechanism, proof/intuition, and trap in one answer.",
        },
    ]


def lecture_comparison_question(lecture: dict) -> dict | None:
    concepts = lecture.get('concepts') or []
    if len(concepts) < 2:
        return None
    first, second = concepts[0], concepts[1]
    return {
        'id': f"eq-{lecture['slug']}-comparison-{first['order']}-{second['order']}",
        'kind': 'exam-question',
        'type': 'comparison',
        'difficulty': 'synthesis',
        'lectureSlug': lecture['slug'],
        'lectureTitle': lecture['title'],
        'conceptIds': [first['id'], second['id']],
        'prompt': f"Compare {first['title']} and {second['title']} for {lecture['shortTitle']}.",
        'answer': f"{first['title']}: {first['coreIdea']} {second['title']}: {second['coreIdea']}",
        'explanation': f"A strong comparison names the role of both concepts, then states the practical trade-off: {first['examUse']} {second['examUse']}",
        'formulaHtml': first.get('formulaHtml') or second.get('formulaHtml', ''),
        'citations': study_citations(lecture, first) + study_citations(lecture, second),
    }


def clean_seed_answer(text: str) -> str:
    return re.sub(r'^[-*]\s*', '', text.strip()).strip()


SEED_TOPIC_RULES: list[tuple[str, list[str]]] = [
    ('w3-vae', ['vae', 'elbo', 'reparameterization', 'amortized', 'encoder', 'decoder']),
    ('w2-em-ppca', [' em ', 'e-step', 'm-step', 'gmm', 'ppca', 'probabilistic pca', 'responsibilities']),
    ('w5-ebm', ['energy', 'ebm', 'energy-based', 'pθ(x)=exp(-eθ', 'log-likelihood gradient', 'partition function', 'positive phase', 'negative phase', 'mcmc', 'negative samples', 'contrastive']),
    ('w6-flow-matching', ['continuous normalizing flow', 'continuous normalising flow', 'cnf', 'continuity equation', 'divergence', 'flow matching', 'vector field', 'ode']),
    ('w5-flow', ['normalizing flow', 'normalising flow', ' a flow ', 'flow be preferable', 'change-of-variables', 'change of variables', 'jacobian', 'invertible', 'coupling']),
    ('w6-diffusion', ['diffusion', 'ddpm', 'ddim', 'reverse process', 'forward process', 'noising', 'noise prediction', 'probability-flow ode', 'probability-flow', 'probability flow', 'sde', 'sampler']),
    ('w5-score', ['score function', 'score matching', 'score modeling', 'score model', 'denoising score', 'langevin', 'annealed', 'normalizing constants', 'normalising constants']),
    ('w6-evaluation', ['evaluation', 'one metric', 'multi-objective', 'likelihood and sample quality', 'sample quality', 'fid', 'inception', 'mmd', 'kid', 'precision', 'recall', 'few modes', 'mode collapse', 'overfitting appear', 'overfitting', 'metrics', 'memorizes', 'memorization', 'near-duplicates']),
    ('w7-life', ['conditioning', 'guidance', 'classifier guidance', 'classifier-free', 'protein', 'motif', 'biological', 'domain constraints', 'validity', 'prompts']),
    ('w1-mle-lvm', ['maximum likelihood', 'mle', 'kl', 'marginal likelihood', 'latent variable', 'posterior', 'pθ(x,z)', 'pθ(z|x)']),
    ('w1-intro', ['discriminative', 'model family', 'universal recipe', 'conditional generation', 'distribution is modeled', 'choose between']),
]

SEED_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'between', 'by', 'can', 'does', 'for', 'from', 'give', 'how',
    'in', 'is', 'it', 'model', 'models', 'of', 'one', 'or', 'the', 'this', 'to', 'use', 'what', 'when', 'which',
    'why', 'with', 'would', 'you', 'generative', 'generation', 'generated', 'modeling', 'modelling',
}

MOCK_DETAILED_ANSWERS: dict[int, str] = {
    19: 'Insert q(z|x), rewrite the marginal as an expectation of pθ(x,z)/q(z|x), then apply Jensen. The result is E_q[log pθ(x,z)-log q(z|x)], usually read as reconstruction/data fit minus KL regularization.',
    20: 'For an invertible flow x=gθ(z), use z=fθ(x). The density is pX(x)=pZ(fθ(x))|det J_fθ(x)|, so the log likelihood is base log-density plus log absolute Jacobian determinant.',
    21: 'The gradient has a data term that lowers the energy of observed samples and a model expectation term that raises energy on negative/model samples. The second term is hard because it requires samples from the current model.',
    22: 'DDPM noising can be written x_t=sqrt(alpha_bar_t)x_0+sqrt(1-alpha_bar_t)epsilon. The common objective trains epsilon_theta(x_t,t) to predict the injected noise.',
    23: 'For a CNF with dx/dt=vθ(t,x), log density changes as d log p_t(x_t)/dt = -div vθ(t,x_t). Divergence measures local volume expansion or contraction.',
    24: 'Use Bayes: log p(x|y)=log p(y|x)+log p(x)-log p(y). Since log p(y) is constant in x, the conditional score adds the classifier gradient to the unconditional score.',
    25: 'VAE uses approximate likelihood through an ELBO and flexible latent dimensions; flows give exact likelihood with invertible equal-dimensional transforms. VAEs sample through a decoder but can blur; flows sample/invert exactly but need tractable Jacobians.',
    26: 'EBMs parameterize unnormalized energy and need MCMC for negative samples; score models parameterize ∇x log p(x), avoiding the normalizer in the objective but still requiring iterative sampling.',
    27: 'Both learn score-like denoising directions across noise levels. Diffusion packages this into an explicit forward noising process and learned reverse process; score models emphasize the score field and Langevin/SDE sampling view.',
    28: 'Discrete normalizing flows use invertible layers and tractable Jacobians. CNFs use an ODE and track density with a divergence term. Flow matching trains the velocity field directly, often without simulating the full path during training.',
    29: 'Likelihood measures assigned probability, FID compares feature-space moments, and precision/recall separates fidelity from coverage. A good evaluation answer states the failure mode of each metric.',
    30: 'Classifier guidance uses gradients from a separate classifier/objective during sampling. Classifier-free guidance combines conditional and unconditional model predictions, avoiding a separate classifier but still trading diversity for condition strength.',
    31: 'High likelihood can reward broad density or average-looking reconstructions. Check samples, nearest neighbours, FID/KID, precision/recall, and task-specific metrics rather than trusting likelihood alone.',
    32: 'Use precision/recall-style metrics: high precision with low recall indicates realistic but mode-dropping samples. Diversity checks and nearest-neighbour inspection can support the diagnosis.',
    33: 'The negative phase is failing: the model is not getting useful samples from its current distribution, so the energy landscape is trained against poor negatives.',
    34: 'Exact flow likelihood requires invertibility and tractable Jacobian determinants. Architectures often use coupling/autoregressive structure, which creates memory or compute costs.',
    35: 'Guidance strengthens the conditional/objective signal during sampling. It can improve prompt adherence, but too much guidance can reduce diversity or create artifacts.',
    36: 'Image metrics may only assess perceptual feature similarity. Protein generation also needs geometric, physical, discrete, functional, and domain-validity checks.',
    37: 'Use nearest-neighbour comparisons, train/test split checks, memorization probes, and diversity/recall metrics. The goal is to see whether samples copy training data rather than generalize.',
    38: 'Ask whether exact likelihood is needed, whether latent inference matters, how sampling cost is constrained, whether the density is normalized, how high-dimensional the data is, and which evaluation property matters.',
}


def seed_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r'[a-zA-Z][a-zA-Z0-9_-]{2,}', text.lower()))
    return {token.strip('-_') for token in tokens if token not in SEED_STOP_WORDS}


def concept_search_text(concept: dict) -> str:
    return ' '.join([
        concept.get('title', ''),
        concept.get('mustKnow', ''),
        concept.get('coreIdea', ''),
        concept.get('formulaOrRule', ''),
        concept.get('commonTrap', ''),
        concept.get('examUse', ''),
        ' '.join(concept.get('tags') or []),
        ' '.join(concept.get('checks') or []),
    ]).lower()


def lecture_seed_scores(lectures: list[dict], prompt: str) -> dict[str, int]:
    padded_prompt = f" {prompt.lower()} "
    tokens = seed_tokens(prompt)
    scores = {lecture['slug']: 0 for lecture in lectures if lecture['slug'] not in EXAM_EXCLUDED_LECTURE_SLUGS}
    for slug, phrases in SEED_TOPIC_RULES:
        if slug not in scores:
            continue
        for phrase in phrases:
            if phrase in padded_prompt:
                scores[slug] += 12 + len(phrase.split())
    for lecture in lectures:
        slug = lecture['slug']
        if slug not in scores:
            continue
        lecture_text = ' '.join([
            lecture.get('shortTitle', ''),
            lecture.get('title', ''),
            lecture.get('focus', ''),
            ' '.join(concept_search_text(concept) for concept in lecture.get('concepts') or []),
        ]).lower()
        scores[slug] += sum(1 for token in tokens if token in lecture_text)
    return scores


def concept_seed_score(concept: dict, prompt: str) -> int:
    prompt_text = prompt.lower()
    text = concept_search_text(concept)
    score = sum(2 for token in seed_tokens(prompt) if token in text)
    if any(term in prompt_text for term in ['overfitting', 'near-duplicates', 'memorizes']) and any(term in text for term in ['memorization', 'train-test', 'overfitting']):
        score += 20
    if any(term in prompt_text for term in ['few modes', 'mode collapse', 'sharp images']) and 'precision' in text and 'recall' in text:
        score += 20
    for field in [concept.get('title', ''), *(concept.get('tags') or [])]:
        field_text = str(field).lower().replace('-', ' ')
        if field_text and field_text in prompt_text:
            score += 10
    return score


def unique_citations(citations: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for citation in citations:
        key = (citation.get('href', ''), citation.get('sourcePath', ''))
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return unique


def seed_mapping(lectures: list[dict], prompt: str, seed_path: str) -> dict:
    scores = lecture_seed_scores(lectures, prompt)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected_slugs = [slug for slug, score in ranked if score > 0][:3]
    quality = 'keyword' if selected_slugs else 'fallback'
    if not selected_slugs:
        selected_slugs = ['w1-intro']
    by_slug = {lecture['slug']: lecture for lecture in lectures}
    selected_lectures = [by_slug[slug] for slug in selected_slugs if slug in by_slug]
    concept_candidates: list[tuple[int, dict, dict]] = []
    for rank, lecture in enumerate(selected_lectures):
        rank_boost = max(0, len(selected_lectures) - rank) * 5
        for concept in lecture.get('concepts') or []:
            score = concept_seed_score(concept, prompt)
            if score > 0:
                concept_candidates.append((score + rank_boost, lecture, concept))
    primary_slug = selected_lectures[0]['slug']
    concept_candidates.sort(key=lambda item: (item[1]['slug'] == primary_slug, item[0]), reverse=True)
    if not concept_candidates and selected_lectures and selected_lectures[0].get('concepts'):
        concept_candidates.append((1, selected_lectures[0], selected_lectures[0]['concepts'][0]))
    selected_concepts: list[tuple[dict, dict]] = []
    seen_concepts: set[str] = set()
    for _, lecture, concept in concept_candidates:
        if concept['id'] in seen_concepts:
            continue
        seen_concepts.add(concept['id'])
        selected_concepts.append((lecture, concept))
        if len(selected_concepts) >= 5:
            break
    primary_lecture = selected_lectures[0]
    if primary_lecture.get('concepts') and not any(lecture['slug'] == primary_lecture['slug'] for lecture, _ in selected_concepts):
        primary_concept = max(primary_lecture['concepts'], key=lambda concept: concept_seed_score(concept, prompt))
        if primary_concept['id'] not in seen_concepts:
            selected_concepts.insert(0, (primary_lecture, primary_concept))
            selected_concepts = selected_concepts[:5]
    citations: list[dict] = []
    if selected_concepts:
        for lecture, concept in selected_concepts:
            citations.extend(study_citations(lecture, concept, seed_path))
    else:
        citations.extend(study_citations(primary_lecture, None, seed_path))
    return {
        'lecture': primary_lecture,
        'conceptIds': [concept['id'] for _, concept in selected_concepts],
        'citations': unique_citations(citations),
        'quality': quality,
        'reason': ', '.join(selected_slugs),
    }


def parse_mock_answers(answer_key: str) -> dict[int, str]:
    answers: dict[int, str] = {}
    for line in answer_key.splitlines():
        match = re.match(r'^\s*(\d+)(?:-(\d+))?\.\s+(.+)$', line)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or start)
        answer = match.group(3).strip()
        for number in range(start, end + 1):
            answers[number] = MOCK_DETAILED_ANSWERS.get(number, answer)
    answers.update(MOCK_DETAILED_ANSWERS)
    return answers


def practice_seed_questions(lectures: list[dict]) -> list[dict]:
    if not PRACTICE_QUESTIONS.is_file() or not lectures:
        return []
    lines = PRACTICE_QUESTIONS.read_text(encoding='utf-8').splitlines()
    questions: list[dict] = []
    for idx, line in enumerate(lines):
        match = re.match(r'^\s*(\d+)\.\s+\*\*(.+?)\*\*\s*$', line)
        if not match:
            continue
        number = int(match.group(1))
        prompt = match.group(2).strip()
        if any(word in prompt.lower() for word in ['causal', 'intervention', 'counterfactual', 'identifiability']):
            continue
        answer = ''
        for next_line in lines[idx + 1:idx + 5]:
            if re.match(r'^\s*\d+\.\s+', next_line):
                break
            if next_line.strip().startswith('-'):
                answer = clean_seed_answer(next_line)
                break
        if not answer:
            answer = 'Use the related concept cards to build a complete answer.'
        mapping = seed_mapping(lectures, prompt, str(PRACTICE_QUESTIONS))
        lecture = mapping['lecture']
        questions.append({
            'id': f'seed-practice-{number}',
            'kind': 'exam-question',
            'type': 'definition' if number % 3 == 1 else 'mechanism' if number % 3 == 2 else 'application',
            'difficulty': 'seed',
            'lectureSlug': lecture['slug'],
            'lectureTitle': lecture['title'],
            'conceptIds': mapping['conceptIds'],
            'prompt': prompt,
            'answer': answer,
            'explanation': 'This is a local practice question. Use it as extra active recall after the generated concept questions.',
            'formulaHtml': '',
            'citations': mapping['citations'],
            'mappingQuality': mapping['quality'],
            'mappingReason': mapping['reason'],
        })
    return questions


def mock_seed_questions(lectures: list[dict]) -> list[dict]:
    if not MOCK_EXAM.is_file() or not lectures:
        return []
    text = MOCK_EXAM.read_text(encoding='utf-8')
    before_key = text.split('## Answer key', 1)[0]
    answer_key = text.split('## Answer key', 1)[1] if '## Answer key' in text else ''
    answers = parse_mock_answers(answer_key)
    questions: list[dict] = []
    for line in before_key.splitlines():
        match = re.match(r'^\s*(\d+)\.\s+(.+)$', line)
        if not match:
            continue
        number = int(match.group(1))
        prompt = match.group(2).strip()
        if any(word in prompt.lower() for word in ['causal', 'intervention', 'counterfactual', 'identifiability']):
            continue
        qtype = 'definition' if number <= 18 else 'mechanism' if number <= 24 else 'comparison' if number <= 30 else 'application'
        mapping = seed_mapping(lectures, prompt, str(MOCK_EXAM))
        lecture = mapping['lecture']
        questions.append({
            'id': f'seed-mixed-{number}',
            'kind': 'exam-question',
            'type': qtype,
            'difficulty': 'seed',
            'lectureSlug': lecture['slug'],
            'lectureTitle': lecture['title'],
            'conceptIds': mapping['conceptIds'],
            'prompt': prompt,
            'answer': answers.get(number, 'Write a structured answer, then compare it against the cited course notes.'),
            'explanation': 'This prompt comes from the local mixed practice set and is included to match the existing exercise style.',
            'formulaHtml': '',
            'citations': mapping['citations'],
            'mappingQuality': mapping['quality'],
            'mappingReason': mapping['reason'],
        })
    return questions


def concept_lookup(lectures: list[dict]) -> dict[str, tuple[dict, dict]]:
    return {
        concept['id']: (lecture, concept)
        for lecture in lectures
        for concept in (lecture.get('concepts') or [])
    }


def citations_for_concept_ids(lectures: list[dict], concept_ids: list[str], seed_path: str) -> list[dict]:
    lookup = concept_lookup(lectures)
    citations: list[dict] = []
    for concept_id in concept_ids:
        if concept_id in lookup:
            lecture, concept = lookup[concept_id]
            citations.extend(study_citations(lecture, concept, seed_path))
    if not citations and lectures:
        citations.extend(study_citations(lectures[0], None, seed_path))
    return unique_citations(citations)


def markdown_block(content: str, marker: str, next_markers: list[str]) -> str:
    start = content.find(marker)
    if start < 0:
        return ''
    start += len(marker)
    end_candidates = [content.find(next_marker, start) for next_marker in next_markers if content.find(next_marker, start) >= 0]
    end = min(end_candidates) if end_candidates else len(content)
    return content[start:end].strip()


def classify_example_part(part_id: str, prompt: str) -> str:
    lower = prompt.lower()
    if 'select all' in lower or 'choose all' in lower or 'which of the following' in lower:
        return 'mcq'
    if any(word in lower for word in ['derive', 'prove', 'show that', 'deduction']):
        return 'proof'
    if any(word in lower for word in ['formula', 'expression', 'write down']):
        return 'mechanism'
    if any(word in lower for word in ['shortcoming', 'impact', 'effect', 'interpretation']):
        return 'application'
    return 'definition' if part_id.endswith(('a', 'c', 'd', 'e')) else 'mechanism'


def parse_example_exam(lectures: list[dict]) -> dict:
    if not EXAMPLE_EXAM.is_file():
        return {'sourcePath': str(EXAMPLE_EXAM), 'title': 'Example exam: 30 June digital test', 'sections': []}
    text = EXAMPLE_EXAM.read_text(encoding='utf-8')
    section_matches = list(re.finditer(r'^##\s+Section\s+(\d+)\s+[—-]\s+(.+)$', text, flags=re.M))
    sections: list[dict] = []
    valid_concepts = set(concept_lookup(lectures).keys())
    for section_index, section_match in enumerate(section_matches):
        section_no = section_match.group(1)
        section_title = section_match.group(2).strip()
        section_start = section_match.end()
        section_end = section_matches[section_index + 1].start() if section_index + 1 < len(section_matches) else len(text)
        section_text = text[section_start:section_end]
        part_matches = list(re.finditer(r'^###\s+([0-9]+[a-z])\s+[—-]\s+([0-9.]+p)\s*$', section_text, flags=re.M))
        parts: list[dict] = []
        for part_index, part_match in enumerate(part_matches):
            part_id = part_match.group(1)
            points = part_match.group(2)
            part_start = part_match.end()
            part_end = part_matches[part_index + 1].start() if part_index + 1 < len(part_matches) else len(section_text)
            body = section_text[part_start:part_end]
            prompt = markdown_block(body, '**Prompt**', ['**Answer target**', '**Explanation**', '**Concept hints:**'])
            answer = markdown_block(body, '**Answer target**', ['**Explanation**', '**Concept hints:**'])
            explanation = markdown_block(body, '**Explanation**', ['**Concept hints:**'])
            hints_match = re.search(r'\*\*Concept hints:\*\*\s*(.+)', body)
            concept_ids = [hint.strip() for hint in (hints_match.group(1).split(',') if hints_match else []) if hint.strip() in valid_concepts]
            prompt_html = markdown_fragment_html(prompt, f'example-{part_id}-prompt')
            answer_html = markdown_fragment_html(answer, f'example-{part_id}-answer')
            qtype = classify_example_part(part_id, prompt)
            parts.append({
                'partId': part_id,
                'points': points,
                'type': qtype,
                'prompt': strip_markdown_for_data(prompt),
                'promptHtml': prompt_html,
                'answer': strip_markdown_for_data(answer),
                'answerHtml': answer_html,
                'explanation': strip_markdown_for_data(explanation),
                'conceptIds': concept_ids,
                'citations': citations_for_concept_ids(lectures, concept_ids, str(EXAMPLE_EXAM)),
            })
        sections.append({
            'sectionId': section_no,
            'title': section_title,
            'parts': parts,
        })
    return {
        'sourcePath': str(EXAMPLE_EXAM),
        'title': 'Example exam: 30 June digital test',
        'sections': sections,
        'subpartCount': sum(len(section['parts']) for section in sections),
    }


def strip_markdown_for_data(text: str) -> str:
    clean = re.sub(r'\$\$(.*?)\$\$', lambda match: match.group(1), text, flags=re.S)
    clean = re.sub(r'`([^`]+)`', r'\1', clean)
    clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def example_exam_questions(example_exam: dict) -> list[dict]:
    questions: list[dict] = []
    for section in example_exam.get('sections') or []:
        for part in section.get('parts') or []:
            questions.append({
                'id': f"example-{part['partId']}",
                'kind': 'exam-question',
                'type': part.get('type', 'definition'),
                'difficulty': 'example-exam',
                'lectureSlug': '',
                'lectureTitle': f"Example exam · {section['title']}",
                'conceptIds': part.get('conceptIds') or [],
                'prompt': part['prompt'],
                'promptHtml': part.get('promptHtml', ''),
                'answer': part['answer'],
                'answerHtml': part.get('answerHtml', ''),
                'explanation': part.get('explanation') or 'Worked answer from the structured example exam.',
                'formulaHtml': '',
                'citations': part.get('citations') or [],
                'sourcePartId': part['partId'],
            })
    return questions


def example_exam_flashcards(example_exam: dict) -> list[dict]:
    cards: list[dict] = []
    for section in example_exam.get('sections') or []:
        for part in section.get('parts') or []:
            cards.append({
                'id': f"fc-example-{part['partId']}",
                'kind': 'flashcard',
                'type': 'example-exam',
                'cardType': 'example-exam',
                'lectureSlug': '',
                'lectureTitle': f"Example exam · {section['title']}",
                'conceptIds': part.get('conceptIds') or [],
                'front': f"Example {part['partId']} ({section['title']}): {part['prompt']}",
                'answer': part['answer'],
                'answerHtml': part.get('answerHtml', ''),
                'explanation': part.get('explanation') or 'This card follows the exact example-exam wording.',
                'formulaHtml': '',
                'citations': part.get('citations') or [],
            })
    return cards


def generated_part_type(order: int, concept: dict) -> str:
    if concept.get('formula'):
        return 'proof' if order % 2 else 'mechanism'
    return ['definition', 'application', 'comparison', 'mechanism'][order % 4]


def generated_prompt(lecture: dict, concept: dict, qtype: str) -> str:
    if qtype == 'proof':
        return f"We study {lecture['shortTitle']}. Derive or justify the formula/rule for {concept['title']}, and explain the intuition behind each term."
    if qtype == 'mechanism':
        return f"For {lecture['shortTitle']}, state the mechanism behind {concept['title']} and explain how it is used in training or sampling."
    if qtype == 'comparison':
        return f"Place {concept['title']} in the course map. What does it solve, what does it not solve, and which neighbouring model family should you compare it with?"
    if qtype == 'application':
        return f"An exam subquestion targets {concept['title']}. What should a complete answer include, and what common mistake should it avoid?"
    return f"Define {concept['title']} in exam-ready terms and state why it matters."


def generated_answer(concept: dict, qtype: str) -> str:
    if qtype == 'proof':
        return f"Formula/rule: {concept['formulaOrRule']} Explanation: {concept['coreIdea']} Exam use: {concept['examUse']}"
    if qtype == 'comparison':
        return f"It solves/represents: {concept['coreIdea']} It should be used as: {concept['examUse']} Do not confuse it with: {concept['commonTrap']}"
    return f"{concept['coreIdea']} Key rule: {concept['formulaOrRule']} Trap: {concept['commonTrap']}"


def build_generated_exam_sets(lectures: list[dict]) -> list[dict]:
    sets: list[dict] = []
    for lecture in lectures:
        if lecture['slug'] in EXAM_EXCLUDED_LECTURE_SLUGS:
            continue
        parts: list[dict] = []
        for index, concept in enumerate(lecture.get('concepts') or [], start=1):
            suffix = chr(ord('a') + ((index - 1) % 26))
            part_id = f"{lecture['order']}{suffix}"
            qtype = generated_part_type(index, concept)
            prompt = generated_prompt(lecture, concept, qtype)
            answer = generated_answer(concept, qtype)
            prompt_html = markdown_fragment_html(prompt, f'generated-{lecture["slug"]}-{part_id}-prompt')
            answer_html = markdown_fragment_html(answer, f'generated-{lecture["slug"]}-{part_id}-answer')
            parts.append({
                'partId': part_id,
                'points': '1.0p',
                'type': qtype,
                'conceptIds': [concept['id']],
                'prompt': prompt,
                'promptHtml': prompt_html,
                'answer': answer,
                'answerHtml': answer_html,
                'explanation': f"Generated from the example-exam pattern for {lecture['shortTitle']}: prompt → derivation/mechanism → trap → repair link.",
                'citations': study_citations(lecture, concept, str(EXAMPLE_EXAM)),
            })
        sets.append({
            'setId': f"generated-{lecture['slug']}",
            'lectureSlug': lecture['slug'],
            'title': f"Lecture {lecture['order']} generated section — {lecture['shortTitle']}",
            'parts': parts,
        })
    return sets


def generated_exam_questions(generated_sets: list[dict]) -> list[dict]:
    questions: list[dict] = []
    for exam_set in generated_sets:
        for part in exam_set.get('parts') or []:
            questions.append({
                'id': f"{exam_set['setId']}-{part['partId']}",
                'kind': 'exam-question',
                'type': part.get('type', 'definition'),
                'difficulty': 'generated-section',
                'lectureSlug': exam_set.get('lectureSlug', ''),
                'lectureTitle': exam_set['title'],
                'conceptIds': part.get('conceptIds') or [],
                'prompt': part['prompt'],
                'promptHtml': part.get('promptHtml', ''),
                'answer': part['answer'],
                'answerHtml': part.get('answerHtml', ''),
                'explanation': part.get('explanation', ''),
                'formulaHtml': '',
                'citations': part.get('citations') or [],
            })
    return questions


def build_speedrun_goals(lectures: list[dict]) -> list[dict]:
    groups = [
        ('speedrun-foundations', 'Foundations + MLE/LVM', ['w1-intro', 'w1-mle-lvm', 'w2-mle-lvm']),
        ('speedrun-em-vae', 'EM, PPCA, VAE', ['w2-em-ppca', 'w3-vae']),
        ('speedrun-energy-flow', 'EBM + Normalising Flows', ['w5-ebm', 'w5-flow']),
        ('speedrun-score-diffusion', 'Score + Diffusion', ['w5-score', 'w6-diffusion']),
        ('speedrun-flow-eval', 'Flow Matching + Evaluation', ['w6-flow-matching', 'w6-evaluation']),
        ('speedrun-life', 'Life Sciences + Guidance', ['w7-life']),
        ('speedrun-final-mix', 'Mixed final review', [lecture['slug'] for lecture in lectures if lecture['slug'] not in EXAM_EXCLUDED_LECTURE_SLUGS]),
    ]
    by_slug = {lecture['slug']: lecture for lecture in lectures}
    goals = []
    for order, (goal_id, title, slugs) in enumerate(groups, start=1):
        goal_lectures = [by_slug[slug] for slug in slugs if slug in by_slug and slug not in EXAM_EXCLUDED_LECTURE_SLUGS]
        concept_ids = [concept['id'] for lecture in goal_lectures for concept in lecture.get('concepts', [])]
        goals.append({
            'id': goal_id,
            'order': order,
            'title': title,
            'lectureSlugs': [lecture['slug'] for lecture in goal_lectures],
            'conceptIds': concept_ids,
            'instructions': 'Read the concept, flip its flashcards, answer one exam prompt, then mark it known, weak, or redo.',
        })
    return goals



def latex_math(expr: str) -> str:
    return render_math(expr, display=True)


CHEATSHEET_TOPICS = [
    {
        'id': 'cs-density-mle',
        'title': '1. Density, KL, MLE',
        'family': 'foundations',
        'slugs': ['w1-intro', 'w1-mle-lvm', 'w2-mle-lvm'],
        'formula_latex': [
            r'\mathrm{KL}(p_{\mathrm{data}}\|p_\theta)=\mathbb{E}_{p_{\mathrm{data}}}[\log p_{\mathrm{data}}(x)-\log p_\theta(x)]',
            r'\theta^*=\arg\max_\theta\frac{1}{N}\sum_{i=1}^N\log p_\theta(x_i)',
            r'p_\theta(x)=\int p_\theta(x,z)\,dz\quad\text{or}\quad \sum_z p_\theta(x,z)',
        ],
        'bullets': [
            'Why MLE: minimizing KL(pdata||pθ) differs only by fixed data entropy, so maximize expected log pθ(x).',
            'Latent model: specify easy joint pθ(x,z); likelihood needs summing/integrating all latent explanations z.',
            'Hardness: posterior pθ(z|x)=pθ(x,z)/pθ(x) needs the same marginal; exact inference is rare.',
            'Exam intuition: if z explains x, learning must reward both good reconstruction and plausible latent mass.',
        ],
    },
    {
        'id': 'cs-em-ppca',
        'title': '2. EM and PPCA',
        'family': 'em ppca',
        'slugs': ['w2-em-ppca'],
        'formula_latex': [
            r'\mathcal{L}(q,\theta)=\mathbb{E}_{q(z)}[\log p_\theta(x,z)]+H(q)',
            r'\mathrm{E\!:\ }q(z)\leftarrow p_{\theta_{old}}(z|x)\qquad \mathrm{M\!:\ }\theta\leftarrow\arg\max_\theta\mathbb{E}_{q(z)}[\log p_\theta(x,z)]',
            r'z\sim\mathcal{N}(0,I),\quad x|z\sim\mathcal{N}(Wz+\mu,\sigma^2I),\quad x\sim\mathcal{N}(\mu,WW^\top+\sigma^2I)',
        ],
        'bullets': [
            'EM reasoning: E-step makes the bound tight at θold; M-step improves θ under that posterior weighting.',
            'ELBO view: q handles uncertainty over z; entropy H(q) prevents choosing one latent explanation too cheaply.',
            'PPCA intuition: Wz creates correlated low-dimensional structure; σ²I is leftover isotropic noise.',
            'Limitation: EM/PPCA are clean because Gaussian/linear; nonlinear decoders break closed-form updates.',
        ],
    },
    {
        'id': 'cs-vae',
        'title': '3. VAE / amortized VI',
        'family': 'vae',
        'slugs': ['w3-vae'],
        'formula_latex': [
            r'\log p_\theta(x)=\log\int p_\theta(x,z)\,dz\ge\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x,z)-\log q_\phi(z|x)]',
            r'\mathcal{L}(\theta,\phi;x)=\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]-\mathrm{KL}(q_\phi(z|x)\|p(z))',
            r'\log p_\theta(x)=\mathcal{L}(\theta,\phi;x)+\mathrm{KL}(q_\phi(z|x)\|p_\theta(z|x))',
            r'z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)',
        ],
        'bullets': [
            'Cannot directly optimize log pθ(x) because the marginal integral and true posterior are intractable for nonlinear decoders.',
            'Lower-bound proof: insert q(z|x), rewrite as expectation, apply Jensen; gap is KL(qφ(z|x)||pθ(z|x)) ≥ 0.',
            'Two terms: reconstruction/data-fit makes x likely given z; KL-to-prior regularizes latent space for sampling.',
            'Amortized VI: encoder network predicts qφ parameters per x instead of optimizing a separate q for every datapoint.',
            'Reparameterization: z=μ+σ⊙ε makes stochastic sampling differentiable w.r.t. μ,σ,φ.',
        ],
    },
    {
        'id': 'cs-ebm',
        'title': '4. Energy-based models',
        'family': 'energy',
        'slugs': ['w5-ebm'],
        'formula_latex': [
            r'p_\theta(x)=\frac{\exp(-E_\theta(x))}{Z_\theta},\qquad Z_\theta=\int\exp(-E_\theta(x))\,dx',
            r'\log p_\theta(x)=-E_\theta(x)-\log Z_\theta',
            r'\nabla_\theta\log p_\theta(x)=-\nabla_\theta E_\theta(x)+\mathbb{E}_{x^{\prime}\sim p_\theta}[\nabla_\theta E_\theta(x^{\prime})]',
        ],
        'bullets': [
            'PDF proof: exp(-Eθ(x)) > 0; Zθ is the integral of exp(-Eθ), so dividing by Zθ makes total mass 1.',
            'Flexibility: Eθ can be almost any neural scoring function; no need for normalized directed likelihood architecture.',
            'Gradient derivation: ∇logp=-∇E-∇logZ and ∇logZ = -Epθ[∇E], giving data phase minus model phase.',
            'Qualitative terms: lower data energies; raise energies of samples the model currently believes too much.',
            'MCMC need: model expectation over pθ is intractable; simulate chain with stationary distribution pθ. Higher energy = lower probability.',
        ],
    },
    {
        'id': 'cs-flows',
        'title': '5. Normalising flows',
        'family': 'flow',
        'slugs': ['w5-flow'],
        'formula_latex': [
            r'x=g_\theta(z),\quad z=f_\theta(x)=g_\theta^{-1}(x)',
            r'p_X(x)=p_Z(f_\theta(x))\left|\det J_{f_\theta}(x)\right|',
            r'\log p_X(x)=\log p_Z(f_\theta(x))+\log\left|\det J_{f_\theta}(x)\right|',
        ],
        'bullets': [
            'Validity: invertible map preserves probability mass; determinant corrects local volume change.',
            'Exact likelihood: evaluate base density at inverse fθ(x) plus log|detJ|, no variational lower bound needed.',
            'Trade-off: expressive transforms are expensive; tractable triangular/coupling designs constrain architecture.',
            'Exam trap: forgetting the log determinant means ignoring whether the transform stretches or compresses space.',
        ],
    },
    {
        'id': 'cs-score',
        'title': '6. Score-based models',
        'family': 'score',
        'slugs': ['w5-score'],
        'formula_latex': [
            r's(x)=\nabla_x\log p(x)',
            r'p(x)=\frac{\exp(-E(x))}{Z}\Rightarrow \nabla_x\log p(x)=-\nabla_xE(x)',
            r'\min_\theta\mathbb{E}_{p_{data}}\left\|s_\theta(x)-\nabla_x\log p_{data}(x)\right\|^2',
            r'x_{t+1}=x_t+\eta s_\theta(x_t)+\sqrt{2\eta}\epsilon_t',
        ],
        'bullets': [
            'Direct density modelling needs nonnegative normalized pθ; normalization is the hard part.',
            'Score benefit: ∇x log p(x) ignores constants like Z and gives direction toward higher density.',
            'Score matching terms: norm penalizes large/uncontrolled scores; divergence term prevents trivial zero field and matches data geometry.',
            'Finite data issue: empirical distribution is spikes/deltas, so its score is useless between samples.',
            'Fix: add noise and learn noise-conditional/denoising scores; Langevin adds noise to sample the distribution, not just ascend one mode.',
        ],
    },
    {
        'id': 'cs-diffusion',
        'title': '7. Diffusion models',
        'family': 'diffusion',
        'slugs': ['w6-diffusion'],
        'formula_latex': [
            r'q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_tI)',
            r'q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)',
            r'x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon',
            r'\mathbb{E}_{x_0,\epsilon,t}\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2',
        ],
        'bullets': [
            'Forward noising is fixed and tractable: every xt is a noisy version of x0 with known Gaussian form.',
            'Learning target εθ: if the model predicts added noise, it also knows how to denoise one reverse step.',
            'Sampling intuition: start from pure noise and repeatedly remove predicted noise while preserving uncertainty.',
            'Relation to score: denoising/noise prediction estimates the direction toward higher noisy-data density.',
            'Reason for stochasticity: avoids deterministic collapse and represents the whole reverse distribution.',
        ],
    },
    {
        'id': 'cs-cnf-fm',
        'title': '8. Continuous flows / Flow Matching',
        'family': 'flow-matching',
        'slugs': ['w6-flow-matching'],
        'formula_latex': [
            r'\frac{dx_t}{dt}=v_\theta(t,x_t)',
            r'\frac{d\log p_t(x_t)}{dt}=-\operatorname{div}v_\theta(t,x_t)',
            r'\min_\theta\mathbb{E}_{t,x_t}\left\|v_\theta(t,x_t)-u_t(x_t)\right\|^2',
        ],
        'bullets': [
            'CNF intuition: particles move continuously; density decreases where velocity field expands volume.',
            'Divergence is the continuous analogue of log determinant in normalizing flows.',
            'Flow matching target: choose an interpolation path, then train vθ to match the true velocity u_t at sampled points.',
            'Reason it helps: supervised velocity regression avoids likelihood ODE simulation inside each training objective.',
        ],
    },
    {
        'id': 'cs-guidance',
        'title': '9. Guidance / conditioning',
        'family': 'guidance',
        'slugs': ['w7-life'],
        'formula_latex': [
            r'\nabla_x\log p(x|y)=\nabla_x\log p(x)+\nabla_x\log p(y|x)',
            r'\epsilon_{guided}=\epsilon_{uncond}+w(\epsilon_{cond}-\epsilon_{uncond})',
        ],
        'bullets': [
            'Classifier guidance follows Bayes: conditional score = prior/sample score + gradient making y more likely.',
            'Classifier-free guidance approximates the same push by subtracting unconditional from conditional predictions.',
            'Guidance scale w is a trade-off: stronger condition adherence can reduce diversity or realism.',
            'For scientific generation, constraints and physical validity are part of the model objective, not decoration.',
        ],
    },
    {
        'id': 'cs-eval',
        'title': '10. Evaluation of generative models',
        'family': 'evaluation',
        'slugs': ['w6-evaluation'],
        'formula_latex': [
            r'\mathrm{IS}=\exp\left(\mathbb{E}_x\mathrm{KL}(p(y|x)\|p(y))\right)',
            r'\mathrm{FID}=\|\mu_r-\mu_g\|^2+\operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})',
            r'\mathrm{MMD}^2(P,Q)=\mathbb{E}k(x,x^{\prime})+\mathbb{E}k(y,y^{\prime})-2\mathbb{E}k(x,y)',
        ],
        'bullets': [
            'No single metric proves generative quality: likelihood, fidelity, diversity, and memorization can disagree.',
            'FID intuition: compare Gaussian moments of real/generated features; lower means closer feature distribution.',
            'Precision vs recall: precision = samples look real/on manifold; recall = model covers data modes.',
            'IS rewards confident diverse classifier labels but can miss memorization and domain mismatch.',
            'Always state metric failure mode: feature extractor bias, finite-sample noise, mode dropping, copying.',
        ],
    },
    {
        'id': 'cs-life',
        'title': '11. Generative models for life sciences',
        'family': 'life sciences',
        'slugs': ['w7-life'],
        'formula_latex': [],
        'bullets': [
            'Scientific generative modelling is constrained generation: samples must be valid under chemistry/biology rules.',
            'Novelty alone is not enough; check usefulness, stability, toxicity, synthesizability, and uncertainty.',
            'Conditional generation/property optimization creates a quality-diversity trade-off like guidance.',
            'Main risk: model exploits proxy metrics or memorizes known structures instead of discovering useful candidates.',
        ],
    },
]


def parse_formula_sections() -> list[dict]:
    if not FORMULA_SHEET.is_file():
        return []
    text = FORMULA_SHEET.read_text(encoding='utf-8')
    matches = list(re.finditer(r'^##\s+(.+)$', text, flags=re.M))
    sections: list[dict] = []
    for index, match in enumerate(matches, start=1):
        start = match.end()
        end = matches[index].start() if index < len(matches) else len(text)
        title = match.group(1).strip()
        body = text[start:end].strip()
        code_blocks = [re.sub(r'\s+', ' ', block.strip()) for block in re.findall(r'```text\s*(.*?)```', body, flags=re.S)]
        bullets = [line.strip('- ').strip() for line in body.splitlines() if line.strip().startswith('-')]
        compact_parts = code_blocks[:2] or bullets[:2] or [strip_markdown_for_data(body)]
        compact = ' ; '.join(part for part in compact_parts if part)
        sections.append({
            'id': f'formula-{index:02d}',
            'order': index,
            'title': re.sub(r'^\d+\.\s*', '', title),
            'compact': compact[:420],
        })
    return sections


def concepts_for_slugs(lectures: list[dict], slugs: list[str]) -> list[dict]:
    allowed = set(slugs)
    return [
        {**concept, 'lectureSlug': lecture['slug'], 'lectureTitle': lecture['shortTitle']}
        for lecture in lectures
        if lecture['slug'] in allowed
        for concept in (lecture.get('concepts') or [])
    ]


def master_formula_ids(start: int, end: int) -> list[str]:
    return [f'formula-{index:02d}' for index in range(start, end + 1)]


CHEATSHEET_FORMULA_IDS = {
    'cs-density-mle': master_formula_ids(1, 2),
    'cs-em-ppca': master_formula_ids(4, 5),
    'cs-vae': master_formula_ids(3, 6),
    'cs-ebm': master_formula_ids(7, 8),
    'cs-flows': master_formula_ids(9, 9),
    'cs-score': master_formula_ids(10, 11),
    'cs-diffusion': master_formula_ids(12, 14),
    'cs-cnf-fm': master_formula_ids(15, 16),
    'cs-guidance': master_formula_ids(17, 18),
    'cs-eval': master_formula_ids(19, 19),
    'cs-life': [],
}


def build_cheatsheet_cell(topic: dict, lectures: list[dict]) -> dict:
    concepts = concepts_for_slugs(lectures, topic['slugs'])
    latex_items = topic.get('formula_latex') or []
    return {
        'id': topic['id'],
        'title': topic['title'],
        'kind': 'cheatsheet-topic',
        'family': topic['family'],
        'body': '',
        'bullets': topic['bullets'],
        'compact': ' · '.join(concept['title'] for concept in concepts[:6]),
        'conceptIds': [concept['id'] for concept in concepts],
        'lectureSlugs': sorted({concept.get('lectureSlug', '') for concept in concepts if concept.get('lectureSlug')}),
        'formulaIds': CHEATSHEET_FORMULA_IDS.get(topic['id'], []),
        'formulaLatex': latex_items,
        'formulaHtml': ''.join(latex_math(expr) for expr in latex_items),
        'formulaSnippets': [],
    }


def build_master_sheet(lectures: list[dict]) -> dict:
    exam_lectures = [lecture for lecture in lectures if lecture['slug'] not in EXAM_EXCLUDED_LECTURE_SLUGS]
    formula_sections = parse_formula_sections()
    cells = [build_cheatsheet_cell(topic, exam_lectures) for topic in CHEATSHEET_TOPICS]
    all_required_concepts = [concept for lecture in exam_lectures for concept in (lecture.get('concepts') or [])]
    covered_concepts = {concept_id for cell in cells for concept_id in cell.get('conceptIds', [])}
    covered_formulas = {formula_id for cell in cells for formula_id in cell.get('formulaIds', [])}
    side_a_cells = cells[:6]
    side_b_cells = cells[6:]
    return {
        'version': 2,
        'format': 'a4-double-sided',
        'density': 'dense-exam-cheatsheet',
        'sourcePaths': [str(FORMULA_SHEET), str(LECTURE_NOTES_BLUEPRINTS), str(CONCEPT_BLUEPRINTS)],
        'coverageFamilies': ['foundations', 'vae', 'energy', 'flow', 'score', 'diffusion', 'flow-matching', 'evaluation', 'guidance', 'life sciences'],
        'contentBudget': {
            'maxCellsPerSide': 6,
            'maxCharsPerCell': 1400,
            'maxCharsPerSide': 7200,
            'targetPrintPages': 2,
            'pageSize': 'A4',
        },
        'coverage': {
            'requiredConceptCount': len(all_required_concepts),
            'coveredConceptCount': len(covered_concepts),
            'formulaSectionCount': len(formula_sections),
            'coveredFormulaCount': len(covered_formulas),
            'questionArchetypes': ['definition', 'derive/prove', 'interpret terms', 'limitations', 'sampling/training procedure', 'multiple choice'],
        },
        'sides': [
            {
                'side': 'A',
                'title': 'Side A — core likelihoods and model families',
                'subtitle': 'Lecture-grounded formulas and facts only.',
                'sections': [{'id': 'a-cheatsheet', 'title': 'Formulas + concepts', 'cells': side_a_cells}],
            },
            {
                'side': 'B',
                'title': 'Side B — diffusion, flow matching, guidance, evaluation',
                'subtitle': 'Use as a memory sheet: equation, object meaning, limitation.',
                'sections': [{'id': 'b-cheatsheet', 'title': 'Formulas + concepts', 'cells': side_b_cells}],
            },
        ],
    }

def build_study_bank(lectures: list[dict]) -> dict:
    exam_lectures = [lecture for lecture in lectures if lecture['slug'] not in EXAM_EXCLUDED_LECTURE_SLUGS]
    example_exam = parse_example_exam(exam_lectures)
    generated_sets = build_generated_exam_sets(exam_lectures)
    flashcards: list[dict] = []
    exam_questions: list[dict] = []
    for lecture in exam_lectures:
        for concept in lecture.get('concepts', []):
            flashcards.extend(concept_flashcards(lecture, concept))
            exam_questions.extend(concept_exam_questions(lecture, concept))
        comparison = lecture_comparison_question(lecture)
        if comparison:
            exam_questions.append(comparison)
    exam_questions.extend(practice_seed_questions(exam_lectures))
    exam_questions.extend(mock_seed_questions(exam_lectures))
    exam_questions.extend(example_exam_questions(example_exam))
    exam_questions.extend(generated_exam_questions(generated_sets))
    flashcards.extend(example_exam_flashcards(example_exam))
    return {
        'version': 1,
        'examScope': {
            'excludedLectureSlugs': sorted(EXAM_EXCLUDED_LECTURE_SLUGS),
            'excludedReason': 'Causal generative modelling is optional/non-exam for this study path.',
        },
        'sourceSeeds': [str(PRACTICE_QUESTIONS), str(MOCK_EXAM), str(EXAMPLE_EXAM), str(FLASHCARDS_TSV)],
        'exampleExam': example_exam,
        'generatedExamSets': generated_sets,
        'flashcards': flashcards,
        'examQuestions': exam_questions,
        'speedrunGoals': build_speedrun_goals(lectures),
        'masterSheet': build_master_sheet(lectures),
    }


def build_lectures() -> list[dict]:
    inventory = json.loads(LECTURE_INVENTORY.read_text(encoding='utf-8'))
    lecture_notes = json.loads(LECTURE_NOTES_BLUEPRINTS.read_text(encoding='utf-8'))
    concept_map = json.loads(CONCEPT_BLUEPRINTS.read_text(encoding='utf-8'))
    if len(inventory) != len(lecture_notes):
        raise RuntimeError(f'lecture inventory/notes mismatch: {len(inventory)} inventory items, {len(lecture_notes)} note definitions')
    lectures = []
    for index, (definition, source) in enumerate(zip(lecture_notes, inventory), start=1):
        highlights = source.get('candidate_lines') or []
        concepts = prepare_concepts(definition['slug'], concept_map.get(definition['slug'], []))
        text = definition['notes'].strip() + concept_markdown(concepts) + source_highlight_markdown(highlights)
        body, sections, images, formulas = convert_markdown(text, definition['slug'])
        lectures.append({
            'kind': 'lecture',
            'order': index,
            'slug': definition['slug'],
            'week': definition['week'],
            'shortTitle': definition['shortTitle'],
            'title': definition['title'],
            'focus': definition['focus'],
            'sourceTitle': source['title'],
            'sourcePath': source['path'],
            'textPath': source['text_path'],
            'pages': source.get('pages'),
            'wordCount': source.get('word_count'),
            'sourceHighlightsLabel': 'Source highlights',
            'sourceHighlights': highlights[:8],
            'concepts': concepts,
            'relatedFriendChapter': definition.get('relatedFriendChapter'),
            'html': body,
            'sections': sections,
            'images': images,
            'formulaCount': len(formulas),
            'questions': definition['questions'],
        })
    return lectures


def main() -> None:
    if not SRC.is_dir():
        raise FileNotFoundError(f'friend notes source folder not found: {SRC}')
    ASSET_OUT.mkdir(parents=True, exist_ok=True)
    for old in ASSET_OUT.glob('*.png'):
        old.unlink()

    friend_manifest = json.loads(FRIEND_NOTES_MANIFEST.read_text(encoding='utf-8'))
    lectures = build_lectures()
    chapters = build_friend_chapters(friend_manifest, lectures)

    for name in sorted(used_images):
        copy_attachment(name)

    data = {
        'sourceRootLabel': PUBLIC_SOURCE_ROOT_LABEL,
        'courseInventoryPath': str(LECTURE_INVENTORY),
        'lectureNotesBlueprintPath': str(LECTURE_NOTES_BLUEPRINTS),
        'conceptBlueprintPath': str(CONCEPT_BLUEPRINTS),
        'conceptCoverageContractPath': str(CONCEPT_COVERAGE_CONTRACT),
        'friendNotesManifestPath': str(FRIEND_NOTES_MANIFEST),
        'conceptCoverage': {
            'status': 'covered',
            'contract': str(CONCEPT_COVERAGE_CONTRACT),
            'totalLectures': len(lectures),
            'totalConcepts': sum(len(lecture['concepts']) for lecture in lectures),
            'minimumConceptsPerLecture': min(len(lecture['concepts']) for lecture in lectures) if lectures else 0,
            'auditScript': str(CONCEPT_AUDIT_SCRIPT),
        },
        'lectures': lectures,
        'chapters': chapters,
        'studyBank': build_study_bank(lectures),
        'slides': friend_manifest['slides'],
        'generatedFrom': [item['filename'] for item in friend_manifest['chapterFiles']] + [str(LECTURE_NOTES_BLUEPRINTS), str(CONCEPT_BLUEPRINTS), str(FRIEND_NOTES_MANIFEST)] + [lecture['textPath'] for lecture in lectures],
        'imageCount': len(used_images),
    }
    js = 'window.GM_NOTES = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';\n'
    (OUT / 'notes-data.js').write_text(js, encoding='utf-8')
    print(f'wrote {OUT / "notes-data.js"} lectures={len(lectures)} chapters={len(chapters)} images={len(used_images)}')


if __name__ == '__main__':
    main()
