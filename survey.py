"""Shared prompt rendering and answer parsing for model runs and fine-tuning data."""
import json
import re
import pandas as pd
from build import ROOT, QUESTIONS
from build21 import CONDITIONS

LABELS = {
    'region': 'Region of Canada', 'age_band': 'Age', 'gender': 'Gender', 'province': 'Province or territory',
    'education': 'Highest level of education', 'income': 'Household income (before taxes)',
    'first_language': 'Language(s) learned as a child and still understood', 'religion': 'Religion',
    'born_in_canada': 'Born in Canada', 'party_id': 'Federal party you usually identify with',
    'left_right': 'Left–right self-placement', 'vote_2019': 'Vote in the 2019 federal election',
}
AGREE_STEM = 'How much do you agree or disagree with this statement?'
INVALID = 'Invalid output'


def load_questions():
    items = json.loads((ROOT / 'questions.json').read_text())
    assert [item['variable'] for item in items] == QUESTIONS
    return items


def load_profiles():
    return pd.read_csv(ROOT / 'profiles_2021.csv', dtype=str, keep_default_na=False)


def question_block(items):
    blocks = []
    for number, item in enumerate(items, 1):
        text = item['question']
        if item['variable'].startswith('cps25_pos_'):
            text = f'{AGREE_STEM} "{text}"'
        options = '\n'.join(f'   {code}. {label}' for code, label in item['options'].items())
        blocks.append(f'{number}. {text}\n{options}')
    return '\n\n'.join(blocks)


def messages(profile, condition, items, template=None):
    template = template or (ROOT / 'prompt.txt').read_text()
    system, user = re.match(r'=== system ===\n(.*?)\n=== user ===\n(.*)', template, re.S).groups()
    lines = '\n'.join(f'- {LABELS[field]}: {profile[field]}' for field in CONDITIONS[condition])
    user = user.format(profile=lines, questions=question_block(items), n_questions=len(items)).strip()
    return [{'role': 'system', 'content': system.strip()}, {'role': 'user', 'content': user}]


def parse(text, items):
    """Read 'n: k' lines; anything missing, repeated inconsistently or out of range is invalid for that item only."""
    # If the model reasons despite thinking being off, read only what follows its last closing tag.
    text = re.split(r'</(?:think|tool_call)>', text or '')[-1]
    found = {}
    for number, choice in re.findall(r'^\s*\**\s*(\d{1,2})\s*[:.)\-]\s*\**\s*(\d{1,2})\b', text, re.M):
        found.setdefault(int(number), set()).add(choice)
    answers = {}
    for number, item in enumerate(items, 1):
        choices = found.get(number, set())
        answers[item['variable']] = item['options'][choices.pop()] if len(choices) == 1 and next(iter(choices)) in item['options'] else INVALID
    return answers
