"""CES 2021 baselines, one answer per profile, scored exactly like a model run."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from build import ROOT, QUESTIONS, COLUMNS, GROUPS, MISSING
from build21 import CONDITIONS, SEED


def output(profiles, answers, name):
    result = pd.DataFrame({'cps25_ResponseId': profiles.synthetic_id, **answers, 'Age': profiles.Age, 'Region': profiles.Region})[COLUMNS]
    assert len(result) == 1500
    result.to_csv(ROOT / f'results_{name}.csv', index=False)


def mode(answers, codes):
    counts = answers.value_counts()
    # Ties go to the smallest CES code.
    return min(counts[counts == counts.max()].index, key=codes.get)


def main():
    read = dict(dtype=str, keep_default_na=False)
    profiles = pd.read_csv(ROOT / 'profiles_2021.csv', **read)
    train = pd.read_csv(ROOT / 'train_2021.csv', **read)
    items = {item['variable']: item for item in json.loads((ROOT / 'questions.json').read_text())}
    codes = {q: {label: int(code) for code, label in items[q]['options'].items()} for q in QUESTIONS}
    answered = {q: train[train[q] != MISSING] for q in QUESTIONS}

    output(profiles, {q: mode(answered[q][q], codes[q]) for q in QUESTIONS}, 'base21_overall_mode')

    group_modes = {}
    for q in QUESTIONS:
        by_group = {(r, a): mode(frame[q], codes[q]) for (r, a), frame in answered[q].groupby(['Region', 'Age'])}
        assert set(by_group) == set(GROUPS)
        group_modes[q] = [by_group[(r, a)] for r, a in zip(profiles.Region, profiles.Age)]
    output(profiles, group_modes, 'base21_group_mode')

    rng = np.random.default_rng(SEED)
    for condition, fields in CONDITIONS.items():
        drawn = {}
        for q in QUESTIONS:
            frame = answered[q]
            encoder = OneHotEncoder(handle_unknown='ignore').fit(frame[fields])
            model = LogisticRegression(C=1.0, max_iter=5000).fit(encoder.transform(frame[fields]), frame[q])
            probabilities = model.predict_proba(encoder.transform(profiles[fields]))
            cumulative = probabilities.cumsum(axis=1)
            picks = (rng.random(len(profiles))[:, None] > cumulative).sum(axis=1).clip(max=len(model.classes_) - 1)
            drawn[q] = model.classes_[picks]
        output(profiles, drawn, f'base21_logit_{condition}')
    print('Wrote results_base21_{overall_mode,group_mode,logit_basic,logit_rich,logit_partisan}.csv')


if __name__ == '__main__':
    main()
