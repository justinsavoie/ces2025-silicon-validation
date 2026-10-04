"""Build CES 2021 profiles (1,500 people) and the 2021 training set used by the baselines."""
import hashlib
import json
import re
import numpy as np
import pandas as pd
from build import ROOT, QUESTIONS, GROUPS, AGES, MISSING

SOURCE21 = ROOT.parent / 'opinionbench-research/data/raw/CES2021/dataverse_files (6)/2021 Canadian Election Study v2.0.dta'
SEED = 2021
PER_GROUP = 100
NOT_STATED = 'Not stated'
REGION_OF_PROVINCE = {2: 'BC', 1: 'AB+SK+MB+Terr', 12: 'AB+SK+MB+Terr', 3: 'AB+SK+MB+Terr', 6: 'AB+SK+MB+Terr', 8: 'AB+SK+MB+Terr',
                      13: 'AB+SK+MB+Terr', 9: 'ON', 11: 'QC', 4: 'ATL', 5: 'ATL', 7: 'ATL', 10: 'ATL'}
REGION_TEXT = {
    'BC': 'British Columbia',
    'AB+SK+MB+Terr': 'the Prairies or the North (Alberta, Saskatchewan, Manitoba, Yukon, Northwest Territories or Nunavut)',
    'ON': 'Ontario',
    'QC': 'Quebec',
    'ATL': 'Atlantic Canada (New Brunswick, Nova Scotia, Prince Edward Island or Newfoundland and Labrador)',
}
AGE_TEXT = {'18-34': '18 to 34', '35-54': '35 to 54', '55+': '55 or older'}
# Profile fields, cumulative across conditions; values are already readable text.
FIELDS = {
    'basic': ['region', 'age_band', 'gender'],
    'rich': ['province', 'education', 'income', 'first_language', 'religion', 'born_in_canada'],
    'partisan': ['party_id', 'left_right', 'vote_2019'],
}
CONDITIONS = {'basic': FIELDS['basic'], 'rich': FIELDS['basic'] + FIELDS['rich'],
              'partisan': FIELDS['basic'] + FIELDS['rich'] + FIELDS['partisan']}
LANGUAGES = {1: 'English', 2: 'French', 3: 'an Indigenous language', 4: 'Arabic', 5: 'Chinese', 6: 'Filipino/Tagalog', 7: 'German',
             8: 'Hindi/Gujarati', 9: 'Italian', 10: 'Korean', 11: 'Punjabi/Urdu', 12: 'Persian/Farsi', 13: 'Russian', 14: 'Spanish',
             15: 'Tamil', 16: 'Vietnamese', 17: 'another language'}
INCOME_BRACKETS = [(0, 'No income'), (30000, '$1 to $30,000'), (60000, '$30,001 to $60,000'), (90000, '$60,001 to $90,000'),
                   (110000, '$90,001 to $110,000'), (150000, '$110,001 to $150,000'), (200000, '$150,001 to $200,000'),
                   (np.inf, 'More than $200,000')]
RELIGION = {1: 'No religion', 2: 'Agnostic', 3: 'Buddhist', 4: 'Hindu', 5: 'Jewish', 6: 'Muslim', 7: 'Sikh', 8: 'Anglican', 9: 'Baptist',
            10: 'Catholic', 11: 'Eastern Orthodox', 12: "Jehovah's Witness", 13: 'Lutheran', 14: 'Latter-day Saints', 15: 'Evangelical/Pentecostal',
            16: 'Presbyterian', 17: 'Protestant', 18: 'United Church of Canada', 19: 'Christian Reformed', 20: 'Salvation Army',
            21: 'Mennonite', 22: 'Another religion'}
PARTY = {1: 'Liberal', 2: 'Conservative', 3: 'NDP', 4: 'Bloc Québécois', 5: 'Green', 6: 'Another party', 7: 'No party'}
VOTE = {1: 'Liberal', 2: 'Conservative', 3: 'NDP', 4: 'Bloc Québécois', 5: 'Green', 6: 'Another party'}


def clean(label):
    return re.sub(r'^\d+\.\s*', '', label).replace('’', "'").replace('know/ Prefer', 'know/Prefer')


def income(number, category, labels):
    # Use the reported amount when plausible, otherwise the bracket question asked to those who declined.
    if pd.notna(number) and 0 <= number < 5_000_000:
        return next(text for upper, text in INCOME_BRACKETS if number <= upper)
    if pd.notna(category) and int(category) in range(1, 9):
        return clean(labels[int(category)])
    return NOT_STATED


def first_language(row):
    chosen = [name for code, name in LANGUAGES.items() if row[f'cps21_language_{code}'] == 1]
    return ' and '.join(chosen) if chosen else NOT_STATED


def main():
    raw = pd.read_stata(SOURCE21, convert_categoricals=False)
    labels = pd.io.stata.StataReader(SOURCE21).value_labels()
    questions = json.loads((ROOT / 'questions.json').read_text())
    data = pd.DataFrame({'ces21_id': raw.cps21_ResponseId})
    data['Region'] = raw.cps21_province.map(REGION_OF_PROVINCE)
    data['Age'] = pd.cut(raw.cps21_age, [18, 35, 55, np.inf], right=False, labels=AGES).astype('string')
    if data[['Region', 'Age']].isna().any().any() or not data.ces21_id.is_unique:
        raise ValueError('Missing group fields or duplicate 2021 IDs')
    data['region'] = data.Region.map(REGION_TEXT)
    data['age_band'] = data.Age.map(AGE_TEXT)
    data['gender'] = raw.cps21_genderid.map({1: 'Man', 2: 'Woman', 3: 'Non-binary', 4: 'Another gender'}).fillna(NOT_STATED)
    data['province'] = raw.cps21_province.map({k: v for k, v in labels['cps21_province'].items()})
    data['education'] = raw.cps21_education.map({k: clean(v).replace('/ ', '/') for k, v in labels['cps21_education'].items() if k != 12}).fillna(NOT_STATED)
    data['income'] = [income(n, c, labels['cps21_income_cat']) for n, c in zip(raw.cps21_income_number, raw.cps21_income_cat)]
    data['first_language'] = raw.apply(first_language, axis=1)
    data['religion'] = raw.cps21_religion.map(RELIGION).fillna(NOT_STATED)
    data['born_in_canada'] = raw.cps21_bornin_canada.map({1: 'Yes', 2: 'No'}).fillna(NOT_STATED)
    data['party_id'] = raw.cps21_fed_id.map(PARTY).fillna(NOT_STATED)
    data['left_right'] = raw.cps21_lr_scale_bef_1.where(raw.cps21_lr_scale_bef_1.between(0, 10)).map(
        lambda v: NOT_STATED if pd.isna(v) else f'{int(v)} (0 = left, 10 = right)')
    data['vote_2019'] = raw.cps21_vote_2019.map(VOTE).fillna(NOT_STATED)

    # Answers, mapped to the 2025 option text by code after checking that the labels match.
    for item in questions:
        key21 = item['variable'].replace('cps25_', 'cps21_')
        options21 = {code: clean(label) for code, label in labels[key21].items()}
        options25 = {int(code): label for code, label in item['options'].items()}
        if options21 != options25:
            raise ValueError(f'{key21}: 2021 options differ from 2025: {options21}')
        unknown = set(raw[key21].dropna().unique()) - set(options25)
        if unknown:
            raise ValueError(f'{key21}: unknown codes {unknown}')
        data[item['variable']] = raw[key21].map(options25).fillna(MISSING)

    rng = np.random.default_rng(SEED)
    picks = []
    for number, (region, age) in enumerate(GROUPS, 1):
        pool = data[(data.Region == region) & (data.Age == age)].sort_values('ces21_id')
        chosen = pool.iloc[np.sort(rng.choice(len(pool), PER_GROUP, replace=False))].copy()
        chosen.insert(0, 'synthetic_id', [f's21_g{number:02d}_{i:03d}' for i in range(1, PER_GROUP + 1)])
        picks.append(chosen)
    profiles = pd.concat(picks, ignore_index=True)
    profile_columns = ['synthetic_id', 'ces21_id', 'Region', 'Age', *CONDITIONS['partisan']]
    profiles[profile_columns].to_csv(ROOT / 'profiles_2021.csv', index=False)
    train = data[~data.ces21_id.isin(profiles.ces21_id)]
    train[['ces21_id', 'Region', 'Age', *CONDITIONS['partisan'], *QUESTIONS]].to_csv(ROOT / 'train_2021.csv', index=False)

    manifest = json.loads((ROOT / 'manifest.json').read_text())
    manifest['source_2021'] = str(SOURCE21)
    manifest['source_2021_sha256'] = hashlib.sha256(SOURCE21.read_bytes()).hexdigest()
    manifest['profiles_2021'] = {'rows': len(profiles), 'per_group': PER_GROUP, 'seed': SEED, 'train_rows': len(train)}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Created {len(profiles):,} profiles and {len(train):,} training rows from {len(data):,} CES 2021 respondents.')
    for field in CONDITIONS['partisan']:
        share = (profiles[field] == NOT_STATED).mean()
        if share:
            print(f'  {field}: {share:.1%} not stated')


if __name__ == '__main__':
    main()
