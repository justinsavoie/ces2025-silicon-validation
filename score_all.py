"""Score every run and baseline once, each model run against the logit baseline with the same profile fields."""
import subprocess
import sys
import pandas as pd
from build import ROOT

RUNS = [f'{model}_{condition}' for model in ('small', 'large') for condition in ('basic', 'rich', 'partisan')]
BASELINES = ['base21_overall_mode', 'base21_group_mode', 'base21_logit_basic', 'base21_logit_rich', 'base21_logit_partisan']


def main():
    rows = []
    for name in BASELINES + RUNS:
        path = ROOT / f'results_{name}.csv'
        if not path.exists():
            print(f'skip {name}: no results')
            continue
        condition = name.rsplit('_', 1)[-1]
        reference = f'results_base21_logit_{condition}.csv' if name in RUNS else 'results_base21_overall_mode.csv'
        subprocess.run([sys.executable, 'score.py', path.name, '--baseline', reference], cwd=ROOT, check=True, capture_output=True)
        headline = pd.read_csv(ROOT / f'scores_{name}_headline.csv').iloc[0].to_dict()
        overall = pd.read_csv(ROOT / f'scores_{name}_overall.csv').set_index('question').TV_overall
        rows.append({'run': name, **headline, **{q.removeprefix('cps25_'): tv for q, tv in overall.items()}})
    table = pd.DataFrame(rows).drop(columns='results')
    table.to_csv(ROOT / 'scores_summary.csv', index=False)
    with pd.option_context('display.width', 200, 'display.float_format', '{:.3f}'.format):
        print(table[['run', 'headline_TV', 'headline_lo', 'headline_hi', 'invalid_rate', 'baseline', 'mean_improvement', 'cells_improved']].to_string(index=False))


if __name__ == '__main__':
    main()
