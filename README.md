# CES 2025 silicon validation

Start with [the two-page plan](plan.pdf). Five regions × three ages = **15 groups**.

- `validation.csv`: 20,180 human respondents; 13 columns; original CES IDs, ten readable answers, Age and Region.
- `results_stupid_baseline.csv`: 1,500 example model-result rows; each question always gets its overall CES modal answer.
- `questions.json`: exact English questions, allowed labels, CES option codes and baseline answers.
- `group_weights.csv`: 150 question–group combinations, human counts and synthetic-row weights.
- `scores_stupid_baseline_*.csv`: baseline results at the overall and subgroup levels.
- `manifest.json`: source location and SHA-256 fingerprint.

Weights are **question-specific CES respondent shares, not official survey weights**. This includes the Territories, which have no official CES weights. `Not asked / missing` is excluded from human denominators; don't-know answers remain. Invalid model answers remain in the denominator as their own category.

## Reproduce

Python with pandas, numpy, scikit-learn and httpx is required. From this directory:

```sh
PY=../opinionbench/.venv/bin/python
$PY build.py        # CES 2025 answer key: validation.csv, questions.json
$PY build21.py      # CES 2021: profiles_2021.csv (1,500), train_2021.csv
$PY baselines.py    # results_base21_{overall_mode,group_mode,logit_*}.csv
$PY run.py --model small --condition basic --key-file ~/path/to/together-key.txt   # or TOGETHER_API_KEY
$PY score.py results_small_basic.csv --baseline results_base21_logit_basic.csv
quarto render plan.qmd --to pdf
```

`run.py` takes `--model small|large`, `--condition basic|rich|partisan`, `--backend mock` for a dry run, and `--per-group 1` for a 15-call smoke test. Raw replies go to `raw/<run>.jsonl` (an interrupted run resumes), parsed answers to `results_<run>.csv`, and settings, tokens and invalid rates to `runs.json`. The prompt is `prompt.txt`, rendered by `survey.py`.

`score.py` writes overall scores with bootstrap 95% CIs, subgroup scores, comparisons with the chosen baseline (default: 2021 overall mode) and a one-row `scores_<run>_headline.csv`. Positive `improvement` means lower TV than the baseline. Results do not claim respondent-level prediction: synthetic IDs have no human counterpart.

`results_stupid_baseline.csv` takes its modes from CES 2025 itself, so it breaks the temporal holdout; it is kept only as a diagnostic. Respondent data and generated CSVs are ignored by Git.
