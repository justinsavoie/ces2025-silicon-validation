# Project Handoff: CES 2025 Silicon-Sampling Validation

**Date**: October 12, 2026  
**Repository**: `ces2025-silicon-validation`  
**Collaborators**: Justin Savoie & Hubert Cadieux  

---

## 1. Executive Summary & Current State

This project tests whether Large Language Models (LLMs) can reliably simulate Canadian public opinion across 15 demographic subgroups (5 regions $\times$ 3 age cohorts) on 10 policy items from the **2025 Canadian Election Study (CES 2025)** under strict temporal holdouts.

Two parallel research branches have been executed, scored, and fully synthesized:
1. **Justin Savoie (Prompting Branch)**: Evaluated zero-shot hosted models (`Qwen3.5-9B` and `Qwen3.8-2.4T-A95B` on Together API) with 1,500 CES 2021 personas across three information conditions (**Basic**, **Rich**, **Rich + Partisan**).
2. **Hubert Cadieux (Fine-Tuning Branch)**: Fine-tuned `Llama-3.3-70B-Instruct` on Azure AI Foundry using 8 Canadian surveys (2019–2024; 20k/50k/100k samples) comparing Fixed vs. Semantic context formats across 15,000 synthetic respondents. Research note: `~/Downloads/note_recherche_2026-10-08.pdf`.

A complete, standalone 2-page synthesis document comparing both branches has been compiled to PDF at [`synthesis.pdf`](synthesis.pdf) (source: [`synthesis.qmd`](synthesis.qmd)).

---

## 2. Benchmark Scoreboard Summary (Mean Total Variation)

All scoring measures Total Variation (TV) distance against actual CES 2025 subgroup distributions ($0.0 = \text{perfect match}$; lower is better):

| Approach / Paradigm | Model / Configuration | Mean TV | Notes |
|:---|:---|:---:|:---|
| **Human Floor** | CES 2025 split-halves | **0.062** | Measurement error lower bound |
| **National Reference** | Real 2025 national (no subgroup gaps) | **0.085** | 9 / 100 out of place |
| **Statistical Baseline** | **CES 2021 Multinomial Logit (Partisan)** | **0.150** | Savoie baseline; trained on CES 2021 |
| **Historical Survey** | **CES 2021 Subgroups Unchanged** | **0.151** | Cadieux baseline; beats LLMs on 8/10 items |
| **Cadieux (Fine-Tuned)** | **Llama 70B: Fixed context, 50k** | **0.178** | **Best fine-tuned model** (18 / 100 error) |
| Cadieux (Fine-Tuned) | Llama 70B: Semantic context, 50k | 0.188 | Fixed beats Semantic format |
| Cadieux (Fine-Tuned) | Llama 70B: Fixed context, 100k | 0.191 | Slight degradation from 50k |
| Cadieux (Fine-Tuned) | Llama 70B: Fixed context, 20k | 0.200 | Data scaling helps up to 50k |
| **Savoie (Prompted)** | **Qwen 9B: Rich + Partisan** | **0.258** | **Best prompted model**; partisan adds large gain |
| Savoie (Prompted) | Qwen 9B: Rich profile | 0.323 | Basic + demographics |
| Savoie (Prompted) | Qwen 2.4T MoE: Rich + Partisan | 0.323 | Larger model ties smaller model + partisan |
| **Untrained Baseline** | Untrained Llama-3.3-70B (Cadieux) | **0.325** | Raw pretraining prior baseline |
| Savoie (Prompted) | Qwen 9B: Basic demographics | 0.357 | Region, age, gender only |
| Savoie (Prompted) | Qwen 2.4T MoE: Rich | 0.421 | Degrades significantly |
| Savoie (Prompted) | Qwen 2.4T MoE: Basic | 0.477 | Severe mode collapse on neutral options |
| **Modal Baseline** | CES 2021 Group Mode (Savoie) | **0.504** | Most common answer per subgroup in 2021 |
| **Modal Baseline** | CES 2021 / 2025 Overall Mode | **0.540–0.615** | Single national modal answer to all |

---

## 3. Core Substantive Findings

1. **Fine-Tuning roughly halves raw model error**:
   * Untrained models start around TV ~0.325. Prompting with rich partisan profiles lowers this to 0.258. Fine-tuning on domain survey data brings it down to 0.178.
2. **The "Flattening" Problem (Level vs. Differences)**:
   * Models severely compress demographic differences: model subgroup gaps are only **~31% of real size** (correlation 0.42), whereas CES 2021 gaps are **69% of real size** (correlation 0.71).
   * If you give CES 2021 subgroup gaps the real 2025 national figure, TV drops to **0.064**—almost reaching the human floor (0.062).
3. **The Scale Paradox**:
   * In zero-shot prompting, scaling to the 2.4T MoE model worsened accuracy across all conditions (Basic: 0.477 vs 0.357). Larger models aggressively clustered around neutral/status-quo options ("about the same" 78% on immigration and 75% on defence).
4. **Where Models Win vs. Fail**:
   * **Defence Spending (Model Win)**: Opinions shifted sharply post-2021 ($56\%$ want more spending). Models beat the 2021 poll (TV 0.17 vs. 0.31) because post-2021 data (Democracy Checkup up to 2024) informed them.
   * **Immigration (Failure for Both)**: 62% of 2025 Canadians chose "fewer immigrants". Zero-shot models chose it 2–4% of the time. Azure Content Safety flagged and stripped immigration questions from fine-tuning, creating a blind spot.
   * **Indigenous Reconciliation**: Models exhibited strong progressive skew, predicting "spend more" 68–91% of the time vs. 22% of actual Canadians.

---

## 4. Critical Methodological Clarification: Repeated vs. Novel Items

* **Why the logit model existed**: In this first test, the 10 questions were repeated from CES 2021. The multinomial logit served as a sanity check to verify whether LLMs process demographic information as effectively as a simple statistical model. They did not (0.258 vs 0.150).
* **Why the comparison is irrelevant for real silicon sampling**: A logit requires historical labels ($Y$). On **novel, newly emerging, or unpolled questions**, $Y$ does not exist, so **a logit is mathematically impossible to fit**.
* **The Pivot**: Testing silicon sampling on repeated questions gives an artificial advantage to past polls and regressions. The true test of silicon sampling is **novel items with zero past survey data**.

---

## 5. Prioritized Next Steps for the Next Session

1. **Extract Novel CES 2025 Items**:
   * Identify 10–15 policy questions in `validation.csv` / CES 2025 that were **never asked in 2019 or 2021**.
   * Run the prompting and fine-tuning models on this novel benchmark where no past poll or logit baseline exists.
2. **Implement a Hybrid "Historical Gaps + LLM Level" Estimator**:
   * Formulate $\hat{P}_{jgk} = \text{LLM National Shift}_{jk} + \text{Historical Subgroup Delta}_{jgk}$.
   * Leverages the structural stability of survey demographic gaps while using LLMs to detect temporal macro-shifts.
3. **Unfiltered Fine-Tuning Pipeline**:
   * Re-train Llama 70B on unmoderated infrastructure (RunPod / Together / local compute) without Azure Content Safety to include immigration and moral value items.
4. **Fix Decoding / Subgroup Flattening**:
   * Test temperature calibration (< 1.0 vs 1.0), nucleus sampling thresholds, and profile weighting to restore authentic subgroup variance.
5. **Joint Paper with Hubert**:
   * Draft the joint methodological paper: *When Does Silicon Sampling Beat a Past Poll? Evidence from the 2025 Canadian Election Study*.

---

## 6. Key Files in This Repository

* [`synthesis.pdf`](synthesis.pdf): The 2-page compiled PDF synthesis comparing Savoie vs. Cadieux.
* [`synthesis.qmd`](synthesis.qmd): Quarto source file for the 2-pager (renders cleanly via `quarto render synthesis.qmd --to pdf`).
* [`plan.pdf`](plan.pdf) / [`plan.qmd`](plan.qmd): Justin's original 2-page study plan and Together AI setup.
* [`scores_summary.csv`](scores_summary.csv): Full summary scoreboard of TV metrics for all 6 runs and baselines.
* [`run.py`](run.py): Together API inference script for running prompt conditions.
* [`score.py`](score.py) / [`score_all.py`](score_all.py): Total Variation calculation scripts.
* [`build.py`](build.py) / [`build21.py`](build21.py): Preprocessing scripts for CES 2025 answer keys and CES 2021 profiles.
* [`validation.csv`](validation.csv): Ground truth CES 2025 answer key ($N = 20,180$).
* `~/Downloads/note_recherche_2026-10-08.pdf`: Hubert Cadieux's 4-page research report on fine-tuning Llama 3.3 70B.

---

## 7. Quick Reproduction Commands

```bash
# Re-render the 2-page synthesis PDF:
quarto render synthesis.qmd --to pdf

# Re-score all prompted runs and generate scores_summary.csv:
python3 score_all.py

# Inspect summary scoreboard:
cat scores_summary.csv
```
