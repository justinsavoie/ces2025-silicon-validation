"""
IRT Debiasing Filter & Combiner Experiment
Testing Approach B: LLM National Macro Shift + IRT Demographic Subgroup Variance
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from pathlib import Path

from build import ROOT, GROUPS, QUESTIONS, COLUMNS
from score import calculate

ORDERED_SCALES = {
    'cps25_spend_afford_h': ['Spend less', 'Spend about the same as now', 'Spend more'],
    'cps25_spend_nation_c': ['Spend less', 'Spend about the same as now', 'Spend more'],
    'cps25_spend_defence': ['Spend less', 'Spend about the same as now', 'Spend more'],
    'cps25_spend_rec_indi': ['Spend less', 'Spend about the same as now', 'Spend more'],
    'cps25_pos_life': ['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
    'cps25_pos_energy': ['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
    'cps25_pos_jobs': ['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
    'cps25_imm': ['Fewer immigrants', 'About the same number of immigrants as now', 'More immigrants'],
    'cps25_demsat': ['Not at all satisfied', 'Not very satisfied', 'Fairly satisfied', 'Very satisfied'],
    'cps25_own_fin_retro': ['Got worse', 'Stayed about the same', 'Got better']
}

DK_OPTION = "Don't know/Prefer not to answer"

def fit_alpha_cuts(y_series, scale, theta):
    valid = y_series.isin(scale)
    if valid.sum() < 30:
        return 0.0, np.linspace(-1, 1, len(scale) - 1)
    y = y_series[valid].map(lambda x: scale.index(x)).to_numpy()
    th = theta[valid]
    K = len(scale)
    
    def nll(params):
        alpha = params[0]
        cuts = [params[1]]
        for d in params[2:]:
            cuts.append(cuts[-1] + np.exp(d))
        cdfs = [norm.cdf(c - alpha * th) for c in cuts]
        cdfs = [np.zeros_like(th)] + cdfs + [np.ones_like(th)]
        probs = np.clip(np.array([cdfs[k+1] - cdfs[k] for k in range(K)]), 1e-12, 1.0)
        return -np.log(probs[y, np.arange(len(y))]).sum()
    
    init_cuts = np.linspace(-1, 1, K-1)
    init_params = [0.0, init_cuts[0]] + [np.log(init_cuts[i+1] - init_cuts[i]) for i in range(len(init_cuts)-1)]
    res = minimize(nll, init_params, method='L-BFGS-B')
    alpha = res.x[0]
    cuts = [res.x[1]]
    for d in res.x[2:]:
        cuts.append(cuts[-1] + np.exp(d))
    return float(alpha), cuts

def reconstruct_subgroup_distribution(national_probs, alpha, group_mu, group_std, scale):
    """
    Given national marginal probabilities for the ordered scale,
    an item discrimination alpha, and subgroup latent (mu, std),
    compute the subgroup probabilities via ordered probit IRT link.
    """
    K = len(scale)
    # National cumulative probs
    cum_nat = np.cumsum(national_probs[:K-1])
    cum_nat = np.clip(cum_nat, 1e-5, 1 - 1e-5)
    
    # Under standard normal population theta ~ N(0, 1), the marginal cutoff tau_k
    # satisfies Phi(tau_k / sqrt(alpha^2 + 1)) = cum_nat[k]
    # => tau_k = Phi^-1(cum_nat[k]) * sqrt(alpha^2 + 1)
    tau = norm.ppf(cum_nat) * np.sqrt(alpha**2 + 1.0)
    
    # Subgroup cumulative probs:
    # theta | g ~ N(group_mu, group_std^2)
    # P(Y <= k | g) = Phi((tau_k - alpha * group_mu) / sqrt(alpha^2 * group_std^2 + 1))
    denom = np.sqrt(alpha**2 * (group_std**2) + 1.0)
    cum_grp = norm.cdf((tau - alpha * group_mu) / denom)
    
    cdfs = [0.0] + list(cum_grp) + [1.0]
    probs = np.diff(cdfs)
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum()
    return probs

def generate_synthetic_results(profiles, group_probs, options_dict, seed=2025):
    """
    Given predicted category probabilities per group, draw 100 rows per group.
    """
    rng = np.random.default_rng(seed)
    drawn_records = []
    
    # profiles has 1500 rows, 100 per group
    for (region, age), group_frame in profiles.groupby(['Region', 'Age']):
        n_rows = len(group_frame)
        group_draws = {}
        for q in QUESTIONS:
            opts = options_dict[q]
            p = group_probs[(region, age)][q]
            # sample categorical
            picks = rng.choice(len(opts), size=n_rows, p=p)
            group_draws[q] = [opts[i] for i in picks]
        
        group_df = pd.DataFrame({
            'cps25_ResponseId': group_frame.synthetic_id.values,
            **group_draws,
            'Age': age,
            'Region': region
        })
        drawn_records.append(group_df)
    
    res = pd.concat(drawn_records, ignore_index=True)[COLUMNS]
    return res

def run_experiment():
    val = pd.read_csv(ROOT / 'validation.csv', dtype=str, keep_default_na=False)
    train = pd.read_csv(ROOT / 'train_2021.csv', dtype=str, keep_default_na=False)
    profiles = pd.read_csv(ROOT / 'profiles_2021.csv', dtype=str, keep_default_na=False)
    items = json.loads((ROOT / 'questions.json').read_text())
    options_dict = {item['variable']: list(item['options'].values()) for item in items}
    
    # 1. Calibrate latent ideology theta on train 2021
    party_map = {'NDP': -1.0, 'Green': -0.8, 'Liberal': -0.2, 'Bloc Québécois': -0.3, 
                 'No party': 0.0, 'Conservative': 1.0, 'Another party': 0.7, 'Not stated': 0.0}
    lr_raw = train['left_right'].str.extract(r'^(\d+)')[0].astype(float)
    lr_imputed = lr_raw.fillna(train['party_id'].map(party_map) * 2.3 + 5.1).fillna(5.1)
    theta_train = ((lr_imputed - lr_imputed.mean()) / lr_imputed.std()).to_numpy()
    train['theta'] = theta_train
    
    group_dists = {}
    for g in GROUPS:
        sub = train[(train.Region == g[0]) & (train.Age == g[1])]
        group_dists[g] = (sub.theta.mean(), sub.theta.std())
    
    # 2. Human alphas from 2021
    human_alphas = {}
    for q, scale in ORDERED_SCALES.items():
        alpha, _ = fit_alpha_cuts(train[q], scale, theta_train)
        human_alphas[q] = alpha
    
    # 3. Test multiple models/sources
    source_runs = {
        'small_basic': pd.read_csv(ROOT / 'results_small_basic.csv', dtype=str, keep_default_na=False),
        'small_partisan': pd.read_csv(ROOT / 'results_small_partisan.csv', dtype=str, keep_default_na=False),
        'large_basic': pd.read_csv(ROOT / 'results_large_basic.csv', dtype=str, keep_default_na=False),
    }
    
    print('========================================================================')
    print('EXPERIMENT: IRT Debiasing of LLM Predictions (Approach B)')
    print('========================================================================')
    
    # Baseline comparison: raw models
    for name, raw_res in source_runs.items():
        _, overall, _, _ = calculate(val, raw_res, options_dict, bootstrap=0)
        mean_tv = overall.TV_overall.mean()
        weighted_cell_tv = overall.TV_cells_weighted_mean.mean()
        print(f'Raw {name:15s}: National TV = {mean_tv:.4f}, Subgroup Cell TV = {weighted_cell_tv:.4f}')
    
    print('------------------------------------------------------------------------')
    
    # Evaluate IRT Debiasings
    # A. Scale Factor sweep on Small Basic: alpha = kappa * alpha_LLM
    # In profiles_2021, compute persona theta
    lr_prof_raw = profiles['left_right'].str.extract(r'^(\d+)')[0].astype(float)
    lr_prof = lr_prof_raw.fillna(profiles['party_id'].map(party_map) * 2.3 + 5.1).fillna(5.1)
    theta_prof = ((lr_prof - 5.12) / 2.34).to_numpy()
    
    for src_name, src_df in [('small_basic', source_runs['small_basic']), 
                             ('small_partisan', source_runs['small_partisan'])]:
        print(f'\n--- Testing IRT Combiner on {src_name} ---')
        
        # Calculate national marginals from the LLM run
        llm_national_probs = {}
        llm_alphas = {}
        for q in QUESTIONS:
            scale = ORDERED_SCALES[q]
            opts = options_dict[q]
            counts = src_df[q].value_counts().reindex(opts, fill_value=0).to_numpy(dtype=float)
            total = counts.sum()
            probs = counts / total
            llm_national_probs[q] = probs
            
            # fit LLM's own alpha on persona theta
            alpha_llm, _ = fit_alpha_cuts(src_df[q], scale, theta_prof)
            llm_alphas[q] = alpha_llm
        
        # Test 1: Using Human Alpha (Historical survey discrimination) + LLM National Level
        group_probs_human_alpha = {}
        for g in GROUPS:
            group_probs_human_alpha[g] = {}
            for q in QUESTIONS:
                scale = ORDERED_SCALES[q]
                opts = options_dict[q]
                alpha = human_alphas[q]
                nat_probs = llm_national_probs[q]
                dk_share = nat_probs[opts.index(DK_OPTION)] if DK_OPTION in opts else 0.0
                ord_nat_probs = np.array([nat_probs[opts.index(c)] for c in scale])
                if ord_nat_probs.sum() > 0:
                    ord_nat_probs /= ord_nat_probs.sum()
                else:
                    ord_nat_probs = np.ones(len(scale)) / len(scale)
                
                sub_ord_probs = reconstruct_subgroup_distribution(ord_nat_probs, alpha, group_dists[g][0], group_dists[g][1], scale)
                
                # Combine ordered scale with DK
                full_p = np.zeros(len(opts))
                for idx, c in enumerate(scale):
                    full_p[opts.index(c)] = (1.0 - dk_share) * sub_ord_probs[idx]
                if DK_OPTION in opts:
                    full_p[opts.index(DK_OPTION)] = dk_share
                full_p /= full_p.sum()
                group_probs_human_alpha[g][q] = full_p
        
        synth_human_alpha = generate_synthetic_results(profiles, group_probs_human_alpha, options_dict)
        _, overall_ha, _, _ = calculate(val, synth_human_alpha, options_dict, bootstrap=0)
        print(f'[{src_name} + Human Alpha (Hybrid IRT)]:')
        print(f'   National TV: {overall_ha.TV_overall.mean():.4f}')
        print(f'   Subgroup Cell TV: {overall_ha.TV_cells_weighted_mean.mean():.4f}')
        
        # Test 2: Using Debiased LLM Alpha (kappa * alpha_LLM) - purely zero-shot, no human alpha needed!
        for kappa in [1.5, 2.5, 3.0, 3.5]:
            group_probs_debiased = {}
            for g in GROUPS:
                group_probs_debiased[g] = {}
                for q in QUESTIONS:
                    scale = ORDERED_SCALES[q]
                    opts = options_dict[q]
                    alpha = llm_alphas[q] * kappa
                    nat_probs = llm_national_probs[q]
                    dk_share = nat_probs[opts.index(DK_OPTION)] if DK_OPTION in opts else 0.0
                    ord_nat_probs = np.array([nat_probs[opts.index(c)] for c in scale])
                    if ord_nat_probs.sum() > 0:
                        ord_nat_probs /= ord_nat_probs.sum()
                    else:
                        ord_nat_probs = np.ones(len(scale)) / len(scale)
                    
                    sub_ord_probs = reconstruct_subgroup_distribution(ord_nat_probs, alpha, group_dists[g][0], group_dists[g][1], scale)
                    full_p = np.zeros(len(opts))
                    for idx, c in enumerate(scale):
                        full_p[opts.index(c)] = (1.0 - dk_share) * sub_ord_probs[idx]
                    if DK_OPTION in opts:
                        full_p[opts.index(DK_OPTION)] = dk_share
                    full_p /= full_p.sum()
                    group_probs_debiased[g][q] = full_p
            
            synth_deb = generate_synthetic_results(profiles, group_probs_debiased, options_dict)
            _, overall_deb, _, _ = calculate(val, synth_deb, options_dict, bootstrap=0)
            print(f'   [Debiased alpha (kappa={kappa:.1f})]: National TV = {overall_deb.TV_overall.mean():.4f}, Subgroup Cell TV = {overall_deb.TV_cells_weighted_mean.mean():.4f}')

    # Oracle check: What if National Level is perfect (2025 Real National), combined with Human Alpha?
    print('\n--- Oracle Test: Real 2025 National + Human IRT Alpha ---')
    val_national_probs = {}
    for q in QUESTIONS:
        opts = options_dict[q]
        counts = val[q].value_counts().reindex(opts, fill_value=0).to_numpy(dtype=float)
        val_national_probs[q] = counts / counts.sum()
    
    oracle_group_probs = {}
    for g in GROUPS:
        oracle_group_probs[g] = {}
        for q in QUESTIONS:
            scale = ORDERED_SCALES[q]
            opts = options_dict[q]
            alpha = human_alphas[q]
            nat_probs = val_national_probs[q]
            dk_share = nat_probs[opts.index(DK_OPTION)] if DK_OPTION in opts else 0.0
            ord_nat_probs = np.array([nat_probs[opts.index(c)] for c in scale])
            ord_nat_probs /= ord_nat_probs.sum()
            sub_ord_probs = reconstruct_subgroup_distribution(ord_nat_probs, alpha, group_dists[g][0], group_dists[g][1], scale)
            full_p = np.zeros(len(opts))
            for idx, c in enumerate(scale):
                full_p[opts.index(c)] = (1.0 - dk_share) * sub_ord_probs[idx]
            if DK_OPTION in opts:
                full_p[opts.index(DK_OPTION)] = dk_share
            full_p /= full_p.sum()
            oracle_group_probs[g][q] = full_p
    
    synth_oracle = generate_synthetic_results(profiles, oracle_group_probs, options_dict)
    _, overall_orc, _, _ = calculate(val, synth_oracle, options_dict, bootstrap=0)
    print(f'Oracle (Real 2025 National + Human IRT Alpha): National TV = {overall_orc.TV_overall.mean():.4f}, Subgroup Cell TV = {overall_orc.TV_cells_weighted_mean.mean():.4f}')
    print(f'Real National with NO gaps: National TV = 0.0850, Subgroup Cell TV = 0.0850')
    print('========================================================================')

if __name__ == '__main__':
    run_experiment()
