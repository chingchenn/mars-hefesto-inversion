#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:27:06 2026

@author: chingchen

MCMC analysis
====================================
free parameters：T_lit, P_lit, dTdP, Mg#
fixed parameters：Si, Ca, Al, Na, Cr（YM2020）


    python analyze_mcmc_4param.py --mcmc_dir /Users/chingchen/Desktop/HeFESTo/mcmc
    python analyze_mcmc_4param.py --mcmc_dir /path/to/mcmc --burnin 0.3 --prefix chain03
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# parameters 

PARAMS = ['T_lit', 'P_lit', 'dTdP', 'Mg#']

PARAM_LABELS = {
    'T_lit': r'$T_{lit}$ [K]',
    'P_lit': r'$P_{lit}$ [GPa]',
    'dTdP':  r'$dT/dP$ [K/GPa]',
    'Mg#':   r'Mg# = Mg/(Mg+Fe)',
}

PRIOR = {
    'T_lit': (1000.0, 2600.0),
    'P_lit': (1.5,    9.0),
    'dTdP':  (5.0,    20.0),
    'Mg#':   (0.50,   0.86),
}

# YM2020 composition
YM_BASE = {
    'Si': 4.01931,
    'Mg': 4.08235,
    'Fe': 1.08599,
    'Ca': 0.27259,
    'Al': 0.37376,
    'Na': 0.10105,
    'Cr': 0.06146,
}
YM_MGNUM = YM_BASE['Mg'] / (YM_BASE['Mg'] + YM_BASE['Fe'])  # ≈ 0.790

# read_chain

def load_chains(mcmc_dir, prefix='chain'):
    pattern = os.path.join(mcmc_dir, f'{prefix}_*', 'chain.json')
    chain_files = sorted(glob.glob(pattern))

    if not chain_files:
        # no prefix 
        pattern = os.path.join(mcmc_dir, 'chain_*', 'chain.json')
        chain_files = sorted(glob.glob(pattern))

    chains = []
    for fpath in chain_files:
        with open(fpath) as f:
            data = json.load(f)
        if len(data) == 0:
            continue

        # read parameters 
        param_data = {}
        for p in PARAMS:
            param_data[p] = np.array([r['params'][p] for r in data])
            
        chains.append({
            'name':        os.path.basename(os.path.dirname(fpath)),
            'steps':       np.array([r['step'] for r in data]),
            'misfit':      np.array([r['misfit'] for r in data]),
            'accepted':    np.array([r['accepted'] for r in data]),
            'accept_rate': np.array([r['accept_rate'] for r in data]),
            'params':      param_data,
        })

    print(f"read {len(chains)} chain")
    for c in chains:
        print(f"  {c['name']}: {len(c['steps'])} steps, "
              f"final misfit={c['misfit'][-1]:.4f}, "
              f"acc rate={c['accept_rate'][-1]:.1f}%")
    return chains


def apply_burnin(chains, burnin_frac=0.2):
    """drop chain first burnin_frac steps"""
    trimmed = []
    for c in chains:
        n     = len(c['steps'])
        start = int(n * burnin_frac)
        trimmed.append({
            'name':        c['name'],
            'misfit':      c['misfit'][start:],
            'accepted':    c['accepted'][start:],
            'accept_rate': c['accept_rate'][start:],
            'params':      {p: c['params'][p][start:] for p in PARAMS},
        })
    total = sum(len(c['misfit']) for c in trimmed)
    print(f"after Burn-in {burnin_frac*100:.0f}%, total sample {total} ")
    return trimmed


def get_all_samples(trimmed_chains):
    
    samples = {p: np.concatenate([c['params'][p] for c in trimmed_chains])
               for p in PARAMS}
    misfits = np.concatenate([c['misfit'] for c in trimmed_chains])
    return samples, misfits


def autocorr(x, maxlag=100):
    x   = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.zeros(maxlag)
    result = [1.0]
    for lag in range(1, maxlag):
        c = np.mean(x[lag:] * x[:-lag]) / var
        result.append(c)
    return np.array(result)


def effective_sample_size(x):
    n  = len(x)
    ac = autocorr(x, maxlag=min(200, n // 2))
    cutoff = next((i for i, v in enumerate(ac) if v < 0), len(ac))
    tau    = 1 + 2 * np.sum(ac[1:cutoff])
    return n / max(tau, 1)


def gelman_rubin(chains_param):
    m  = len(chains_param)
    if m < 2:
        return np.nan
    n  = min(len(c) for c in chains_param)
    cp = [c[:n] for c in chains_param]
    chain_means = np.array([np.mean(c) for c in cp])
    chain_vars  = np.array([np.var(c, ddof=1) for c in cp])
    grand_mean  = np.mean(chain_means)
    B   = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    W   = np.mean(chain_vars)
    if W == 0:
        return np.nan
    var_hat = (1 - 1/n) * W + B / n
    return np.sqrt(var_hat / W)


# figure 1：Trace plots

def plot_trace(chains, output_dir):
    n_rows = len(PARAMS) + 1  # +1 for misfit
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.8))
    fig.suptitle('Trace Plots', fontsize=14, fontweight='bold')

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(chains), 1)))

    # Misfit
    ax = axes[0, 0]
    for i, c in enumerate(chains):
        ax.plot(c['steps'], c['misfit'], alpha=0.7, lw=0.8, color=colors[i])
    ax.set_ylabel('misfit/datum')
    ax.set_title('Misfit')
    ax.set_xlabel('Step')

    # Acceptance rate
    ax = axes[0, 1]
    for i, c in enumerate(chains):
        ax.plot(c['steps'], c['accept_rate'], alpha=0.7, lw=0.8, color=colors[i])
    ax.axhline(20, color='red',   ls='--', lw=1.2, label='20%')
    ax.axhline(40, color='green', ls='--', lw=1.2, label='40%')
    ax.set_ylabel('Accept rate (%)')
    ax.set_title('Acceptance Rate (best 20-40%)')
    ax.set_xlabel('Step')
    ax.legend(fontsize=9)

    # Parameters
    for idx, p in enumerate(PARAMS):
        row = idx + 1
        # left: full trace
        ax_l = axes[row, 0]
        for i, c in enumerate(chains):
            ax_l.plot(c['steps'], c['params'][p], alpha=0.6, lw=0.7, color=colors[i])
        lo, hi = PRIOR[p]
        ax_l.axhline(lo, color='gray', ls=':', lw=0.8)
        ax_l.axhline(hi, color='gray', ls=':', lw=0.8)
        if p == 'Mg#':
            ax_l.axhline(YM_MGNUM, color='orange', ls='--', lw=1.2, label='YM2020')
            ax_l.legend(fontsize=8)
        ax_l.set_ylabel(PARAM_LABELS[p])
        ax_l.set_title(p)
        ax_l.set_xlabel('Step')

        # right: running mean
        ax_r = axes[row, 1]
        for i, c in enumerate(chains):
            x   = c['params'][p]
            cum = np.cumsum(x) / np.arange(1, len(x) + 1)
            ax_r.plot(c['steps'], cum, alpha=0.7, lw=0.8, color=colors[i])
        ax_r.set_ylabel(f'Running mean')
        ax_r.set_title(f'{p} running mean')
        ax_r.set_xlabel('Step')

    plt.tight_layout()
    out = os.path.join(output_dir, '01_trace_plots.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


# figure 2：Autocorrelation

def plot_autocorr(trimmed_chains, output_dir):
    n_chains = len(trimmed_chains)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('Autocorrelation (Burn-in)',
                 fontsize=13, fontweight='bold')

    c      = trimmed_chains[0]
    maxlag = min(100, len(c['misfit']) // 2)

    for idx, p in enumerate(PARAMS):
        ax = axes[idx // 2][idx % 2]
        ac = autocorr(c['params'][p], maxlag=maxlag)
        ax.bar(range(maxlag), ac, color='steelblue', alpha=0.7, width=1.0)
        ax.axhline(0,    color='black', lw=0.8)
        ax.axhline(0.05, color='red',   ls='--', lw=0.8)
        ess = effective_sample_size(c['params'][p])
        ax.set_title(f"{p}\nESS≈{ess:.0f}")
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')

    plt.tight_layout()
    out = os.path.join(output_dir, '02_autocorrelation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

# figure 3：marginal posterior distributions


def plot_marginal_posteriors(samples, misfits, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('marginal posterior distributions (all chain)',
                 fontsize=13, fontweight='bold')

    for idx, p in enumerate(PARAMS):
        ax  = axes[idx]
        x   = samples[p]
        lo, hi = PRIOR[p]

        ax.hist(x, bins=50, density=True,
                color='steelblue', alpha=0.7, edgecolor='none')

        ax.axvline(lo, color='gray', ls='--', lw=1.2, label='prior')
        ax.axvline(hi, color='gray', ls='--', lw=1.2)

        med    = np.median(x)
        ci_lo  = np.percentile(x, 2.5)
        ci_hi  = np.percentile(x, 97.5)
        ax.axvline(med, color='red', lw=1.8, label=f'median={med:.3f}')
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color='red', label='95% CI')

        # YM2020 refenerce line
        if p == 'Mg#':
            ax.axvline(YM_MGNUM, color='orange', ls='--',
                       lw=1.5, label=f'YM2020={YM_MGNUM:.3f}')

        ax.set_xlabel(PARAM_LABELS[p])
        ax.set_ylabel('Density')
        ax.set_title(f"{p}\n{med:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(output_dir, '03_marginal_posteriors.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


# figure 4：Corner plot（4×4）


def plot_corner(samples, output_dir, max_samples=5000):
    n   = len(PARAMS)
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    fig.suptitle('Corner Plot',
                 fontsize=13, fontweight='bold')

    total = len(samples[PARAMS[0]])
    idx   = np.random.choice(total, min(max_samples, total), replace=False)

    for i, pi in enumerate(PARAMS):
        for j, pj in enumerate(PARAMS):
            ax = axes[i][j]
            xi = samples[pi][idx]
            xj = samples[pj][idx]

            if i == j:
                ax.hist(xi, bins=40, color='steelblue',
                        alpha=0.7, density=True, edgecolor='none')
                ax.set_xlim(PRIOR[pi])
                if pi == 'Mg#':
                    ax.axvline(YM_MGNUM, color='orange',
                               ls='--', lw=1.2, label='YM2020')
            elif i > j:
                ax.hist2d(xj, xi, bins=30, cmap='Blues',
                          range=[PRIOR[pj], PRIOR[pi]])
                r = np.corrcoef(xj, xi)[0, 1]
                color = 'red' if abs(r) > 0.4 else 'black'
                ax.text(0.05, 0.92, f'r={r:.2f}',
                        transform=ax.transAxes, fontsize=8,
                        va='top', color=color, fontweight='bold')
            else:
                ax.set_visible(False)

            if j == 0 and i > 0:
                ax.set_ylabel(PARAM_LABELS[pi], fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(PARAM_LABELS[pj], fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = os.path.join(output_dir, '04_corner_plot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

# figure 5：Misfit vs parameters（sensitivity）

def plot_misfit_vs_params(samples, misfits, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Misfit vs posterior sample',
                 fontsize=13, fontweight='bold')

    for idx, p in enumerate(PARAMS):
        ax = axes[idx]
        ax.scatter(samples[p], misfits, alpha=0.3, s=5,
                   c=misfits, cmap='viridis_r', rasterized=True)
        ax.axhline(1.0, color='red', ls='--', lw=1.2, label='misfit=1')
        ax.set_xlabel(PARAM_LABELS[p])
        ax.set_ylabel('misfit/datum')
        ax.set_title(p)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, '05_misfit_vs_params.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

# results

def print_convergence_report(trimmed_chains):
    print("\n" + "=" * 55)
    print("Gelman-Rubin R-hat")
    print("R-hat < 1.1 → converge；> 1.2 → need more steps")
    print("=" * 55)
    all_ok = True
    for p in PARAMS:
        chains_p = [c['params'][p] for c in trimmed_chains]
        rhat     = gelman_rubin(chains_p)
        status   = "✓" if rhat < 1.1 else ("△" if rhat < 1.2 else "✗")
        if rhat >= 1.1:
            all_ok = False
        print(f"  {status} {p:<8s}: R-hat = {rhat:.4f}")
    print("=" * 55)
    if all_ok:
        print("  → all parameters converge! ")
    else:
        print("  → need more step")
    return all_ok

# ============================================================
# 統計摘要
# ============================================================

def print_summary(samples, misfits):
    print("\n" + "=" * 65)
  
    print(f"{'parameters':<8} {'mdeian':>10} {'average':>10} "
          f"{'std':>8} {'2.5%':>10} {'97.5%':>10}")
    print("=" * 65)
    for p in PARAMS:
        x    = samples[p]
        med  = np.median(x)
        mean = np.mean(x)
        std  = np.std(x)
        lo   = np.percentile(x, 2.5)
        hi   = np.percentile(x, 97.5)
        print(f"{p:<8} {med:>10.4f} {mean:>10.4f} "
              f"{std:>8.4f} {lo:>10.4f} {hi:>10.4f}")
    print("=" * 65)
    print(f"Misfit median:{np.median(misfits):.4f}")
    print(f"Misfit min:{np.min(misfits):.4f}")
    print(f"total smaple: {len(misfits)}")

    print("\n ESS:")
    for p in PARAMS:
        ess = effective_sample_size(samples[p])
        print(f"  {p:<8}: ESS ≈ {ess:.0f}")

    # 換算成物理量
    print("\n posterior from Mg# YM2020: ")
    mgfe_total = YM_BASE['Mg'] + YM_BASE['Fe']
    mg_med = np.median(samples['Mg#']) * mgfe_total
    fe_med = (1 - np.median(samples['Mg#'])) * mgfe_total
    mg_lo  = np.percentile(samples['Mg#'], 2.5)  * mgfe_total
    mg_hi  = np.percentile(samples['Mg#'], 97.5) * mgfe_total
    fe_lo  = (1 - np.percentile(samples['Mg#'], 97.5)) * mgfe_total
    fe_hi  = (1 - np.percentile(samples['Mg#'], 2.5))  * mgfe_total
    print(f"  Mg# median = {np.median(samples['Mg#']):.4f} "
          f"[{np.percentile(samples['Mg#'],2.5):.4f}, "
          f"{np.percentile(samples['Mg#'],97.5):.4f}]")
    print(f"  Mg = {mg_med:.3f} mol [{mg_lo:.3f}, {mg_hi:.3f}]")
    print(f"  Fe = {fe_med:.3f} mol [{fe_lo:.3f}, {fe_hi:.3f}]")
    print(f"  YM2020 Mg# = {YM_MGNUM:.4f}")

# ============================================================
# 主程式
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mcmc_dir', type=str,
                        default='/Users/chingchen/Desktop/HeFESTo/mcmc',
                        help='MCMC 結果資料夾')
    parser.add_argument('--burnin', type=float, default=0.2,
                        help='Burn-in 比例（預設 0.2）')
    parser.add_argument('--prefix', type=str, default='chain',
                        help='chain 資料夾前綴（預設 chain）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出資料夾（預設與 mcmc_dir 相同）')
    args = parser.parse_args()

    output_dir = args.output_dir or args.mcmc_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 55)

    # 讀取
    chains = load_chains(args.mcmc_dir, prefix=args.prefix)

    # Trace plots
    print("\n[1] Trace plots")
    plot_trace(chains, output_dir)

    # Burn-in
    print(f"\n[2] burn-in = {args.burnin*100:.0f}%")
    trimmed = apply_burnin(chains, burnin_frac=args.burnin)

    # Autocorrelation
    print("\n[3]  Autocorrelation")
    plot_autocorr(trimmed, output_dir)

    # Gelman-Rubin
    print("\n[4] Gelman-Rubin R-hat")
    print_convergence_report(trimmed)

    # 合併樣本
    samples, misfits = get_all_samples(trimmed)

    # 統計摘要
    print_summary(samples, misfits)

    # 邊際後驗
    print("\n[5] plot_marginal_posteriors")
    plot_marginal_posteriors(samples, misfits, output_dir)

    # Corner plot
    print("\n[6] plot_corner")
    plot_corner(samples, output_dir)

    # Misfit vs 參數
    print("\n[7] plot_misfit_vs_params")
    plot_misfit_vs_params(samples, misfits, output_dir)





if __name__ == '__main__':
    main()