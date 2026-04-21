#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:39:58 2026

@author: chingchen
"""

"""
MCMC 診斷與後驗分析
===================
使用方式：
    python analyze_mcmc.py --mcmc_dir ./mcmc_results
    python analyze_mcmc.py --mcmc_dir ./mcmc_results --burnin 0.3
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

# parameters

model = 'chain02'
PARAMS = ['T_lit', 'P_lit', 'dTdP', 'Si', 'Mg', 'Fe', 'Ca', 'Al']

PARAM_LABELS = {
    'T_lit': r'$T_{lit}$ [K]',
    'P_lit': r'$P_{lit}$ [GPa]',
    'dTdP':  r'$dT/dP$ [K/GPa]',
    'Si':    'Si [mol]',
    'Mg':    'Mg [mol]',
    'Fe':    'Fe [mol]',
    'Ca':    'Ca [mol]',
    'Al':    'Al [mol]',
}

PRIOR = {
    'T_lit': (1000.0, 2000.0),
    'P_lit': (1.5,    9.0),
    'dTdP':  (5.0,    20.0),
    'Si':    (3.0,    5.5),
    'Mg':    (2.5,    5.5),
    'Fe':    (0.4,    2.5),
    'Ca':    (0.1,    0.6),
    'Al':    (0.1,    0.8),
}

# ============================================================
# 讀取資料
# ============================================================

def load_chains(mcmc_dir):
    chain_dirs = sorted(glob.glob(os.path.join(mcmc_dir, f'{model}_*')))
    chains = []
    for d in chain_dirs:
        fpath = os.path.join(d, 'chain.json')
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        if len(data) == 0:
            continue
        chains.append({
            'name': os.path.basename(d),
            'steps': [r['step'] for r in data],
            'misfit': np.array([r['misfit'] for r in data]),
            'accepted': np.array([r['accepted'] for r in data]),
            'accept_rate': np.array([r['accept_rate'] for r in data]),
            'params': {p: np.array([r['params'][p] for r in data]) for p in PARAMS},
        })
    print(f"read {len(chains)} chain")
    for c in chains:
        print(f"  {c['name']}: {len(c['steps'])} steps, "
              f"final misfit={c['misfit'][-1]:.4f}, "
              f"acc rate={c['accept_rate'][-1]:.1f}%")
    return chains


def apply_burnin(chains, burnin_frac=0.3):
    trimmed = []
    for c in chains:
        n = len(c['steps'])
        start = int(n * burnin_frac)
        tc = {
            'name': c['name'],
            'misfit': c['misfit'][start:],
            'accepted': c['accepted'][start:],
            'params': {p: c['params'][p][start:] for p in PARAMS},
        }
        trimmed.append(tc)
    total = sum(len(c['misfit']) for c in trimmed)
    print(f"after Burn-in {burnin_frac*100:.0f}%, total sample {total} ")
    return trimmed




def get_all_samples(trimmed_chains):
    samples = {p: np.concatenate([c['params'][p] for c in trimmed_chains])
               for p in PARAMS}
    misfits = np.concatenate([c['misfit'] for c in trimmed_chains])
    return samples, misfits

# figure 1：Trace plots

def plot_trace(chains, output_dir):
    n_params = len(PARAMS)
    n_cols = 2
    n_rows = (n_params + 1) // n_cols + 1  # +1 for misfit row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.5))
    fig.suptitle('Trace Plots', fontsize=14, fontweight='bold')

    colors = plt.cm.tab20(np.linspace(0, 1, len(chains)))

    # Misfit trace
    ax_misfit = axes[0, 0]
    for i, c in enumerate(chains):
        ax_misfit.plot(c['steps'], c['misfit'], alpha=0.6, lw=0.8,
                       color=colors[i], label=c['name'] if i < 5 else None)
    ax_misfit.set_ylabel('misfit/datum')
    ax_misfit.set_title('Misfit')
    ax_misfit.set_xlabel('Step')

    # Accept rate trace
    ax_acc = axes[0, 1]
    for i, c in enumerate(chains):
        ax_acc.plot(c['steps'], c['accept_rate'], alpha=0.6, lw=0.8,
                    color=colors[i])
    ax_acc.axhline(20, color='red', ls='--', lw=1, label='20%')
    ax_acc.axhline(40, color='green', ls='--', lw=1, label='40%')
    ax_acc.set_ylabel('Accept rate (%)')
    ax_acc.set_title('Acceptance Rate (best 20-40%)')
    ax_acc.set_xlabel('Step')
    ax_acc.legend(fontsize=8)

    # Parameter traces
    for idx, p in enumerate(PARAMS):
        row = (idx // n_cols) + 1
        col = idx % n_cols
        ax = axes[row, col]
        for i, c in enumerate(chains):
            ax.plot(c['steps'], c['params'][p], alpha=0.5, lw=0.6,
                    color=colors[i])
        lo, hi = PRIOR[p]
        ax.axhline(lo, color='gray', ls=':', lw=0.8)
        ax.axhline(hi, color='gray', ls=':', lw=0.8)
        ax.set_ylabel(PARAM_LABELS[p])
        ax.set_title(p)
        ax.set_xlabel('Step')

    # Hide unused axes
    total_subplots = n_rows * n_cols
    used = 2 + n_params
    for idx in range(used, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, f'01_{model}_trace_plots.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()



# figure 2：Autocorrelation

def autocorr(x, maxlag=100):
    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.zeros(maxlag)
    result = []
    for lag in range(maxlag):
        if lag == 0:
            result.append(1.0)
        else:
            c = np.mean(x[lag:] * x[:-lag]) / var
            result.append(c)
    return np.array(result)


def effective_sample_size(x):
    
    n = len(x)
    ac = autocorr(x, maxlag=min(200, n // 2))
    # 找到第一個負值的 lag
    cutoff = next((i for i, v in enumerate(ac) if v < 0), len(ac))
    tau = 1 + 2 * np.sum(ac[1:cutoff])
    return n / tau


def plot_autocorr(trimmed_chains, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle('Autocorrelation', fontsize=13, fontweight='bold')

    c = trimmed_chains[0]  
    maxlag = min(100, len(c['misfit']) // 2)

    for idx, p in enumerate(PARAMS):
        ax = axes[idx // 4][idx % 4]
        ac = autocorr(c['params'][p], maxlag=maxlag)
        ax.bar(range(maxlag), ac, color='steelblue', alpha=0.7, width=1.0)
        ax.axhline(0, color='black', lw=0.8)
        ax.axhline(0.05, color='red', ls='--', lw=0.8, label='0.05')
        ax.set_title(PARAM_LABELS[p])
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ess = effective_sample_size(c['params'][p])
        ax.set_title(f"{p}\nESS≈{ess:.0f}")

    plt.tight_layout()
    out = os.path.join(output_dir, f'02_{model}_autocorrelation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
   


# figure 3：marginal posterior distributions

def gelman_rubin(chains_param):
    m = len(chains_param)
    n = min(len(c) for c in chains_param)
    chains_param = [c[:n] for c in chains_param]

    chain_means = np.array([np.mean(c) for c in chains_param])
    chain_vars  = np.array([np.var(c, ddof=1) for c in chains_param])

    grand_mean = np.mean(chain_means)
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    W = np.mean(chain_vars)

    var_hat = (1 - 1/n) * W + B / n
    R_hat = np.sqrt(var_hat / W) if W > 0 else np.nan
    return R_hat


def print_convergence_report(trimmed_chains):
    print("\n" + "="*55)
    print("Gelman-Rubin R-hat")
    print("R-hat < 1.1 → converge；> 1.2 → need more steps")
    print("="*55)
    all_ok = True
    for p in PARAMS:
        chains_p = [c['params'][p] for c in trimmed_chains]
        rhat = gelman_rubin(chains_p)
        status = "✓" if rhat < 1.1 else ("△" if rhat < 1.2 else "✗")
        if rhat >= 1.1:
            all_ok = False
        print(f"  {status} {p:<8s}: R-hat = {rhat:.4f}")
    print("="*55)
    if all_ok:
        print("  → all parameters converge! ")
    else:
        print("  → need more step")
    return all_ok



def plot_marginal_posteriors(samples, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle('marginal posterior distributions (all chain)',
                 fontsize=13, fontweight='bold')

    for idx, p in enumerate(PARAMS):
        ax = axes[idx // 4][idx % 4]
        x = samples[p]
        lo, hi = PRIOR[p]

        ax.hist(x, bins=50, density=True, color='steelblue',
                alpha=0.7, edgecolor='none')

        # 先驗（uniform）參考線
        ax.axvline(lo, color='gray', ls='--', lw=1.2, label='prior bounds')
        ax.axvline(hi, color='gray', ls='--', lw=1.2)

        # 中位數和 95% CI
        med = np.median(x)
        ci_lo, ci_hi = np.percentile(x, [2.5, 97.5])
        ax.axvline(med, color='red', lw=1.5, label=f'median={med:.3f}')
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color='red', label='95% CI')

        ax.set_xlabel(PARAM_LABELS[p])
        ax.set_ylabel('Density')
        ax.set_title(f"{p}\n{med:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        ax.legend(fontsize=6)

    plt.tight_layout()
    out = os.path.join(output_dir, f'03_{model}_marginal_posteriors.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    
    
# figure 4：Corner plot（4×4）

def plot_corner(samples, output_dir, max_samples=5000):
    n = len(PARAMS)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    fig.suptitle('Corner Plot', fontsize=13, fontweight='bold')


    total = len(samples[PARAMS[0]])
    idx = np.random.choice(total, min(max_samples, total), replace=False)

    for i, pi in enumerate(PARAMS):
        for j, pj in enumerate(PARAMS):
            ax = axes[i][j]
            xi = samples[pi][idx]
            xj = samples[pj][idx]

            if i == j:
                # 對角線：邊際分佈
                ax.hist(xi, bins=40, color='steelblue', alpha=0.7,
                        density=True, edgecolor='none')
                ax.set_xlim(PRIOR[pi])
            elif i > j:
                # 下三角：2D 散佈
                ax.hist2d(xj, xi, bins=30, cmap='Blues',
                          range=[PRIOR[pj], PRIOR[pi]])
                # 相關係數
                r = np.corrcoef(xj, xi)[0, 1]
                ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes,
                        fontsize=7, va='top',
                        color='red' if abs(r) > 0.5 else 'black')
            else:
                ax.set_visible(False)

            # 軸標籤只在邊緣
            if j == 0:
                ax.set_ylabel(PARAM_LABELS[pi], fontsize=7)
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(PARAM_LABELS[pj], fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = os.path.join(output_dir, f'04_{model}_corner_plot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
  

# ============================================================
# 統計摘要
# ============================================================

def print_summary(samples, misfits):
    print("\n" + "="*65)
    print(f"{'parameters':<8} {'mdeian':>10} {'average':>10} "
          f"{'std':>8} {'2.5%':>10} {'97.5%':>10}")
    print("="*65)
    for p in PARAMS:
        x = samples[p]
        med  = np.median(x)
        mean = np.mean(x)
        std  = np.std(x)
        lo   = np.percentile(x, 2.5)
        hi   = np.percentile(x, 97.5)
        print(f"{p:<10} {med:>10.4f} {mean:>10.4f} "
              f"{std:>8.4f} {lo:>10.4f} {hi:>10.4f}")
    print("="*65)
    print(f"Misfit median:{np.median(misfits):.4f}")
    print(f"Misfit min:{np.min(misfits):.4f}")
    print(f"total smaple: {len(misfits)}")
    # ESS
    
    print("\n ESS:")
    for p in PARAMS:
        ess = effective_sample_size(samples[p])
        print(f"  {p:<10}: ESS ≈ {ess:.0f}")



# ============================================================
# 主程式
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mcmc_dir', type=str, default='./mcmc_results',
                        help='MCMC 結果資料夾（包含 chain_XX 子資料夾）')
    parser.add_argument('--burnin', type=float, default=0.3,
                        help='Burn-in 比例（預設 0.3）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出資料夾（預設與 mcmc_dir 相同）')
    args = parser.parse_args()

    output_dir = args.output_dir or args.mcmc_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 55)

    # read
    chains = load_chains(args.mcmc_dir)


    # Trace plots
    print("\n[1] Trace plots")
    plot_trace(chains, output_dir)

    # burn-in
    print(f"\n[2] burn-in = {args.burnin*100:.0f}%")
    trimmed = apply_burnin(chains, burnin_frac=args.burnin)

    # Autocorrelation
    print("\n[3]  Autocorrelation")
    plot_autocorr(trimmed, output_dir)

    # 5. Gelman-Rubin 收斂診斷
    print("\n[4] Gelman-Rubin R-hat")
    print_convergence_report(trimmed)

    # 6. 合併樣本
    samples, misfits = get_all_samples(trimmed)

    # 7. 統計摘要
    print_summary(samples, misfits)

    # 8. 邊際後驗
    print("\n[5] plot_marginal_posteriors")
    plot_marginal_posteriors(samples, output_dir)

    # 9. Corner plot
    print("\n[6] plot_corner")
    plot_corner(samples, output_dir)



if __name__ == '__main__':
    main()