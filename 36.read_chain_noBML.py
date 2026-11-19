#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMC Diagnostics and Posterior Analysis — noBML version
========================================================
Reads and analyses chain.json produced by 33_mcmc_noBML.py.
Differences from 16_read_chain_BML.py:
  - No BML-specific parameters (T_bml, Mg#_bml)
  - Auto-detects free parameters from the first valid chain.json
  - Adds Mg# -> Mg/Fe mol conversion (carried over from 14)
  - All other plot/analysis logic is identical to 16

Usage:
    python 17_read_chain_noBML.py --mcmc_dir ./mcmc --prefix chain
    python 17_read_chain_noBML.py --mcmc_dir ./mcmc --prefix chain_nb01 --burnin 0.3
    python 17_read_chain_noBML.py --mcmc_dir ./mcmc --prefix chain --output_dir ./output
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mars_config import FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

# ============================================================
# Parameter definitions (labels and priors, looked up after auto-detection)
# ============================================================

ALL_PARAM_LABELS = {
    'T_lit': r'$T_{lit}$ [K]',
    'P_lit': r'$P_{lit}$ [GPa]',
    'Mg#':   r'Mg# = Mg/(Mg+Fe)',
    'dTdP':  r'$dT/dP$ [K/GPa]',
    'Si':    'Si [mol]',
    'Mg':    'Mg [mol]',
    'Fe':    'Fe [mol]',
    'Ca':    'Ca [mol]',
    'Al':    'Al [mol]',
    'Na':    'Na [mol]',
    'Cr':    'Cr [mol]',
}

ALL_PRIOR = {
    'T_lit': (1000.0, 2600.0),
    'P_lit': (1.5,    9.0),
    'Mg#':   (0.50,   0.86),
    'dTdP':  (5.0,    20.0),
    'Si':    (3.0,    5.5),
    'Mg':    (2.5,    5.5),
    'Fe':    (0.4,    2.5),
    'Ca':    (0.1,    0.6),
    'Al':    (0.1,    0.8),
    'Na':    (0.0,    0.3),
    'Cr':    (0.0,    0.2),
}

# YM2020 bulk composition (used for Mg# -> mol conversion)
YM_BASE = {
    'Si': 4.01931, 'Mg': 4.08235, 'Fe': 1.08599,
    'Ca': 0.27259, 'Al': 0.37376, 'Na': 0.10105, 'Cr': 0.06146,
}
YM_MGNUM    = YM_BASE['Mg'] / (YM_BASE['Mg'] + YM_BASE['Fe'])   # ≈ 0.790
MGFE_TOTAL  = YM_BASE['Mg'] + YM_BASE['Fe']

# ============================================================
# Auto-detect free parameters
# ============================================================

def detect_params(mcmc_dir, prefix):
    """Read the first valid chain.json and return the list of parameter keys."""
    files = sorted(glob.glob(os.path.join(mcmc_dir, f'{prefix}_*', 'chain.json')))
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if data:
                keys = list(data[0]['params'].keys())
                print(f"Auto-detected params: {keys}  "
                      f"(from {os.path.basename(os.path.dirname(fpath))})")
                return keys
        except Exception:
            continue
    raise RuntimeError(f"No valid chain.json found under {mcmc_dir}/{prefix}_*")

# ============================================================
# Load chains
# ============================================================

def load_chains(mcmc_dir, prefix, params):
    chain_dirs = sorted(glob.glob(os.path.join(mcmc_dir, f'{prefix}_*')))
    chains = []
    for d in chain_dirs:
        fpath = os.path.join(d, 'chain.json')
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        if len(data) == 0:
            continue
        try:
            chains.append({
                'name':        os.path.basename(d),
                'steps':       [r['step'] for r in data],
                'misfit':      np.array([r['misfit']                  for r in data]),
                'misfit_tt':   np.array([r.get('misfit_tt',   np.nan) for r in data]),
                'misfit_mass': np.array([r.get('misfit_mass', np.nan) for r in data]),
                'misfit_moi':  np.array([r.get('misfit_moi',  np.nan) for r in data]),
                'accepted':    np.array([r['accepted']                for r in data]),
                'accept_rate': np.array([r['accept_rate']             for r in data]),
                'params':      {p: np.array([r['params'][p] for r in data]) for p in params},
            })
        except KeyError as e:
            print(f"  skip {os.path.basename(d)}: missing key {e}")
            continue

    print(f"Loaded {len(chains)} chains")
    for c in chains:
        print(f"  {c['name']}: {len(c['steps'])} steps, "
              f"final misfit={c['misfit'][-1]:.4f}, "
              f"acc rate={c['accept_rate'][-1]:.1f}%")
    return chains


def apply_burnin(chains, params, burnin_frac=0.3):
    trimmed = []
    for c in chains:
        n     = len(c['steps'])
        start = int(n * burnin_frac)
        trimmed.append({
            'name':        c['name'],
            'misfit':      c['misfit'][start:],
            'misfit_tt':   c['misfit_tt'][start:],
            'misfit_mass': c['misfit_mass'][start:],
            'misfit_moi':  c['misfit_moi'][start:],
            'accepted':    c['accepted'][start:],
            'params':      {p: c['params'][p][start:] for p in params},
        })
    total = sum(len(c['misfit']) for c in trimmed)
    print(f"After burn-in {burnin_frac*100:.0f}%: {total} samples remaining")
    return trimmed


def get_all_samples(trimmed_chains, params):
    samples = {p: np.concatenate([c['params'][p] for c in trimmed_chains])
               for p in params}
    misfits = np.concatenate([c['misfit'] for c in trimmed_chains])
    return samples, misfits

# ============================================================
# Statistical utilities
# ============================================================

def autocorr(x, maxlag=100):
    x   = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.zeros(maxlag)
    result = [1.0]
    for lag in range(1, maxlag):
        result.append(np.mean(x[lag:] * x[:-lag]) / var)
    return np.array(result)


def effective_sample_size(x):
    n  = len(x)
    ac = autocorr(x, maxlag=min(200, n // 2))
    cutoff = next((i for i, v in enumerate(ac) if v < 0), len(ac))
    tau = 1 + 2 * np.sum(ac[1:cutoff])
    return n / max(tau, 1)


def gelman_rubin(chains_param):
    m = len(chains_param)
    if m < 2:
        return np.nan
    n            = min(len(c) for c in chains_param)
    chains_param = [c[:n] for c in chains_param]
    chain_means  = np.array([np.mean(c) for c in chains_param])
    chain_vars   = np.array([np.var(c, ddof=1) for c in chains_param])
    grand_mean   = np.mean(chain_means)
    B            = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    W            = np.mean(chain_vars)
    var_hat      = (1 - 1/n) * W + B / n
    return np.sqrt(var_hat / W) if W > 0 else np.nan

# ============================================================
# Figure 1：Trace plots
# ============================================================

def plot_trace(chains, params, param_labels, prior, prefix, output_dir):
    n_params     = len(params)
    n_cols       = 2
    n_header     = 2   # row 0: total misfit + accept rate;  row 1: tt + mass/moi
    n_param_rows = (n_params + n_cols - 1) // n_cols
    n_rows       = n_header + n_param_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.5))
    fig.suptitle(f'Trace Plots  [{prefix}]  (noBML)',
                 fontsize=14, fontweight='bold')
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(chains), 1)))

    # Row 0 left: total misfit
    ax = axes[0, 0]
    for i, c in enumerate(chains):
        ax.plot(c['steps'], c['misfit'], alpha=0.6, lw=0.8,
                color=colors[i], label=c['name'] if i < 5 else None)
    ax.set_ylabel('misfit/datum')
    ax.set_title('Total Misfit')
    ax.set_xlabel('Step')
    ax.legend(fontsize=6)

    # Row 0 right: acceptance rate
    ax = axes[0, 1]
    for i, c in enumerate(chains):
        ax.plot(c['steps'], c['accept_rate'], alpha=0.6, lw=0.8, color=colors[i])
    ax.axhline(20, color='red',   ls='--', lw=1, label='20%')
    ax.axhline(40, color='green', ls='--', lw=1, label='40%')
    ax.set_ylabel('Accept rate (%)')
    ax.set_title('Acceptance Rate (target 20-40%)')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)

    # Row 1 left: travel-time misfit
    ax = axes[1, 0]
    has_tt = not np.all(np.isnan(chains[0]['misfit_tt']))
    if has_tt:
        for i, c in enumerate(chains):
            ax.plot(c['steps'], c['misfit_tt'], alpha=0.6, lw=0.8, color=colors[i])
    ax.set_title('Travel-Time Misfit' + ('' if has_tt else ' (N/A)'))
    ax.set_ylabel('misfit_tt')
    ax.set_xlabel('Step')

    # Row 1 right: mass + MoI misfit
    ax = axes[1, 1]
    has_constraints = not np.all(np.isnan(chains[0]['misfit_mass']))
    if has_constraints:
        for i, c in enumerate(chains):
            ax.plot(c['steps'], c['misfit_mass'], alpha=0.6, lw=0.8,
                    color=colors[i], ls='-',  label='mass' if i == 0 else None)
            ax.plot(c['steps'], c['misfit_moi'],  alpha=0.6, lw=0.8,
                    color=colors[i], ls='--', label='MoI'  if i == 0 else None)
        ax.legend(fontsize=8)
    ax.set_title('Mass & MoI Misfit' + ('' if has_constraints else ' (N/A)'))
    ax.set_ylabel('misfit (σ)')
    ax.set_xlabel('Step')

    # Parameter rows
    for idx, p in enumerate(params):
        row = n_header + idx // n_cols
        col = idx % n_cols
        ax  = axes[row, col]
        for i, c in enumerate(chains):
            ax.plot(c['steps'], c['params'][p], alpha=0.5, lw=0.6, color=colors[i])
        lo, hi = prior[p]
        ax.axhline(lo, color='gray', ls=':', lw=0.8)
        ax.axhline(hi, color='gray', ls=':', lw=0.8)
        if p == 'Mg#':
            ax.axhline(YM_MGNUM, color='orange', ls='--', lw=1.2, label='YM2020')
            ax.legend(fontsize=7)
        ax.set_ylabel(param_labels[p])
        ax.set_title(p)
        ax.set_xlabel('Step')

    # Hide unused subplot cells
    for idx in range(n_header * n_cols + n_params, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, f'{prefix}_01_trace_plots.png')
    if SAVE_FIG: plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out}")

# ============================================================
# Figure 2：Autocorrelation
# ============================================================

def plot_autocorr(trimmed_chains, params, param_labels, prefix, output_dir):
    n     = len(params)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(f'Autocorrelation (post Burn-in)  [{prefix}]',
                 fontsize=13, fontweight='bold')

    c      = trimmed_chains[0]
    maxlag = min(100, len(c['misfit']) // 2)

    for idx, p in enumerate(params):
        ax  = axes[idx]
        ac  = autocorr(c['params'][p], maxlag=maxlag)
        ax.bar(range(maxlag), ac, color='steelblue', alpha=0.7, width=1.0)
        ax.axhline(0,    color='black', lw=0.8)
        ax.axhline(0.05, color='red',   ls='--', lw=0.8)
        ess = effective_sample_size(c['params'][p])
        ax.set_title(f"{p}\nESS≈{ess:.0f}")
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, f'{prefix}_02_autocorrelation.png')
    if SAVE_FIG: plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out}")

# ============================================================
# Gelman-Rubin 收斂報告
# ============================================================

def print_convergence_report(trimmed_chains, params):
    print("\n" + "=" * 55)
    print("Gelman-Rubin R-hat")
    print("R-hat < 1.1 → converge；> 1.2 → need more steps")
    print("=" * 55)
    all_ok = True
    for p in params:
        chains_p = [c['params'][p] for c in trimmed_chains]
        rhat     = gelman_rubin(chains_p)
        status   = "✓" if rhat < 1.1 else ("△" if rhat < 1.2 else "✗")
        if rhat >= 1.1:
            all_ok = False
        print(f"  {status} {p:<12s}: R-hat = {rhat:.4f}")
    print("=" * 55)
    print("  → all parameters converge!" if all_ok else "  → need more steps")

# ============================================================
# Figure 3：Marginal posteriors
# ============================================================

def plot_marginal_posteriors(samples, params, param_labels, prior, prefix, output_dir):
    n     = len(params)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(f'Marginal Posterior Distributions  [{prefix}]  (noBML)',
                 fontsize=13, fontweight='bold')

    for idx, p in enumerate(params):
        ax     = axes[idx]
        x      = samples[p]
        lo, hi = prior[p]

        ax.hist(x, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='none')
        ax.axvline(lo, color='gray', ls='--', lw=1.2, label='prior bounds')
        ax.axvline(hi, color='gray', ls='--', lw=1.2)

        med   = np.median(x)
        ci_lo = np.percentile(x, 2.5)
        ci_hi = np.percentile(x, 97.5)
        ax.axvline(med, color='red', lw=1.5, label=f'median={med:.4f}')
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color='red', label='95% CI')

        if p == 'Mg#':
            ax.axvline(YM_MGNUM, color='orange', ls='--',
                       lw=1.5, label=f'YM2020={YM_MGNUM:.3f}')

        ax.set_xlabel(param_labels[p])
        ax.set_ylabel('Density')
        ax.set_title(f"{p}\n{med:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
        ax.legend(fontsize=6)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, f'{prefix}_03_marginal_posteriors.png')
    if SAVE_FIG: plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out}")

# ============================================================
# Figure 4：Corner plot
# ============================================================

def plot_corner(samples, params, param_labels, prior, prefix, output_dir, max_samples=5000):
    n     = len(params)
    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
    if n == 1:
        axes = np.array([[axes]])
    fig.suptitle(f'Corner Plot  [{prefix}]  (noBML)', fontsize=13, fontweight='bold')

    total = len(samples[params[0]])
    idx   = np.random.choice(total, min(max_samples, total), replace=False)

    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            ax = axes[i][j]
            xi = samples[pi][idx]
            xj = samples[pj][idx]

            if i == j:
                ax.hist(xi, bins=40, color='steelblue', alpha=0.7,
                        density=True, edgecolor='none')
                ax.set_xlim(prior[pi])
                if pi == 'Mg#':
                    ax.axvline(YM_MGNUM, color='orange', ls='--',
                               lw=1.2, label='YM2020')
                    ax.legend(fontsize=6)
            elif i > j:
                ax.hist2d(xj, xi, bins=30, cmap='Blues',
                          range=[prior[pj], prior[pi]])
                r = np.corrcoef(xj, xi)[0, 1]
                ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes,
                        fontsize=7, va='top',
                        color='red' if abs(r) > 0.5 else 'black',
                        fontweight='bold')
            else:
                ax.set_visible(False)

            if j == 0:
                ax.set_ylabel(param_labels[pi], fontsize=7)
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(param_labels[pj], fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = os.path.join(output_dir, f'{prefix}_04_corner_plot.png')
    if SAVE_FIG: plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out}")

# ============================================================
# Figure 5: Misfit decomposition (tt / mass / moi)
# ============================================================

def plot_misfit_components(trimmed_chains, prefix, output_dir):
    tt   = np.concatenate([c['misfit_tt']   for c in trimmed_chains])
    mass = np.concatenate([c['misfit_mass'] for c in trimmed_chains])
    moi  = np.concatenate([c['misfit_moi']  for c in trimmed_chains])

    if np.all(np.isnan(tt)) and np.all(np.isnan(mass)):
        print("  misfit components not available (old chain format), skipping Fig 5")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'Misfit Components  [{prefix}]  (noBML)',
                 fontsize=13, fontweight='bold')

    for ax, data, label, color in zip(
            axes,
            [tt, mass, moi],
            ['Travel-Time misfit', 'Mass misfit (σ)', 'MoI misfit (σ)'],
            ['steelblue', 'tomato', 'seagreen']):
        valid  = data[~np.isnan(data) & (data < 990)]
        n_fail = np.sum(data >= 990)
        if len(valid) == 0:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
        else:
            ax.hist(valid, bins=50, color=color, alpha=0.75,
                    edgecolor='none', density=True)
            med = np.median(valid)
            ax.axvline(med, color='black', lw=1.5, ls='--',
                       label=f'median={med:.3f}')
            ax.legend(fontsize=8)
        ax.set_title(f'{label}\n(excluded {n_fail} failed=999)')
        ax.set_xlabel(label)
        ax.set_ylabel('Density')

    plt.tight_layout()
    out = os.path.join(output_dir, f'{prefix}_05_misfit_components.png')
    if SAVE_FIG: plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out}")

# ============================================================
# Figure 6: Misfit vs params (excluding misfit=999)
# ============================================================

def plot_misfit_vs_params(samples, misfits, params, param_labels, prior, prefix, output_dir):
    mask   = misfits < 990
    n_fail = np.sum(~mask)
    ms     = misfits[mask]

    n     = len(params)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(f'Misfit vs posterior sample  [{prefix}]  (noBML)\n'
                 f'(excluded {n_fail} failed steps with misfit=999)',
                 fontsize=13, fontweight='bold')

    for idx, p in enumerate(params):
        ax = axes[idx]
        ps = samples[p][mask]
        sc = ax.scatter(ps, ms, c=ms, cmap='plasma_r', s=4, alpha=0.5, rasterized=True)
        ax.axhline(1.0, color='red', ls='--', lw=1, label='misfit=1')
        ax.set_xlabel(param_labels[p])
        ax.set_ylabel('misfit/datum')
        ax.set_xlim(prior[p])
        ax.legend(fontsize=7)
        plt.colorbar(sc, ax=ax, label='misfit')

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, f'{prefix}_06_misfit_vs_params.png')
    if SAVE_FIG: plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out}")

# ============================================================
# Summary statistics (includes Mg# -> mol conversion)
# ============================================================

def print_summary(samples, misfits, params):
    print("\n" + "=" * 70)
    print(f"{'param':<12} {'median':>10} {'mean':>10} "
          f"{'std':>8} {'2.5%':>10} {'97.5%':>10}")
    print("=" * 70)
    for p in params:
        x = samples[p]
        print(f"{p:<12} {np.median(x):>10.4f} {np.mean(x):>10.4f} "
              f"{np.std(x):>8.4f} {np.percentile(x,2.5):>10.4f} "
              f"{np.percentile(x,97.5):>10.4f}")
    print("=" * 70)
    print(f"Misfit  median={np.median(misfits):.4f}  min={np.min(misfits):.4f}")
    print(f"Total samples: {len(misfits)}")

    print("\nESS:")
    for p in params:
        ess = effective_sample_size(samples[p])
        print(f"  {p:<12}: ESS ≈ {ess:.0f}")

    # Mg# -> Mg / Fe mol conversion (only if Mg# is a free parameter)
    if 'Mg#' in samples:
        print("\n--- Mg# → mol (based on YM2020 Mg+Fe total) ---")
        mgnum = samples['Mg#']
        mg_med  = np.median(mgnum)    * MGFE_TOTAL
        fe_med  = (1 - np.median(mgnum)) * MGFE_TOTAL
        mg_lo   = np.percentile(mgnum, 2.5)  * MGFE_TOTAL
        mg_hi   = np.percentile(mgnum, 97.5) * MGFE_TOTAL
        fe_lo   = (1 - np.percentile(mgnum, 97.5)) * MGFE_TOTAL
        fe_hi   = (1 - np.percentile(mgnum, 2.5))  * MGFE_TOTAL
        print(f"  Mg# median = {np.median(mgnum):.4f} "
              f"[{np.percentile(mgnum,2.5):.4f}, {np.percentile(mgnum,97.5):.4f}]")
        print(f"  Mg = {mg_med:.3f} mol  [{mg_lo:.3f}, {mg_hi:.3f}]")
        print(f"  Fe = {fe_med:.3f} mol  [{fe_lo:.3f}, {fe_hi:.3f}]")
        print(f"  YM2020 Mg# = {YM_MGNUM:.4f}")

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='MCMC posterior analysis — noBML version')
    parser.add_argument('--mcmc_dir',   type=str,   default='./mcmc',
                        help='Directory containing MCMC chain folders (default: ./mcmc)')
    parser.add_argument('--prefix',     type=str,   default='chain',
                        help='Chain folder prefix, e.g. chain or chain_nb01 (default: chain)')
    parser.add_argument('--burnin',     type=float, default=0.3,
                        help='Burn-in fraction to discard (default: 0.3)')
    parser.add_argument('--output_dir', type=str,   default=None,
                        help='Output directory for figures (default: same as mcmc_dir)')
    args = parser.parse_args()

    output_dir = args.output_dir or FIG_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 55)
    print(f"model     : noBML")
    print(f"prefix    : {args.prefix}")
    print(f"mcmc_dir  : {args.mcmc_dir}")
    print(f"burnin    : {args.burnin*100:.0f}%")
    print(f"output_dir: {output_dir}")
    print("=" * 55)

    # 0. Auto-detect parameters
    params       = detect_params(args.mcmc_dir, args.prefix)
    param_labels = {p: ALL_PARAM_LABELS.get(p, p) for p in params}
    prior        = {p: ALL_PRIOR.get(p, (0, 1))   for p in params}

    # 1. Load chains
    chains = load_chains(args.mcmc_dir, args.prefix, params)
    if not chains:
        print("No chains found. Check --mcmc_dir and --prefix.")
        return

    # 2. Trace plots
    print("\n[1] Trace plots")
    plot_trace(chains, params, param_labels, prior, args.prefix, output_dir)

    # 3. Burn-in
    print(f"\n[2] Burn-in = {args.burnin*100:.0f}%")
    trimmed = apply_burnin(chains, params, burnin_frac=args.burnin)

    # 4. Autocorrelation
    print("\n[3] Autocorrelation")
    plot_autocorr(trimmed, params, param_labels, args.prefix, output_dir)

    # 5. Gelman-Rubin
    print("\n[4] Gelman-Rubin R-hat")
    print_convergence_report(trimmed, params)

    # 6. Combine samples
    samples, misfits = get_all_samples(trimmed, params)

    # 7. Summary statistics
    print_summary(samples, misfits, params)

    # 8. Marginal posteriors
    print("\n[5] Marginal posteriors")
    plot_marginal_posteriors(samples, params, param_labels, prior,
                             args.prefix, output_dir)

    # 9. Corner plot
    print("\n[6] Corner plot")
    plot_corner(samples, params, param_labels, prior, args.prefix, output_dir)

    # 10. Misfit decomposition
    print("\n[7] Misfit components")
    plot_misfit_components(trimmed, args.prefix, output_dir)

    # 11. Misfit vs params
    print("\n[8] Misfit vs params")
    plot_misfit_vs_params(samples, misfits, params, param_labels, prior,
                          args.prefix, output_dir)


if __name__ == '__main__':
    main()