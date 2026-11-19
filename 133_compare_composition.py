#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 133: compare the three mantle compositions (YM a52B, TA a53B, EH45 a54B)
# in one set of figures.
#   fig 1  fit quality: misfit components (advisor's criterion)
#   fig 2  mantle parameters (converged in TA)
#   fig 3  BML / core parameters (NOT mixed yet; per-chain means as ticks)
#   fig 4  Mg# trade-off: MoI vs travel times
# All records are used (rejected steps repeat the state = posterior weight).

import json, glob, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

# ══════════════════════════════════════════════════════════════════════════
# config
# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR   = MCMC_DIR
RUNS        = [('YM',   'chain_a52B', '#4E6E8E'),
               ('TA',   'chain_a53B', '#7B9E87'),
               ('EH45', 'chain_a54B', '#C47F3E')]
EXCLUDE     = ['chain_a52B_04', 'chain_a54B_00', 'chain_a54B_09']
BURNIN_FRAC = 0.4
DROP_LAST   = 1
MIN_STEPS   = 400
N_SCATTER   = 3000        # points per composition in fig 4
SAVE_FIG    = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
FS          = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (   1.4,    9.0),
    'Mg#':           (  0.50,   0.90),
    'T_core':        (1550.0, 3600.0),
    'Mg#_bulk_bml':  (  0.20,   0.80),
    'BML_thickness': (  0.0,  400.0),
    'w_S':           (  0.05,   0.30),
    'R_cmb':         (1400.0, 1900.0),
}
MG_PUBLISHED = {'YM': 0.79, 'TA': 0.75, 'EH45': 0.72}          # Yoshizaki & McDonough 2020 Table 3
MG_DRILLEAU  = {'YM': (0.753, 0.010), 'TA': (0.773, 0.012),     # Drilleau 2026 Table 3, BML set
                'EH45': (0.755, 0.016)}

# misfit_nofesi = misfit_tt + moi_sigma + mass_sigma (Fe/Si anchor is CI, unfair to EH45)
FIT_KEYS = [('misfit',        'total misfit'),
            ('misfit_nofesi', 'TT + MoI + mass'),
            ('misfit_tt',     'TT misfit'),
            ('moi_sigma',     r'MoI ($\sigma$)'),
            ('mass_sigma',    r'mass ($\sigma$)'),
            ('fesi_sigma',    r'Fe/Si ($\sigma$)')]
MANTLE = [('T_lit', 'T_lit (K)'), ('P_lit', 'P_lit (GPa)'), ('Mg#', 'Mg# (mantle)')]
DEEP   = [('Mg#_bulk_bml',  'Mg#_bulk_bml'),
          ('BML_thickness', 'BML thickness (km)'),
          ('T_core',        'T_core (K)'),
          ('R_cmb',         'R_cmb (km)'),
          ('R_ScS',         'R_ScS = R_cmb + h_liq (km)'),
          ('w_S',           'w_S')]

# ══════════════════════════════════════════════════════════════════════════
# load: post burn-in records of every usable chain, per composition
# ══════════════════════════════════════════════════════════════════════════
D = {}
for name, prefix, c in RUNS:
    recs, cid, n_ch = [], [], 0
    for dd in sorted(glob.glob(os.path.join(CHAIN_DIR, prefix + '_*'))):
        if os.path.basename(dd) in EXCLUDE:
            continue
        f = os.path.join(dd, 'chain.jsonl')
        if not os.path.exists(f):
            continue
        ch = []
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                ch.append(json.loads(line))
            except json.JSONDecodeError:      # half-written last line
                break
        ch = ch[:-DROP_LAST] if len(ch) > DROP_LAST else []
        if len(ch) < MIN_STEPS:
            continue
        b = int(len(ch) * BURNIN_FRAC)
        recs += ch[b:]
        cid  += [n_ch] * (len(ch) - b)
        n_ch += 1
    d = {'chain': np.array(cid), 'n_chain': n_ch}
    for k in PRIOR:
        d[k] = np.array([float(r['params'][k]) for r in recs])
    for k in ['misfit', 'misfit_tt', 'moi_sigma', 'mass_sigma', 'fesi_sigma', 'h_liquid_km']:
        d[k] = np.array([np.nan if r.get(k) is None else float(r[k]) for r in recs])
    d['misfit_nofesi'] = d['misfit_tt'] + d['moi_sigma'] + d['mass_sigma']
    # ScS reflects at the top of the liquid: R_ScS = R_cmb + h_liquid
    d['R_ScS'] = d['R_cmb'] + d['h_liquid_km']
    D[name] = d
    print(f'{name:5s} {prefix}: {n_ch} chains, {len(recs)} post burn-in samples')

# ══════════════════════════════════════════════════════════════════════════
# summary table: median [5, 95]
# ══════════════════════════════════════════════════════════════════════════
keys_print = [k for k, _ in FIT_KEYS] + [k for k, _ in MANTLE] + [k for k, _ in DEEP]
print('\n' + '=' * 100)
print('median [5%, 95%] per composition' + '   (min for misfit terms in last column)')
print('=' * 100)
print(f"{'key':<16s}" + ''.join(f'{n:>28s}' for n, _, _ in RUNS))
for k in keys_print:
    row = f'{k:<16s}'
    for n, _, _ in RUNS:
        v = D[n][k][np.isfinite(D[n][k])]
        lo, md, hi = np.percentile(v, [5, 50, 95])
        row += f'{md:10.3f} [{lo:7.3f},{hi:7.3f}]'
    print(row)
print('\nminimum misfit per composition:')
for n, _, _ in RUNS:
    print(f"  {n:5s} total {np.nanmin(D[n]['misfit']):8.3f}   "
          f"TT+MoI+mass {np.nanmin(D[n]['misfit_nofesi']):8.3f}   "
          f"TT {np.nanmin(D[n]['misfit_tt']):8.3f}")

# ══════════════════════════════════════════════════════════════════════════
# fig 1: fit quality
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (k, lab) in zip(axes.ravel(), FIT_KEYS):
    pooled = np.concatenate([D[n][k][np.isfinite(D[n][k])] for n, _, _ in RUNS])
    lo, hi = np.percentile(pooled, [0.5, 99])
    bins = np.linspace(lo, hi, 50)
    for n, _, c in RUNS:
        v = D[n][k][np.isfinite(D[n][k])]
        ax.hist(v, bins=bins, density=True, histtype='step', lw=2.2, color=c, label=n)
        ax.axvline(np.median(v), color=c, ls='--', lw=1.2)
    ax.set_xlabel(lab, fontsize=FS)
    ax.set_ylabel('density', fontsize=FS)
    ax.tick_params(labelsize=14)
axes.ravel()[-1].legend(loc='lower right', fontsize=FS - 2)
fig.suptitle('Fit quality by mantle composition (dashed = median)', fontsize=FS + 2)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '133_fit_quality.png'), dpi=200)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# fig 2: mantle parameters
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (k, lab) in zip(axes, MANTLE):
    bins = np.linspace(*PRIOR[k], 60)
    for n, _, c in RUNS:
        ax.hist(D[n][k], bins=bins, density=True, histtype='step', lw=2.2, color=c, label=n)
        if k == 'Mg#':
            ax.axvline(MG_PUBLISHED[n], color=c, ls=':', lw=2)
            m, s = MG_DRILLEAU[n]
            ax.axvspan(m - s, m + s, color=c, alpha=0.12, lw=0)
    if k == 'Mg#':
        lo = min(np.percentile(D[n][k], 0.5) for n, _, _ in RUNS) - 0.03
        hi = max(np.percentile(D[n][k], 99.5) for n, _, _ in RUNS) + 0.03
        ax.set_xlim(lo, hi)
    else:
        lo = min(np.percentile(D[n][k], 0.5) for n, _, _ in RUNS)
        hi = max(np.percentile(D[n][k], 99.5) for n, _, _ in RUNS)
        ax.set_xlim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.set_xlabel(lab, fontsize=FS)
    ax.set_ylabel('density', fontsize=FS)
    ax.tick_params(labelsize=14)
axes[-1].legend(loc='lower right', fontsize=FS - 2)
fig.suptitle('Mantle parameters (Mg# panel: dotted = published, band = Drilleau 2026 BML)',
             fontsize=FS + 1)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '133_mantle_params.png'), dpi=200)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# fig 3: BML / core parameters, with per-chain means as ticks
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (k, lab) in zip(axes.ravel(), DEEP):
    rng_k = PRIOR[k] if k in PRIOR else PRIOR['R_cmb']
    bins = np.linspace(*rng_k, 45)
    ymax = 0.0
    for n, _, c in RUNS:
        h, _, _ = ax.hist(D[n][k], bins=bins, density=True, histtype='step',
                          lw=2.2, color=c, label=n)
        ymax = max(ymax, h.max())
    # prior density (uniform) for the sampled parameters
    if k in PRIOR:
        ax.axhline(1.0 / (rng_k[1] - rng_k[0]), color='0.4', lw=1.5)
    # per-chain means: chains that sit in different places = not mixed
    for i, (n, _, c) in enumerate(RUNS):
        cm = [D[n][k][D[n]['chain'] == j].mean() for j in range(D[n]['n_chain'])]
        ax.plot(cm, np.full(len(cm), ymax * (1.10 + 0.08 * i)), '|',
                color=c, ms=14, mew=2)
    ax.set_ylim(0, ymax * 1.40)
    ax.set_xlim(*rng_k)
    ax.set_xlabel(lab, fontsize=FS)
    ax.set_ylabel('density', fontsize=FS)
    ax.tick_params(labelsize=14)
axes.ravel()[-1].legend(loc='lower right', fontsize=FS - 2)
fig.suptitle('BML / core parameters: NOT mixed yet (ticks = per-chain means, grey = prior)',
             fontsize=FS + 1)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '133_deep_params.png'), dpi=200)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# fig 4: Mg# trade-off, MoI vs travel times
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print('Spearman r of mantle Mg# with each misfit term')
print('=' * 64)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
rng = np.random.default_rng(0)
for n, _, c in RUNS:
    r_moi, _ = spearmanr(D[n]['Mg#'], D[n]['moi_sigma'], nan_policy='omit')
    r_tt,  _ = spearmanr(D[n]['Mg#'], D[n]['misfit_tt'], nan_policy='omit')
    print(f'  {n:5s}  MoI r = {r_moi:+.3f}   TT r = {r_tt:+.3f}')
    idx = rng.choice(len(D[n]['Mg#']), min(N_SCATTER, len(D[n]['Mg#'])), replace=False)
    axes[0].scatter(D[n]['Mg#'][idx], D[n]['moi_sigma'][idx], s=4, alpha=0.4,
                    color=c, linewidths=0, label=n)
    axes[1].scatter(D[n]['Mg#'][idx], D[n]['misfit_tt'][idx], s=4, alpha=0.4,
                    color=c, linewidths=0, label=n)
axes[0].set_ylabel(r'MoI misfit ($\sigma$)', fontsize=FS)
axes[1].set_ylabel('TT misfit', fontsize=FS)
for ax in axes:
    ax.set_xlabel('Mg# (mantle)', fontsize=FS)
    ax.tick_params(labelsize=14)
axes[1].legend(loc='lower right', fontsize=FS - 2, markerscale=4)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '133_mg_tradeoff.png'), dpi=200)
plt.show()

# EH45: which chains populate the low-Mg# tail?  (index j = sorted chain order after EXCLUDE)
n = 'EH45'
print(f"{'j':>3s} {'n':>6s} {'Mg# med':>8s} {'TT med':>8s} {'MoI med':>8s} {'frac Mg#<0.75':>14s}")
for j in range(D[n]['n_chain']):
    m = D[n]['chain'] == j
    print(f"{j:3d} {m.sum():6d} {np.median(D[n]['Mg#'][m]):8.3f} "
          f"{np.median(D[n]['misfit_tt'][m]):8.2f} {np.median(D[n]['moi_sigma'][m]):8.2f} "
          f"{np.mean(D[n]['Mg#'][m] < 0.75):14.1%}")