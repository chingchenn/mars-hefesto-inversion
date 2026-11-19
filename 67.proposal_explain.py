#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 67: how the proposal moves the chains: diagonal STEP vs fixed covariance
#
#   python 67.proposal_explain.py
#
# Compare two runs with the SAME target (same likelihood, prior, liquid EoS); only propose() differs:
#   a60  (Fo, old prior, diagonal) vs a61t (Fo, old prior, cov from a60,  prop_cov_a60_YM20.npz)
#   a62  (Hd, new prior, diagonal) vs a64  (Hd, new prior, cov from a62,  prop_cov_a62_YM20.npz)
#   a62 vs a63 is NOT a clean pair: a63 is Fo, a62 is Hd
#
#   diagonal   : theta'_k = theta_k + STEP_k z_k,            z_k ~ N(0, 1)
#   covariance : theta'   = theta + L z,  L = chol(s^2 Sigma), z ~ N(0, I_8)
#                s^2   = scale * 2.38^2 / d,  d = 8
#                Sigma = (1 - b) Sigma_post + b diag(STEP^2), b = 0.2   (blended in 81, saved in the npz)
#   accept     : ln u < -(Phi' - Phi) / T, T = 1; out of prior or failed forward = reject
#
# (a) run A: SHOW_N_STEPS consecutive steps of several chains (one color per chain) on the posterior cloud
# (b) run B: same
#     blue ellipses = 1 and 2 sigma of the proposal projected on the w_S - R_cmb plane
# (c) posterior cloud colored by BML thickness: the plane is a projection of an 8-D space
# (d) |dR_cmb| of accepted steps for both runs, with proposal sd and posterior sd
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mars_config as cfg

MCMC_DIR = cfg.MCMC_DIR
OUT_DIR  = cfg.FIG_DIR

# run A and run B: prefix, chains to load, proposal (npz = None -> diagonal STEP), chains to draw, first step drawn
A = dict(prefix='chain_a62_YM20', chains=list(range(16)), npz=None,                    scale=None,
         show=[0, 3, 6, 9, 12], start=4000)
B = dict(prefix='chain_a64_YM20', chains=list(range(16)), npz='prop_cov_a62_YM20.npz', scale=0.25,
         show=[0, 3, 6, 9, 12], start=2000)
# a60 / a61t pair: A = chain_a60_YM20, npz None, start 6000
#                  B = chain_a61t_YM20, chains [4, 5, 6, 7] (scale 0.25 chains), npz prop_cov_a60_YM20.npz, start 2200
CLOUD_FROM   = 'B'           # posterior cloud from the better-mixed run
BURN_FRAC    = 0.4           # for the posterior cloud
BURN_STEP    = 200           # for the step statistics
SHOW_N_STEPS = 300
THIN         = 5             # cloud thinning for plotting
ELL_CENTER   = (0.30, 1480.0)
PARAM_ORDER = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness', 'w_S', 'R_cmb']
STEP = {'w_S': 0.015, 'R_cmb': 25.0}      # diagonal STEP of the main script for the two plotted parameters
IW, IR, IH = 6, 7, 5
FS = 14
CHAIN_COLORS = plt.cm.tab10.colors
plt.rcParams['font.family'] = 'DejaVu Sans'

for R in (A, B):
    R['name'] = R['prefix'].split('_')[1]                       # 'chain_a62_YM20' -> 'a62'
    R['prop'] = 'diagonal proposal' if R['npz'] is None else f'covariance proposal (s = {R["scale"]})'

# ── load ────────────────────────────────────────────────────────────────────
for R in (A, B):
    R['X'] = {}
    for i in R['chains']:
        recs = []
        for line in open(os.path.join(MCMC_DIR, f'{R["prefix"]}_{i:02d}', 'chain.jsonl')):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break
        R['X'][i] = np.array([[r['params'][k] for k in PARAM_ORDER] for r in recs])
    n = [len(x) for x in R['X'].values()]
    print(f'{R["name"]}: {len(n)} chains, steps {min(n)}-{max(n)}, {R["prop"]}')

C = A if CLOUD_FROM == 'A' else B
cloud = np.vstack([x[int(BURN_FRAC * len(x)):] for x in C['X'].values()])

# proposal covariance projected on the w_S - R_cmb plane: 2x2 block of s^2 Sigma
for R in (A, B):
    if R['npz'] is None:
        R['S'] = np.diag([STEP['w_S']**2, STEP['R_cmb']**2])
    else:
        d = np.load(R['npz'])
        assert list(d['param_order']) == PARAM_ORDER
        s2 = R['scale'] * 2.38**2 / len(PARAM_ORDER)
        R['S'] = s2 * d['cov'][np.ix_([IW, IR], [IW, IR])]

# ── step statistics (accepted moves only) ───────────────────────────────────
print(f"\n{'run':>6} {'accept':>7} {'med |dw_S|':>11} {'med |dR|':>9} {'prop sd w_S':>12} {'prop sd R':>10} {'prop corr':>10}")
for R in (A, B):
    dw, dR, n_mv, n = [], [], 0, 0
    for x in R['X'].values():
        dX = np.diff(x[BURN_STEP:], axis=0)
        mv = np.any(dX != 0, axis=1)                 # rejected steps repeat the state
        n_mv += mv.sum()
        n += len(mv)
        dw += list(np.abs(dX[mv, IW]))
        dR += list(np.abs(dX[mv, IR]))
    R['dR'] = np.array(dR)
    R['acc'] = n_mv / n
    S = R['S']
    print(f'{R["name"]:>6} {R["acc"]:7.3f} {np.median(dw):11.4f} {np.median(dR):9.1f} '
          f'{np.sqrt(S[0, 0]):12.4f} {np.sqrt(S[1, 1]):10.1f} {S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]):+10.2f}')
print(f'posterior sd ({C["name"]}): w_S {cloud[:, IW].std():.4f}, R_cmb {cloud[:, IR].std():.1f} km')

# ── figure ──────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
for ax, R, tag in ((axs[0, 0], A, '(a)'), (axs[0, 1], B, '(b)')):
    ax.scatter(cloud[::THIN, IW], cloud[::THIN, IR], s=2, c='0.85', rasterized=True,
               label=f'{C["name"]} posterior (all chains)')
    print(f'\n{tag} {R["name"]}: steps {R["start"]}-{R["start"] + SHOW_N_STEPS}')
    for j, i in enumerate(R['show']):
        X = R['X'][i][R['start']:R['start'] + SHOW_N_STEPS]
        col = CHAIN_COLORS[j % len(CHAIN_COLORS)]
        mv = np.r_[True, np.any(np.diff(X, axis=0) != 0, axis=1)]
        ax.plot(X[:, IW], X[:, IR], '-', c=col, lw=0.9, alpha=0.9, label=f'chain {i:02d}')
        ax.plot(X[mv, IW], X[mv, IR], '.', c=col, ms=3)
        ax.plot(X[0, IW], X[0, IR], 'o', c=col, mec='k', ms=9)
        ax.plot(X[-1, IW], X[-1, IR], 's', c=col, mec='k', ms=8)
        print(f'  chain {i:02d}: net dR {X[-1, IR] - X[0, IR]:+5.0f} km, '
              f'path length in R {np.sum(np.abs(np.diff(X[:, IR]))):5.0f} km')
    ax.plot([], [], 'o', c='w', mec='k', ms=9, label='start')
    ax.plot([], [], 's', c='w', mec='k', ms=8, label='end')
    # ellipse of a 2-D Gaussian: axes = n sqrt(eigenvalues), angle of the major eigenvector
    w, V = np.linalg.eigh(R['S'])
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    for n_sig, ls, lw in ((1, '-', 2), (2, '--', 1)):
        ax.add_patch(Ellipse(ELL_CENTER, 2 * n_sig * np.sqrt(w[1]), 2 * n_sig * np.sqrt(w[0]), angle=ang,
                             fill=False, color='C0', ls=ls, lw=lw, label=f'proposal {n_sig}-sigma'))
    ax.set_xlim(0, 0.37)
    ax.set_ylim(1400, 1900)
    ax.set_xlabel('w_S (S mass fraction in core)', fontsize=FS)
    ax.set_ylabel('R_cmb (km)', fontsize=FS)
    ax.set_title(f'{tag} {R["name"]}: {R["prop"]}, {SHOW_N_STEPS} steps per chain', fontsize=FS)
    ax.tick_params(labelsize=14)
    ax.legend(loc='upper left', fontsize=10, ncol=2)

ax = axs[1, 0]
o = np.argsort(cloud[:, IH])[::THIN]
sc = ax.scatter(cloud[o, IW], cloud[o, IR], s=3, c=cloud[o, IH], cmap='viridis', rasterized=True)
cb = fig.colorbar(sc, ax=ax)
cb.set_label('BML thickness h (km)', fontsize=FS)
cb.ax.tick_params(labelsize=14)
ax.set_xlim(0, 0.37)
ax.set_ylim(1400, 1900)
ax.set_xlabel('w_S', fontsize=FS)
ax.set_ylabel('R_cmb (km)', fontsize=FS)
ax.set_title(f'(c) {C["name"]} posterior, colored by a 3rd parameter', fontsize=FS)
ax.tick_params(labelsize=14)

ax = axs[1, 1]
bins = np.linspace(0, 150, 31)
for R, col in ((A, 'C3'), (B, 'C0')):
    ax.hist(R['dR'], bins=bins, histtype='step', lw=2, color=col, density=True,
            label=f'{R["name"]}: accept {100 * R["acc"]:.0f}%, median |dR| {np.median(R["dR"]):.0f} km')
    ax.axvline(np.sqrt(R['S'][1, 1]), color=col, ls=':', lw=1.5,
               label=f'{R["name"]} proposal sd = {np.sqrt(R["S"][1, 1]):.0f} km')
ax.axvline(cloud[:, IR].std(), color='k', ls='--', lw=1.5, label=f'posterior sd = {cloud[:, IR].std():.0f} km')
ax.set_xlabel('|dR_cmb| per accepted step (km)', fontsize=FS)
ax.set_ylabel('density', fontsize=FS)
ax.set_title('(d) step size, accepted moves only', fontsize=FS)
ax.tick_params(labelsize=14)
ax.legend(loc='upper right', fontsize=10)

plt.show()
fig.tight_layout()
out = os.path.join(OUT_DIR, f'67_proposal_explain_{A["name"]}_{B["name"]}.png')
fig.savefig(out, dpi=150)
print(f'\nsaved {out}')
