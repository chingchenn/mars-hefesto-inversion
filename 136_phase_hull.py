#!/usr/bin/env python3
import os
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:35:58 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 136_phase_hull.py
# Full Fo-Fa phase diagram (alpha, beta, gamma, liquid) from the SAME thermodynamic
# model as the MCMC code, computed by minimising the Gibbs energy (lower convex hull)
# instead of solving one solid+L loop at a time. All stable fields, including
# alpha+beta, beta+gamma, alpha+gamma and the invariant lines, come out automatically.
#
#   G_phi(X) = X G_Fo^phi + (1-X) G_Fa^phi + 2RT[X ln X + (1-X) ln(1-X)] + W_phi X(1-X)
#   reference: alpha end-members (G = 0)
#   beta, gamma: G_i = Delta G_i^{alpha->phi}(P, T)               (Akaogi 1989, code values)
#   liquid:      G_i = Delta H_fus,i (1 - T/T_m,i(P))             (ideal, W_L = 0)
#   W_alpha = 2.5 kJ/mol (SLB24; not used in the MCMC, which has only beta and gamma)
#
# Figure 1: subsolidus isothermal P-X sections at 1473 and 1873 K vs Katsura & Ito (1989)
# Figure 2: isobaric T-X sections (with the rigid Pierru shift) vs Pierru (2026) tie-lines
# Composition axis: mol% Fa = 100 (1 - Mg#), as in Katsura & Ito (1989).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import fsolve, brentq
from mars_config import FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

plt.rcParams['font.family'] = 'DejaVu Sans'

# ── config ───────────────────────────────────────────────────────────────
SRC     = '86.change_traveltime_a52.py'
W_ALPHA = 2500.0
NX      = 801
P_SUB   = np.arange(8.0, 24.01, 0.05)            # GPa, subsolidus P-X sections
T_SUB   = [1473.0, 1873.0]                       # K (1200 and 1600 C of KI89)
P_TX    = [13.0, 14.1, 17.7, 21.0]               # GPa, isobaric T-X sections
T_TX    = np.arange(1500.0, 3200.1, 4.0)         # K, AFTER the shift
fontsize = 14

# Katsura & Ito (1989), 1600 C: alpha-beta-gamma at 13 GPa with Mg# 0.84, 0.72, 0.61;
# Fo alpha->beta 15 GPa, beta->gamma 21 GPa. 1200 C: 12 GPa, Mg# 0.88, 0.75, 0.62;
# Fo 14 and 19 GPa. Coexisting pairs from Table 2 (Mg#). Run 147 omitted (gamma more
# Mg-rich than beta in the extracted table; check the original).
KI89 = {1873.0: dict(univ=(13.0, [0.84, 0.72, 0.61]), fo=[(15.0, 'a/b'), (21.0, 'b/g')],
                     pairs=[(14.4, 0.924, 0.877), (14.5, 0.782, 0.674), (16.5, 0.833, 0.765),
                            (13.7, 0.840, 0.740), (14.0, 0.947, 0.907), (13.5, 0.720, 0.636)]),
        1473.0: dict(univ=(12.0, [0.88, 0.75, 0.62]), fo=[(14.0, 'a/b'), (19.0, 'b/g')],
                     pairs=[(13.9, 0.841, 0.753), (12.6, 0.760, 0.646), (13.1, 0.921, 0.859),
                            (12.2, 0.880, 0.744), (17.3, 0.957, 0.926)])}
PIERRU = [dict(P=7.95, Tsol=1890., XS=0.804, XL=0.525, name='Ol'),
          dict(P=9.60, Tsol=1960., XS=0.910, XL=0.670, name='Ol'),
          dict(P=14.10, Tsol=2090., XS=0.797, XL=0.507, name='Wad'),
          dict(P=17.70, Tsol=2200., XS=0.740, XL=0.499, name='Rw')]

# ── load MCMC phase-diagram block ───────────────────────────────────────
lines = open(SRC).read().splitlines()
i0 = [i for i, l in enumerate(lines) if l.startswith('_R_GAS')][0]
i1 = [i for i, l in enumerate(lines) if l.startswith('# ── BML thermal state')][0]
ns = {'np': np, 'fsolve': fsolve, 'brentq': brentq}
exec('\n'.join(lines[i0:i1]), ns)
R, SHIFT = ns['_R_GAS'], ns['_DELTA_SHIFT']
print(f'using shift = {SHIFT:.0f} K from {SRC}')

X = np.linspace(1e-4, 1 - 1e-4, NX)               # X = X_Fo = Mg#
SMIX = X*np.log(X) + (1 - X)*np.log(1 - X)
PH = ['a', 'b', 'g', 'L']

def end_G(P, T):
    """(G_Fo, G_Fa, W) per phase, alpha end-members as reference"""
    TmFo, TmFa = ns['_Tm_Fo'](P), ns['_Tm_Fa'](P)
    return {'a': (0.0, 0.0, W_ALPHA),
            'b': (ns['_dG_Fo_ab'](P, T), ns['_dG_Fa_ab'](P, T), ns['_W_BETA']),
            'g': (ns['_dG_Fo_ag'](P, T), ns['_dG_Fa_ag'](P, T), ns['_W_GAMMA']),
            'L': (ns['_DH_FO_FUS']*(1 - T/TmFo), ns['_DH_FA_FUS']*(1 - T/TmFa), 0.0)}

def lower_hull(x, y):
    idx = np.lexsort((y, x)); h = []
    for i in idx:
        while len(h) >= 2:
            x1, y1 = x[h[-2]], y[h[-2]]; x2, y2 = x[h[-1]], y[h[-1]]
            if (x2 - x1)*(y[i] - y1) - (y2 - y1)*(x[i] - x1) <= 0:
                h.pop()
            else:
                break
        h.append(i)
    return h

def assemblage(P, T, phases=PH):
    """label of the stable assemblage at each X (e.g. 'a', 'a+b', 'g+L')"""
    eg = end_G(P, T)
    xs, gs, ph = [], [], []
    for k in phases:
        gFo, gFa, W = eg[k]
        xs.append(X); gs.append(X*gFo + (1 - X)*gFa + 2*R*T*SMIX + W*X*(1 - X)); ph += [k]*NX
    xs, gs, ph = np.concatenate(xs), np.concatenate(gs), np.array(ph)
    h = lower_hull(xs, gs)
    lab = np.empty(NX, dtype=object)
    dx = X[1] - X[0]
    for a, b in zip(h[:-1], h[1:]):
        xa, xb = xs[a], xs[b]
        m = (X >= xa - 1e-12) & (X <= xb + 1e-12)
        if ph[a] == ph[b] and xb - xa < 1.5*dx:
            lab[m] = ph[a]
        else:
            pa, pb = sorted({ph[a], ph[b]}, key=PH.index) if ph[a] != ph[b] else (ph[a], ph[a])
            lab[m & (lab == None)] = pa if pa == pb else f'{pa}+{pb}'
    return lab

ORDER = ['a', 'a+b', 'b', 'b+g', 'g', 'a+g', 'a+L', 'b+L', 'g+L', 'L']
COL   = ['#E8C9A6', '#D9B88C', '#9CC7CC', '#B5A8C9', '#C9A6BE', '#E0B3A6',
         '#F2E3CF', '#DCEBEC', '#EBDDE6', '#FFFFFF']
NAME  = {'a': r'$\alpha$', 'b': r'$\beta$', 'g': r'$\gamma$', 'L': 'L'}
def pretty(s): return '+'.join(NAME[p] for p in s.split('+'))
cmap = ListedColormap(COL)
fa = lambda Xfo: 100*(1 - np.asarray(Xfo))      # mol% Fa

def field_map(grid, fixed, mode):
    M = np.full((len(grid), NX), np.nan)
    for i, v in enumerate(grid):
        lab = assemblage(v, fixed) if mode == 'P' else assemblage(fixed, v - SHIFT)
        M[i] = [ORDER.index(l) if l in ORDER else np.nan for l in lab]
    return M

def label_fields(ax, M, grid):
    for k, name in enumerate(ORDER):
        ii, jj = np.where(M == k)
        if len(ii) < 200:
            continue
        ax.text(fa(X[int(np.median(jj))]), grid[int(np.median(ii))], pretty(name),
                ha='center', va='center', fontsize=12)

# ── figure 1: subsolidus P-X sections vs Katsura & Ito (1989) ────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, T in zip(axes, T_SUB):
    M = field_map(P_SUB, T, 'P')
    ax.pcolormesh(fa(X), P_SUB, M, cmap=cmap, vmin=-0.5, vmax=len(ORDER) - 0.5, shading='auto')
    label_fields(ax, M, P_SUB)
    d = KI89[T]
    Pu, xu = d['univ']
    ax.plot(fa(xu), [Pu]*3, 'k-', lw=1.5); ax.plot(fa(xu), [Pu]*3, 'ko', ms=7, label=r'KI89 $\alpha$+$\beta$+$\gamma$')
    for Pp, x1, x2 in d['pairs']:
        ax.plot(fa([x1, x2]), [Pp, Pp], 'k-', lw=1); ax.plot(fa([x1, x2]), [Pp, Pp], 'k|', ms=10)
    for Pf, lab in d['fo']:
        ax.plot(0.5, Pf, 'k>', ms=10)
        ax.text(2, Pf, f'Fo {lab}', fontsize=11, va='center')
    ax.set_xlim(0, 100); ax.set_ylim(P_SUB[0], P_SUB[-1])
    ax.set_xlabel(r'Fa (mol %)', fontsize=fontsize); ax.set_ylabel('P (GPa)', fontsize=fontsize)
    ax.set_title(f'{T:.0f} K ({T - 273.15:.0f} °C), model vs Katsura & Ito (1989)', fontsize=fontsize)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=11, loc='lower right')
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '136_subsolidus_PX.png'), dpi=150)

# ── figure 2: isobaric T-X sections (shifted) vs Pierru (2026) ──────────
fig, axes = plt.subplots(1, len(P_TX), figsize=(5.2*len(P_TX), 6))
for ax, P in zip(axes, P_TX):
    M = field_map(T_TX, P, 'T')
    ax.pcolormesh(fa(X), T_TX, M, cmap=cmap, vmin=-0.5, vmax=len(ORDER) - 0.5, shading='auto')
    label_fields(ax, M, T_TX)
    for t in PIERRU:
        if abs(t['P'] - P) < 0.3:
            ax.plot(fa([t['XS'], t['XL']]), [t['Tsol']]*2, '-', color='#CD5C5C', lw=3)
            ax.plot(fa([t['XS']]), [t['Tsol']], 'o', color='#CD5C5C', mec='k', ms=9)
            ax.plot(fa([t['XL']]), [t['Tsol']], 'D', color='white', mec='#CD5C5C', mew=2, ms=8)
            ax.text(fa(t['XL']) + 2, t['Tsol'], f"Pierru {t['name']}", fontsize=11, color='#CD5C5C', va='center')
    ax.axvline(25, color='k', ls=':', lw=1)
    ax.set_xlim(0, 100); ax.set_ylim(T_TX[0], T_TX[-1])
    ax.set_xlabel('Fa (mol %)', fontsize=fontsize); ax.set_ylabel('T (K)', fontsize=fontsize)
    ax.set_title(f'P = {P} GPa (shift {SHIFT:.0f} K)', fontsize=fontsize)
    ax.tick_params(labelsize=14)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '136_isobaric_TX.png'), dpi=150)
plt.show()