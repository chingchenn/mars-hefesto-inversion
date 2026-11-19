#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
113 v5: grouped corner plots for the a60 / a61t runs (paper Fig. 4)
     v4 was the a52B/a53B version (qual figures). To redo those, set PREFIX back,
     w_S upper bound 0.40 and BURNIN_FRAC 0.3.

Two figures:
  (a) mantle parameters   3x3   T_lit, P_lit, Mg#
  (b) BML and core        4x4   T_core, w_S, R_cmb, BML_thickness
                                (5x5 with Mg#_bulk_bml when DROP_MGBML = False)
      (T_core / w_S / R_cmb adjacent so the panels next to the diagonal show the ridge)

Panels
  diagonal = posterior / prior of all samples (grey line at 1.0 = prior level)
  lower    = posterior PDF of all samples (normalised per panel)
  upper    = 68% KDE contour of each state. Each state is normalised on its own, so the
             contours show WHERE the states sit, not how much weight each has. Safe to
             report even though solid_frac is not converged.
             Parameters whose solid/molten marginals overlap > DISCRIM_OVL get the text
             "no state discrimination" instead of contours.
             Both corners have this upper triangle (a61t: mantle Mg# differs between
             states, OVL 0.60, so the mantle corner is not left empty).

Changes from 113 v4
  - a60 settings: PREFIX, w_S prior 0.05-0.365, burn-in 40% and MIN_STEPS 400 (same as 63v2)
  - lower-triangle colour map: turbo + PowerNorm(gamma 0.5), smoothing 2.5 bins
  - tick labels via tick_params(labelsize=14) (project convention)
  - suptitle off by default (goes into the caption); PAPER = False brings it back
  - 'phi' upper-triangle option removed (it depends on state weights)
  - DROP_MGBML (default on): interior corner without Mg#_bulk_bml (4x4)
  - diagonal shows all samples only (per-state lines removed: their heights reflect the
    state definition, not data constraint)
  - mantle corner also gets the per-state upper triangle
  - legend moved outside the panels
"""

import json, glob, os
import numpy as np
from scipy.stats import spearmanr, gaussian_kde
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════
# settings
# ══════════════════════════════════════════════════════════════════════════
import mars_config as cfg
from mars_config import FIG_DIR
CHAIN_DIR = cfg.MCMC_DIR          # figures are saved next to this script
PREFIX    = 'chain_a64_YM20'     # main result (YM20, Hd, gravity penalty); was chain_a61t_YM20
EXCLUDE   = []

BURNIN_FRAC = 0.4          # same as 63v2
DROP_LAST   = 1            # last line may be half written
MIN_STEPS   = 100          # a64 is still running; raise to 400 when finished
H_STATE_TOL = 1.0          # km, sublayer thickness threshold for the state labels
SAVE_FIG    = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
PAPER       = True         # True = no suptitle (text goes into the caption)
DROP_MGBML  = False        # False = interior 5x5 with Mg#_bulk_bml (Fig. 4b, 11/16 plan); True = 4x4

FS          = 16           # axis labels
AXIS_RANGE  = 'prior'      # 'prior' shows constraint strength; 'data' = 0.5-99.5 percentiles
NBINS_1D    = 45
NBINS_2D    = 60
SMOOTH_BINS = 2.5          # larger smoothing removes fake peaks in flat panels
CONT_LEVEL  = 0.68
STATE_MIN_N = 50
CORNER_CMAP = 'turbo'      # rainbow-like but perceptually smoother than 'jet'; 'jet' also works
DENSITY_GAMMA = 0.5        # < 1 lifts the low-density tails of the ridge (PowerNorm)

# KDE for the upper triangle
KDE_MAX_N = 4000           # subsample per state (gaussian_kde is slow on 1e5 points)
KDE_BW    = 0.40           # deliberately wider than Scott (~0.26) so flat regions do not
                           # break into islands. State in the caption that contours are smoothed.
KDE_GRID  = 80
KDE_SEED  = 0

DISCRIM_OVL = 0.90         # solid/molten marginal overlap above this -> no contours

PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (   1.4,    9.0),
    'Mg#':           (  0.50,   0.90),
    'T_core':        (1550.0, 3600.0),
    'Mg#_bulk_bml':  (  0.20,   0.90),   # a62+ (a60/a61t: 0.80)
    'BML_thickness': (   0.0,  400.0),
    'w_S':           (  0.00,  0.365),   # a62+ (a60/a61t: 0.05)
    'R_cmb':         (1400.0, 1900.0),
}
LABELS = {'T_lit': r'$T_{lit}$ (K)', 'P_lit': r'$P_{lit}$ (GPa)',
          'Mg#': 'Mg# (mantle)', 'T_core': r'$T_{core}$ (K)',
          'Mg#_bulk_bml': r'Mg#$_{BML}$', 'BML_thickness': 'BML thickness (km)',
          'w_S': r'$w_S$', 'R_cmb': r'$R_{CMB}$ (km)'}

GROUPS = {
    'mantle':   ['T_lit', 'P_lit', 'Mg#'],
    'interior': ['T_core', 'w_S', 'R_cmb', 'BML_thickness']
                + ([] if DROP_MGBML else ['Mg#_bulk_bml']),
}
TITLES = {'mantle': 'mantle parameters', 'interior': 'BML and core parameters'}

STATES      = ['none', 'solid', 'layered', 'molten']
STATE_COLOR = {'none': '#999999', 'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
plt.rcParams['font.family'] = 'DejaVu Sans'


# ══════════════════════════════════════════════════════════════════════════
# state labels
#   solid   : h_liquid < tol and h_solid >= tol
#   molten  : h_solid  < tol and h_liquid >= tol
#   layered : both >= tol
#   fail    : both < tol, or NaN
# ══════════════════════════════════════════════════════════════════════════
def classify_state(h_sol, h_liq, h_bml=None, tol=H_STATE_TOL):
    # none: H_BML < tol (a62+, mantle connects to the core); checked last so it overrides
    h_sol, h_liq = np.asarray(h_sol, float), np.asarray(h_liq, float)
    st = np.full(h_sol.shape, 'fail', dtype='<U8')
    ok = np.isfinite(h_sol) & np.isfinite(h_liq)
    s_y, l_y = ok & (h_sol >= tol), ok & (h_liq >= tol)
    st[s_y & l_y] = 'layered'
    st[~s_y & l_y] = 'molten'
    st[s_y & ~l_y] = 'solid'
    if h_bml is not None:
        st[np.asarray(h_bml, float) < tol] = 'none'
    return st


# ══════════════════════════════════════════════════════════════════════════
# load (records are chain STATES; rejected steps are repeated records, keep them)
# ══════════════════════════════════════════════════════════════════════════
def load_jsonl(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return recs


samples, chain_id, names = [], [], []
for d in sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '_*'))):
    name = os.path.basename(d)
    path = os.path.join(d, 'chain.jsonl')
    if name in EXCLUDE or not os.path.exists(path):
        print(f'  {name:<22s} skipped')
        continue
    recs = load_jsonl(path)[:-DROP_LAST or None]
    if len(recs) < MIN_STEPS:
        print(f'  {name:<22s} skipped ({len(recs)} steps)')
        continue
    b = int(len(recs) * BURNIN_FRAC)
    samples += recs[b:]
    chain_id += [len(names)] * (len(recs) - b)
    names.append(name)
if not names:
    raise SystemExit('no usable chains')
chain_id = np.array(chain_id)
print(f'{PREFIX}: {len(names)} chains, {len(samples)} post burn-in samples')

PLIST = sorted(set(sum(GROUPS.values(), [])))
post = {p: np.array([float(s['params'].get(p, np.nan)) for s in samples]) for p in PLIST}
h_sol = np.array([np.nan if s.get('h_solid_km') is None else float(s['h_solid_km'])
                  for s in samples])
h_liq = np.array([np.nan if s.get('h_liquid_km') is None else float(s['h_liquid_km'])
                  for s in samples])
h_bml = np.array([float(s['params'].get('BML_thickness', np.nan)) for s in samples])
state = classify_state(h_sol, h_liq, h_bml)
print('state counts:', {s: int((state == s).sum()) for s in STATES + ['fail']},
      '  (weights not reportable)')

rng = {}
for p in PLIST:
    if AXIS_RANGE == 'prior':
        rng[p] = PRIOR[p]
    else:
        lo, hi = np.nanpercentile(post[p], [0.5, 99.5])
        rng[p] = (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))


# ══════════════════════════════════════════════════════════════════════════
# state discrimination per parameter
#   OVL = sum(min(p_solid, p_molten)) * binwidth : 1 = identical marginals, 0 = separate
#   Uses state locations only, not weights.
# ══════════════════════════════════════════════════════════════════════════
ovl_of = {}
print('\n' + '=' * 72)
print(f"{'param':<16s}{'OVL(s,m)':>10s}{'|dmed|/sd':>11s}{'med solid':>12s}{'med molten':>12s}")
print('=' * 72)
for p in PLIST:
    a, b = PRIOR[p]
    ms = (state == 'solid') & np.isfinite(post[p])
    mm = (state == 'molten') & np.isfinite(post[p])
    if ms.sum() < STATE_MIN_N or mm.sum() < STATE_MIN_N:
        ovl_of[p] = np.nan
        print(f'{p:<16s}{"--":>10s}')
        continue
    hs, e = np.histogram(post[p][ms], bins=NBINS_1D, range=(a, b), density=True)
    hm, _ = np.histogram(post[p][mm], bins=NBINS_1D, range=(a, b), density=True)
    ovl_of[p] = np.sum(np.minimum(hs, hm)) * (e[1] - e[0])
    d = abs(np.median(post[p][ms]) - np.median(post[p][mm])) / np.nanstd(post[p], ddof=1)
    print(f'{p:<16s}{ovl_of[p]:10.3f}{d:11.3f}'
          f'{np.median(post[p][ms]):12.4g}{np.median(post[p][mm]):12.4g}')
print(f'  OVL > {DISCRIM_OVL:.2f}: upper-triangle panels show "no state discrimination"')

print('\nSpearman r within each group')
for gname, plist in GROUPS.items():
    for i in range(len(plist)):
        for j in range(i):
            r = spearmanr(post[plist[j]], post[plist[i]], nan_policy='omit')[0]
            print(f'  {gname:<9s} r({plist[j]:<14s}, {plist[i]:<14s}) = {r:+.3f}')


# ══════════════════════════════════════════════════════════════════════════
# KDE contour enclosing CONT_LEVEL of the probability mass
# ══════════════════════════════════════════════════════════════════════════
kde_rng = np.random.default_rng(KDE_SEED)
from matplotlib.colors import PowerNorm
NORM = PowerNorm(gamma=DENSITY_GAMMA, vmin=0, vmax=1)


def kde_contour(xs, ys, rx, ry):
    if len(xs) > KDE_MAX_N:
        k = kde_rng.choice(len(xs), KDE_MAX_N, replace=False)
        xs, ys = xs[k], ys[k]
    if np.std(xs) <= 0 or np.std(ys) <= 0:
        return None
    try:
        kd = gaussian_kde(np.vstack([xs, ys]), bw_method=KDE_BW)
    except np.linalg.LinAlgError:
        return None
    gx, gy = np.linspace(*rx, KDE_GRID), np.linspace(*ry, KDE_GRID)
    GX, GY = np.meshgrid(gx, gy)
    D = kd(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    if D.sum() <= 0:
        return None
    d = np.sort(D.ravel())[::-1]
    lev = d[np.searchsorted(np.cumsum(d) / d.sum(), CONT_LEVEL)]
    return gx, gy, D, lev


# ══════════════════════════════════════════════════════════════════════════
# corner
# ══════════════════════════════════════════════════════════════════════════
for gname, plist in GROUPS.items():
    n = len(plist)
    sz = 3.6 if n <= 3 else 2.9
    fig, axes = plt.subplots(n, n, figsize=(sz * n, sz * n * 0.93))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            xi, yi = plist[j], plist[i]
            a, b = rng[xi]

            if i == j:                                   # diagonal: posterior / prior
                v = post[xi][np.isfinite(post[xi])]
                h, e = np.histogram(v, bins=NBINS_1D, range=(a, b), density=True)
                ctr = 0.5 * (e[1:] + e[:-1])
                ax.bar(ctr, h * (b - a), width=(b - a) / NBINS_1D, color='0.55',
                       edgecolor='0.35', linewidth=0.5, zorder=2)
                ax.axhline(1.0, color='0.35', lw=1.5, zorder=4)
                ax.set_xlim(a, b)
                if i == 0:
                    ax.set_ylabel('posterior / prior', fontsize=FS)

            elif i > j:                                  # lower: posterior PDF
                m = np.isfinite(post[xi]) & np.isfinite(post[yi])
                H, xe, ye = np.histogram2d(post[xi][m], post[yi][m], bins=NBINS_2D,
                                           range=[rng[xi], rng[yi]])
                H = gaussian_filter(H, SMOOTH_BINS)
                if H.max() > 0:
                    H = H / H.max()
                ax.imshow(H.T, origin='lower', aspect='auto', cmap=CORNER_CMAP, norm=NORM,
                          extent=[xe[0], xe[-1], ye[0], ye[-1]])
                ax.set_xlim(*rng[xi]); ax.set_ylim(*rng[yi])

            else:                                        # upper: state contours
                ox, oy = ovl_of.get(xi, np.nan), ovl_of.get(yi, np.nan)
                if (np.isfinite(ox) and ox > DISCRIM_OVL) or (np.isfinite(oy) and oy > DISCRIM_OVL):
                    ax.text(0.5, 0.5, 'no state\ndiscrimination', transform=ax.transAxes,
                            ha='center', va='center', fontsize=FS - 3, color='0.55')
                else:
                    for st in STATES:
                        m = (state == st) & np.isfinite(post[xi]) & np.isfinite(post[yi])
                        if m.sum() < STATE_MIN_N:
                            continue
                        out = kde_contour(post[xi][m], post[yi][m], rng[xi], rng[yi])
                        if out is not None:
                            gx, gy, D, lev = out
                            ax.contour(gx, gy, D, levels=[lev], colors=STATE_COLOR[st],
                                       linewidths=1.8)
                ax.set_xlim(*rng[xi]); ax.set_ylim(*rng[yi])

            if i == n - 1:
                ax.set_xlabel(LABELS[xi], fontsize=FS)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LABELS[yi], fontsize=FS)
            elif j != 0 and i != j:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=14)

    # legend outside the panels, next to the colour bar
    handles = [plt.Line2D([], [], color=STATE_COLOR[s], lw=2, label=s) for s in STATES]
    fig.legend(handles=handles, fontsize=FS - 2, loc='lower right', ncol=3,
               bbox_to_anchor=(0.99, 0.03), frameon=False,
               title=f'{CONT_LEVEL:.0%} contour per state', title_fontsize=FS - 2)
    if not PAPER:
        fig.suptitle(f'{PREFIX}  {TITLES[gname]}   diagonal = posterior / prior   '
                     f'lower = posterior PDF   upper = {CONT_LEVEL:.0%} KDE per state',
                     fontsize=FS)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    cax = fig.add_axes([0.10, 0.06, 0.38, 0.016])     # colour bar bottom left, legend bottom right
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CORNER_CMAP, norm=NORM),
                      cax=cax, orientation='horizontal')
    cb.set_label('posterior PDF (normalised per panel)', fontsize=FS)
    cb.ax.tick_params(labelsize=14)
    if SAVE_FIG:
        fig.savefig(os.path.join(FIG_DIR, f'113_{PREFIX}_{gname}.png'), dpi=200)
    plt.show()
