#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
63v2: posterior summary for the a60 runs (rewrite of 63.mcmc_marginal_prior.py)

Output
  Tables (printed)
    1. state counts per run
    2. parameter summary: median, 5-95%, width ratio vs prior, density at the prior walls
    3. derived quantities by state: R_cmb, R_ScS, w_S, BML density, core density, core Vp, P_cmb
    4. a few flags used in the text (P_cmb > 23 GPa, w_S > 0.17, inner core, Mg# correlations)
    5. convergence: R-hat, ESS, per-chain state counts and transitions
  Figures
    A. prior vs posterior, shallow row / deep row        (draft of paper Fig. 3)
    B. BML state and its consequences                    (draft of paper Fig. 5)
    C. corner plot coloured by state                     (SI)

Removed from v1: chain.json reader, a36 guard, misfit/MoI corners, tt_per_datum,
core diagnostics vs Bi/Irving, self-consistency figure, KL/bits tables,
step-size suggestions, running means, stale PRIOR_FRAC.
"""

import json, glob, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════
# settings
# ══════════════════════════════════════════════════════════════════════════
import mars_config as cfg
CHAIN_DIR = cfg.MCMC_DIR          # figures are saved next to this script

PREFIX       = 'chain_a61t_YM20'                        # main run
COMPARE      = ['chain_a60_TA', 'chain_a60_EH45']      # thin lines in Fig A; [] to skip
PRIOR_PREFIX = None    # prior-only run (likelihood off); None = uniform prior from PRIOR

EXCLUDE     = ['chain_a60_EH45_00']
BURNIN_FRAC = 0.4
DROP_LAST   = 1
MIN_STEPS   = 400
SAVE        = False
N_SCATTER   = 20000    # points per scatter plot (subsampled for speed)

PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (   1.4,    9.0),
    'Mg#':           (  0.50,   0.90),
    'T_core':        (1550.0, 3600.0),
    'Mg#_bulk_bml':  (  0.20,   0.80),
    'BML_thickness': (   0.0,  400.0),
    'w_S':           (  0.05,  0.365),   # a60 value (v1 still had 0.30)
    'R_cmb':         (1400.0, 1900.0),
}
SHALLOW = ['T_lit', 'P_lit', 'Mg#']
DEEP    = ['T_core', 'Mg#_bulk_bml', 'BML_thickness', 'R_cmb', 'w_S']
PARAMS  = SHALLOW + DEEP

LABEL = {'T_lit': r'$T_{lit}$ (K)', 'P_lit': r'$P_{lit}$ (GPa)', 'Mg#': 'Mg# (mantle)',
         'T_core': r'$T_{core}$ (K)', 'Mg#_bulk_bml': r'Mg#$_{BML}$',
         'BML_thickness': 'BML thickness (km)', 'R_cmb': r'$R_{CMB}$ (km)',
         'w_S': r'$w_S$'}

STATES      = ['solid', 'layered', 'molten']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
POST_COLOR  = '#3d3d3d'
PRIOR_COLOR = '#d9d9d9'
COMPARE_LS  = ['--', ':', '-.']

LIT_RHO_BML = {'Khan 2023': 4.05, 'Samuel 2023': 4.88}   # g/cm3, reference lines only
WS_GEOCHEM  = 0.17
P_EXP_MAX   = 23.0                                       # GPa, top of the phase-diagram tests

FS = 14
plt.rcParams['font.family'] = 'DejaVu Sans'


# ══════════════════════════════════════════════════════════════════════════
# loading
# ══════════════════════════════════════════════════════════════════════════
def load_jsonl(path):
    """Read chain.jsonl line by line; a half-written last line is dropped."""
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


def load_run(prefix, verbose=True):
    """Return (chains, names). Each chain is a list of records after DROP_LAST."""
    chains, names = [], []
    for d in sorted(glob.glob(os.path.join(CHAIN_DIR, f'{prefix}_*'))):
        name = os.path.basename(d)
        path = os.path.join(d, 'chain.jsonl')
        if name in EXCLUDE or not os.path.exists(path):
            if verbose:
                print(f'  {name:<22s} skipped')
            continue
        recs = load_jsonl(path)[:-DROP_LAST or None]
        if len(recs) < MIN_STEPS:
            if verbose:
                print(f'  {name:<22s} skipped ({len(recs)} steps)')
            continue
        chains.append(recs)
        names.append(name)
    if not chains:
        raise SystemExit(f'no usable chains for {prefix}')
    if verbose:
        print(f'  {prefix}: {len(chains)} chains')
    return chains, names


def _get(rec, key):
    v = rec['params'].get(key) if key in PRIOR else rec.get(key)
    return np.nan if v is None else float(v)


def classify(h_sol, h_liq):
    st = np.full(h_sol.shape, 'fail', dtype='<U8')
    st[(h_sol >= 1) & (h_liq >= 1)] = 'layered'
    st[(h_sol <  1) & (h_liq >= 1)] = 'molten'
    st[(h_sol >= 1) & (h_liq <  1)] = 'solid'
    return st


EXTRA = ['misfit_tt', 'moi_sigma', 'h_solid_km', 'h_liquid_km', 'rho_solid_bml',
         'rho_liquid_bml', 'rho_core_mean', 'Vp_core_cmb', 'P_cmb', 'R_ic_km', 'sc_dP']


def to_post(chains):
    """Apply burn-in per chain and return a dict of arrays plus derived quantities."""
    recs, cid = [], []
    for c, ch in enumerate(chains):
        b = int(len(ch) * BURNIN_FRAC)
        recs += ch[b:]
        cid += [c] * (len(ch) - b)
    post = {k: np.array([_get(r, k) for r in recs]) for k in PARAMS + EXTRA}
    post['chain'] = np.array(cid)

    hs, hl = post['h_solid_km'], post['h_liquid_km']
    post['state'] = classify(hs, hl)
    post['R_scs'] = post['R_cmb'] + hl                    # ScS reflects off the top of the liquid
    post['R_bml_top'] = post['R_cmb'] + post['BML_thickness']

    # BML density (g/cm3): liquid, solid, or thickness-weighted mean for layered
    rs, rl = post['rho_solid_bml'], post['rho_liquid_bml']
    if np.nanmedian(np.r_[rs, rl]) > 100:                 # stored in kg/m3
        rs, rl = rs / 1e3, rl / 1e3
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = (np.nan_to_num(hs) * np.nan_to_num(rs) + np.nan_to_num(hl) * np.nan_to_num(rl)) / (hs + hl)
    post['rho_bml'] = np.where(hs + hl > 0, rho, np.nan)
    return post


print('loading')
chains, names = load_run(PREFIX)
post = to_post(chains)
others = {p: to_post(load_run(p, verbose=False)[0]) for p in COMPARE}
prior_post = to_post(load_run(PRIOR_PREFIX)[0]) if PRIOR_PREFIX else None


# ══════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════
def q(v, p=(5, 50, 95)):
    v = v[np.isfinite(v)]
    return np.percentile(v, p) if v.size else np.full(len(p), np.nan)


def prior_q90(k):
    if prior_post is not None:
        lo, hi = q(prior_post[k], (5, 95))
        return hi - lo
    a, b = PRIOR[k]
    return 0.9 * (b - a)


def wall_density(v, k, nb=25):
    a, b = PRIOR[k]
    h, _ = np.histogram(v[np.isfinite(v)], bins=nb, range=(a, b))
    h = h / max(h.max(), 1)
    return h[0], h[-1]


def line(n=78):
    print('=' * n)


# ══════════════════════════════════════════════════════════════════════════
# table 1  state counts
# ══════════════════════════════════════════════════════════════════════════
print()
line()
print('1. state counts  (not posterior weights: chains rarely cross states, see table 5)')
line()
runs = {PREFIX: post, **others}
if prior_post is not None:
    runs['PRIOR'] = prior_post
for name, P in runs.items():
    n = len(P['state'])
    print(f'  {name:<22s}' + ''.join(f'{s:>9s} {100*np.mean(P["state"] == s):5.1f}%' for s in STATES)
          + f'   fail {np.sum(P["state"] == "fail")}')


# ══════════════════════════════════════════════════════════════════════════
# table 2  parameter summary
# ══════════════════════════════════════════════════════════════════════════
print()
line()
print(f'2. parameters ({PREFIX})')
print('   ratio = posterior 90% width / prior 90% width   (1 = data said nothing)')
print('   wall  = posterior density at prior bounds, relative to peak (> 30% = bound matters)')
line()
print(f"  {'':<15s}{'median':>9s}{'5%':>10s}{'95%':>10s}{'ratio':>8s}{'wall L':>9s}{'wall R':>8s}")
for k in PARAMS:
    lo, md, hi = q(post[k])
    wl, wr = wall_density(post[k], k)
    flag = '  <--' if max(wl, wr) > 0.3 else ''
    if k == DEEP[0]:
        print('  ' + '-' * 67)
    print(f'  {k:<15s}{md:9.4g}{lo:10.4g}{hi:10.4g}{(hi-lo)/prior_q90(k):8.2f}'
          f'{wl:9.0%}{wr:8.0%}{flag}')


# ══════════════════════════════════════════════════════════════════════════
# table 3  derived quantities by state
# ══════════════════════════════════════════════════════════════════════════
DERIVED = [('R_cmb', 'km'), ('R_scs', 'km'), ('R_bml_top', 'km'), ('w_S', ''),
           ('rho_bml', 'g/cm3'), ('rho_core_mean', 'g/cm3'), ('Vp_core_cmb', 'km/s'),
           ('P_cmb', 'GPa')]
print()
line(96)
print(f'3. derived quantities by state, median [5, 95]  ({PREFIX})')
line(96)
print(f"  {'':<15s}" + ''.join(f'{s:>27s}' for s in STATES))
for k, u in DERIVED:
    cells = []
    for s in STATES:
        lo, md, hi = q(post[k][post['state'] == s])
        cells.append(f'{md:8.4g} [{lo:7.4g},{hi:7.4g}]')
    print(f'  {k:<15s}' + ''.join(f'{c:>27s}' for c in cells) + f'  {u}')

print('\n  sd(R_ScS) / sd(R_cmb)   (< 1 means ScS pins the reflector better than the metal core)')
for s in ['all'] + STATES:
    m = np.ones(len(post['state']), bool) if s == 'all' else post['state'] == s
    a, b = np.nanstd(post['R_scs'][m]), np.nanstd(post['R_cmb'][m])
    print(f'    {s:<8s} {a:6.1f} / {b:6.1f} km = {a/b:.2f}')


# ══════════════════════════════════════════════════════════════════════════
# table 4  numbers quoted in the text
# ══════════════════════════════════════════════════════════════════════════
print()
line()
print('4. flags')
line()
f = lambda m: f'{100*np.nanmean(m):5.1f}%'
print(f"  P(P_cmb > {P_EXP_MAX:.0f} GPa)   = {f(post['P_cmb'] > P_EXP_MAX)}   (above phase-diagram tests)")
print(f"  P(w_S > {WS_GEOCHEM})       = {f(post['w_S'] > WS_GEOCHEM)}   (above geochemical limit)")
print(f"  P(R_ic > 0)          = {f(post['R_ic_km'] > 0)}   (whole core frozen, see Methods 2.4)")
print(f"  max |sc_dP|          = {np.nanmax(np.abs(post['sc_dP'])):.3g} GPa   (P-g iteration residual)")
for other in ('misfit_tt', 'moi_sigma'):
    r, _ = spearmanr(post['Mg#'], post[other], nan_policy='omit')
    print(f'  Spearman(Mg#, {other:<9s}) = {r:+.2f}')


# ══════════════════════════════════════════════════════════════════════════
# table 5  convergence
# ══════════════════════════════════════════════════════════════════════════
def rhat_ess(mat):
    """Split-free R-hat and Geyer initial-positive-sequence ESS for an (m, n) array."""
    m, n = mat.shape
    W = mat.var(axis=1, ddof=1).mean()
    B = n * mat.mean(axis=1).var(ddof=1)
    if W <= 0:
        return np.nan, np.nan
    vp = (n - 1) / n * W + B / n

    def acov(x):
        x = x - x.mean()
        fx = np.fft.rfft(x, 2 * n)
        return np.fft.irfft(fx * np.conj(fx))[:n] / n

    rho = 1.0 - (W - np.mean([acov(r) for r in mat], axis=0)) / vp
    s, t = 0.0, 1
    while t + 1 < n and rho[t] + rho[t + 1] > 0:
        s += rho[t] + rho[t + 1]
        t += 2
    return np.sqrt(vp / W), m * n / (1 + 2 * s)


print()
line()
print('5. convergence (chains truncated to the shortest)')
line()
n = min(len(c) for c in chains)
b = int(BURNIN_FRAC * n)
print(f'  {len(chains)} chains x {n - b} post burn-in steps')
print(f"  {'':<15s}{'R-hat':>8s}{'ESS':>8s}")
for k in PARAMS + ['solid_frac']:
    if k == 'solid_frac':
        mat = np.array([(classify(np.array([_get(r, 'h_solid_km') for r in c[-n:][b:]]),
                                  np.array([_get(r, 'h_liquid_km') for r in c[-n:][b:]]))
                         == 'solid').astype(float) for c in chains])
    else:
        mat = np.array([[_get(r, k) for r in c[-n:][b:]] for c in chains])
    r, e = rhat_ess(mat)
    print(f'  {k:<15s}{r:8.3f}{e:8.0f}' + ('  *' if r > 1.1 else ''))

print(f"\n  {'chain':<22s}{'trans':>6s}" + ''.join(f'{s:>9s}' for s in STATES))
for c, nm in enumerate(names):
    s = post['state'][post['chain'] == c]
    print(f'  {nm:<22s}{int(np.sum(s[1:] != s[:-1])):6d}'
          + ''.join(f'{int(np.sum(s == x)):9d}' for x in STATES))


# ══════════════════════════════════════════════════════════════════════════
# Fig A  prior vs posterior (draft of paper Fig. 3)
# ══════════════════════════════════════════════════════════════════════════
def save(fig, tag):
    if SAVE:
        fig.savefig(f'63v2_{PREFIX}_{tag}.png', dpi=200)
    plt.show()


fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
for row, keys in enumerate([SHALLOW, DEEP]):
    for col in range(5):
        ax = axes[row, col]
        if col >= len(keys):
            ax.set_visible(False)
            continue
        k = keys[col]
        a, bnd = PRIOR[k]
        bins = np.linspace(a, bnd, 41)
        if prior_post is None:
            ax.fill_between([a, bnd], 0, 1 / (bnd - a), color=PRIOR_COLOR, lw=0, label='prior')
        else:
            ax.hist(prior_post[k], bins=bins, density=True, color=PRIOR_COLOR, label='prior')
        ax.hist(post[k], bins=bins, density=True, color=POST_COLOR, alpha=0.6,
                label=PREFIX.split('_')[-1])   # translucent so the prior shows through
        for (name, P), ls in zip(others.items(), COMPARE_LS):
            ax.hist(P[k], bins=bins, density=True, histtype='step', color='0.5', ls=ls,
                    lw=1.5, label=name.split('_')[-1])
        lo, hi = q(post[k], (5, 95))
        ax.text(0.97, 0.95, f'{(hi - lo) / prior_q90(k):.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=FS)
        ax.set_xlim(a, bnd)
        ax.set_yticks([])
        ax.set_xlabel(LABEL[k], fontsize=FS)
        ax.tick_params(labelsize=14)
axes[0, 0].set_ylabel('shallow', fontsize=FS)
axes[1, 0].set_ylabel('deep', fontsize=FS)
axes[0, 2].legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.1, 1.0))
fig.tight_layout()
save(fig, 'A_prior_post')


# ══════════════════════════════════════════════════════════════════════════
# Fig B  BML state and its consequences (draft of paper Fig. 5)
# ══════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(0)
idx = rng.choice(len(post['state']), min(N_SCATTER, len(post['state'])), replace=False)

fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
ax_a = fig.add_subplot(gs[:, 0])
ax_b1, ax_b2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
ax_c = fig.add_subplot(gs[:, 2])

# (a) state in Mg#_BML - T_core space (same axes as the phase diagram)
for s in ['molten', 'layered', 'solid']:
    m = idx[post['state'][idx] == s]
    ax_a.scatter(post['Mg#_bulk_bml'][m], post['T_core'][m], s=3, alpha=0.4,
                 color=STATE_COLOR[s], linewidths=0)
ax_a.set_xlabel(LABEL['Mg#_bulk_bml'], fontsize=FS)
ax_a.set_ylabel(LABEL['T_core'], fontsize=FS)
ax_a.set_xlim(*PRIOR['Mg#_bulk_bml']); ax_a.set_ylim(*PRIOR['T_core'])
ax_a.legend(handles=[mpatches.Patch(color=STATE_COLOR[s], label=s) for s in STATES],
            fontsize=12, loc='lower right')

# (b) R_cmb vs R_ScS, molten and solid
rbins = np.linspace(*PRIOR['R_cmb'], 51)
for ax, s in [(ax_b1, 'molten'), (ax_b2, 'solid')]:
    m = post['state'] == s
    ax.hist(post['R_cmb'][m], bins=rbins, density=True, histtype='step', lw=2,
            color='k', label=r'$R_{CMB}$')
    ax.hist(post['R_scs'][m], bins=rbins, density=True, histtype='step', lw=2,
            color=STATE_COLOR[s], label=r'$R_{ScS}$')
    ax.set_yticks([])
    ax.text(0.03, 0.9, s, transform=ax.transAxes, fontsize=FS, color=STATE_COLOR[s])
    ax.tick_params(labelsize=14)
ax_b1.legend(fontsize=12, loc='upper right')
ax_b1.set_xticklabels([])
ax_b2.set_xlabel('radius (km)', fontsize=FS)

# (c) BML density by state
dbins = np.linspace(np.nanpercentile(post['rho_bml'], 0.5),
                    max(np.nanpercentile(post['rho_bml'], 99.5), 5.0), 50)
for s in STATES:
    ax_c.hist(post['rho_bml'][post['state'] == s], bins=dbins, density=True,
              histtype='step', lw=2, color=STATE_COLOR[s], label=s)
for name, v in LIT_RHO_BML.items():
    ax_c.axvline(v, color='0.5', lw=1)
    ax_c.text(v, ax_c.get_ylim()[1] * 0.97, f' {name}', rotation=90, va='top',
              fontsize=11, color='0.4')
ax_c.set_xlabel(r'BML density (g/cm$^3$)', fontsize=FS)
ax_c.set_yticks([])
ax_c.legend(fontsize=12, loc='center right')

for ax in (ax_a, ax_c):
    ax.tick_params(labelsize=14)
fig.tight_layout()
save(fig, 'B_state')


# ══════════════════════════════════════════════════════════════════════════
# Fig C  corner plot coloured by state (SI)
# ══════════════════════════════════════════════════════════════════════════
nP = len(PARAMS)
fig, axes = plt.subplots(nP, nP, figsize=(2.1 * nP, 2.0 * nP))
for i in range(nP):
    for j in range(nP):
        ax = axes[i, j]
        if j > i:
            ax.set_visible(False)
            continue
        if i == j:
            for s in STATES:
                ax.hist(post[PARAMS[i]][post['state'] == s], bins=30, range=PRIOR[PARAMS[i]],
                        density=True, histtype='step', lw=1.5, color=STATE_COLOR[s])
            ax.set_yticks([])
        else:
            for s in ['molten', 'layered', 'solid']:
                m = idx[post['state'][idx] == s]
                ax.scatter(post[PARAMS[j]][m], post[PARAMS[i]][m], s=2, alpha=0.4,
                           color=STATE_COLOR[s], linewidths=0)
            ax.set_ylim(*PRIOR[PARAMS[i]])
        ax.set_xlim(*PRIOR[PARAMS[j]])
        if i == nP - 1:
            ax.set_xlabel(LABEL[PARAMS[j]], fontsize=12)
        else:
            ax.set_xticklabels([])
        if j == 0 and i > 0:
            ax.set_ylabel(LABEL[PARAMS[i]], fontsize=12)
        elif j > 0:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=14)
fig.legend(handles=[mpatches.Patch(color=STATE_COLOR[s], label=s) for s in STATES],
           fontsize=FS, loc='upper right')
fig.tight_layout()
save(fig, 'C_corner_state')
