#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
76: is a compositionally distinct BML required?  (paper Fig. 3, conclusion A)

Panel (a)  posterior / prior density of H_BML
           first bin = [0, H_MIN_BML) = state 'none' (no BML), then BINS_KM-wide bins;
           ratio ~ 1 everywhere -> the data neither favour nor exclude a layer.
           With a Savage-Dickey reading, the ratio in the 'none' bin is the Bayes factor
           for 'no BML' (only if the prior is the true sampled prior, see PRIOR_PREFIX).
Panel (b)  H2 test: Mg#_BML - Mg#_mantle vs H_BML, coloured by state;
           grey band = |dMg#| < DMG_H2 (a solid BML of mantle composition, i.e. H2).
Panel (c)  data fit by state: distribution of MISFIT_KEY (default misfit_tt).
           overlapping curves -> the states fit the data equally well.

Prior
  PRIOR_PREFIX = None   -> uniform prior from PRIOR (analytic for H_BML, Monte Carlo for dMg#).
                           Ignores forward-model failures, so it is only approximate.
  PRIOR_PREFIX = 'chain_prior_...' -> histogram of a --prior_only run (preferred:
                           includes rejections by the forward model and the prior volume
                           of each state).

Uncertainty: ratios are also computed per chain; the printed 16-84% range across chains
is a rough error bar (samples within a chain are autocorrelated).

Notes
  * a64 (Hd, with gravity penalty) has a 0.63 penalty jump at H_BML = 1 km, which favours
    'none'. Compare with a66 (no penalty) via COMPARE before reading panel (a).
  * Paths come from mars_config; figure is saved to cfg.FIG_DIR.
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

import mars_config as cfg

# ══════════════════════════════════════════════════════════════════════════
# settings
# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR    = cfg.MCMC_DIR
PREFIX       = 'chain_a64_YM20'          # main result (Hd, gravity penalty)
COMPARE      = 'chain_a66_YM20'          # same model without gravity penalty; None to skip
PRIOR_PREFIX = None                      # e.g. 'chain_prior_hd_YM20' once it exists
EXCLUDE      = []
BURNIN_FRAC  = 0.4
DROP_LAST    = 1
MIN_STEPS    = 100
H_MIN_BML    = 1.0                       # km, below this the state is 'none'
BIN_KM       = 20.0                      # bin width for H_BML >= H_MIN_BML
DMG_H2       = 0.02                      # |Mg#_BML - Mg#_mantle| defining H2
MISFIT_KEY   = 'misfit_tt'               # or 'misfit' (total, includes penalties)
N_SCATTER    = 6000
N_MC_PRIOR   = 2_000_000
SEED         = 0
SAVE_FIG     = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
OUT_DIR      = cfg.FIG_DIR

PRIOR = {                                # must match 26fall_main_mcmc_a63.py
    'Mg#':           (0.50, 0.90),
    'Mg#_bulk_bml':  (0.20, 0.90),
    'BML_thickness': (0.0, 400.0),
}

STATES      = ['none', 'solid', 'layered', 'molten']
STATE_COLOR = {'none': '#999999', 'solid': '#4E6E8E',
               'layered': '#7B9E87', 'molten': '#CD5C5C'}
FS = 14
plt.rcParams['font.family'] = 'DejaVu Sans'


# ══════════════════════════════════════════════════════════════════════════
# loading
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


def load_run(prefix):
    chains = []
    for d in sorted(glob.glob(os.path.join(CHAIN_DIR, f'{prefix}_*'))):
        name = os.path.basename(d)
        path = os.path.join(d, 'chain.jsonl')
        if name in EXCLUDE or not os.path.exists(path):
            continue
        recs = load_jsonl(path)[:-DROP_LAST or None]
        if len(recs) < MIN_STEPS:
            print(f'  {name:<26s} skipped ({len(recs)} steps)')
            continue
        chains.append(recs)
    if not chains:
        raise SystemExit(f'no usable chains for {prefix} in {CHAIN_DIR}')
    print(f'  {prefix}: {len(chains)} chains, steps {min(map(len, chains))}-{max(map(len, chains))}')
    return chains


def get(rec, key):
    v = rec.get('params', {}).get(key, rec.get(key))
    return np.nan if v is None else float(v)


KEYS = ['Mg#', 'Mg#_bulk_bml', 'BML_thickness', 'h_solid_km', 'h_liquid_km',
        'misfit', 'misfit_tt', 'grav_sigma']


def to_post(chains):
    recs, cid = [], []
    for c, ch in enumerate(chains):
        b = int(len(ch) * BURNIN_FRAC)
        recs += ch[b:]
        cid += [c] * (len(ch) - b)
    post = {k: np.array([get(r, k) for r in recs]) for k in KEYS}
    post['chain'] = np.array(cid)
    hs, hl, H = post['h_solid_km'], post['h_liquid_km'], post['BML_thickness']
    st = np.full(len(H), 'fail', dtype='<U8')
    st[(hs >= 1) & (hl >= 1)] = 'layered'
    st[(hs < 1) & (hl >= 1)] = 'molten'
    st[(hs >= 1) & (hl < 1)] = 'solid'
    st[H < H_MIN_BML] = 'none'
    post['state'] = st
    post['dMg'] = post['Mg#_bulk_bml'] - post['Mg#']
    return post


# ══════════════════════════════════════════════════════════════════════════
# prior
# ══════════════════════════════════════════════════════════════════════════
H_LO, H_HI = PRIOR['BML_thickness']
EDGES = np.r_[H_LO, np.arange(H_MIN_BML, H_HI + 1e-9, BIN_KM)]
if EDGES[-1] < H_HI:
    EDGES = np.r_[EDGES, H_HI]
WIDTH = np.diff(EDGES)


def prior_bin_frac(prior_post=None):
    """prior probability of each H_BML bin"""
    if prior_post is not None:
        h, _ = np.histogram(prior_post['BML_thickness'], bins=EDGES)
        return h / h.sum()
    return WIDTH / (H_HI - H_LO)


def prior_h2_frac(prior_post=None):
    """prior probability that |dMg#| < DMG_H2"""
    if prior_post is not None:
        return np.mean(np.abs(prior_post['dMg']) < DMG_H2)
    rng = np.random.default_rng(SEED + 1)
    m = rng.uniform(*PRIOR['Mg#'], N_MC_PRIOR)
    b = rng.uniform(*PRIOR['Mg#_bulk_bml'], N_MC_PRIOR)
    return np.mean(np.abs(b - m) < DMG_H2)


def ratio_per_chain(post, frac_prior):
    """posterior/prior ratio of each H_BML bin, overall and per chain"""
    h, _ = np.histogram(post['BML_thickness'], bins=EDGES)
    overall = (h / h.sum()) / frac_prior
    per = []
    for c in np.unique(post['chain']):
        x = post['BML_thickness'][post['chain'] == c]
        hc, _ = np.histogram(x, bins=EDGES)
        per.append((hc / max(hc.sum(), 1)) / frac_prior)
    return overall, np.array(per)


# ══════════════════════════════════════════════════════════════════════════
# run
# ══════════════════════════════════════════════════════════════════════════
print('loading')
post = to_post(load_run(PREFIX))
cmp_post = to_post(load_run(COMPARE)) if COMPARE else None
prior_post = to_post(load_run(PRIOR_PREFIX)) if PRIOR_PREFIX else None
if prior_post is None:
    print('  prior: analytic uniform (ignores forward-model failures; use a --prior_only run when available)')

f_prior = prior_bin_frac(prior_post)
ratio, ratio_ch = ratio_per_chain(post, f_prior)
if cmp_post is not None:
    ratio_cmp, ratio_cmp_ch = ratio_per_chain(cmp_post, f_prior)

print('\n1. states (posterior)')
for name, p in [(PREFIX, post)] + ([(COMPARE, cmp_post)] if cmp_post is not None else []):
    n = len(p['state'])
    print(f'  {name}: ' + '  '.join(f'{s} {np.mean(p["state"] == s):.3f}' for s in STATES)
          + f'  (n={n}, fail {np.mean(p["state"] == "fail"):.3f})')


def q(a):
    a = a[np.isfinite(a)]
    return np.percentile(a, [16, 50, 84]) if a.size else [np.nan] * 3


print(f'\n2. H1: posterior/prior ratio for none (H_BML < {H_MIN_BML:g} km)')
lo, med, hi = q(ratio_ch[:, 0])
print(f'  {PREFIX}: {ratio[0]:.2f}   per chain median {med:.2f} [16-84%: {lo:.2f}, {hi:.2f}]')
if cmp_post is not None:
    lo, med, hi = q(ratio_cmp_ch[:, 0])
    print(f'  {COMPARE}: {ratio_cmp[0]:.2f}   per chain median {med:.2f} [16-84%: {lo:.2f}, {hi:.2f}]')
print('  (Savage-Dickey Bayes factor for no BML only if the prior is a sampled --prior_only run)')

print(f'\n3. H2: solid BML with |Mg#_BML - Mg#_mantle| < {DMG_H2}')
p_h2_prior = prior_h2_frac(prior_post)
for name, p in [(PREFIX, post)] + ([(COMPARE, cmp_post)] if cmp_post is not None else []):
    solid = p['state'] == 'solid'
    in_h2 = solid & (np.abs(p['dMg']) < DMG_H2)
    frac_in_solid = in_h2.sum() / max(solid.sum(), 1)
    print(f'  {name}: posterior fraction {in_h2.mean():.4f};  within solid {frac_in_solid:.4f};'
          f'  prior P(|dMg#|<{DMG_H2}) {p_h2_prior:.4f}  ->  ratio within solid {frac_in_solid / p_h2_prior:.2f}')
print('  (ratio within solid ~ 1: the solid models do not avoid mantle composition)')

print(f'\n4. data fit by state: {MISFIT_KEY}, median [5, 95]')
for s in STATES:
    m = (post['state'] == s) & np.isfinite(post[MISFIT_KEY])
    if m.any():
        a = np.percentile(post[MISFIT_KEY][m], [5, 50, 95])
        print(f'  {s:<8s} {a[1]:7.2f} [{a[0]:.2f}, {a[2]:.2f}]  (n={m.sum()})')


# ══════════════════════════════════════════════════════════════════════════
# figure
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 3, figsize=(20, 6.2))

# (a) posterior / prior of H_BML
a = ax[0]
lo_c, hi_c = np.nanpercentile(ratio_ch, 16, axis=0), np.nanpercentile(ratio_ch, 84, axis=0)
E = EDGES[1:]                      # edges of the bins with H_BML >= H_MIN_BML (len = nbins)
a.stairs(hi_c[1:], E, baseline=lo_c[1:], fill=True, color='k', alpha=0.12, lw=0,
         label='16-84% across chains')
a.stairs(ratio[1:], E, color='k', lw=2, label=PREFIX.split('_')[1])
if cmp_post is not None:
    a.stairs(ratio_cmp[1:], E, color='k', lw=1.5, ls='--',
             label=COMPARE.split('_')[1] + ' (no grav. penalty)')
# 'none' bin drawn as a bar at the left edge (its width is only H_MIN_BML)
a.bar(-12, ratio[0], width=16, color=STATE_COLOR['none'], edgecolor='k',
      yerr=[[max(ratio[0] - lo_c[0], 0)], [max(hi_c[0] - ratio[0], 0)]], capsize=4, label='none (H$_{BML}$<1 km)')
if cmp_post is not None:
    a.bar(-12, ratio_cmp[0], width=16, fill=False, edgecolor='k', ls='--')
a.axhline(1.0, color='0.4', lw=1, ls=':')
a.set_xlim(-25, H_HI)
a.set_xlabel(r'$H_{\mathrm{BML}}$ (km)', fontsize=FS)
a.set_ylabel('posterior / prior density', fontsize=FS)
a.set_title('(a) BML thickness', fontsize=FS)
a.legend(fontsize=10, loc='upper right')

# (b) H2: dMg# vs H_BML
b = ax[1]
rng = np.random.default_rng(SEED)
idx = rng.choice(len(post['state']), size=min(N_SCATTER, len(post['state'])), replace=False)
for s in STATES:
    m = post['state'][idx] == s
    if m.any():
        b.scatter(post['dMg'][idx][m], post['BML_thickness'][idx][m], s=6, alpha=0.4,
                  color=STATE_COLOR[s], label=f'{s} ({np.sum(post["state"] == s)})',
                  rasterized=True)
b.axvspan(-DMG_H2, DMG_H2, color='k', alpha=0.10, lw=0, label=f'|$\\Delta$Mg#| < {DMG_H2} (H2)')
b.axvline(0, color='k', lw=0.8)
b.set_xlabel(r'Mg#$_{\mathrm{BML}}$ $-$ Mg#$_{\mathrm{mantle}}$', fontsize=FS)
b.set_ylabel(r'$H_{\mathrm{BML}}$ (km)', fontsize=FS)
b.set_title('(b) composition contrast', fontsize=FS)
b.legend(fontsize=10, loc='upper left', markerscale=2)

# (c) misfit by state
c = ax[2]
fin = np.isfinite(post[MISFIT_KEY])
bins = np.linspace(*np.nanpercentile(post[MISFIT_KEY][fin], [0.5, 99.5]), 40)
for s in STATES:
    m = (post['state'] == s) & fin
    if m.sum() > 20:
        c.hist(post[MISFIT_KEY][m], bins=bins, density=True, histtype='step', lw=2,
               color=STATE_COLOR[s], label=s)
c.set_xlabel({'misfit_tt': r'travel-time misfit $\Phi_{\mathrm{TT}}$',
              'misfit': r'total misfit $\Phi$'}.get(MISFIT_KEY, MISFIT_KEY), fontsize=FS)
c.set_ylabel('density', fontsize=FS)
c.set_title('(c) data fit by state', fontsize=FS)
c.legend(fontsize=11)

for p in ax:
    p.tick_params(labelsize=14)
    p.grid(alpha=0.3)

prior_tag = PRIOR_PREFIX if PRIOR_PREFIX else 'analytic uniform prior'
fig.suptitle(f'{PREFIX}   (prior: {prior_tag})', fontsize=FS)
fig.tight_layout()
if SAVE_FIG:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'76_{PREFIX}_bml_requirement.png')
    fig.savefig(out, dpi=200)
    print('\nsaved', out)
plt.show()
