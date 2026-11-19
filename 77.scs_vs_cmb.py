#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
77: ScS reflector vs metallic core radius  (paper Fig. 5, conclusion B)
    Based on EXAM/python_code/15_radius_ScS_Rcmb.py (exam figure R3, a52B), updated for a64.

Rows (shared x = radius, km)
  (a) literature radii (RoG Table 11), two families:
        'fluid' = measures the fluid outer edge (ScS reflector / spin-state fluid core)
        'metal' = reports the metallic core itself (models with a molten layer)
      plus this study: R_ScS (pooled) and R_CMB per state (16-84%), because R_CMB is bimodal
  (b) posterior histograms of R_CMB and R_ScS = R_CMB + h_liq, with sd in the legend;
      grey band = outside the R_CMB prior (left of 1400 km)
  (c) optional (PANEL_C): w_S vs R_CMB by state, with w_S = 0 (pure Fe, no Ni)

Changes from 15_radius_ScS_Rcmb.py
  * a64, mars_config paths, burn-in 0.4 (same as 63v2, 75, 76, 89, 113)
  * 'none' state (H_BML < 1 km): R_ScS = R_CMB
  * this study's R_CMB rows added (per state), the prior wall is shaded
  * colours no longer clash with the state colours (molten #CD5C5C was the R_ScS colour)
  * optional panel (c); figure saved to cfg.FIG_DIR
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

import mars_config as cfg

# ══════════════════════════════════════════════════════════════════════════
# settings
# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR   = cfg.MCMC_DIR
PREFIX      = 'chain_a64_YM20'
EXCLUDE     = []
BURNIN_FRAC = 0.4
DROP_LAST   = 1
MIN_STEPS   = 100            # a64 is still running; raise to 400 when finished
H_STATE_TOL = 1.0
STATE_MIN_N = 50
SAVE_FIG    = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
OUT_DIR     = cfg.FIG_DIR
FS          = 14

XLIM    = (1380.0, 2260.0)   # right margin is the label column of panel (a)
NBIN    = 70
ALPHA   = 0.55
LABEL_X = 2250.0
PRIOR_RCMB = (1400.0, 1900.0)

# colours: kept away from the state colours used everywhere else
C_RCMB = '#5E3C99'           # metal core, purple
C_RSCS = '#E08E2B'           # fluid outer edge / ScS reflector, orange
STATES      = ['none', 'solid', 'layered', 'molten']
STATE_COLOR = {'none': '#999999', 'solid': '#4E6E8E',
               'layered': '#7B9E87', 'molten': '#CD5C5C'}

SHOW_OWN     = True
OWN_BY_STATE = True          # R_CMB is bimodal: one row per state instead of a pooled bar
OWN_STATES   = ['molten', 'layered', 'solid']
OWN_Q        = (16.0, 84.0)
PANEL_C      = True          # w_S vs R_CMB below the histograms
N_SCATTER    = 6000
SEED         = 0

# literature radii, RoG (Khan et al., submitted) Table 11, mean radius ranges
#   use = False rows are printed only
LIT = [
    # (label,                        lo,     hi,   fam,     use)
    ('Stahler 2021 (ScS)',         1790.0, 1870.0, 'fluid', True),
    ('Duran 2022 (ScS)',           1830.0, 1870.0, 'fluid', True),
    ('Drilleau 2022 (ScS)',        1730.0, 1875.0, 'fluid', True),
    ('Khan 2022 (ScS)',            1830.0, 1850.0, 'fluid', False),
    ('Wang & Tkalcic 2022',        1790.0, 1830.0, 'fluid', False),
    ('Deng & Levander 2023',       1790.0, 1870.0, 'fluid', False),
    ('Le Maistre 2023 (spin)',     1780.0, 1890.0, 'fluid', True),
    ('Irving 2023 (ScS, SKS)',     1780.0, 1810.0, 'fluid', True),
    ('Samuel 2023 (BML)',          1630.0, 1670.0, 'metal', True),
    ('Khan 2023 (BML)',            1645.0, 1705.0, 'metal', True),
    ('Khan subm. (P-PMM)',         1700.0, 1816.0, 'metal', True),
]

plt.rcParams['font.family'] = 'DejaVu Sans'


# ══════════════════════════════════════════════════════════════════════════
# load
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


samples = []
for d in sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '_*'))):
    name = os.path.basename(d)
    path = os.path.join(d, 'chain.jsonl')
    if name in EXCLUDE or not os.path.exists(path):
        continue
    recs = load_jsonl(path)[:-DROP_LAST or None]
    if len(recs) < MIN_STEPS:
        print(f'  {name:<24s} skipped ({len(recs)} steps)')
        continue
    samples += recs[int(len(recs) * BURNIN_FRAC):]
if not samples:
    raise SystemExit(f'no usable chains for {PREFIX}')
print(f'{PREFIX}: {len(samples)} post burn-in samples')


def g(s, k):
    v = s.get('params', {}).get(k, s.get(k))
    return np.nan if v is None else float(v)


R_cmb = np.array([g(s, 'R_cmb') for s in samples])
w_S   = np.array([g(s, 'w_S') for s in samples])
H_bml = np.array([g(s, 'BML_thickness') for s in samples])
h_sol = np.array([g(s, 'h_solid_km') for s in samples])
h_liq = np.array([g(s, 'h_liquid_km') for s in samples])

state = np.full(len(samples), 'fail', dtype='<U8')
s_y, l_y = np.nan_to_num(h_sol) >= H_STATE_TOL, np.nan_to_num(h_liq) >= H_STATE_TOL
state[s_y & l_y] = 'layered'
state[~s_y & l_y] = 'molten'
state[s_y & ~l_y] = 'solid'
state[H_bml < H_STATE_TOL] = 'none'
h_liq = np.where(state == 'none', 0.0, np.nan_to_num(h_liq))

good = np.isfinite(R_cmb) & (state != 'fail')
rc, hl, st, ws = R_cmb[good], h_liq[good], state[good], w_S[good]
rs = rc + hl                                     # ScS reflects from the top of the liquid
sd_rc, sd_rs = rc.std(ddof=1), rs.std(ddof=1)
sd_prior = (PRIOR_RCMB[1] - PRIOR_RCMB[0]) / np.sqrt(12.0)

# ══════════════════════════════════════════════════════════════════════════
# numbers (printed, not on the figure)
# ══════════════════════════════════════════════════════════════════════════
print('\nmarginals (pooled)')
for nm, x, sd in [('R_cmb', rc, sd_rc), ('R_ScS', rs, sd_rs)]:
    print(f'  {nm}  5/50/95 = {np.percentile(x, [5, 50, 95]).round(0)}   sd {sd:.1f} km'
          f'   shrinkage vs prior {100 * (1 - sd / sd_prior):.0f}%')
print(f'  prior sd (uniform {PRIOR_RCMB[0]:.0f}-{PRIOR_RCMB[1]:.0f}) = {sd_prior:.1f} km')

print('\nper state')
print(f"  {'state':<9s}{'n':>7s}{'med R_cmb':>11s}{'sd R_cmb':>10s}{'med R_ScS':>11s}"
      f"{'sd R_ScS':>10s}{'med w_S':>9s}{'R_cmb<1420':>12s}")
for s in STATES:
    m = st == s
    if m.sum() < 2:
        continue
    print(f'  {s:<9s}{m.sum():>7d}{np.median(rc[m]):>11.0f}{rc[m].std(ddof=1):>10.1f}'
          f'{np.median(rs[m]):>11.0f}{rs[m].std(ddof=1):>10.1f}{np.median(ws[m]):>9.3f}'
          f'{np.mean(rc[m] < PRIOR_RCMB[0] + 20):>12.3f}')
print('  solid and none: R_ScS = R_cmb by construction (h_liq = 0)')

print('\nliterature: fraction of this study inside each range (not a compatibility test)')
for nm, lo, hi, fam, use in LIT:
    x = rs if fam == 'fluid' else rc
    print(f'  {nm:<26s}{lo:>6.0f}-{hi:<6.0f}{fam:>7s}  {np.mean((x >= lo) & (x <= hi)):.3f}')

# ══════════════════════════════════════════════════════════════════════════
# figure
# ══════════════════════════════════════════════════════════════════════════
rows = [(nm, lo, 0.5 * (lo + hi), hi, fam, False) for nm, lo, hi, fam, use in LIT if use]
if SHOW_OWN:
    ql, qh = OWN_Q
    if OWN_BY_STATE:
        for s in OWN_STATES:
            m = st == s
            if m.sum() >= STATE_MIN_N:
                a, b, c = np.percentile(rc[m], [ql, 50.0, qh])
                rows.append((f'This study $R_{{CMB}}$ ({s})', a, b, c, 'metal', True))
    else:
        a, b, c = np.percentile(rc, [ql, 50.0, qh])
        rows.append(('This study $R_{CMB}$', a, b, c, 'metal', True))
    a, b, c = np.percentile(rs, [ql, 50.0, qh])
    rows.append(('This study $R_{ScS}$', a, b, c, 'fluid', True))

order = (sorted([t for t in rows if t[4] == 'metal'], key=lambda t: t[2])
         + sorted([t for t in rows if t[4] == 'fluid'], key=lambda t: t[2]))

h_top = 0.34 * len(order) + 0.5
heights = [h_top, 5.0] + ([4.0] if PANEL_C else [])
fig, axs = plt.subplots(len(heights), 1, figsize=(10.5, sum(heights) + 0.5), sharex=True,
                        gridspec_kw=dict(height_ratios=heights, hspace=0.08))
axt, axb = axs[0], axs[1]

# (a) literature
for i, (nm, lo, mid, hi, fam, own) in enumerate(order):
    y = i + 1
    col = C_RSCS if fam == 'fluid' else C_RCMB
    if own:
        axt.plot([lo, hi], [y, y], lw=9, color=col, solid_capstyle='butt', zorder=5)
        axt.plot([mid], [y], 'o', ms=7, color='w', markeredgecolor='k',
                 markeredgewidth=1.2, zorder=6)
    else:
        axt.plot([lo, hi], [y, y], lw=7, color=col, alpha=0.45, solid_capstyle='butt', zorder=3)
        axt.plot([mid], [y], 'o', ms=6, color=col, zorder=4)
    axt.plot([hi + 8, LABEL_X - 6], [y, y], ls=':', lw=0.8, color='0.65', zorder=2)
    axt.text(LABEL_X, y, nm, fontsize=FS - 3, va='center', ha='right', color=col,
             fontweight='bold' if own else 'normal')
axt.set_ylim(0.3, len(order) + 0.7)
axt.set_yticks([])
for s in ['left', 'right', 'top']:
    axt.spines[s].set_visible(False)
axt.plot([], [], lw=7, color=C_RSCS, alpha=0.55, label='measures the fluid outer edge ($R_{ScS}$)')
axt.plot([], [], lw=7, color=C_RCMB, alpha=0.55, label='reports the metal core ($R_{CMB}$)')
fig.legend(*axt.get_legend_handles_labels(), loc='upper left', fontsize=FS - 2,
           frameon=False, bbox_to_anchor=(0.1, 0.99))

# (b) posteriors
bins = np.linspace(XLIM[0], XLIM[1], NBIN + 1)
for x, col in [(rc, C_RCMB), (rs, C_RSCS)]:
    axb.hist(x, bins=bins, density=True, color=col, alpha=ALPHA, lw=0, zorder=2)
    axb.hist(x, bins=bins, density=True, histtype='step', color=col, lw=2.0, zorder=3)
    axb.axvline(np.median(x), color=col, ls='--', lw=1.6, zorder=4)
axb.plot([], [], lw=7, color=C_RCMB, alpha=ALPHA, label=f'$R_{{CMB}}$ (metal core)   sd {sd_rc:.0f} km')
axb.plot([], [], lw=7, color=C_RSCS, alpha=ALPHA,
         label=f'$R_{{ScS}} = R_{{CMB}} + h_{{liq}}$   sd {sd_rs:.0f} km')
axb.set_ylabel('posterior density', fontsize=FS)
axb.legend(loc='upper right', fontsize=FS - 3, framealpha=0.92)

# (c) w_S vs R_CMB
if PANEL_C:
    axc = axs[2]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(rc), size=min(N_SCATTER, len(rc)), replace=False)
    for s in STATES:
        m = st[idx] == s
        if m.any():
            axc.scatter(rc[idx][m], ws[idx][m], s=6, alpha=0.45, color=STATE_COLOR[s],
                        label=f'{s} ({np.sum(st == s)})', rasterized=True)
    axc.axhline(0.0, color='k', lw=1.2, ls='--')
    axc.text(PRIOR_RCMB[0] + 5, 0.006, 'pure Fe (no Ni)', fontsize=FS - 3, va='bottom')
    axc.set_ylabel(r'$w_S$', fontsize=FS)
    axc.legend(loc='upper left', fontsize=FS - 4, markerscale=2, ncol=2)

# prior wall (outside the R_CMB prior) on the posterior rows
for ax in axs[1:]:
    ax.axvspan(XLIM[0], PRIOR_RCMB[0], color='0.85', zorder=0, lw=0)
for ax in axs:
    ax.set_xlim(*XLIM)
    ax.tick_params(labelsize=14)
axs[-1].set_xlabel('radius (km)', fontsize=FS)

fig.tight_layout(rect=[0, 0, 1, 0.96])
if SAVE_FIG:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'77_{PREFIX}_radius.png')
    fig.savefig(out, dpi=200)
    print('\nsaved', out)
plt.show()
