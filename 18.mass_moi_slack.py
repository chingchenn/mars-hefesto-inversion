#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 18: how much of the loosened mass / MoI sigma do the posteriors actually use?
#
#   python 18.mass_moi_slack.py                (SAVE_FIG=1 to save to FIG_DIR)
#
# The main script uses MARS_MASS_SIGMA = 0.27 % (9x the 0.031 % formal) and MOI_SIGMA = 6.4e-4
# (11x the Konopliv 2020 formal 6e-5), both loosened for the fixed Samuel 2023 0-100 km column.
# If the posterior residuals sit well inside the formal errors, the slack is unused and
# tightening changes little. If they spread to ~1 sigma, the slack is absorbing something
# (likely the R_cmb / BML trade-off) and the crust has to become a free parameter.
#
# Terminal: per run and state, signed residuals in units of the formal error, the fraction
# inside the Drilleau 2026 window (|dMoI| <= 4 x 6e-5) and inside 3 formal sigma of mass,
# and the state fractions after applying those windows as a filter.
# Panels: (a) MoI residual, (b) mass residual, (c) MoI residual vs R_cmb coloured by state.
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

RUNS = [('a64', 'chain_a64_YM20', '#333333'),
        ('a70', 'chain_a70_YM20', '#CD5C5C'),
        ('a72', 'chain_a72_YM20', '#8B0000'),
        ('a74', 'chain_a74_YM20', '#FF8C00'),
        ('a71', 'chain_a71_YM20', '#7FA7D9'),
        ('a73', 'chain_a73_YM20', '#1F4E9E')]
EXCLUDE   = ['chain_a71_YM20_10', 'chain_a71_YM20_12']   # stuck without SKS (155)
BURN_FRAC = 0.4
MIN_STEPS = 300

MASS_OBS   = 6.4171e23
MOI_OBS    = 0.3634
MASS_FORM  = 3.1e-4          # 0.031 % formal (mean density 3935 +- 1.2 kg/m3)
MOI_FORM   = 6.0e-5          # Konopliv 2020 formal
MASS_USED  = 2.7e-3          # MARS_MASS_SIGMA / MARS_MASS_OBS in the main script
MOI_USED   = 6.4e-4          # MOI_SIGMA in the main script
DRIL_MOI   = 4*MOI_FORM      # Drilleau 2026: reject if MoI is outside 4 formal sigma
MASS_WIN   = 3*MASS_FORM     # 3 formal sigma on mass

STATE_COL = {'solid': '#4C72B0', 'layered': '#8C8C8C', 'molten': '#D9534F', 'none': '#BBBBBB'}
SAVE_FIG  = os.environ.get('SAVE_FIG') == '1'
FS        = 14
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_jsonl(path):
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return recs


def state(r):
    if r['params']['BML_thickness'] < 1.0:
        return 'none'
    hs, hl = r.get('h_solid_km'), r.get('h_liquid_km')
    if hs is None or hl is None:
        return 'fail'
    if hl < 1:
        return 'solid'
    if hs < 1:
        return 'molten'
    return 'layered'


data = {}
for lab, prefix, col in RUNS:
    recs = []
    for d in sorted(glob.glob(os.path.join(MCMC_DIR, prefix + '_[0-9][0-9]'))):
        fp = os.path.join(d, 'chain.jsonl')
        if os.path.basename(d) in EXCLUDE or not os.path.exists(fp):
            continue
        R = load_jsonl(fp)
        if len(R) >= MIN_STEPS:
            recs += R[int(len(R)*BURN_FRAC):]
    recs = [r for r in recs if r.get('moi_pred') is not None and r.get('M_pred') is not None]
    if not recs:
        print(f'{lab}: no chains')
        continue
    dmoi  = np.array([r['moi_pred'] for r in recs]) - MOI_OBS
    dmass = np.array([r['M_pred'] for r in recs])/MASS_OBS - 1.0
    st    = np.array([state(r) for r in recs])
    rcmb  = np.array([r['params']['R_cmb'] for r in recs])
    data[lab] = dict(dmoi=dmoi, dmass=dmass, st=st, rcmb=rcmb, col=col)

print('\nresiduals in FORMAL sigma: median [5, 95];  |.| > used sigma;  inside windows')
print(f'  used sigma = {MOI_USED/MOI_FORM:.1f} formal (MoI), {MASS_USED/MASS_FORM:.1f} formal (mass)')
for lab, v in data.items():
    print(f'\n{lab}  n={len(v["st"])}')
    for s in ['all', 'solid', 'layered', 'molten']:
        m = np.ones(len(v['st']), bool) if s == 'all' else v['st'] == s
        if m.sum() < 30:
            continue
        zm, zs = v['dmoi'][m]/MOI_FORM, v['dmass'][m]/MASS_FORM
        in_moi  = np.abs(v['dmoi'][m]) <= DRIL_MOI
        in_mass = np.abs(v['dmass'][m]) <= MASS_WIN
        print(f'  {s:>8s} n={m.sum():6d}  MoI {np.median(zm):+6.1f} [{np.percentile(zm, 5):+6.1f}, {np.percentile(zm, 95):+6.1f}]'
              f'  >1used {np.mean(np.abs(v["dmoi"][m]) > MOI_USED):.2f}'
              f'  | mass {np.median(zs):+6.1f} [{np.percentile(zs, 5):+6.1f}, {np.percentile(zs, 95):+6.1f}]'
              f'  >1used {np.mean(np.abs(v["dmass"][m]) > MASS_USED):.2f}'
              f'  | in Drilleau MoI {np.mean(in_moi):.2f}  in 3sig mass {np.mean(in_mass):.2f}'
              f'  both {np.mean(in_moi & in_mass):.2f}')
    keep = (np.abs(v['dmoi']) <= DRIL_MOI) & (np.abs(v['dmass']) <= MASS_WIN)
    if keep.sum() >= 30:
        f0 = '  '.join(f'{s} {np.mean(v["st"] == s):.2f}' for s in ['solid', 'layered', 'molten'])
        f1 = '  '.join(f'{s} {np.mean(v["st"][keep] == s):.2f}' for s in ['solid', 'layered', 'molten'])
        print(f'  states all      : {f0}   R_cmb {np.median(v["rcmb"]):.0f}')
        print(f'  states filtered : {f1}   R_cmb {np.median(v["rcmb"][keep]):.0f}   (n={keep.sum()})')
    else:
        print(f'  filtered: only {keep.sum()} samples left')

fig, axs = plt.subplots(1, 3, figsize=(19, 6))
bins_m = np.linspace(-3*MOI_USED, 3*MOI_USED, 61)
bins_s = np.linspace(-3*MASS_USED, 3*MASS_USED, 61)
for lab, v in data.items():
    axs[0].hist(v['dmoi'], bins=bins_m, density=True, histtype='step', lw=2, color=v['col'], label=lab)
    axs[1].hist(100*v['dmass'], bins=100*bins_s, density=True, histtype='step', lw=2, color=v['col'], label=lab)
for ax, used, form, win in [(axs[0], MOI_USED, MOI_FORM, DRIL_MOI), (axs[1], 100*MASS_USED, 100*MASS_FORM, 100*MASS_WIN)]:
    ax.axvspan(-used, used, color='0.85', zorder=0, label='used 1 sigma')
    ax.axvspan(-win, win, color='#9ACD32', alpha=0.35, zorder=0, label='formal window')
    ax.axvline(0, color='k', lw=0.8)
axs[0].set_xlabel('MoI pred - obs', fontsize=FS)
axs[1].set_xlabel('mass pred / obs - 1 [%]', fontsize=FS)
for ax in axs[:2]:
    ax.set_ylabel('density', fontsize=FS)
    ax.legend(loc='lower right', fontsize=FS - 4)
axs[0].set_title('(a) MoI residual', fontsize=FS)
axs[1].set_title('(b) mass residual', fontsize=FS)

ax = axs[2]
for lab, v in data.items():
    for s, c in STATE_COL.items():
        m = v['st'] == s
        ax.scatter(v['rcmb'][m], v['dmoi'][m], s=3, alpha=0.25, color=c, rasterized=True,
                   label=s if lab == list(data)[0] else None)
ax.axhspan(-DRIL_MOI, DRIL_MOI, color='#9ACD32', alpha=0.35, zorder=0)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('R_cmb [km]', fontsize=FS)
ax.set_ylabel('MoI pred - obs', fontsize=FS)
ax.set_title('(c) MoI residual vs R_cmb (all runs)', fontsize=FS)
ax.legend(loc='lower right', fontsize=FS - 4, markerscale=4)
for ax in axs:
    ax.tick_params(labelsize=14)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(os.path.join(FIG_DIR, '18_mass_moi_slack.png'), dpi=200)
plt.show()
