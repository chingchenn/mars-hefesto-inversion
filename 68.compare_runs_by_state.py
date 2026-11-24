#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 68: parameter distributions of several runs, one figure per BML state
#      (e.g. a64 Fe-S vs a70/a72/a74 +C: does adding a light element move R_cmb, T_core, ...
#       within the same state?)   (was 157; renumbered 11/23 to fill a gap)
#
#   python 68.compare_runs_by_state.py             C series (default)
#   RUN_SET=H python 68.compare_runs_by_state.py   H series
#   SAVE_FIG=1 to save to FIG_DIR
#
# Panels: step histograms (density), one colour per run, dashed line = run median.
# A run without a parameter (a64 has no w_C) is simply absent from that panel.
# States from h_solid_km / h_liquid_km (same as 153/63v3); 'all' = every state pooled.
# Burn-in 0.4 per chain (same as 63v2/81). Stage-1 runs (a70/a71, 1500 steps) are not converged:
# compare shapes, not small shifts.
# The terminal table prints median [5, 95] per run and state for every panel.
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

RUN_SET = os.environ.get('RUN_SET', 'C')
RUNS_C = {'a64': ('chain_a64_YM20', 'g'),                 # Fe-S, no SKS
          'a70': ('chain_a70_YM20', '#CD5C5C'),           # C <= 2%, soft SKS, stage 1
          'a72': ('chain_a72_YM20', '#8B0000'),           # C <= 2%, hard SKS
          'a74': ('chain_a74_YM20', '#FF8C00')}           # C <= 5%, hard SKS, stage 1
RUNS_H = {'a64': ('chain_a64_YM20', 'g'),
          'a71': ('chain_a71_YM20', '#4E6E8E'),           # H <= 1%, soft SKS, stage 1
          'a73': ('chain_a73_YM20', '#00008B')}           # H <= 1%, hard SKS
RUNS = RUNS_C if RUN_SET == 'C' else RUNS_H
STATES_SHOW = ['solid', 'layered', 'molten', 'all']
BURN_FRAC   = 0.4
MIN_STEPS   = 300
EXCLUDE     = ['chain_a71_YM20_10', 'chain_a71_YM20_12']   # stuck without SKS (155)
MIN_N       = 50          # fewer samples than this in a state -> run not drawn in that figure
NBINS       = 40
SAVE_FIG    = os.environ.get('SAVE_FIG') == '1'
FS          = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# panel: (key, label, range)   key in params or in the record; R_scs, R_bml_top derived
PANELS = [('T_lit', 'T_lit [K]', (1000, 2600)), ('P_lit', 'P_lit [GPa]', (1.4, 9.0)),
          ('Mg#', 'Mg#', (0.5, 0.9)), ('T_core', 'T_core [K]', (1550, 3600)),
          ('Mg#_bulk_bml', 'Mg#_bulk_bml', (0.2, 0.9)), ('BML_thickness', 'BML_thickness [km]', (0, 400)),
          ('w_S', 'w_S', (0.0, 0.365)), ('R_cmb', 'R_cmb [km]', (1400, 1900)),
          ('w_C', 'w_C', (0.0, 0.05)), ('w_H', 'w_H', (0.0, 0.01)),
          ('R_bml_top', 'R_bml_top [km]', (1550, 2200)), ('R_scs', 'R_scs [km]', (1550, 2000)),
          ('sks_pp_pred', 'SKS-PP pred [s] (obs 339.3)', (280, 420)), ('rho_core_mean', r'core $\rho$ mean [g/cm$^3$]', (5.8, 8.2)),
          ('Vp_core_cmb', 'core Vp at CMB [km/s]', (4.3, 5.6)), ('misfit_tt', 'misfit_tt', (15, 30))]


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


# ── load ────────────────────────────────────────────────────────────────────
data = {}
for run, (prefix, col) in RUNS.items():
    recs = []
    for d in sorted(glob.glob(os.path.join(MCMC_DIR, prefix + '_[0-9][0-9]'))):
        name, fp = os.path.basename(d), os.path.join(d, 'chain.jsonl')
        if name in EXCLUDE or not os.path.exists(fp):
            continue
        R = load_jsonl(fp)
        if len(R) < MIN_STEPS:
            continue
        recs += R[int(len(R)*BURN_FRAC):]
    if not recs:
        print(f'{run}: no chains in {MCMC_DIR}')
        continue
    v = {}
    for key, _, _ in PANELS:
        if key == 'R_scs':
            v[key] = np.array([r['params']['R_cmb'] + (r.get('h_liquid_km') or 0.0) for r in recs])
        elif key == 'R_bml_top':
            v[key] = np.array([r['params']['R_cmb'] + r['params']['BML_thickness'] for r in recs])
        else:
            v[key] = np.array([float(r['params'][key]) if key in r['params']
                               else (float(r[key]) if r.get(key) is not None else np.nan) for r in recs])
    v['state'] = np.array([state(r) for r in recs])
    data[run] = v
    print(f'{run}: {len(recs)} samples  ' + '  '.join(f'{s} {np.mean(v["state"] == s):.3f}'
                                                     for s in ('none', 'solid', 'layered', 'molten')))

# ── table ───────────────────────────────────────────────────────────────────
for st in STATES_SHOW:
    print(f'\n== {st}: median [5, 95] ' + '=' * 60)
    print(f"{'':>15s}" + ''.join(f'{run:>26s}' for run in data))
    for key, _, _ in PANELS:
        row = f'{key:>15s}'
        for run, v in data.items():
            m = np.ones(len(v['state']), bool) if st == 'all' else v['state'] == st
            x = v[key][m]
            x = x[np.isfinite(x)]
            if len(x) < MIN_N:
                row += f'{"":>26s}'
                continue
            a = np.percentile(x, [5, 50, 95])
            row += f'{a[1]:10.4g} [{a[0]:.4g}, {a[2]:.4g}]'.rjust(26)
        print(row)

# ── figures: one per state ──────────────────────────────────────────────────
for st in STATES_SHOW:
    fig, axes = plt.subplots(4, 4, figsize=(17, 14))
    for ax, (key, lab, rng) in zip(axes.ravel(), PANELS):
        bins = np.linspace(*rng, NBINS + 1)
        drawn = False
        for run, v in data.items():
            m = np.ones(len(v['state']), bool) if st == 'all' else v['state'] == st
            x = v[key][m]
            x = x[np.isfinite(x)]
            if len(x) < MIN_N:
                continue
            ax.hist(x, bins=bins, density=True, histtype='step', lw=1.8, color=RUNS[run][1],
                    label=f'{run} (n={len(x)})')
            ax.axvline(np.median(x), color=RUNS[run][1], ls='--', lw=1.3)
            drawn = True
        if not drawn:
            ax.set_visible(False)
            continue
        ax.set_xlim(*rng)
        ax.set_xlabel(lab, fontsize=FS)
        ax.tick_params(labelsize=14)
    h, l = axes[0, 0].get_legend_handles_labels()
    empty = [ax for ax in axes.ravel() if not ax.get_visible()]
    if empty:                      # put the legend in an unused panel (e.g. w_H for the C series)
        empty[0].set_visible(True)
        empty[0].axis('off')
        empty[0].legend(h, l, loc='lower right', fontsize=FS, frameon=False)
    else:
        fig.legend(h, l, loc='lower right', fontsize=FS, frameon=False)
    fig.suptitle(f'BML state: {st}   (' + ' vs '.join(data) + ')', fontsize=FS + 2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    if SAVE_FIG:
        fig.savefig(os.path.join(FIG_DIR, f'68_{"_".join(data)}_{st}.png'), dpi=200)
    plt.show()
