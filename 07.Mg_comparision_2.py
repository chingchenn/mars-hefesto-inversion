#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:42:17 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fe# of each phase vs Pressure and Mg#
Style: matches your existing BML phase diagram code
Usage on your machine:
  - Set BASE_DIR to your HeFESTo folder
  - CASES lists (folder_name, Mg#) pairs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────────────
BASE_DIR = '/Users/chingchen/Desktop/HeFESTo'

CASES = [
    ('21_EM1_BML_Mg75', 0.75),
    ('22_EM1_BML_Mg70', 0.70),
    ('23_EM1_BML_Mg65', 0.65),
    ('24_EM1_BML_Mg60', 0.60),
    ('25_EM1_BML_Mg55', 0.55),
    ('26_EM1_BML_Mg50', 0.50),
]

P_FIXED   = [18, 19, 20, 21, 22]   # GPa for right panels
fontsize  = 13
colors    = ['#282130','#849DAB','#CD5C5C','#35838D','#97795D','#414F67']
colors_P  = ['#E07B54','#CD5C5C','#35838D','#414F67','#282130']

# ── HELPER: read fort.99 and compute Fe# for each phase ──────────────────
def compute_fe_content(path):
    """
    Returns DataFrame with columns:
      P, T, ri_Fe, ri_vol, mw_Fe, mw_vol, gt_Fe, gt_vol
    Fe# = molar Fe / (Mg+Fe) for each phase
    """
    f99 = pd.read_csv(os.path.join(path, 'fort.99'), sep=r'\s+')
    f66 = pd.read_csv(os.path.join(path, 'fort.66'), sep=r'\s+')

    out = pd.DataFrame()
    out['P'] = f99['Pi']
    out['T'] = f99['Ti']

    # ── Ringwoodite: (Mg,Fe)2SiO4 ──
    ri_mg = f99['mgri']
    ri_fe = f99['feri']
    ri_sum = ri_mg + ri_fe
    out['ri_Fe']  = np.where(ri_sum > 0.01, ri_fe / ri_sum, np.nan)
    out['ri_vol'] = f66['ri'] * 100

    # ── Ferropericlase: (Mg,Fe)O ──
    mw_mg = f99['pe']
    mw_fe = f99['wu']
    mw_sum = mw_mg + mw_fe
    out['mw_Fe']  = np.where(mw_sum > 0.005, mw_fe / mw_sum, np.nan)
    out['mw_vol'] = f66['mw'] * 100

    # ── Garnet: py=Mg3, al=Fe3, gr=Ca3, mgmj=Mg4Si, namj=NaMgSi ──
    # Fe# = almandine / (pyrope + almandine + grossular + majorites)
    gt_mg = f99['py'] + f99['mgmj']
    gt_fe = f99['al']
    gt_ca = f99['gr'] + f99['andr']
    gt_na = f99['namj']
    gt_sum = gt_mg + gt_fe + gt_ca + gt_na
    out['gt_Fe']  = np.where(gt_sum > 0.01, gt_fe / gt_sum, np.nan)
    out['gt_Mg']  = np.where(gt_sum > 0.01, gt_mg / gt_sum, np.nan)
    out['gt_vol'] = f66['gt'] * 100

    return out

# ── LOAD ALL CASES ──────────────────────────────────────────────────────────
all_data = {}
for folder, mg in CASES:
    path = os.path.join(BASE_DIR, folder)
    try:
        all_data[mg] = compute_fe_content(path)
    except Exception as e:
        print(f"  Warning: could not load {folder}: {e}")

mg_vals_loaded = sorted(all_data.keys(), reverse=True)

# ── PLOT ────────────────────────────────────────────────────────────────────
phases = [
    ('ri',  'Ringwoodite',     'ri_Fe',  'ri_vol'),
    ('mw',  'Ferropericlase',  'mw_Fe',  'mw_vol'),
    ('gt',  'Garnet',          'gt_Fe',  'gt_vol'),
]

fig, axes = plt.subplots(3, 2, figsize=(13, 13))
fig.suptitle('Fe# of BML Phases vs Pressure and Mg#', fontsize=fontsize+2, y=1.01)

for row, (phase_key, phase_name, fe_col, vol_col) in enumerate(phases):
    ax_left  = axes[row, 0]   # Fe# vs P
    ax_right = axes[row, 1]   # Fe# vs Mg# at fixed P

    # ── LEFT PANEL: Fe# vs Pressure (one line per Mg#) ──
    for uu, mg in enumerate(mg_vals_loaded):
        df = all_data[mg]
        mask = df[vol_col] > 0.5     # only plot where phase exists (>0.5 vol%)
        if mask.sum() < 2:
            continue
        ax_left.plot(df['P'][mask], df[fe_col][mask],
                     label=f'Mg#={mg:.2f}',
                     color=colors[uu], linewidth=1.8)

    ax_left.set_xlabel('Pressure (GPa)', fontsize=fontsize)
    ax_left.set_ylabel('Fe#', fontsize=fontsize)
    ax_left.set_title(f'{phase_name}  —  Fe# vs P', fontsize=fontsize)
    ax_left.tick_params(labelsize=fontsize)
    ax_left.legend(fontsize=fontsize-3)
    ax_left.grid(alpha=0.4)
    ax_left.set_xlim(16, 23)
    ax_left.set_ylim(bottom=0)

    # ── RIGHT PANEL: Fe# vs Mg# at fixed P ──
    for uu, p_fix in enumerate(P_FIXED):
        fe_at_p  = []
        mg_avail = []
        for mg in mg_vals_loaded:
            df = all_data[mg]
            idx = np.abs(df['P'] - p_fix).argmin()
            vol = df[vol_col].iloc[idx]
            fe  = df[fe_col].iloc[idx]
            if vol > 0.5 and not np.isnan(fe):
                fe_at_p.append(fe)
                mg_avail.append(mg)
        if len(mg_avail) < 2:
            continue
        ax_right.plot(mg_avail, fe_at_p,
                      marker='o', linewidth=1.8,
                      label=f'P={p_fix} GPa',
                      color=colors_P[uu])

    ax_right.set_xlabel('Bulk Mg#', fontsize=fontsize)
    ax_right.set_ylabel('Fe#', fontsize=fontsize)
    ax_right.set_title(f'{phase_name}  —  Fe# vs Mg#', fontsize=fontsize)
    ax_right.tick_params(labelsize=fontsize)
    ax_right.legend(fontsize=fontsize-3)
    ax_right.grid(alpha=0.4)
    ax_right.set_xlim(0.48, 0.77)
    ax_right.set_ylim(bottom=0)

plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/BML_Fe_content.pdf', bbox_inches='tight', dpi=150)
# plt.savefig('/mnt/user-data/outputs/BML_Fe_content.png', bbox_inches='tight', dpi=150)
# print("Saved: BML_Fe_content.pdf / .png")
plt.show()