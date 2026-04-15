#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:44:34 2026

@author: chingchen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── CONFIG ─────────────────────────────────────────
file_path = '/Users/chingchen/Desktop/HeFESTo/'
bulk_model = '02_YM_bulk_mantle_BML_T'
bml_model  = '21_EM1_BML_Mg75'
SPLICE_P   = 17.0   
THRESHOLD  = 0.005
fontsize   = 15

# ── READ 兩個fort.66 ───────────────────────────────
df_bulk = pd.read_csv(file_path + bulk_model + '/fort.66', sep=r'\s+')
df_bml  = pd.read_csv(file_path + bml_model  + '/fort.66', sep=r'\s+')

# ── 切割 ───────────────────────────────────────────
df_upper = df_bulk[df_bulk['Pi'] <= SPLICE_P].copy()
df_lower = df_bml [df_bml ['Pi'] >  SPLICE_P].copy()

# ── 合併，缺少的欄位補0 ────────────────────────────
all_cols = set(df_upper.columns) | set(df_lower.columns)
for col in all_cols:
    if col not in df_upper.columns:
        df_upper[col] = 0.0
    if col not in df_lower.columns:
        df_lower[col] = 0.0

df = pd.concat([df_upper, df_lower], ignore_index=True)
P  = df['Pi'].values

# ── 以下跟原本一樣 ─────────────────────────────────
IGNORE = ['Pi', 'Ti', 'depth']
phases = []
for col in df.columns:
    if col in IGNORE:
        continue
    if df[col].max() > THRESHOLD:
        phases.append(col)



COLOR_MAP = {
    'gt':  '#E6862F',   # Garnet
    'ri':  '#4C72B0',   # Ringwoodite

    'mw':  '#D62728',   # Ferropericlase（紅）
    'fp':  '#D62728',

    'st':  '#9467BD',   # silica phase
    'pv':  '#8C564B',   # bridgmanite / pv
    'capv':'#17BECF',   # Ca-pv

    'cpx': '#1F77B4',   # 深藍
    'opx': '#FFBF00',   # 黃
    'ol':  '#2CA02C',   # 綠
    'wa':  '#00A65A',   # 深綠

    'fea': '#E377C2',   # Fe phase
    'feg': '#BCBD22',   # 
    
    'c2c': '#AEC7E8',   # high-pressure clinopyroxene
    'plg': '#C5B0D5',   # 
    'ky':  '#FFBB78',   # 
    'qtz': '#98DF8A',   # 
    'sp':  '#FF9896',   # 
    'feg': '#BCBD22',   # 
}

# ── LABEL MAP（讓圖比較漂亮）──────────────────────
LABEL_MAP = {
    'ri':'Ringwoodite',
    'gt':'Garnet',
    'st':'Stishovite',
    'fp':'Ferropericlase',
    'mw':'Ferropericlase',
    'capv':'Ca-pv',
    'mgpv':'Bridgmanite',
    'cpx':'Cpx',
    'opx':'Opx',
    'ol':'Olivine',
    'wa':'Wadsleyite',
    'c2c':'high-pressure clinopyroxene'
}

# ── SORT────────────────────────
PHASE_ORDER = [
    'ol', 'wa', 'ri',        # olivine group
    'opx',                    # orthopyroxene
    'cpx', 'c2c',            # clinopyroxene group（相鄰！）
    'gt',                     # garnet
    'st',                     # stishovite
    'fp', 'mw',              # ferropericlase
    'capv',                   # Ca-pv
    'pv',                     # bridgmanite
    'plg', 'sp', 'ky', 'qtz', 'feg',  # minor phases
]

phases_sorted = [p for p in PHASE_ORDER if p in phases]
# ── PLOT ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,9))


ax.axhline(y=SPLICE_P, color='black', 
           linewidth=1.5, linestyle=':', alpha=0.7)
ax.text(2, SPLICE_P + 0.3, 'BML boundary', 
        fontsize=fontsize-4, color='black', alpha=0.7)


# ── PLOT phases ───────────────────────────────────
legend_patches = []
bottom = np.zeros(len(P))

for col in phases_sorted:
    if col not in df.columns:
        continue
    vals = df[col].values * 100
    color = COLOR_MAP.get(col, 'gray')
    label = LABEL_MAP.get(col, col)
    ax.fill_betweenx(P, bottom, bottom + vals,
                     color=color, alpha=0.85, linewidth=0)
    if label not in [p.get_label() for p in legend_patches]:
        legend_patches.append(mpatches.Patch(color=color, label=label))
    bottom += vals

# ── 溫度 twin axis ────────────────────────────────
PT_bulk = np.loadtxt(file_path + bulk_model + '/ad.in')
PT_bml  = np.loadtxt(file_path + bml_model  + '/ad.in')

PT_upper = PT_bulk[PT_bulk[:, 0] <= SPLICE_P]
PT_lower = PT_bml [PT_bml [:, 0] >  SPLICE_P]
PT = np.vstack([PT_upper, PT_lower])
PPP, _, TTT = PT.T

ax2 = ax.twiny()
ax2.plot(TTT, PPP, color='red', linewidth=2,
         linestyle='--', label='Temperature (K)')
ax2.set_xlabel('Temperature (K)', color='red', fontsize=fontsize)
ax2.tick_params(axis='x', labelcolor='red', labelsize=fontsize)
ax2.set_xlim(500, 2500)

# ── AXES ──────────────────────────────────────────
ax.set_ylim(23, 0)
ax.set_xlim(0, 100)
ax.set_yticks(np.arange(0, 24, 2))
ax.set_ylabel('Pressure (GPa)', fontsize=fontsize)
ax.set_xlabel('Volume fraction (%)', fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize)
ax.set_title(f'{bulk_model}\n+ {bml_model} (BML > {SPLICE_P} GPa)',
             fontsize=fontsize-2)

# ── LEGEND ────────────────────────────────────────
temp_line = mlines.Line2D([], [], color='red', linewidth=2,
                           linestyle='--', label='Temperature (K)')
splice_line = mlines.Line2D([], [], color='black', linewidth=1.5,
                             linestyle=':', label=f'BML boundary ({SPLICE_P} GPa)')
all_handles = legend_patches + [temp_line, splice_line]

ax.legend(handles=all_handles,
          loc='upper left',
          bbox_to_anchor=(1.02, 1),
          framealpha=0.9,
          fontsize=fontsize-3)

plt.tight_layout()