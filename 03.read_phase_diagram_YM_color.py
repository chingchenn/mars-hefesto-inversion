#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:57:19 2026

@author: chingchen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ─────────────────────────────────────────
# model = [01_YM_bulk_mantle_noBML_T,
#          02_YM_bulk_mantle_BML_T,
#          21_EM1_BML_Mg75,
#          22_EM1_BML_Mg70,
#          23_EM1_BML_Mg65,
#          24_EM1_BML_Mg60,
#          25_EM1_BML_Mg55,
#          26_EM1_BML_Mg50]
model='02_YM_bulk_mantle_BML_T'
file_path = '/Users/chingchen/Desktop/HeFESTo/'
phase_file = file_path+model+'/fort.66'
temp_file = file_path+model+'/ad.in'

# 
THRESHOLD = 0.0001   
fontsize=15

# ── READ ───────────────────────────────────────────
df = pd.read_csv(phase_file, sep=r'\s+')
P  = df['Pi'].values
# T  = df['Ti'].values

# ── AUTO DETECT PHASES ─────────────────────────────
IGNORE = ['Pi', 'Ti', 'depth']
legend_patches = []
phases = []
for col in df.columns:
    if col in IGNORE:
        continue
    if df[col].max() > THRESHOLD:
        phases.append(col)


# ── COLOR MAP────────────────────
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

bottom = np.zeros(len(P))

for col in phases_sorted:
    vals = df[col].values * 100

    color = COLOR_MAP.get(col, 'gray')
    label = LABEL_MAP.get(col, col)

    ax.fill_betweenx(P, bottom, bottom + vals,
                    color=color, alpha=0.85, linewidth=0)

   
    if label not in [p.get_label() for p in legend_patches]:
        legend_patches.append(mpatches.Patch(color=color, label=label))

    bottom += vals
# ── AXES ─────────────────────────────────────────
ax.set_ylim(22,0)
ax.set_yticks(np.arange(0, 23, 2))
ax.set_xlim(0, 100)
ax.set_ylabel('Pressure (GPa)',fontsize=fontsize)
ax.set_xlabel('Volume fraction (%)',fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize )
ax.set_title(model,fontsize=fontsize)

import matplotlib.lines as mlines

temp_line = mlines.Line2D([], [], 
                           color='red', 
                           linewidth=2, 
                           linestyle='--',
                           label='Temperature (K)')

all_handles = legend_patches + [temp_line]

ax.legend(handles=all_handles,
          loc='upper left',
          bbox_to_anchor=(1.02, 1),
          framealpha=0.9,
          fontsize=fontsize-3)
# ── TOP AXIS（temperature）────────────────────────
PPP,_,TTT = np.loadtxt(temp_file).T
ax2 = ax.twiny()   # twiny = 共享y軸（壓力），新增上方x軸

ax2.plot(TTT, PPP,
         color='red',
         linewidth=2,
         linestyle='--',
         label='Temperature (K)')

ax2.set_xlim(0,2500)

ax2.set_xlabel('Temperature (K)', color='red', fontsize=fontsize-3)
ax2.tick_params(axis='x', labelcolor='red', labelsize=fontsize-3)

plt.tight_layout()
# plt.savefig('phase_BML_Mg60.pdf', dpi=150)
# plt.show()