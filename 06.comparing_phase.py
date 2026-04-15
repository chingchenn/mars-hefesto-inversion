#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:33:14 2026

@author: chingchen
"""

"""
BML Phase Diagrams - 2x3 Subplot
6 folders: 21_EM1_BML_Mg75 ~ 26_EM1_BML_Mg50
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── CONFIG ─────────────────────────────────────────
BASE_DIR   = '/Users/chingchen/Desktop/HeFESTo'
OUTPUT_PDF = '/Users/chingchen/Desktop/HeFESTo/phase_BML_comparison.pdf'

CASES = [
    ('21_EM1_BML_Mg75', 'BML  Mg#=0.75'),
    ('22_EM1_BML_Mg70', 'BML  Mg#=0.70'),
    ('23_EM1_BML_Mg65', 'BML  Mg#=0.65'),
    ('24_EM1_BML_Mg60', 'BML  Mg#=0.60'),
    ('25_EM1_BML_Mg55', 'BML  Mg#=0.55'),
    ('26_EM1_BML_Mg50', 'BML  Mg#=0.50'),
]

THRESHOLD = 0.001
fontsize=12
IGNORE    = {'Pi', 'Ti', 'depth'}

# ── COLOR / LABEL MAP ──────────────────────────────
COLOR_MAP = {
    'gt':   '#E6862F',   # Garnet
    'ri':   '#4C72B0',   # Ringwoodite
    'mw':   '#D62728',   # Ferropericlase
    'fp':   '#D62728',
    'st':   '#9467BD',   # Stishovite
    'pv':   '#8C564B',   # Bridgmanite
    'capv': '#17BECF',   # Ca-perovskite
    'cpx':  '#1F77B4',   # Clinopyroxene
    'opx':  '#FFBF00',   # Orthopyroxene
    'ol':   '#2CA02C',   # Olivine
    'wa':   '#00A65A',   # Wadsleyite
    'fea':  '#E377C2',
    'feg':  '#BCBD22',
}

LABEL_MAP = {
    'gt':   'Garnet',
    'ri':   'Ringwoodite',
    'mw':   'Ferropericlase',
    'fp':   'Ferropericlase',
    'st':   'Stishovite',
    'pv':   'Bridgmanite',
    'capv': 'Ca-Pv',
    'cpx':  'Cpx',
    'opx':  'Opx',
    'ol':   'Olivine',
    'wa':   'Wadsleyite',
    'fea':  'Fe-phase A',
    'feg':  'Fe-phase G',
}

# ── PLOT FUNCTION ──────────────────────────────────
def plot_one(ax, df, title):
    P = df['Pi'].values

    phases = [c for c in df.columns
              if c not in IGNORE and df[c].max() > THRESHOLD]
    
    PHASE_ORDER = [
        'ol', 'wa', 'ri',
        'opx',
        'cpx', 'c2c',
        'gt',
        'st',
        'fp', 'mw',
        'capv',
        'pv',
        'fea', 'feg',
    ]
    phases_sorted = [p for p in PHASE_ORDER if p in phases]
    for p in phases:
        if p not in phases_sorted:
            phases_sorted.append(p)

    bottom = np.zeros(len(P))
    seen   = {}

    for col in phases_sorted:
        vals  = df[col].values * 100
        color = COLOR_MAP.get(col, 'gray')
        label = LABEL_MAP.get(col, col)
        
        # fill_between → fill_betweenx
        ax.fill_betweenx(P, bottom, bottom + vals,
                         color=color, alpha=0.85, linewidth=0)
        if label not in seen:
            seen[label] = mpatches.Patch(color=color, label=label)
        bottom += vals

    # 軸設定
    ax.set_ylim(23, 16)          # 壓力範圍，上小下大
    ax.set_xlim(90, 100)
    ax.set_yticks(np.arange(16, 24, 1))
    ax.set_ylabel('Pressure (GPa)', fontsize=fontsize)
    ax.set_xlabel('Volume fraction (%)', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.tick_params(axis='both', labelsize=fontsize )
    return seen

# ── MAIN ───────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

all_patches = {}   # merged legend across all subplots

for i, (folder, title) in enumerate(CASES):
    fort66 = os.path.join(BASE_DIR, folder, 'fort.66')
    if not os.path.exists(fort66):
        print(f'[SKIP] not found: {fort66}')
        axes[i].set_visible(False)
        continue

    df      = pd.read_csv(fort66, sep=r'\s+')
    patches = plot_one(axes[i], df, title)
    for lbl, patch in patches.items():
        all_patches.setdefault(lbl, patch)   # keep first occurrence

# ── SHARED LEGEND (right side) ────────────────────
fig.legend(
    handles=list(all_patches.values()),
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    fontsize=fontsize,
    framealpha=0.9,
    title='Phase',
    title_fontsize=10,
)

fig.suptitle('BML Phase Diagrams  (Mg# 0.50 – 0.75)',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
# plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
print(f'[SAVED] {OUTPUT_PDF}')
plt.show()