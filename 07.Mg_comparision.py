#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:39:14 2026

@author: chingchen
"""

#!/usr/bin/env python3
"""
Ferropericlase fraction analysis:
  Panel A: fp% vs Pressure (6 lines, one per Mg#)
  Panel B: fp% vs Mg# (fixed pressure points)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR   = '/Users/chingchen/Desktop/HeFESTo'
fontsize=13
CASES = [
    ('21_EM1_BML_Mg75', 0.75),
    ('22_EM1_BML_Mg70', 0.70),
    ('23_EM1_BML_Mg65', 0.65),
    ('24_EM1_BML_Mg60', 0.60),
    ('25_EM1_BML_Mg55', 0.55),
    ('26_EM1_BML_Mg50', 0.50),
]

# ── PHASES CONFIG ────────────────────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 5))
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(13, 5))
fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(13, 5))
fig5, (ax9, ax10) = plt.subplots(1, 2, figsize=(13, 5))
PHASES = {
    'mw': ('Ferropericlase',    ax1, ax2),
    'ri': ('Ringwoodite',       ax3, ax4),
    'gt': ('Garnet',            ax5, ax6),
    'st': ('Stishovite',        ax7, ax8),
    'pv': ('Bridgmanite (+ppv)',ax9, ax10),
}


colors = ['#282130','#849DAB','#CD5C5C','#35838D',
          '#97795D','#414F67','#4198B9','#2F4F4F']




for uu, (folder, mg) in enumerate(CASES):
    df = pd.read_csv(os.path.join(BASE_DIR, folder, 'fort.66'), sep=r'\s+')

    for phase, (label, ax_left, _) in PHASES.items():

        if phase == 'pv':
            y = (df['pv'] + df['ppv']) * 100
        else:
            y = df[phase] * 100

        ax_left.plot(df['Pi'], y,
                     label=f'Mg#={mg}',
                     color=colors[uu])
        
P_FIXED = [19, 20, 21, 22, 23]

for uu, p in enumerate(P_FIXED):

    for phase, (label,_, ax_right) in PHASES.items():

        y_vals = []
        mg_vals = []

        for folder, mg in CASES:
            df = pd.read_csv(os.path.join(BASE_DIR, folder, 'fort.66'), sep=r'\s+')

            idx = np.abs(df['Pi'] - p).argmin()

            if phase == 'pv':
                val = (df['pv'].iloc[idx] + df['ppv'].iloc[idx]) * 100
            else:
                val = df[phase].iloc[idx] * 100

            y_vals.append(val)
            mg_vals.append(mg)

        ax_right.plot(mg_vals, y_vals,marker='o',label=f'P={p} GPa',color=colors[uu])


for aa in [ax1,ax3,ax5,ax7,ax9]:
    aa.set_xlabel('Pressure', fontsize=fontsize)
    aa.legend( fontsize=fontsize-3)
    aa.set_ylabel(' (%)', fontsize=fontsize)
    aa.tick_params(axis='both', labelsize=fontsize)
    aa.grid()
    
for aa in [ax2,ax4,ax6,ax8,ax10]:
    aa.set_xlabel('Mg#', fontsize=fontsize)
    aa.legend( fontsize=fontsize-3)
    aa.set_ylabel(' (%)', fontsize=fontsize)
    aa.tick_params(axis='both', labelsize=fontsize)
    aa.grid()
    
for label, ax_left, ax_right in PHASES.values():
    ax_left.set_title(label, fontsize=fontsize)
    ax_right.set_title(label, fontsize=fontsize)