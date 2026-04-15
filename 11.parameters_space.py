#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:06:24 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# CONFIG

JSON_FILE = "/Users/chingchen/Desktop/HeFESTo/runs/all_misfits_samuel.json"

PARAM_KEYS = [
    'T_lit', 'P_lit', 'dTdP',
    'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr'
]

MISFIT_THRESHOLD = 1.0   # adjustable
# LOAD DATA
with open(JSON_FILE) as f:
    results = json.load(f)

rows = []
for r in results:
    row = {
        'misfit': r['misfit_per_datum']
    }
    for k in PARAM_KEYS:
        row[k] = r['params'].get(k, np.nan)
    rows.append(row)

df = pd.DataFrame(rows)

print(df.describe())

# 1. Single Parameter vs Misfit
ncols = 3
nrows = int(np.ceil(len(PARAM_KEYS)/ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
axes = axes.flatten()

for ax, key in zip(axes, PARAM_KEYS):
    ax.scatter(df[key], df['misfit'], alpha=0.7)
    ax.set_xlabel(key)
    ax.set_ylabel('Misfit / datum')
    ax.axhline(MISFIT_THRESHOLD, ls='--', c='r')
    ax.set_ylim(0,4)

for ax in axes[len(PARAM_KEYS):]:
    ax.axis('off')
    

plt.tight_layout()
plt.show()


# 2. Thermal Pair Plot

plt.figure(figsize=(6,5))
sc = plt.scatter(
    df['T_lit'],
    df['P_lit'],
    c=df['misfit'],
    cmap='viridis_r',
    s=60, 
)
plt.colorbar(sc, label='Misfit / datum')
plt.xlabel('T_lit (K)')
plt.ylabel('P_lit (GPa)')
plt.title('Thermal Parameter Tradeoff')
plt.tight_layout()
plt.show()


# ================================
# 3. Acceptable Model Histograms
# ================================
acc = df[df['misfit'] < MISFIT_THRESHOLD]

print(f"\nAcceptable models: {len(acc)} / {len(df)}")

ncols = 3
nrows = int(np.ceil(len(PARAM_KEYS)/ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
axes = axes.flatten()

for ax, key in zip(axes, PARAM_KEYS):
    ax.hist(acc[key], bins=15)
    ax.set_xlabel(key)
    ax.set_ylabel('Count')

for ax in axes[len(PARAM_KEYS):]:
    ax.axis('off')

plt.tight_layout()
plt.show()


# ================================
# 4. Corner Plot of Acceptable Models
# ================================
subset_keys = ['T_lit', 'P_lit', 'dTdP', 'Fe', 'Mg']

sns.pairplot(
    acc[subset_keys],
    corner=True,
    diag_kind='hist'
)
plt.show()