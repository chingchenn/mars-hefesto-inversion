#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:16:06 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- config ----------------
SAMUEL_BASE  = '/Users/chingchen/Desktop/Lunar/Mars_Samuel_2023/Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG2/'
PANEL_BML    = os.path.join(SAMUEL_BASE, 'PANEL_B')
PANEL_NOBML  = os.path.join(SAMUEL_BASE, 'PANEL_A')

KHAN_DIR     = '/Users/chingchen/Desktop/Lunar/Mars_Khan_2023/LSL_Models'
# HEFESTO_1500 = '/Users/chingchen/Desktop/HeFESTo/ad_samuel_noBML/fort.56'
HEFESTO_1400 = '/Users/chingchen/Desktop/HeFESTo/ad_samuel_noBML/fort.56'

MARS_RADIUS = 3389.5   
G_MARS      = 3.7      

labelsize = 14
bwith     = 2.5

# ---------------- HeFESTo ----------------
# with open(HEFESTO_1500) as f:
#     f.readline()
#     cols = f.readline().split()
# df1500 = pd.read_csv(HEFESTO_1500, sep=r'\s+', skiprows=2, names=cols)

# P1500   = df1500['P(GPa)'].values
# rho1500 = df1500['rho(g/cm^3)'].values
# vp1500  = df1500['VPQ(km/s)'].values * 1000 * 0.97
# vs1500  = df1500['VSQ(km/s)'].values * 1000 * 0.97

# dP1500      = np.diff(P1500) * 1e9
# rho_mid1500 = (rho1500[:-1] + rho1500[1:]) / 2
# dz1500      = dP1500 / (rho_mid1500 * 1000 * G_MARS) / 1000
# depth1500   = np.zeros(len(P1500))
# depth1500[1:] = np.cumsum(dz1500)

with open(HEFESTO_1400) as f:
    f.readline()
    cols = f.readline().split()
df1400 = pd.read_csv(HEFESTO_1400, sep=r'\s+', skiprows=2, names=cols)

P1400   = df1400['P(GPa)'].values
rho1400 = df1400['rho(g/cm^3)'].values
vp1400  = df1400['VPQ(km/s)'].values * 1000 
vs1400  = df1400['VSQ(km/s)'].values * 1000 

dP1400      = np.diff(P1400) * 1e9
rho_mid1400 = (rho1400[:-1] + rho1400[1:]) / 2
dz1400      = dP1400 / (rho_mid1400 * 1000 * G_MARS) / 1000
depth1400   = np.zeros(len(P1400))
depth1400[1:] = np.cumsum(dz1400)

# ---------------- Samuel no BML ----------------
sam_nobml_vp = []
sam_nobml_vs = []

for fpath in sorted(glob.glob(os.path.join(PANEL_NOBML, 'vp*.dat'))):
    d = np.loadtxt(fpath)
    vel = d[:, 0]
    radius = d[:, 1]
    depth = MARS_RADIUS - radius
    idx = np.argsort(depth)
    sam_nobml_vp.append((depth[idx], vel[idx]))

for fpath in sorted(glob.glob(os.path.join(PANEL_NOBML, 'vs*.dat'))):
    d = np.loadtxt(fpath)
    vel = d[:, 0]
    radius = d[:, 1]
    depth = MARS_RADIUS - radius
    idx = np.argsort(depth)
    sam_nobml_vs.append((depth[idx], vel[idx]))

# ---------------- Samuel with BML ----------------
sam_bml_vp = []
sam_bml_vs = []

for fpath in sorted(glob.glob(os.path.join(PANEL_BML, 'vp*.dat'))):
    d = np.loadtxt(fpath)
    vel = d[:, 0]
    radius = d[:, 1]
    depth = MARS_RADIUS - radius
    idx = np.argsort(depth)
    sam_bml_vp.append((depth[idx], vel[idx]))

for fpath in sorted(glob.glob(os.path.join(PANEL_BML, 'vs*.dat'))):
    d = np.loadtxt(fpath)
    vel = d[:, 0]
    radius = d[:, 1]
    depth = MARS_RADIUS - radius
    idx = np.argsort(depth)
    sam_bml_vs.append((depth[idx], vel[idx]))

# ---------------- Khan ----------------
khan_vp = []
khan_vs = []

for fpath in sorted(glob.glob(os.path.join(KHAN_DIR, 'Model_*.txt'))):
    d = np.loadtxt(fpath, comments='#')
    depth = d[:, 0]
    vp = d[:, 1] * 1000
    vs = d[:, 2] * 1000

    mask = vs > 0.01
    khan_vp.append((depth[mask], vp[mask]))
    khan_vs.append((depth[mask], vs[mask]))

# ---------------- common depth grid ----------------
# zmax = max(depth1500.max(), depth1400.max())
zmax=1600
Z_common = np.linspace(0, zmax, 400)

# Samuel no BML median
mat = np.full((len(sam_nobml_vp), len(Z_common)), np.nan)
for i, (z, v) in enumerate(sam_nobml_vp):
    mat[i] = np.interp(Z_common, z, v, left=np.nan, right=np.nan)
sam_nb_vp_med = np.nanmedian(mat, axis=0)

mat = np.full((len(sam_nobml_vs), len(Z_common)), np.nan)
for i, (z, v) in enumerate(sam_nobml_vs):
    mat[i] = np.interp(Z_common, z, v, left=np.nan, right=np.nan)
sam_nb_vs_med = np.nanmedian(mat, axis=0)

# Samuel with BML median
mat = np.full((len(sam_bml_vp), len(Z_common)), np.nan)
for i, (z, v) in enumerate(sam_bml_vp):
    mat[i] = np.interp(Z_common, z, v, left=np.nan, right=np.nan)
sam_bml_vp_med = np.nanmedian(mat, axis=0)

mat = np.full((len(sam_bml_vs), len(Z_common)), np.nan)
for i, (z, v) in enumerate(sam_bml_vs):
    mat[i] = np.interp(Z_common, z, v, left=np.nan, right=np.nan)
sam_bml_vs_med = np.nanmedian(mat, axis=0)

# Khan median
mat = np.full((len(khan_vp), len(Z_common)), np.nan)
for i, (z, v) in enumerate(khan_vp):
    mat[i] = np.interp(Z_common, z, v, left=np.nan, right=np.nan)
khan_vp_med = np.nanmedian(mat, axis=0)

mat = np.full((len(khan_vs), len(Z_common)), np.nan)
for i, (z, v) in enumerate(khan_vs):
    mat[i] = np.interp(Z_common, z, v, left=np.nan, right=np.nan)
khan_vs_med = np.nanmedian(mat, axis=0)

# ---------------- plot ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), sharey=True)

# ----- ax1 : Vp -----
for z, v in khan_vp:
    ax1.plot(v, z, color='blue', alpha=0.03, lw=0.8)
ax1.plot(khan_vp_med, Z_common, color='blue', lw=2.0, label='Khan median')


for z, v in sam_bml_vp:
    ax1.plot(v, z, color='red', alpha=0.05, lw=0.8)
ax1.plot(sam_bml_vp_med, Z_common, color='red', lw=2.0, label='Samuel with BML')

for z, v in sam_nobml_vp:
    ax1.plot(v, z, color='black', alpha=0.05, lw=0.8)
ax1.plot(sam_nb_vp_med, Z_common, color='black', lw=2.0, ls='--', label='Samuel no BML')


ax1.plot(vp1400, depth1400, color='green', lw=2.0, ls='--', label='HeFESTo Tm1400')
# ax1.plot(vp1500, depth1500, color='k', lw=2.0, ls='--', label='HeFESTo Tm1500')

ax1.set_xlabel('Vp (m/s)', fontsize=labelsize)
ax1.set_ylabel('Depth (km)', fontsize=labelsize)
ax1.set_xlim(3500, 10500)
ax1.set_ylim(zmax, 0)
ax1.grid(True, ls='--', alpha=0.3)
ax1.legend(fontsize=10)

# ----- ax2 : Vs -----
for z, v in khan_vs:
    ax2.plot(v, z, color='blue', alpha=0.03, lw=0.8)
ax2.plot(khan_vs_med, Z_common, color='blue', lw=2.0, label='Khan median')

for z, v in sam_bml_vs:
    ax2.plot(v, z, color='red', alpha=0.05, lw=0.8)
ax2.plot(sam_bml_vs_med, Z_common, color='red', lw=2.0, label='Samuel with BML')

for z, v in sam_nobml_vs:
    ax2.plot(v, z, color='black', alpha=0.05, lw=0.8)
ax2.plot(sam_nb_vs_med, Z_common, color='black', lw=2.0, ls='--', label='Samuel no BML')

ax2.plot(vs1400, depth1400, color='green', lw=2.0, ls='--', label='HeFESTo Tm1400')

ax2.set_xlabel('Vs (m/s)', fontsize=labelsize)
ax2.set_xlim(0, 6000)
ax2.grid(True, ls='--', alpha=0.3)
ax2.legend(fontsize=10)

for ax in [ax1, ax2]:
    ax.tick_params(labelsize=labelsize)
    for spine in ax.spines.values():
        spine.set_linewidth(bwith)

