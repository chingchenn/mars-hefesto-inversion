#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:57:47 2026

@author: chingchen
"""

"""
Combined plot: HeFESTo (YM, no BML) vs Khan et al. (2023) 1000 seismic models
Y-axis: pressure (GPa) — common between both datasets
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d

scale=0.97
# ── CONFIG ────────────────────────────────────────────────────────────────────

# model = [01_YM_bulk_mantle_noBML_T,
#          02_YM_bulk_mantle_BML_T,
#          21_EM1_BML_Mg75,
#          22_EM1_BML_Mg70,
#          23_EM1_BML_Mg65,
#          24_EM1_BML_Mg60,
#          25_EM1_BML_Mg55,
#          26_EM1_BML_Mg50]
HEFESTO_FILE = '/Users/chingchen/Desktop/HeFESTo/01_YM_bulk_mantle_noBML_T/fort.56'
MODEL_DIR    = '/Users/chingchen/Desktop/Lunar/Mars_Khan_2023/LSL_Models'   
SAMUEL_BASE  = '/Users/chingchen/Desktop/Lunar/Mars_Samuel_2023/Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1/'
# OUTPUT_FIG   = 'combined_HeFESTo_Khan2023.pdf'

g_mars = 3.45   # m/s²
MARS_RADIUS = 3389.5   
labelsize = 14
bwith     = 2.5

colors = ['#282130', '#849DAB', '#CD5C5C', '#35838D',
          '#97795D', '#414F67', '#4198B9', '#2F4F4F']

# ── LOAD HeFESTo ──────────────────────────────────────────────────────────────
with open(HEFESTO_FILE) as f:
    f.readline()
    cols = f.readline().split()
df56 = pd.read_csv(HEFESTO_FILE, sep=r'\s+', skiprows=2, names=cols)

hef_P   = df56['P(GPa)'].values
hef_rho = df56['rho(g/cm^3)'].values
hef_vs  = df56['VS(km/s)'].values 
hef_vp  = df56['VP(km/s)'].values 
hef_T   = df56['T(K)'].values

# Depth from pressure integration (for right-axis ticks)
dP        = np.diff(hef_P) * 1e9                    # Pa
rho_mid   = (hef_rho[:-1] + hef_rho[1:]) / 2        # g/cm³
dz        = dP / (rho_mid * 1000 * g_mars) / 1000   # km
hef_depth = np.zeros(len(hef_P))
hef_depth[1:] = np.cumsum(dz)

P_MAX      = hef_P.max()
DEPTH_MAX  = 2000
 


# ── LOAD Khan et al. 1000 MODELS ──────────────────────────────────────────────
pattern = os.path.join(MODEL_DIR, 'Model_*.txt')
files   = sorted(glob.glob(pattern))

 
khan_depth_list, khan_vp_list, khan_vs_list, khan_rho_list = [], [], [], []
for fpath in files:
    try:
        data     = np.loadtxt(fpath, comments='#')
        depth_km = data[:, 0]
        vp_km    = data[:, 1]
        vs_km    = data[:, 2]
        rho_gcc  = data[:, 3]
        p_pa     = data[:, 6]
        p_gpa    = p_pa / 1e9
        mask = (vs_km > 0.01) & (p_gpa <= P_MAX + 1)
        if mask.sum() < 5: continue
        khan_depth_list.append(depth_km[mask])
        khan_vp_list.append(vp_km[mask])
        khan_vs_list.append(vs_km[mask])
        khan_rho_list.append(rho_gcc[mask])
    except: pass
 
n_models = len(khan_depth_list)
print(f'Khan: {n_models} models loaded')
 
# Khan statistics on common depth grid
Z_common = np.linspace(0, DEPTH_MAX, 400)
def make_stats(lst):
    mat = np.full((n_models, len(Z_common)), np.nan)
    for i,(z,v) in enumerate(zip(khan_depth_list, lst)):
        if z.max() < Z_common.max()*0.8: continue
        mat[i] = np.interp(Z_common, z, v, left=np.nan, right=np.nan)
    return (np.nanmedian(mat,0),
            np.nanpercentile(mat,16,0),
            np.nanpercentile(mat,84,0))
 
vp_med, vp_p16, vp_p84 = make_stats(khan_vp_list)
vs_med, vs_p16, vs_p84 = make_stats(khan_vs_list)
rho_med, rho_p16, rho_p84 = make_stats(khan_rho_list)


# ── LOAD Samuel ───────────────────────────────────────────────────────────────
Tliq,     depTliq     = np.loadtxt(SAMUEL_BASE+'PANEL_C/Tliq.dat').T
Tprofile, depTprofile = np.loadtxt(SAMUEL_BASE+'PANEL_C/Tprofile.dat').T
Tsol,     depTsol     = np.loadtxt(SAMUEL_BASE+'PANEL_C/Tsol.dat').T
rho_sam,  deprho_sam  = np.loadtxt(SAMUEL_BASE+'PANEL_D/rho_profile.dat').T
samvp,    depvp_sam   = np.loadtxt(SAMUEL_BASE+'PANEL_E/vp_profile.dat').T
samvs,    depvs_sam   = np.loadtxt(SAMUEL_BASE+'PANEL_E/vs_profile.dat').T


def r2d(r): return MARS_RADIUS - r
# hef_d = p2depth(hef_P)

def setup_ax(ax, xlabel):
    ax.set_ylim(DEPTH_MAX, 0)
    ax.set_ylabel('depth (km)', fontsize=labelsize)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.grid(True, ls='--', alpha=0.35)
    ax.tick_params(labelsize=labelsize)
    for sp in ax.spines.values(): sp.set_linewidth(bwith)

    axr = ax.twinx()
    axr.set_ylim(DEPTH_MAX, 0)  # 跟左軸一樣！

    
    p_ticks = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    depth_at_p = np.interp(p_ticks, hef_P, hef_depth)

    axr.set_yticks(depth_at_p)
    axr.set_yticklabels([f'{p:.0f}' for p in p_ticks], color='gray')

    axr.tick_params(labelsize=labelsize)
    for sp in axr.spines.values(): sp.set_linewidth(bwith)
    return axr

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
fig.subplots_adjust(wspace=0.35, left=0.10, right=0.93, top=0.92, bottom=0.09)


# ─── Panel 1: Vp and Vs ───────────────────────────────────────────────────────

ax = ax1
# Khan
for i in range(n_models):
    ax.plot(khan_vp_list[i], khan_depth_list[i],
            color='#4198B9', alpha=0.05, lw=0.5, rasterized=True)
    ax.plot(khan_vs_list[i], khan_depth_list[i],
            color='#97795D', alpha=0.05, lw=0.5, rasterized=True)
ax.fill_betweenx(Z_common, vp_p16, vp_p84, color='#4198B9', alpha=0.15)
ax.fill_betweenx(Z_common, vs_p16, vs_p84, color='#97795D', alpha=0.15)
ax.plot(vp_med, Z_common, color='#4198B9', lw=2.0, label='Khan Vp (median)')
ax.plot(vs_med, Z_common, color='#97795D', lw=2.0, label='Khan Vs (median)')
# Samuel
ax.plot(samvp/1000, r2d(depvp_sam), color='red',  lw=2, label='Samuel Vp')
ax.plot(samvs/1000, r2d(depvs_sam), color='blue', lw=2, label='Samuel Vs')
# HeFESTo
axr = setup_ax(ax1, 'velocity (km/s)')

axr.plot(hef_vp*scale, hef_depth, color=colors[2], lw=2.2, ls='--', label='HeFESTo Vp')
axr.plot(hef_vs*scale, hef_depth, color=colors[0], lw=2.2, ls='--', label='HeFESTo Vs')

 
ax.set_xlim(0, 11)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = axr.get_legend_handles_labels()

ax.legend(lines1 + lines2,
          labels1 + labels2,
          fontsize=labelsize-5,
          loc='upper left',
          framealpha=0.85)


# ─── Panel 2: density ─────────────────────────────────────────
ax = ax2
for i in range(n_models):
    ax.plot(khan_rho_list[i], khan_depth_list[i],
            color='#35838D', alpha=0.05, lw=0.5, rasterized=True)
ax.fill_betweenx(Z_common, rho_p16, rho_p84, color='#35838D', alpha=0.20)
ax.plot(rho_med, Z_common,  color='#35838D', lw=2.0, label='Khan ρ (median)')
axr = setup_ax(ax2, 'rho ')
# axr.plot(hef_rho, hef_P,     color=colors[3], lw=2.2, ls='--', label='HeFESTo ρ (YM)')
axr.plot(hef_rho, hef_depth, color=colors[3], lw=2.2, ls='--', label='HeFESTo ρ')

ax.plot(rho_sam/1000, r2d(deprho_sam), color='k', lw=2, label='Samuel 2023 noBML')
 
ax.set_xlim(1.5, 7)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = axr.get_legend_handles_labels()

ax.legend(lines1 + lines2,
          labels1 + labels2,
          fontsize=labelsize-5,
          loc='lower left',
          framealpha=0.85)

ax.set_ylabel('')   


# ─── Panel 3: temperature ──────────────────────────────────────────
ax = ax3
axr = setup_ax(ax3, 'temperature ')
# axr.plot(hef_T,    hef_P,              color=colors[0], lw=1.8, ls=':', label='T (HeFESTo)')
axr.plot(hef_T, hef_depth, color=colors[0], lw=1.8, ls=':', label='T (HeFESTo)')
ax.plot(Tprofile, r2d(depTprofile),   color='k',       lw=1.8, label='Samuel 2023 noBML')
ax.plot(Tliq,     r2d(depTliq),       color='brown',   lw=1.8, label='liquidus')
ax.plot(Tsol,     r2d(depTsol),       color='orange',  lw=1.8, label='solidus')
 
axr.set_ylabel('pressure (GPa)\n[HeFESTo]', fontsize=labelsize, color='gray')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = axr.get_legend_handles_labels()

ax.legend(lines1 + lines2,
          labels1 + labels2,
          fontsize=labelsize-5,
          loc='lower left',
          framealpha=0.85)

ax.set_ylabel('')


 
fig.suptitle(
    f'Mars mantle: HeFESTo (YM, no BML) [g={g_mars} m/s²] and velocity decreasing {round((1-scale)*100,0)} %',
    fontsize=16, y=0.97
)
