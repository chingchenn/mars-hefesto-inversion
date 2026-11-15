#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:38:03 2026

@author: chingchen
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


scale = 1

# ── CONFIG ────────────────────────────────────────────────────────────────────
HEFESTO_FILE = '/Users/chingchen/Desktop/HeFESTo/02_YM_bulk_mantle_BML_T/fort.56'
MODEL_DIR    = '/Users/chingchen/Desktop/Lunar/Mars_Khan_2023/LSL_Models'
SAMUEL_BASE  = '/Users/chingchen/Desktop/Lunar/Mars_Samuel_2023/Nature_Samuel/METADATA_BML/DATA_FIG1/'

# ── 新增：best_run 的路徑 ──────────────────────────────────────────────────────
BEST_MANTLE_FORT56 = '/Users/chingchen/Desktop/HeFESTo/best_M04/files/mnt/user-data/outputs/best_run/mantle_step2/fort.56'
BEST_BML_FORT56    = '/Users/chingchen/Desktop/HeFESTo/best_M04/files/mnt/user-data/outputs/best_run/bml/fort.56'

g_mars      = 3.35
MARS_RADIUS = 3389.5
labelsize   = 14
bwith       = 2.5

colors = ['#282130', '#849DAB', '#CD5C5C', '#35838D',
          '#97795D', '#414F67', '#4198B9', '#2F4F4F']

# ── 讀 fort.56 的共用函式 ──────────────────────────────────────────────────────
def load_fort56(fpath, g=g_mars):
    with open(fpath) as f:
        f.readline()
        cols = f.readline().split()
    df = pd.read_csv(fpath, sep=r'\s+', skiprows=2, names=cols)
    P   = df['P(GPa)'].values
    rho = df['rho(g/cm^3)'].values
    vs  = df['VS(km/s)'].values
    vp  = df['VP(km/s)'].values
    T   = df['T(K)'].values
    # depth integration
    dP      = np.diff(P) * 1e9
    rho_mid = (rho[:-1] + rho[1:]) / 2
    dz      = dP / (rho_mid * 1000 * g) / 1000
    depth   = np.zeros(len(P))
    depth[1:] = np.cumsum(dz)
    return P, rho, vs, vp, T, depth

# ── LOAD original HeFESTo ──────────────────────────────────────────────────────
hef_P, hef_rho, hef_vs, hef_vp, hef_T, hef_depth = load_fort56(HEFESTO_FILE)

P_MAX     = hef_P.max()
DEPTH_MAX = 2000

# ── LOAD best_run mantle ───────────────────────────────────────────────────────
best_mantle_ok = os.path.exists(BEST_MANTLE_FORT56)
P_BML_TOP = 17.05   # GPa，BML 頂部壓力，mantle 只畫到這裡

if best_mantle_ok:
    bm_P, bm_rho, bm_vs, bm_vp, bm_T, bm_depth = load_fort56(BEST_MANTLE_FORT56)
    # 只保留 P <= P_BML_TOP 的部分
    bm_mask = bm_P <= P_BML_TOP
    bm_P, bm_rho, bm_vs, bm_vp, bm_T, bm_depth = (
        bm_P[bm_mask], bm_rho[bm_mask], bm_vs[bm_mask],
        bm_vp[bm_mask], bm_T[bm_mask],  bm_depth[bm_mask]
    )
    print(f"best mantle loaded: {len(bm_P)} points, P={bm_P.min():.1f}~{bm_P.max():.1f} GPa")
else:
    print(f"WARNING: {BEST_MANTLE_FORT56} not found, skipping best mantle")

# ── LOAD best_run BML ──────────────────────────────────────────────────────────
best_bml_ok = os.path.exists(BEST_BML_FORT56)
if best_bml_ok:
    bb_P, bb_rho, bb_vs, bb_vp, bb_T, bb_depth_rel = load_fort56(BEST_BML_FORT56)
    # BML depth offset：BML top = CMB - 150 km，CMB ≈ 1560 km（依你的 khan_cache）
    BML_TOP_KM = 1560 - 150   # 調整成你的 CMB depth
    bb_depth   = bb_depth_rel + BML_TOP_KM
    print(f"best BML loaded: {len(bb_P)} points, P={bb_P.min():.1f}~{bb_P.max():.1f} GPa")
else:
    print(f"WARNING: {BEST_BML_FORT56} not found, skipping best BML")

# ── LOAD Khan et al. ──────────────────────────────────────────────────────────
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
        if mask.sum() < 5:
            continue
        khan_depth_list.append(depth_km[mask])
        khan_vp_list.append(vp_km[mask])
        khan_vs_list.append(vs_km[mask])
        khan_rho_list.append(rho_gcc[mask])
    except:
        pass

n_models = len(khan_depth_list)
print(f'Khan: {n_models} models loaded')

Z_common = np.linspace(0, DEPTH_MAX, 400)
def make_stats(lst):
    mat = np.full((n_models, len(Z_common)), np.nan)
    for i, (z, v) in enumerate(zip(khan_depth_list, lst)):
        if z.max() < Z_common.max() * 0.8:
            continue
        mat[i] = np.interp(Z_common, z, v, left=np.nan, right=np.nan)
    return (np.nanmedian(mat, 0),
            np.nanpercentile(mat, 16, 0),
            np.nanpercentile(mat, 84, 0))

vp_med,  vp_p16,  vp_p84  = make_stats(khan_vp_list)
vs_med,  vs_p16,  vs_p84  = make_stats(khan_vs_list)
rho_med, rho_p16, rho_p84 = make_stats(khan_rho_list)

# ── LOAD Samuel ───────────────────────────────────────────────────────────────
Tliq,     depTliq     = np.loadtxt(SAMUEL_BASE + 'PANEL_I/Tliq.dat').T
Tprofile, depTprofile = np.loadtxt(SAMUEL_BASE + 'PANEL_I/Tprofile.dat').T
Tsol,     depTsol     = np.loadtxt(SAMUEL_BASE + 'PANEL_I/Tsol.dat').T
rho_sam,  deprho_sam  = np.loadtxt(SAMUEL_BASE + 'PANEL_J/rho_profile.dat').T
samvp,    depvp_sam   = np.loadtxt(SAMUEL_BASE + 'PANEL_K/vp_profile.dat').T
samvs,    depvs_sam   = np.loadtxt(SAMUEL_BASE + 'PANEL_K/vs_profile.dat').T

def r2d(r):
    return MARS_RADIUS - r

# ── AXES SETUP ────────────────────────────────────────────────────────────────
def setup_ax(ax, xlabel):
    ax.set_ylim(DEPTH_MAX, 0)
    ax.set_ylabel('depth (km)', fontsize=labelsize)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.grid(True, ls='--', alpha=0.35)
    ax.tick_params(labelsize=labelsize)
    for sp in ax.spines.values():
        sp.set_linewidth(bwith)

    axr = ax.twinx()
    axr.set_ylim(DEPTH_MAX, 0)
    p_ticks       = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    depth_at_p    = np.interp(p_ticks, hef_P, hef_depth)
    axr.set_yticks(depth_at_p)
    axr.set_yticklabels([f'{p:.0f}' for p in p_ticks], color='gray')
    axr.tick_params(labelsize=labelsize)
    for sp in axr.spines.values():
        sp.set_linewidth(bwith)
    return axr

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
fig.subplots_adjust(wspace=0.35, left=0.10, right=0.93, top=0.92, bottom=0.09)

BEST_COLOR  = 'magenta'
BEST_LW     = 2.2
BEST_LABEL  = 'Best (MCMC a04)'

# ─── Panel 1: Vp and Vs ───────────────────────────────────────────────────────
ax = ax1
for i in range(n_models):
    ax.plot(khan_vp_list[i], khan_depth_list[i], color='#4198B9', alpha=0.05, lw=0.5, rasterized=True)
    ax.plot(khan_vs_list[i], khan_depth_list[i], color='#97795D', alpha=0.05, lw=0.5, rasterized=True)
ax.fill_betweenx(Z_common, vp_p16, vp_p84, color='#4198B9', alpha=0.15)
ax.fill_betweenx(Z_common, vs_p16, vs_p84, color='#97795D', alpha=0.15)
ax.plot(vp_med, Z_common, color='#4198B9', lw=2.0, label='Khan Vp (median)')
ax.plot(vs_med, Z_common, color='#97795D', lw=2.0, label='Khan Vs (median)')
ax.plot(samvp / 1000, r2d(depvp_sam), color='red',  lw=2, label='Samuel Vp')
ax.plot(samvs / 1000, r2d(depvs_sam), color='blue', lw=2, label='Samuel Vs')

axr = setup_ax(ax1, 'velocity (km/s)')
axr.plot(hef_vp * scale, hef_depth, color=colors[2], lw=2.2, ls='--', label='HeFESTo Vp')
axr.plot(hef_vs * scale, hef_depth, color=colors[0], lw=2.2, ls='--', label='HeFESTo Vs')

# ── best mantle + BML
if best_mantle_ok:
    axr.plot(bm_vp, bm_depth, color=BEST_COLOR, lw=BEST_LW, ls='-',  label=f'{BEST_LABEL} Vp')
    axr.plot(bm_vs, bm_depth, color=BEST_COLOR, lw=BEST_LW, ls='--', label=f'{BEST_LABEL} Vs')
if best_bml_ok:
    axr.plot(bb_vp, bb_depth, color=BEST_COLOR, lw=BEST_LW, ls='-')
    axr.plot(bb_vs, bb_depth, color=BEST_COLOR, lw=BEST_LW, ls='--')

ax.set_xlim(0, 11)
ax.legend(fontsize=8.5, loc='lower left', framealpha=0.85)

# ─── Panel 2: density ─────────────────────────────────────────────────────────
ax = ax2
for i in range(n_models):
    ax.plot(khan_rho_list[i], khan_depth_list[i], color='#35838D', alpha=0.05, lw=0.5, rasterized=True)
ax.fill_betweenx(Z_common, rho_p16, rho_p84, color='#35838D', alpha=0.20)
ax.plot(rho_med, Z_common, color='#35838D', lw=2.0, label='Khan ρ (median)')
ax.plot(rho_sam / 1000, r2d(deprho_sam), color='k', lw=2, label='Samuel 2023 noBML')

axr = setup_ax(ax2, 'rho')
axr.plot(hef_rho, hef_depth, color=colors[3], lw=2.2, ls='--', label='HeFESTo ρ')

# ── best mantle + BML
if best_mantle_ok:
    axr.plot(bm_rho, bm_depth, color=BEST_COLOR, lw=BEST_LW, ls='-', label=f'{BEST_LABEL} ρ')
if best_bml_ok:
    axr.plot(bb_rho, bb_depth, color=BEST_COLOR, lw=BEST_LW, ls='-')

ax.set_xlim(1.5, 7)
ax.legend(fontsize=8.5, loc='lower right', framealpha=0.85)
ax.set_ylabel('')

# ─── Panel 3: temperature ─────────────────────────────────────────────────────
ax = ax3
axr = setup_ax(ax3, 'temperature')
axr.plot(hef_T, hef_depth, color=colors[0], lw=1.8, ls=':', label='T (HeFESTo)')

# ── best mantle + BML
if best_mantle_ok:
    axr.plot(bm_T, bm_depth, color=BEST_COLOR, lw=BEST_LW, ls='-', label=f'{BEST_LABEL} T')
if best_bml_ok:
    axr.plot(bb_T, bb_depth, color=BEST_COLOR, lw=BEST_LW, ls='-')

ax.plot(Tprofile, r2d(depTprofile), color='k',      lw=1.8, label='Samuel 2023 noBML')
ax.plot(Tliq,     r2d(depTliq),     color='brown',  lw=1.8, label='liquidus')
ax.plot(Tsol,     r2d(depTsol),     color='orange', lw=1.8, label='solidus')

axr.set_ylabel('pressure (GPa)\n[HeFESTo]', fontsize=labelsize, color='gray')
ax.legend(fontsize=8.5, loc='upper right', framealpha=0.85)
ax.set_ylabel('')

#fig.suptitle(
#    f'Mars mantle: HeFESTo (YM, with BML) [g={g_mars} m/s²] and velocity decreasing {round((1-scale)*100, 0)} %',
#    fontsize=16, y=0.97
#)

#plt.savefig('/Users/chingchen/Desktop/best_run_profile.png', dpi=150, bbox_inches='tight')
#plt.show()
#print('saved: ~/Desktop/best_run_profile.png')