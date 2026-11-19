#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:31:09 2026

@author: chingchen
"""

# 106.ppt_mantle_figs.py
# quals PPT / mantle：一支腳本，四張分開存的圖
#
#   FIG 1  106_lithosphere.png     T_lit - P_lit 聯合後驗 -> 岩石圈厚度 z_lit
#   FIG 2  106_Mg.png              地函 Mg# 後驗，與文獻組成對照
#   FIG 3  106_olivine_ab.png      橄欖石 α-β 轉換深度，與 Huang 2022 對照
#   FIG 4  106_geotherm.png        地溫線包絡（T-z 與 T-P）
#
# 資料來源：
#   FIG 1, 2  讀 chain.jsonl —— 鏈的 STATE，被拒步原地重複 = 天然後驗權重，
#             全部記錄都要用，不可過濾成 accepted-only
#   FIG 3, 4  讀 profile npz —— 只在 accept 時寫出，沒有原地重複的權重，
#             所以是 accepted 樣本的分布，不是嚴格後驗邊際。圖說要註明
#
# TOGGLE 可以只跑其中幾張（掃 npz 很慢）

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR

# ── config ────────────────────────────────────────────────────────────────────
TOGGLE = {'lithosphere': True, 'Mg': True, 'olivine': True, 'geotherm': True}

CHAIN_DIR = MCMC_DIR
PREFIX    = 'chain_a52B'          # 必須寫到 A 或 B，否則會 glob 到兩套 EoS
BURNIN    = 200
MIN_STEPS = 300

PROF_GLOB = os.path.join(CHAIN_DIR, PREFIX + '*', '*.npz')  # ← 確認實際路徑
STRIDE    = 1                     # profile 太多時抽樣

MARS_RADIUS = 3389.5              # km

PRIOR_T_LIT  = (1000.0, 2600.0)   # K
PRIOR_P_LIT  = (1.4, 9.0)         # GPa
PRIOR_MG     = (0.50, 0.90)
PRIOR_MG_BML = (0.20, 0.80)

H_MIN = 1.0                       # km，三態分類門檻（與主程式 H_MIN_LAYER 一致）

LAYER_CRUST  = 0
LAYER_MANTLE = 1
LAYER_BML    = 2
LAYER_CORE   = 3

# 固相線：T_sol = -4.877 P^2 + 120.2 P + 600 * Mg# + 911.15   (P <= 10 GPa)
SOLIDUS_MG = 0.79

# 文獻的岩石圈厚度（km）—— 數值核對後再定稿
LIT_ZLIT = {
    'Khan 2023 prior 100-600': (100.0, 600.0),
    'Samuel 2023':  (None, None),
    'Drilleau 2026': (None, None),
}

# 文獻的地函 Mg#（bulk silicate Mars）
# 這幾個數字是憑記憶填的，定稿前逐一核對原文，尤其 TA / YM
LIT_MG = {
    'DW (Wanke & Dreibus 1994)': 0.75,
    'EH45 (Sanloup 1999)':       0.745,
    'TA (Taylor 2013)':          0.79,
    'YM (Yoshizaki & McDonough 2020)': 0.79,
    'Khan 2023 (fixed)': None,
    'Samuel 2023':       None,
}

# 橄欖石 α-β
Z_SEARCH   = (700.0, 1400.0)      # km，搜尋窗（避開岩石圈折點與 BML 頂）
DVS_MIN    = 0.0005               # (km/s)/km，低於此視為無可辨識的轉換
HUANG_2022 = (1006.0, 40.0)       # km, 1006 +- 40（三重震相）
N_SHOW     = 40                   # FIG 3 左圖畫幾條 Vs 剖面

# 地溫線
Z_GRID = np.arange(0.0, 1900.0, 5.0)      # km
# 文獻地溫線檔案：(路徑, 標籤, 顏色)，格式兩欄 depth_km  T_K
LIT_GEOTHERM = [
    # ('/path/to/samuel2023_geotherm.dat', 'Samuel 2023', 'tab:orange'),
    # ('/path/to/khan2023_geotherm.dat',   'Khan 2023',   'tab:purple'),
]

FS = 14
STATE_COLOR = {'solid': 'tab:blue', 'layered': 'tab:green', 'molten': 'tab:red'}
P_GRID = np.arange(0.0, 21.0, 0.1)        # GPa
# ── 載入 chain ────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '*', 'chain.jsonl')))
print('chain files:', len(files))

T_lit = []
P_lit = []
z_lit = []
Mg = []
Mg_bml = []
state = []

for f in files:
    recs = []
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break                      # 尾端半筆（被 kill 寫壞）就停
    if len(recs) < MIN_STEPS:
        print('  skip (too short):', f, len(recs))
        continue
    recs = recs[BURNIN:]
    for r in recs:
        p = r['params']
        hs = r['h_solid_km']
        hl = r['h_liquid_km']
        if hl < H_MIN:
            s = 'solid'
        elif hs < H_MIN:
            s = 'molten'
        else:
            s = 'layered'
        T_lit.append(p['T_lit'])
        P_lit.append(p['P_lit'])
        z_lit.append(r['z_lit_km'])
        Mg.append(p['Mg#'])
        Mg_bml.append(p['Mg#_bulk_bml'])
        state.append(s)
    print('  loaded:', os.path.basename(os.path.dirname(f)), len(recs))

T_lit  = np.array(T_lit)
P_lit  = np.array(P_lit)
z_lit  = np.array(z_lit)
Mg     = np.array(Mg)
Mg_bml = np.array(Mg_bml)
state  = np.array(state)

print('n chain samples =', len(T_lit))
print('state counts:', {s: int((state == s).sum()) for s in STATE_COLOR})
print('T_lit  5/50/95 :', np.percentile(T_lit, [5, 50, 95]))
print('P_lit  5/50/95 :', np.percentile(P_lit, [5, 50, 95]))
print('z_lit  5/50/95 :', np.percentile(z_lit, [5, 50, 95]))
print('Mg#    5/50/95 :', np.percentile(Mg, [5, 50, 95]))
print('Mg#_bml 5/50/95 :', np.percentile(Mg_bml, [5, 50, 95]))
for s in STATE_COLOR:
    m = state == s
    if m.sum() > 0:
        print('  ', s, 'n =', int(m.sum()),
              ' z_lit', np.percentile(z_lit[m], [5, 50, 95]),
              ' Mg#', np.percentile(Mg[m], [5, 50, 95]))
# 貼牆比例：牆被壓住的話後驗就不是資料說了算
print('Mg# 貼左牆 :', float(np.mean(Mg < PRIOR_MG[0] + 0.01)))
print('Mg# 貼右牆 :', float(np.mean(Mg > PRIOR_MG[1] - 0.01)))

# ── 掃描 profile npz ──────────────────────────────────────────────────────────
z_mid = np.array([])
z_lo = np.array([])
z_hi = np.array([])
show_z = []
show_vs = []
show_zm = []
T_all = np.zeros((0, len(Z_GRID)))
P_all = np.zeros((0, len(Z_GRID)))
zmax_mantle = np.array([])

if TOGGLE['olivine'] or TOGGLE['geotherm']:
    pfiles = sorted(glob.glob(PROF_GLOB))[::STRIDE]
    print('profiles:', len(pfiles))

    z_mid_l = []
    z_lo_l = []
    z_hi_l = []
    T_l = []
    Tp_l = []
    zmax_l = []
    pmax_l = []

    for f in pfiles:
        d = np.load(f)
        z = d['z_km']
        T = d['T_K']
        P = d['P_GPa']
        Vs = d['Vs']
        layer = d['layer']

        # ── 橄欖石 α-β：相分率沒存，從速度梯度反推
        #    w = max(dVs/dz, 0)
        #    mid-depth = sum(z*w)/sum(w)           對應 Huang 2022 的 mid-point
        #    轉換帶 = w 的累積 16 / 84 百分位（雙變相環的視深度範圍）
        m = (layer == LAYER_MANTLE) & (z >= Z_SEARCH[0]) & (z <= Z_SEARCH[1])
        if m.sum() >= 10:
            zz = z[m]
            vv = Vs[m]
            w = np.clip(np.gradient(vv, zz), 0.0, None)
            if w.max() >= DVS_MIN:
                W = w.sum()
                zc = float((zz * w).sum() / W)
                cw = np.cumsum(w) / W
                z_mid_l.append(zc)
                z_lo_l.append(float(np.interp(0.16, cw, zz)))
                z_hi_l.append(float(np.interp(0.84, cw, zz)))
                if len(show_z) < N_SHOW:
                    mm = layer == LAYER_MANTLE
                    show_z.append(z[mm])
                    show_vs.append(Vs[mm])
                    show_zm.append(zc)

        # ── 地溫線：固定殼層 + 地函，內插到共同網格
        m2 = layer <= LAYER_MANTLE
        if m2.sum() >= 10:
            zz = z[m2]
            TT = T[m2]
            PP = P[m2]

            # (a) 深度網格：自變數 = z
            Ti = np.interp(Z_GRID, zz, TT, left=np.nan, right=np.nan)
            Ti[Z_GRID > zz.max()] = np.nan
            T_l.append(Ti)
            zmax_l.append(float(zz.max()))

            # (b) 壓力網格：自變數 = P（每條剖面用自己的壓力，不用跨模型中位數）
            Tp = np.interp(P_GRID, PP, TT, left=np.nan, right=np.nan)
            Tp[P_GRID > PP.max()] = np.nan
            Tp_l.append(Tp)
            pmax_l.append(float(PP.max()))

    z_mid = np.array(z_mid_l)
    z_lo = np.array(z_lo_l)
    z_hi = np.array(z_hi_l)
    T_all = np.array(T_l)
    Tp_all = np.array(Tp_l)
    zmax_mantle = np.array(zmax_l)
    pmax_mantle = np.array(pmax_l)

    print('profiles with detectable transition :', len(z_mid))
    if len(z_mid) > 0:
        print('  mid-depth 5/16/50/84/95 :', np.percentile(z_mid, [5, 16, 50, 84, 95]))
        print('  transition width 5/50/95 :', np.percentile(z_hi - z_lo, [5, 50, 95]))
        print('  在 Huang 2022 1006 +- 40 內的比例 :',
              float(np.mean(np.abs(z_mid - HUANG_2022[0]) <= HUANG_2022[1])))
    print('profiles for geotherm :', len(T_all))
    if len(T_all) > 0:
        print('  BML top depth 5/50/95 :', np.percentile(zmax_mantle, [5, 50, 95]))

# ── FIG 1：岩石圈 ─────────────────────────────────────────────────────────────
if TOGGLE['lithosphere']:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    h = ax[0].hist2d(T_lit, P_lit, bins=[60, 60],
                     range=[list(PRIOR_T_LIT), list(PRIOR_P_LIT)], cmap='Greys')
    cb = fig.colorbar(h[3], ax=ax[0])
    cb.set_label('posterior samples', fontsize=FS)
    cb.ax.tick_params(labelsize=FS)

    pp = np.linspace(PRIOR_P_LIT[0], PRIOR_P_LIT[1], 200)
    Tsol_line = -4.877 * pp**2 + 120.2 * pp + 600.0 * SOLIDUS_MG + 911.15
    ax[0].plot(Tsol_line, pp, 'r-', lw=2, label='solidus (Mg# = %.2f)' % SOLIDUS_MG)

    ax[0].plot([PRIOR_T_LIT[0], PRIOR_T_LIT[1], PRIOR_T_LIT[1], PRIOR_T_LIT[0], PRIOR_T_LIT[0]],
               [PRIOR_P_LIT[0], PRIOR_P_LIT[0], PRIOR_P_LIT[1], PRIOR_P_LIT[1], PRIOR_P_LIT[0]],
               'k--', lw=1.5, label='prior')

    ax[0].set_xlabel('$T_{lit}$ (K)', fontsize=FS)
    ax[0].set_ylabel('$P_{lit}$ (GPa)', fontsize=FS)
    ax[0].invert_yaxis()
    ax[0].tick_params(labelsize=FS)
    ax[0].legend(fontsize=FS - 2, loc='lower right')

    bins = np.linspace(np.floor(z_lit.min() / 10) * 10,
                       np.ceil(z_lit.max() / 10) * 10, 50)
    ax[1].hist(z_lit, bins=bins, color='0.75', edgecolor='none',
               density=True, label='all')
    for s in ['solid', 'layered', 'molten']:
        m = state == s
        if m.sum() > 20:
            ax[1].hist(z_lit[m], bins=bins, histtype='step', lw=2,
                       color=STATE_COLOR[s], density=True, label=s)

    ymax = ax[1].get_ylim()[1]
    for k in LIT_ZLIT:
        lo, hi = LIT_ZLIT[k]
        if lo is None:
            continue
        if hi is None:
            ax[1].axvline(lo, color='k', ls=':', lw=2)
            ax[1].text(lo, ymax * 0.98, k, fontsize=FS - 4, rotation=90,
                       va='top', ha='right')
        else:
            ax[1].axvspan(lo, hi, color='k', alpha=0.08)
            ax[1].text(0.5 * (lo + hi), ymax * 0.98, k, fontsize=FS - 4,
                       ha='center', va='top')

    ax[1].set_xlabel('lithosphere thickness $z_{lit}$ (km)', fontsize=FS)
    ax[1].set_ylabel('posterior density', fontsize=FS)
    ax[1].tick_params(labelsize=FS)
    ax[1].legend(fontsize=FS - 2, loc='lower right')

    fig.tight_layout()
    #fig.savefig('106_lithosphere.png', dpi=200)

# ── FIG 2：Mg# ────────────────────────────────────────────────────────────────
if TOGGLE['Mg']:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    bins = np.linspace(PRIOR_MG[0], PRIOR_MG[1], 60)
    ax[0].hist(Mg, bins=bins, color='0.75', edgecolor='none', density=True, label='all')
    for s in ['solid', 'layered', 'molten']:
        m = state == s
        if m.sum() > 20:
            ax[0].hist(Mg[m], bins=bins, histtype='step', lw=2,
                       color=STATE_COLOR[s], density=True, label=s)

    # 先驗密度 = 1 / (b - a)
    ax[0].axhline(1.0 / (PRIOR_MG[1] - PRIOR_MG[0]), color='k', ls='--', lw=1.5,
                  label='prior')

    ymax = ax[0].get_ylim()[1]
    for i, k in enumerate(LIT_MG):
        v = LIT_MG[k]
        if v is None:
            continue
        ax[0].axvline(v, color='k', ls=':', lw=2)
        ax[0].text(v, ymax * (0.98 - 0.13 * i), k, fontsize=FS - 4,
                   rotation=90, va='top', ha='right')

    ax[0].set_xlabel('mantle Mg#', fontsize=FS)
    ax[0].set_ylabel('posterior density', fontsize=FS)
    ax[0].tick_params(labelsize=FS)
    ax[0].legend(fontsize=FS - 2, loc='lower right')

    for s in ['molten', 'layered', 'solid']:
        m = state == s
        if m.sum() > 0:
            ax[1].plot(Mg[m], Mg_bml[m], '.', ms=2, alpha=0.25,
                       color=STATE_COLOR[s], label=s)

    ax[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Mg#$_{BML}$ = Mg#$_{mantle}$')
    ax[1].set_xlim(PRIOR_MG)
    ax[1].set_ylim(PRIOR_MG_BML)
    ax[1].set_xlabel('mantle Mg#', fontsize=FS)
    ax[1].set_ylabel('BML bulk Mg#', fontsize=FS)
    ax[1].tick_params(labelsize=FS)
    leg = ax[1].legend(fontsize=FS - 2, loc='lower right', markerscale=6)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.tight_layout()
    #fig.savefig('106_Mg.png', dpi=200)

# ── FIG 3：橄欖石 α-β ─────────────────────────────────────────────────────────
if TOGGLE['olivine'] and len(z_mid) > 0:
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))

    for zz, vv in zip(show_z, show_vs):
        ax[0].plot(vv, zz, '-', lw=0.8, color='0.6', alpha=0.6)
    x_mark = min([vv.min() for vv in show_vs]) - 0.2
    for zc in show_zm:
        ax[0].plot(x_mark, zc, '.', color='tab:orange')
    ax[0].set_xlim(x_mark - 0.2, max([vv.max() for vv in show_vs]) + 0.2)

    ax[0].axhspan(HUANG_2022[0] - HUANG_2022[1], HUANG_2022[0] + HUANG_2022[1],
                  color='tab:blue', alpha=0.15, label='Huang et al. 2022')
    ax[0].axhline(HUANG_2022[0], color='tab:blue', lw=2)
    ax[0].axhline(Z_SEARCH[0], color='k', ls=':', lw=1)
    ax[0].axhline(Z_SEARCH[1], color='k', ls=':', lw=1, label='search window')

    ax[0].set_xlabel('$V_S$ (km/s)', fontsize=FS)
    ax[0].set_ylabel('depth (km)', fontsize=FS)
    ax[0].set_ylim(1600, 0)
    ax[0].tick_params(labelsize=FS)
    ax[0].legend(fontsize=FS - 2, loc='lower right')

    bins = np.linspace(Z_SEARCH[0], Z_SEARCH[1], 60)
    ax[1].hist(z_mid, bins=bins, color='0.75', edgecolor='none', density=True,
               label='this study (accepted models)')
    ax[1].axvspan(HUANG_2022[0] - HUANG_2022[1], HUANG_2022[0] + HUANG_2022[1],
                  color='tab:blue', alpha=0.15, label='Huang et al. 2022')
    ax[1].axvline(HUANG_2022[0], color='tab:blue', lw=2)

    ax[1].set_xlabel('olivine $\\alpha$-$\\beta$ mid-transition depth (km)', fontsize=FS)
    ax[1].set_ylabel('density', fontsize=FS)
    ax[1].tick_params(labelsize=FS)
    ax[1].legend(fontsize=FS - 2, loc='lower right')

    fig.tight_layout()
    #fig.savefig('106_olivine_ab.png', dpi=200)

# ── FIG 4：地溫線 ─────────────────────────────────────────────────────────────
if TOGGLE['geotherm'] and len(T_all) > 0:
    # 深度網格
    n_valid = np.sum(np.isfinite(T_all), axis=0)
    ok = n_valid >= max(20, 0.5 * len(T_all))
    T05 = np.nanpercentile(T_all, 5, axis=0)
    T25 = np.nanpercentile(T_all, 25, axis=0)
    T50 = np.nanpercentile(T_all, 50, axis=0)
    T75 = np.nanpercentile(T_all, 75, axis=0)
    T95 = np.nanpercentile(T_all, 95, axis=0)

    # 壓力網格
    np_valid = np.sum(np.isfinite(Tp_all), axis=0)
    okp = np_valid >= max(20, 0.5 * len(Tp_all))
    Q05 = np.nanpercentile(Tp_all, 5, axis=0)
    Q25 = np.nanpercentile(Tp_all, 25, axis=0)
    Q50 = np.nanpercentile(Tp_all, 50, axis=0)
    Q75 = np.nanpercentile(Tp_all, 75, axis=0)
    Q95 = np.nanpercentile(Tp_all, 95, axis=0)

    print('geotherm 深度包絡上限 :', Z_GRID[ok].max())
    print('geotherm 壓力包絡上限 :', P_GRID[okp].max())
    i500 = np.argmin(abs(Z_GRID - 500))
    i1400 = np.argmin(abs(Z_GRID - 1400))
    print('T @ 500 km  5/50/95 :', T05[i500], T50[i500], T95[i500])
    print('T @ 1400 km 5/50/95 :', T05[i1400], T50[i1400], T95[i1400])

    # 固相線：直接畫在 P_GRID 上，自變數就是壓力，不經過任何轉換
    #   T_sol = -4.877 P^2 + 120.2 P + 600 * Mg# + 911.15   (P <= 10 GPa)
    p_sol = P_GRID[P_GRID <= 10.0]
    Tsol = -4.877 * p_sol**2 + 120.2 * p_sol + 600.0 * SOLIDUS_MG + 911.15

    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))

    # 左：T-z（固相線畫不上去，因為固相線是壓力的函數）
    ax[0].fill_betweenx(Z_GRID[ok], T05[ok], T95[ok], color='0.8', label='5-95%')
    ax[0].fill_betweenx(Z_GRID[ok], T25[ok], T75[ok], color='0.6', label='25-75%')
    ax[0].plot(T50[ok], Z_GRID[ok], 'k-', lw=2.5, label='median')

    for path, lab, col in LIT_GEOTHERM:
        a = np.loadtxt(path)
        ax[0].plot(a[:, 1], a[:, 0], '-', lw=2, color=col, label=lab)

    ax[0].axhspan(np.percentile(zmax_mantle, 5), np.percentile(zmax_mantle, 95),
                  color='tab:red', alpha=0.10)
    ax[0].axhline(np.percentile(zmax_mantle, 50), color='tab:red', lw=1.5, ls=':',
                  label='BML top (median)')

    ax[0].set_xlabel('temperature (K)', fontsize=FS)
    ax[0].set_ylabel('depth (km)', fontsize=FS)
    ax[0].set_ylim(Z_GRID[ok].max() + 100, 0)
    ax[0].tick_params(labelsize=FS)
    ax[0].legend(fontsize=FS - 3, loc='lower right')

    # 右：T-P（固相線在這裡才是自洽的）
    ax[1].fill_betweenx(P_GRID[okp], Q05[okp], Q95[okp], color='0.8', label='5-95%')
    ax[1].fill_betweenx(P_GRID[okp], Q25[okp], Q75[okp], color='0.6', label='25-75%')
    ax[1].plot(Q50[okp], P_GRID[okp], 'k-', lw=2.5, label='median')
    ax[1].plot(Tsol, p_sol, 'r--', lw=2,
               label='solidus (Mg# = %.2f, $P \\leq 10$ GPa)' % SOLIDUS_MG)

    ax[1].axhspan(np.percentile(pmax_mantle, 5), np.percentile(pmax_mantle, 95),
                  color='tab:red', alpha=0.10)
    ax[1].axhline(np.percentile(pmax_mantle, 50), color='tab:red', lw=1.5, ls=':',
                  label='BML top (median)')

    ax[1].set_xlabel('temperature (K)', fontsize=FS)
    ax[1].set_ylabel('pressure (GPa)', fontsize=FS)
    ax[1].set_ylim(P_GRID[okp].max() * 1.05, 0)
    ax[1].tick_params(labelsize=FS)
    ax[1].legend(fontsize=FS - 3, loc='lower right')

    fig.tight_layout()
    #fig.savefig('106_geotherm.png', dpi=200)

plt.show()