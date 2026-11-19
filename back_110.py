#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:14:39 2026

@author: chingchen
"""

# 110.ppt_bml_figs.py
# quals PPT / BML：一支腳本，多張分開存的圖
#
#   B1  110_ScS_Rcmb.png     R_ScS = R_cmb + h_liquid  vs  R_cmb   ← 主結果
#   B2  110_bml_profiles.png BML 內部 T(z) 與 rho(z)，三態各一組包絡
#   B3  110_bml_Mg.png       Mg#_bml / X_S / X_L 與地函 Mg# 對照
#   B4  110_Tcore.png        T_core 邊際，分三態，疊 Cheng 2025 band
#   B5  110_Ra.png           Ra 分布（只有含固態層的樣本有定義）
#   B6  110_phase_TX.png     Fo-Fa 二元 loop：T-X，固定 P
#   B7  110_phase_TP.png     solidus/liquidus vs P，固定數個 Mg#，疊後驗界面點
#
# 三態的 apparent ScS bouncing depth 是同一條式子，不是三條：
#     R_ScS = R_cmb + h_liquid      h_liquid = 0 (solid) / D (molten) / 中間 (layered)
# 所以 B1 是一張圖。這也是「R_ScS 被資料定住、R_cmb 沒有」那句話的載體。

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

# ── config ────────────────────────────────────────────────────────────────────
TOGGLE = {'ScS': True, 'profiles': True, 'Mg': True, 'Tcore': True,
          'Ra': True, 'phase_TX': True, 'phase_TP': True}

CHAIN_DIR = MCMC_DIR   # ← 改成本機實際路徑
PREFIX    = 'chain_a52B'
BURNIN    = 200
MIN_STEPS = 300

PROF_GLOB = os.path.join(CHAIN_DIR, PREFIX + '*', 'profiles', '*.npz')
STRIDE    = 1

MARS_RADIUS = 3389.5      # km
H_MIN       = 1.0         # km，三態分類門檻
LAYER_BML   = 2

# chain.jsonl 的欄位名 —— 先跑最下面的「欄位探測」區塊確認，再改這裡
K_RCMB   = 'R_cmb'             # 在 params 裡，不是 top level
K_MGSOL  = 'Mg_solid'
K_MGLIQ  = 'Mg_liquid'
K_RA     = 'Ra'
K_TINT   = 'T_interface'       # 有
K_PINT   = 'P_interface'       # 有
K_HSOL   = 'h_solid_km'
K_HLIQ   = 'h_liquid_km'
 

PRIOR_MG_BML = (0.20, 0.80)
PRIOR_TCORE  = (1800.0, 3200.0)

CHENG_TCORE = (2200.0, 2400.0)   # Cheng 2025 熱演化允許的 T_core 區間

# 相圖模組（不依賴 nGibbs）
PHASE_MODULE = HERE   # ← 放 70.mcmc_core_nGibbs.py 的目錄
# bml_phase_diagram 回傳 dict 的鍵名，探測後改這裡
KEY_TSOL = 'T_sol'
KEY_TLIQ = 'T_liq'

# B6：T-X loop 要畫哪幾個壓力（GPa）
TX_PRESSURES = [14.0, 17.7, 21.7]
TX_MG_GRID = np.linspace(0.05, 0.95, 91)

# B7：T-P 要畫哪幾個 Mg#
TP_MGS = [0.40, 0.50, 0.60, 0.70]
TP_P_GRID = np.linspace(12.0, 24.0, 49)

FS = 14
STATE_COLOR = {'solid': 'tab:blue', 'layered': 'tab:green', 'molten': 'tab:red'}

# ── 欄位探測：第一次跑先看這裡印什麼，再回去改上面的 K_* ─────────────────────
files = sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '*', 'chain.jsonl')))
print('chain files:', len(files))
if len(files) == 0:
    print('CHAIN_DIR 內容:', os.listdir(CHAIN_DIR))
    raise SystemExit('no chain matched: check CHAIN_DIR / PREFIX')

r0 = json.loads(open(files[0]).readline())
print('record top-level keys :', sorted(r0.keys()))
print('params keys           :', sorted(r0['params'].keys()))

# ── 載入 chain ────────────────────────────────────────────────────────────────
R_cmb = []
h_sol = []
h_liq = []
D_bml = []
Mg_man = []
Mg_bml = []
X_S = []
X_L = []
T_core = []
Ra = []
T_int = []
P_int = []
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
            break
    if len(recs) < MIN_STEPS:
        print('  skip (too short):', f, len(recs))
        continue
    recs = recs[BURNIN:]
    for r in recs:
        p = r['params']
        hs = r[K_HSOL]
        hl = r[K_HLIQ]
        if hl < H_MIN:
            s = 'solid'
        elif hs < H_MIN:
            s = 'molten'
        else:
            s = 'layered'
        R_cmb.append(p[K_RCMB])
        h_sol.append(hs)
        h_liq.append(hl)
        D_bml.append(p['BML_thickness'])
        Mg_man.append(p['Mg#'])
        Mg_bml.append(p['Mg#_bulk_bml'])
        X_S.append(r[K_MGSOL] if K_MGSOL in r else np.nan)
        X_L.append(r[K_MGLIQ] if K_MGLIQ in r else np.nan)
        T_core.append(p['T_core'])
        Ra.append(r[K_RA] if K_RA in r else np.nan)
        T_int.append(r[K_TINT] if K_TINT in r else np.nan)
        P_int.append(r[K_PINT] if K_PINT in r else np.nan)
        state.append(s)
    print('  loaded:', os.path.basename(os.path.dirname(f)), len(recs))

R_cmb  = np.array(R_cmb)
h_sol  = np.array(h_sol)
h_liq  = np.array(h_liq)
D_bml  = np.array(D_bml)
Mg_man = np.array(Mg_man)
Mg_bml = np.array(Mg_bml)
X_S    = np.array(X_S, dtype=float)
X_L    = np.array(X_L, dtype=float)
T_core = np.array(T_core)
Ra     = np.array(Ra, dtype=float)
T_int  = np.array(T_int, dtype=float)
P_int  = np.array(P_int, dtype=float)
state  = np.array(state)

# apparent ScS bouncing radius：ScS 在液相 BML 頂反射
#   solid   h_liquid = 0   -> R_ScS = R_cmb
#   layered              -> R_ScS = R_cmb + h_liquid
#   molten  h_liquid = D   -> R_ScS = R_cmb + D
R_ScS = R_cmb + h_liq

print('n =', len(R_cmb))
print('state counts:', {s: int((state == s).sum()) for s in STATE_COLOR})
print('R_cmb  5/50/95 :', np.percentile(R_cmb, [5, 50, 95]), ' sd =', R_cmb.std())
print('R_ScS  5/50/95 :', np.percentile(R_ScS, [5, 50, 95]), ' sd =', R_ScS.std())
print('corr(R_cmb, h_liquid) :', np.corrcoef(R_cmb, h_liq)[0, 1])
print('sd 比 R_cmb / R_ScS   :', R_cmb.std() / R_ScS.std())
for s in STATE_COLOR:
    m = state == s
    if m.sum() > 0:
        print('  ', s, 'n =', int(m.sum()),
              ' R_cmb', np.percentile(R_cmb[m], [5, 50, 95]),
              ' R_ScS', np.percentile(R_ScS[m], [5, 50, 95]),
              ' T_core', np.percentile(T_core[m], [5, 50, 95]))
print('Mg#_bml 貼上牆 :', float(np.mean(Mg_bml > PRIOR_MG_BML[1] - 0.01)))
print('Mg#_bml 貼下牆 :', float(np.mean(Mg_bml < PRIOR_MG_BML[0] + 0.01)))
print('T_core  貼上牆 :', float(np.mean(T_core > PRIOR_TCORE[1] - 10)))
print('T_core  貼下牆 :', float(np.mean(T_core < PRIOR_TCORE[0] + 10)))

# ── B1：R_ScS vs R_cmb ────────────────────────────────────────────────────────
if TOGGLE['ScS']:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    for s in ['molten', 'layered', 'solid']:
        m = state == s
        if m.sum() > 0:
            ax[0].plot(R_cmb[m], R_ScS[m], '.', ms=2, alpha=0.25,
                       color=STATE_COLOR[s], label=s)
    lo = min(R_cmb.min(), R_ScS.min()) - 20
    hi = max(R_cmb.max(), R_ScS.max()) + 20
    ax[0].plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='$R_{ScS} = R_{cmb}$')
    ax[0].set_xlim(lo, hi)
    ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel('$R_{cmb}$ (km)', fontsize=FS)
    ax[0].set_ylabel('apparent ScS bouncing radius $R_{ScS}$ (km)', fontsize=FS)
    ax[0].tick_params(labelsize=FS)
    leg = ax[0].legend(fontsize=FS - 2, loc='lower right', markerscale=6)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    bins = np.linspace(lo, hi, 45)
    ax[1].hist(R_cmb, bins=bins, histtype='step', lw=2.5, color='k',
               density=True, label='$R_{cmb}$  (sd %.0f km)' % R_cmb.std())
    ax[1].hist(R_ScS, bins=bins, histtype='step', lw=2.5, color='tab:orange',
               density=True, label='$R_{ScS}$  (sd %.0f km)' % R_ScS.std())
    ax[1].set_xlabel('radius (km)', fontsize=FS)
    ax[1].set_ylabel('posterior density', fontsize=FS)
    ax[1].tick_params(labelsize=FS)
    ax[1].legend(fontsize=FS - 2, loc='lower right')

    fig.tight_layout()
    #fig.savefig('110_ScS_Rcmb.png', dpi=200)

# ── B4：T_core ────────────────────────────────────────────────────────────────
if TOGGLE['Tcore']:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bins = np.linspace(PRIOR_TCORE[0], PRIOR_TCORE[1], 45)
    ax.hist(T_core, bins=bins, color='0.75', edgecolor='none', density=True,
            label='all')
    for s in ['solid', 'layered', 'molten']:
        m = state == s
        if m.sum() > 20:
            ax.hist(T_core[m], bins=bins, histtype='step', lw=2,
                    color=STATE_COLOR[s], density=True, label=s)
    ax.axhline(1.0 / (PRIOR_TCORE[1] - PRIOR_TCORE[0]), color='k', ls='--', lw=1.5,
               label='prior')
    ax.axvspan(CHENG_TCORE[0], CHENG_TCORE[1], color='tab:purple', alpha=0.15,
               label='Cheng et al. 2025')
    ax.set_xlabel('$T_{core}$ (K)', fontsize=FS)
    ax.set_ylabel('posterior density', fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS - 2, loc='lower right')
    fig.tight_layout()
    #fig.savefig('110_Tcore.png', dpi=200)

# ── B3：BML Mg# ───────────────────────────────────────────────────────────────
if TOGGLE['Mg']:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    bins = np.linspace(PRIOR_MG_BML[0], PRIOR_MG_BML[1], 40)
    ax[0].hist(Mg_bml, bins=bins, color='0.75', edgecolor='none', density=True,
               label='bulk BML (all)')
    for s in ['solid', 'layered', 'molten']:
        m = state == s
        if m.sum() > 20:
            ax[0].hist(Mg_bml[m], bins=bins, histtype='step', lw=2,
                       color=STATE_COLOR[s], density=True, label='bulk, ' + s)
    ax[0].axhline(1.0 / (PRIOR_MG_BML[1] - PRIOR_MG_BML[0]), color='k', ls='--',
                  lw=1.5, label='prior')
    ax[0].axvline(np.median(Mg_man), color='tab:brown', lw=2.5,
                  label='mantle Mg# (median)')
    ax[0].set_xlabel('BML bulk Mg#', fontsize=FS)
    ax[0].set_ylabel('posterior density', fontsize=FS)
    ax[0].tick_params(labelsize=FS)
    ax[0].legend(fontsize=FS - 3, loc='lower right')

    # layered 才有共存兩相；X_S 富 Mg 在上、X_L 富 Fe 在下
    m = (state == 'layered') & np.isfinite(X_S) & np.isfinite(X_L)
    print('layered with X_S/X_L :', int(m.sum()))
    if m.sum() > 0:
        ax[1].hist(X_S[m], bins=bins, histtype='step', lw=2.5, color='tab:blue',
                   density=True, label='$X_S$ (solid)')
        ax[1].hist(X_L[m], bins=bins, histtype='step', lw=2.5, color='tab:red',
                   density=True, label='$X_L$ (liquid)')
        ax[1].hist(Mg_bml[m], bins=bins, histtype='step', lw=2, color='k',
                   ls='--', density=True, label='bulk')
        ax[1].axvline(np.median(Mg_man), color='tab:brown', lw=2.5,
                      label='mantle Mg# (median)')
    ax[1].set_xlabel('Mg# (layered samples only)', fontsize=FS)
    ax[1].set_ylabel('density', fontsize=FS)
    ax[1].tick_params(labelsize=FS)
    ax[1].legend(fontsize=FS - 3, loc='lower right')

    fig.tight_layout()
    #fig.savefig('110_bml_Mg.png', dpi=200)

# ── B5：Ra ────────────────────────────────────────────────────────────────────
if TOGGLE['Ra'] and np.isfinite(Ra).any():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    m_ok = np.isfinite(Ra) & (Ra > 0)
    print('Ra 有定義的樣本 :', int(m_ok.sum()))
    print('log10 Ra 5/50/95 :', np.percentile(np.log10(Ra[m_ok]), [5, 50, 95]))
    bins = np.linspace(np.floor(np.log10(Ra[m_ok]).min()),
                       np.ceil(np.log10(Ra[m_ok]).max()), 40)
    for s in ['solid', 'layered']:
        m = m_ok & (state == s)
        if m.sum() > 20:
            ax.hist(np.log10(Ra[m]), bins=bins, histtype='step', lw=2.5,
                    color=STATE_COLOR[s], density=True, label=s)
    ax.axvline(np.log10(657.0), color='k', ls='--', lw=2, label='$Ra_c$ = 657')
    ax.set_xlabel('$\\log_{10} Ra$ (solid BML sublayer)', fontsize=FS)
    ax.set_ylabel('posterior density', fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS - 2, loc='lower right')
    fig.tight_layout()
    #fig.savefig('110_Ra.png', dpi=200)

# ── B2：BML 內部剖面 ──────────────────────────────────────────────────────────
if TOGGLE['profiles']:
    pfiles = sorted(glob.glob(PROF_GLOB))[::STRIDE]
    print('profiles:', len(pfiles))

    # 以「距 BML 頂的相對深度」為自變數，因為各模型 BML 頂深度不同
    ZR = np.linspace(0.0, 1.0, 101)          # 0 = BML 頂, 1 = CMB
    T_by = {'solid': [], 'layered': [], 'molten': []}
    R_by = {'solid': [], 'layered': [], 'molten': []}

    for f in pfiles:
        d = np.load(f)
        layer = d['layer']
        m = layer == LAYER_BML
        if m.sum() < 5:
            continue
        zz = d['z_km'][m]
        if zz.max() - zz.min() < 1.0:
            continue
        hs = float(d['h_solid_km'])
        hl = float(d['h_liquid_km'])
        if hl < H_MIN:
            s = 'solid'
        elif hs < H_MIN:
            s = 'molten'
        else:
            s = 'layered'
        zr = (zz - zz.min()) / (zz.max() - zz.min())
        T_by[s].append(np.interp(ZR, zr, d['T_K'][m]))
        R_by[s].append(np.interp(ZR, zr, d['rho'][m]))

    for s in T_by:
        print('  BML profiles', s, ':', len(T_by[s]))

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for s in ['solid', 'layered', 'molten']:
        if len(T_by[s]) < 20:
            continue
        A = np.array(T_by[s])
        B = np.array(R_by[s])
        ax[0].fill_betweenx(ZR, np.percentile(A, 5, axis=0),
                            np.percentile(A, 95, axis=0),
                            color=STATE_COLOR[s], alpha=0.15)
        ax[0].plot(np.percentile(A, 50, axis=0), ZR, '-', lw=2.5,
                   color=STATE_COLOR[s], label=s)
        ax[1].fill_betweenx(ZR, np.percentile(B, 5, axis=0),
                            np.percentile(B, 95, axis=0),
                            color=STATE_COLOR[s], alpha=0.15)
        ax[1].plot(np.percentile(B, 50, axis=0), ZR, '-', lw=2.5,
                   color=STATE_COLOR[s], label=s)

    ax[0].set_xlabel('temperature (K)', fontsize=FS)
    ax[0].set_ylabel('relative depth in BML  (0 = top, 1 = CMB)', fontsize=FS)
    ax[0].set_ylim(1, 0)
    ax[0].tick_params(labelsize=FS)
    ax[0].legend(fontsize=FS - 2, loc='lower right')

    ax[1].set_xlabel('density (g/cm$^3$)', fontsize=FS)
    ax[1].set_ylabel('relative depth in BML', fontsize=FS)
    ax[1].set_ylim(1, 0)
    ax[1].tick_params(labelsize=FS)
    ax[1].legend(fontsize=FS - 2, loc='lower right')

    fig.tight_layout()
    #fig.savefig('110_bml_profiles.png', dpi=200)

# ── 相圖：載入模組 ────────────────────────────────────────────────────────────
if TOGGLE['phase_TX'] or TOGGLE['phase_TP']:
    import sys
    sys.path.insert(0, PHASE_MODULE)
    from importlib import import_module
    mod = import_module('70.mcmc_core_nGibbs')
    bml_phase_diagram = mod.bml_phase_diagram

    probe = bml_phase_diagram(2200.0, 17.7, 0.50)
    print('bml_phase_diagram keys :', sorted(probe.keys()))
    print('probe :', probe)

# ── B6：T-X 二元 loop ─────────────────────────────────────────────────────────
if TOGGLE['phase_TX']:
    fig, ax = plt.subplots(figsize=(8, 6))
    for P, ls in zip(TX_PRESSURES, ['-', '--', ':']):
        ts = []
        tl = []
        for X in TX_MG_GRID:
            pdct = bml_phase_diagram(2000.0, P, float(X))
            ts.append(pdct[KEY_TSOL])
            tl.append(pdct[KEY_TLIQ])
        ax.plot(TX_MG_GRID, ts, ls, lw=2, color='tab:blue',
                label='solidus, %.1f GPa' % P)
        ax.plot(TX_MG_GRID, tl, ls, lw=2, color='tab:red',
                label='liquidus, %.1f GPa' % P)

    # 後驗的 bulk Mg# 位置（只是標示範圍，縱軸無意義）
    ax.axvspan(np.percentile(Mg_bml, 5), np.percentile(Mg_bml, 95),
               color='0.8', alpha=0.5, label='posterior bulk Mg# (5-95%)')

    ax.set_xlabel('Mg#  (Fo-Fa binary)', fontsize=FS)
    ax.set_ylabel('temperature (K)', fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS - 3, loc='lower right')
    fig.tight_layout()
    #fig.savefig('110_phase_TX.png', dpi=200)

# ── B7：solidus / liquidus vs P，固定數個 Mg# ─────────────────────────────────
if TOGGLE['phase_TP']:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = plt.get_cmap('viridis')
    for i, X in enumerate(TP_MGS):
        col = cmap(i / max(len(TP_MGS) - 1, 1))
        ts = []
        tl = []
        for P in TP_P_GRID:
            pdct = bml_phase_diagram(2000.0, float(P), X)
            ts.append(pdct[KEY_TSOL])
            tl.append(pdct[KEY_TLIQ])
        ax.plot(ts, TP_P_GRID, '-', lw=2, color=col, label='Mg# = %.2f' % X)
        ax.plot(tl, TP_P_GRID, '--', lw=2, color=col)

    # 後驗的界面點：落在哪一對曲線之間就決定三態
    if np.isfinite(T_int).any():
        for s in ['molten', 'layered', 'solid']:
            m = (state == s) & np.isfinite(T_int) & np.isfinite(P_int)
            if m.sum() > 0:
                ax.plot(T_int[m], P_int[m], '.', ms=2, alpha=0.2,
                        color=STATE_COLOR[s], label=s + ' (interface)')
    else:
        print('T_interface / P_interface 不在 chain 裡 -> B7 只畫相界')

    ax.set_xlabel('temperature (K)', fontsize=FS)
    ax.set_ylabel('pressure (GPa)', fontsize=FS)
    ax.invert_yaxis()
    ax.tick_params(labelsize=FS)
    leg = ax.legend(fontsize=FS - 3, loc='lower right', markerscale=6)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    fig.tight_layout()
    #fig.savefig('110_phase_TP.png', dpi=200)

plt.show()