#!/usr/bin/env python3
"""
plot_phase_diagram.py
Fo-Fa 二元相圖 (同 48_melting_Mg#_W.py 邏輯) + MCMC posterior samples
色碼 = BML_thickness

用法: python plot_phase_diagram.py
(設定直接改檔案頂部的變數)
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ── 設定 ───────────────────────────────────────────────────────────────────────
CHAIN_DIR = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'
PREFIX    = 'chain_a23'
BURNIN    = 0.1   # 丟掉前 30%
OUT_DIR   = '/net/beno3/data1/jcchen2/mars-hefesto-runs/figures'
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
TAG = f'{PREFIX}_' if PREFIX else ''

# ── 讀 chain.json ──────────────────────────────────────────────────────────────
pattern     = f'{PREFIX}*/chain.json' if PREFIX else '*/chain.json'
chain_files = sorted(glob.glob(os.path.join(CHAIN_DIR, pattern)))
print(f"找到 {len(chain_files)} 條 chain")

samples = []
for f in chain_files:
    with open(f) as fh:
        chain = json.load(fh)
    burnin_step = int(chain[-1]['step'] * BURNIN) if chain else 0
    for s in chain:
        if s['step'] > burnin_step:
            samples.append(s)
print(f"Post burn-in samples: {len(samples)}")

if not samples:
    print("ERROR: 沒有 samples，請確認 --chain_dir 和 --prefix")
    raise SystemExit(1)

Mg_bulk = np.array([s['params']['Mg#_bulk_bml'] for s in samples])
T_core  = np.array([s['params']['T_core']        for s in samples])
BML_th  = np.array([s['params']['BML_thickness'] for s in samples])

# ── Phase diagram (直接從 48_melting_Mg#_W.py 搬過來) ─────────────────────────
R   = 8.314
W   = 8000.0   # J/mol, best fit Pierru 2026
X26 = 0.75

pierru26_sol = {7.95: 1890, 9.6: 1960, 14.1: 2090, 17.7: 2200}

pierru4 = [
    (8.0,  1890, 'Ol(cold)', 0.80, 0.53),
    (8.0,  1890, 'Ol(warm)', 0.85, 0.58),
    (9.6,  1960, 'Ol',       0.91, 0.67),
    (14.1, 2090, 'Wad',      0.80, 0.51),
    (17.7, 2200, 'Rw',       0.74, 0.50),
]

def get_DH(P):
    if P < 14.0:   return 142000.0, 89000.0
    elif P < 20.0: return 116000.0, 73000.0
    else:          return 105000.0, 74000.0

def Tm_Fo(P):
    return 2163.0 * (P / 65.67 + 1.0) ** (1.0 / 0.7809)

def get_TmFa(P_key):
    Tsol   = pierru26_sol[P_key]
    TmFo_P = Tm_Fo(P_key)
    DH_Fo, DH_Fa = get_DH(P_key)
    XS = X26
    def eqs(v):
        TmFa, XL = v
        XL = np.clip(XL, 1e-9, 1-1e-9)
        e1 = 2*np.log(XL/XS)         + W/(R*Tsol)*(1-XS)**2 - (DH_Fo/R)*(1/TmFo_P - 1/Tsol)
        e2 = 2*np.log((1-XL)/(1-XS)) + W/(R*Tsol)*XS**2     - (DH_Fa/R)*(1/TmFa   - 1/Tsol)
        return [e1, e2]
    for g in [[1400, 0.40], [1600, 0.35], [1200, 0.45]]:
        sol, info, ier, _ = fsolve(eqs, g, full_output=True)
        if ier == 1 and np.max(np.abs(info['fvec'])) < 1e-8:
            return float(sol[0])
    return np.nan

def solve_XS_XL(T, P, TmFa):
    DH_Fo, DH_Fa = get_DH(P)
    TmFo = Tm_Fo(P)
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        e1 = 2*np.log(XL/XS)         + W/(R*T)*(1-XS)**2 - (DH_Fo/R)*(1/TmFo - 1/T)
        e2 = 2*np.log((1-XL)/(1-XS)) + W/(R*T)*XS**2     - (DH_Fa/R)*(1/TmFa - 1/T)
        return [e1, e2]
    for x0 in [[0.85,0.65],[0.80,0.55],[0.70,0.45],[0.90,0.70]]:
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        XS, XL = sol
        if ier==1 and np.max(np.abs(info['fvec']))<1e-6 and 0<XS<1 and 0<XL<1 and XS>XL:
            return float(np.clip(XS,0,1)), float(np.clip(XL,0,1))
    return np.nan, np.nan

# ── 4 個壓力面板 ──────────────────────────────────────────────────────────────
target_P   = [7.95, 9.6, 14.1, 17.7]
P_label    = {7.95: 8.0, 9.6: 9.6, 14.1: 14.1, 17.7: 17.7}
phase_name = {7.95:'Olivine', 9.6:'Olivine', 14.1:'Wadsleyite', 17.7:'Ringwoodite'}

fig, axes = plt.subplots(2, 2, figsize=(14, 13))
axes = axes.flatten()

vmin, vmax = BML_th.min(), BML_th.max()

for ax, P_target in zip(axes, target_P):
    TmFo_P = Tm_Fo(P_target)
    Tsol26 = pierru26_sol[P_target]
    TmFa   = get_TmFa(P_target)
    if np.isnan(TmFa):
        ax.set_title(f'P={P_label[P_target]} GPa  TmFa=NaN'); continue

    # 掃 T → XS(T), XL(T)  (同 48 的做法)
    T_arr  = np.linspace(TmFa * 0.97, TmFo_P * 1.02, 400)
    XS_arr = np.full(len(T_arr), np.nan)
    XL_arr = np.full(len(T_arr), np.nan)
    for i, T in enumerate(T_arr):
        if T >= TmFo_P:
            XS_arr[i] = 1.0; XL_arr[i] = 1.0
        elif T <= TmFa:
            XS_arr[i] = 0.0; XL_arr[i] = 0.0
        else:
            XS_arr[i], XL_arr[i] = solve_XS_XL(T, P_target, TmFa)

    valid = ~np.isnan(XS_arr)
    ax.plot(XS_arr[valid], T_arr[valid], 'k-',  lw=2.5, label='solidus')
    ax.plot(XL_arr[valid], T_arr[valid], 'k--', lw=2.5, label='liquidus')
    ax.fill_betweenx(T_arr[valid], XL_arr[valid], XS_arr[valid],
                     color='lightblue', alpha=0.3, label='partial melt')

    # Pierru Table 3 anchor ■
    ax.scatter([X26], [Tsol26], marker='s', s=200,
               color='limegreen', edgecolors='darkgreen', zorder=11, lw=2,
               label=f'Pierru T$_{{sol}}$ {Tsol26}K')

    # Pierru Table 4 tie lines ●─◆
    for (Pp, Tp, mineral, XS_obs, XL_obs) in pierru4:
        if abs(Pp - P_label[P_target]) < 1.0:
            ax.scatter([XS_obs], [Tp], marker='o', s=120,
                       color='limegreen', edgecolors='darkgreen', zorder=12, lw=2)
            ax.scatter([XL_obs], [Tp], marker='D', s=120,
                       color='limegreen', edgecolors='darkgreen', zorder=12, lw=2)
            ax.plot([XS_obs, XL_obs], [Tp, Tp], color='darkgreen', lw=2, zorder=9)

    # MCMC posterior scatter (色碼 = BML_thickness)
    sc = ax.scatter(Mg_bulk, T_core,
                    c=BML_th, cmap='plasma', vmin=vmin, vmax=vmax,
                    s=8, alpha=0.4, zorder=5, linewidths=0)

    ax.axvline(X26,  color='darkorange', ls='--', lw=1.5, alpha=0.7)
    ax.text(X26+0.01, TmFa*0.98, 'Mg#=0.75', fontsize=8, color='darkorange')

    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(1400, min(TmFo_P * 1.05, 3200))
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=11)
    ax.set_ylabel('Temperature (K)', fontsize=11)
    ax.set_title(
        f'P = {P_label[P_target]} GPa  |  {phase_name[P_target]}\n'
        f'$T_m^{{Fo}}$={TmFo_P:.0f}K   $T_m^{{Fa}}$={TmFa:.0f}K  (W=8000 J/mol)',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    ax.text(0.38, TmFo_P*0.995, 'Liquid',     fontsize=9, color='darkred',  fontweight='bold')
    ax.text(0.88, TmFa*1.01,   'Solid$_{ss}$', fontsize=9, color='darkblue', fontweight='bold')

# 共用 colorbar
fig.subplots_adjust(right=0.88)
cax  = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label('BML thickness (km)', fontsize=11)

fig.suptitle(
    'Fo-Fa Phase Diagram (Pierru 2026, W=8000 J/mol) + MCMC posterior\n'
    'Black: solidus/liquidus  |  Green ■●◆: Pierru 2026 Table 3&4\n'
    f'Scatter: post-burnin samples (n={len(samples)}), color = BML thickness',
    fontsize=11)

fname = os.path.join(OUT_DIR, TAG + 'phase_diagram_posterior.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"saved {fname}")