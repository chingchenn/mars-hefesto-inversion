#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:09:18 2026

@author: chingchen
"""

#!/usr/bin/env python3
"""
Fo-Fa Phase Diagram — Mars Effective (v12)
==========================================
基礎：45_Mg#_Temperature_2.py 的邏輯

Step 1: 從 Pierru 2026 Table 3 反推 Tm_Fa（XS=0.75, Tm_Fo=Ohtani）
Step 2: 用不同 W 畫相圖
Step 3: 在 Table 4 的每個 (XS_obs, T) 點，用 eq2 預測 XL_pred
        → 跟 Table 4 的 XL_obs 比較，找最佳 W

★ 改動 W_list ★
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ============================================================
# ★ 改動這裡 ★
# ============================================================
W_list = [0, 2500, 8000, 15000]   # J/mol

# ============================================================
# 常數
# ============================================================
R   = 8.314
X26 = 0.75   # Pierru 2026 bulk Mg#（solidus 定義）

def get_DH(P):
    if P < 14.0:   return 142000.0, 89000.0
    elif P < 20.0: return 116000.0, 73000.0
    else:          return 105000.0, 74000.0

def Tm_Fo(P):
    T0, a, c = 2163.0, 65.67, 0.7809
    return T0 * (P / a + 1.0) ** (1.0 / c)

# ============================================================
# Pierru 2026 Table 3: solidus T at Mg#=0.75
# ============================================================
pierru26_sol = {7.95: 1890, 9.6: 1960, 14.1: 2090, 17.7: 2200}

# ============================================================
# Pierru 2026 Table 4: tie lines (XS, XL at T, P)
# (XS, T) 在 solidus 上，(XL, T) 在 liquidus 上
# ============================================================
pierru4 = [
    (8.0,  1890, 'Ol(cold)', 0.80, 0.53),
    (8.0,  1890, 'Ol(warm)', 0.85, 0.58),
    (9.6,  1960, 'Ol',       0.91, 0.67),
    (14.1, 2090, 'Wad',      0.80, 0.51),
    (17.7, 2200, 'Rw',       0.74, 0.50),
]

# Duncan solidus
def T_sol_duncan(P):
    P = float(P)
    if P <= 10.0:
        T_C = -4.877*P**2 + 120.2*P + 1088.0
    elif P <= 23.0:
        T_C = -1.323*(P-10.0)**2 + 38.18*(P-10.0) + 1802.0
    else:
        T_C = 77.75*(P-23.0) + 2075.0
    return T_C + 273.15

duncan_sup_P = np.array([8.1, 9.8, 10.2, 9.9, 16.5, 16.5, 20.0, 23.0, 25.0])
duncan_sup_T = np.array([1675,1815,1850,1875,2050,2110,2100,2150,2300]) + 273.15
duncan_sub_P = np.array([9.9, 16.5, 20.0])
duncan_sub_T = np.array([1775, 2000, 1975]) + 273.15

# ============================================================
# Step 1: 從 Table 3 反推 Tm_Fa 和 XL（給定 W）
#
# eq1: 2*ln(XL/XS) + W/(RT)*(1-XS)^2 = (DH_Fo/R)*(1/Tm_Fo - 1/T)
# eq2: 2*ln((1-XL)/(1-XS)) + W/(RT)*XS^2 = (DH_Fa/R)*(1/Tm_Fa - 1/T)
# XS = 0.75, T = T_sol → 解出 Tm_Fa, XL
# ============================================================
def get_TmFa_XL(P, W):
    Tsol   = pierru26_sol[P]
    TmFo_P = Tm_Fo(P)
    DH_Fo, DH_Fa = get_DH(P)
    XS = X26

    def eqs(v):
        TmFa, XL = v
        XL = np.clip(XL, 1e-9, 1-1e-9)
        e1 = (2*np.log(XL/XS) + W/(R*Tsol)*(1-XS)**2
              - (DH_Fo/R)*(1/TmFo_P - 1/Tsol))
        e2 = (2*np.log((1-XL)/(1-XS)) + W/(R*Tsol)*XS**2
              - (DH_Fa/R)*(1/TmFa - 1/Tsol))
        return [e1, e2]

    for g in [[1400, 0.40], [1600, 0.35], [1200, 0.45]]:
        sol, info, ier, _ = fsolve(eqs, g, full_output=True)
        if ier == 1 and np.max(np.abs(info['fvec'])) < 1e-8:
            return sol[0], sol[1]
    return np.nan, np.nan

# ============================================================
# Step 2: 相圖 solver
# ============================================================
def solve_XS_XL(T, P, TmFa, W):
    DH_Fo, DH_Fa = get_DH(P)
    TmFo = Tm_Fo(P)
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        e1 = (2*np.log(XL/XS) + W/(R*T)*(1-XS)**2
              - (DH_Fo/R)*(1/TmFo - 1/T))
        e2 = (2*np.log((1-XL)/(1-XS)) + W/(R*T)*XS**2
              - (DH_Fa/R)*(1/TmFa - 1/T))
        return [e1, e2]
    for x0 in [[0.85,0.65],[0.80,0.55],[0.70,0.45],[0.90,0.70]]:
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        XS, XL = sol
        if (ier==1 and np.max(np.abs(info['fvec']))<1e-6
                and 0<XS<1 and 0<XL<1 and XS>XL):
            return float(np.clip(XS,0,1)), float(np.clip(XL,0,1))
    return np.nan, np.nan

# ============================================================
# Step 3: 給定 Tm_Fa 和 W，在 Table 4 的 (XS_obs, T) 預測 XL
# eq2 解析求 XL：
#   2*ln((1-XL)/(1-XS)) = (DH_Fa/R)*(1/Tm_Fa - 1/T) - W/(RT)*XS^2
# ============================================================
def predict_XL(P, T, XS_obs, TmFa, W):
    _, DH_Fa = get_DH(P)
    RHS = (DH_Fa/R)*(1/TmFa - 1/T) - W/(R*T)*XS_obs**2
    return float(np.clip(1 - (1-XS_obs)*np.exp(RHS/2), 0, 1))

# ============================================================
# 印出 XL 比較表
# ============================================================
print("="*75)
print("XL 比較：Table 4 觀測 vs 模型預測（不同 W）")
print("="*75)
print(f"{'P':<6} {'Mineral':<10} {'XS':<5} {'XL_obs':<8}", end='')
for W in W_list:
    print(f" W={W/1000:.0f}k", end='')
print()
print("-"*75)

for (P, T, mineral, XS_obs, XL_obs) in pierru4:
    P_key = min(pierru26_sol.keys(), key=lambda k: abs(k-P))
    row = f"{P:<6.1f} {mineral:<10} {XS_obs:<5.2f} {XL_obs:<8.3f}"
    for W in W_list:
        TmFa, _ = get_TmFa_XL(P_key, W)
        xl = predict_XL(P, T, XS_obs, TmFa, W)
        diff = xl - XL_obs
        marker = '✓' if abs(diff) < 0.03 else ('~' if abs(diff) < 0.06 else '✗')
        row += f" {xl:.3f}{marker}"
    print(row)

# ============================================================
# 畫圖：4 個壓力面板
# ============================================================
target_P   = [7.95, 9.6, 14.1, 17.7]
P_label    = {7.95: 8.0, 9.6: 9.6, 14.1: 14.1, 17.7: 17.7}
phase_name = {7.95:'Olivine', 9.6:'Olivine',
              14.1:'Wadsleyite', 17.7:'Ringwoodite'}

colors_W = plt.cm.plasma(np.linspace(0.1, 0.85, len(W_list)))
fontsize = 12

fig, axes = plt.subplots(2, 2, figsize=(14, 13))
axes = axes.flatten()

for ax, P_target in zip(axes, target_P):
    TmFo_P = Tm_Fo(P_target)
    Tsol26 = pierru26_sol[P_target]
    T_dun  = T_sol_duncan(P_target)
    DH_Fo, DH_Fa = get_DH(P_target)

    for W, col in zip(W_list, colors_W):
        TmFa, XL_anchor = get_TmFa_XL(P_target, W)
        if np.isnan(TmFa):
            continue

        # 相圖曲線
        T_arr  = np.linspace(TmFa*0.97, TmFo_P*1.02, 400)
        XS_arr = np.full(len(T_arr), np.nan)
        XL_arr = np.full(len(T_arr), np.nan)
        for i, T in enumerate(T_arr):
            if T >= TmFo_P:   XS_arr[i]=1.0; XL_arr[i]=1.0
            elif T <= TmFa:   XS_arr[i]=0.0; XL_arr[i]=0.0
            else:
                XS_arr[i], XL_arr[i] = solve_XS_XL(T, P_target, TmFa, W)

        valid = ~np.isnan(XS_arr)
        lw  = 2.5 if W == 0 else 1.8
        lbl = f'W={W/1000:.1f} kJ  $T_m^{{Fa}}$={TmFa:.0f}K'
        ax.plot(XS_arr[valid], T_arr[valid], '-',  color=col, lw=lw, label=lbl)
        ax.plot(XL_arr[valid], T_arr[valid], '--', color=col, lw=lw)

        # Fa 端元熔點（W=0 only）
        if W == 0:
            ax.plot(0.0, TmFa, 'v', color=col, ms=9, zorder=6)
            ax.text(0.02, TmFa+25, f'{TmFa:.0f}K', fontsize=9, color=col)

        # Table 4 の XL 預測點（★ 關鍵：liquidus 的驗證）
        for (Pp, Tp, mineral, XS_obs, XL_obs) in pierru4:
            if abs(Pp - P_label[P_target]) < 1.0:
                XL_pred = predict_XL(Pp, Tp, XS_obs, TmFa, W)
                # 畫預測的 XL 點（空心，顏色對應 W）
                ax.scatter([XL_pred], [Tp], marker='D', s=80,
                           facecolors='none', edgecolors=col,
                           zorder=10, lw=2)

    # ── Fo 端元熔點 ──
    ax.plot(1.0, TmFo_P, 'b^', ms=10, zorder=6)
    ax.text(0.87, TmFo_P+25, f'{TmFo_P:.0f}K', fontsize=9, color='blue')

    # ── Pierru Table 3 solidus 錨點（■）——solidus 通過這個點 ──
    ax.scatter([X26], [Tsol26], marker='s', s=250,
               color='limegreen', edgecolors='darkgreen',
               zorder=11, lw=2, label='Pierru T$_{sol}$ ■ (Table 3)')
    ax.text(X26+0.02, Tsol26+25, f'{Tsol26}K',
            fontsize=9, color='darkgreen')

    # ── Pierru Table 4 tie lines（實測 XS●, XL◆）──
    for (Pp, Tp, mineral, XS_obs, XL_obs) in pierru4:
        if abs(Pp - P_label[P_target]) < 1.0:
            ax.scatter([XS_obs], [Tp], marker='o', s=180,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=12, lw=2)
            ax.scatter([XL_obs], [Tp], marker='D', s=180,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=12, lw=2)
            ax.plot([XS_obs, XL_obs], [Tp, Tp],
                    color='darkgreen', lw=2, zorder=9)
            ax.text(XL_obs-0.02, Tp+28,
                    f'{mineral}\n({XS_obs}→{XL_obs})',
                    fontsize=8, color='darkgreen', ha='right')

    # ── Duncan ──
    ax.axhline(T_dun, color='darkorange', ls=':', lw=1.8)
    ax.scatter([0.75], [T_dun], marker='o', s=100,
               color='darkorange', zorder=9)
    ax.text(0.01, T_dun+15, f'Duncan {T_dun:.0f}K',
            fontsize=8, color='darkorange')

    win = 2.5
    mask_sup = np.abs(duncan_sup_P - P_target) < win
    mask_sub = np.abs(duncan_sub_P - P_target) < win
    if mask_sup.any():
        ax.scatter([0.75]*mask_sup.sum(), duncan_sup_T[mask_sup],
                   marker='*', s=150, color='darkorange', zorder=8)
    if mask_sub.any():
        ax.scatter([0.75]*mask_sub.sum(), duncan_sub_T[mask_sub],
                   marker='v', s=100, color='darkorange', zorder=8,
                   facecolors='none', lw=2)

    # ── 垂直線 ──
    ax.axvline(0.75, color='darkorange', ls='--', lw=1.5, alpha=0.7)
    ax.axvline(0.89, color='gray',       ls=':',  lw=1.0, alpha=0.5)
    ax.text(0.64, 1060, 'Mars 0.75', fontsize=8, color='darkorange')
    ax.text(0.90, 1060, '0.89', fontsize=8, color='gray')

    # ── 軸・タイトル ──
    ax.set_xlim(0, 1)
    ax.set_ylim(1000, 3100)
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=fontsize)
    ax.set_ylabel('Temperature (K)', fontsize=fontsize)
    ax.set_title(
        f'P = {P_label[P_target]} GPa  |  {phase_name[P_target]}\n'
        f'$T_m^{{Fo}}$={TmFo_P:.0f}K  |  '
        f'$T_{{sol}}^{{26}}$={Tsol26}K  |  '
        f'$\\Delta H_{{Fo}}$={DH_Fo/1000:.0f}, '
        f'$\\Delta H_{{Fa}}$={DH_Fa/1000:.0f} kJ/mol',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    ax.text(0.05, TmFo_P*1.01, 'Liquid',
            fontsize=10, color='darkred', fontweight='bold')
    ax.text(0.82, 1150, 'Solid$_{ss}$',
            fontsize=10, color='darkblue', fontweight='bold')

plt.suptitle(
    'Fo-Fa Phase Diagram — Mars Effective (v12)\n'
    '$T_m^{Fo}$: Ohtani 1981  |  $T_m^{Fa}$: from Pierru 2026 Table 3 ($X_S$=0.75)\n'
    'Solid=solidus, Dashed=liquidus  |  ■=Table 3 anchor (by construction on solidus)\n'
    'Green ●─◆ = Table 4 observed tie lines  |  '
    'Open ◇ = XL predicted by eq2 (same color as W)',
    fontsize=10)
plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/phase_diagram_v12.png',
#             dpi=150, bbox_inches='tight')
# plt.close()
# print("\nSaved: phase_diagram_v12.png")