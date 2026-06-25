#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Fo-Fa Phase Diagram — Mars Effective
======================================
你原本的邏輯 + 加入 W 項

已知：
  Tm_Fo(P) ← Ohtani & Kumazawa 1981
  T_sol(P) ← Pierru 2026 Table 3 (solidus at Mg#=0.75)
  XS = 0.75 (bulk = solidus 定義)
  DH_Fo, DH_Fa ← Stixrude 2024 + Hess's Law
  W(P) ← 你要測試的值

未知：
  Tm_Fa(P), XL(P) ← 從兩個方程式同時解出

eq1: 2*ln(XL/XS) + W/(RT)*(1-XS)^2 = (DH_Fo/R)*(1/Tm_Fo - 1/T)
eq2: 2*ln((1-XL)/(1-XS)) + W/(RT)*XS^2 = (DH_Fa/R)*(1/Tm_Fa - 1/T)

XS = 0.75, T = T_sol → 解出 Tm_Fa 和 XL

★ 改動 W_list 看不同 W 的效果 ★
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ============================================================
# ★ 改動這裡 ★
# ============================================================
W_list = [0, 2500, 8000, 12000]   # J/mol

# ============================================================
# 常數
# ============================================================
R     = 8.314
X26   = 0.75   # Pierru 2026 bulk Mg#

# ============================================================
# DH(P): Stixrude 2024 + Hess's Law
# ============================================================
def get_DH(P):
    if P < 14.0:
        return 142000.0, 89000.0    # olivine
    elif P < 20.0:
        return 116000.0, 73000.0    # wadsleyite
    else:
        return 105000.0, 74000.0    # ringwoodite

# ============================================================
# Tm_Fo(P): Ohtani & Kumazawa 1981
# ============================================================
def Tm_Fo(P):
    T0, a, c = 2163.0, 65.67, 0.7809
    return T0 * (P / a + 1.0) ** (1.0 / c)

# ============================================================
# Pierru 2026 Table 3: solidus at Mg#=0.75
# ============================================================
pierru26_sol = {7.95: 1890, 9.6: 1960, 14.1: 2090, 17.7: 2200}

# Pierru 2026 Table 4: tie lines (XS, XL at T, P)
pierru_pre = [
    (8.0,  1890, 'Ol (cold)', 0.80, 0.53),
    (8.0,  1890, 'Ol (warm)', 0.85, 0.58),
    (9.6,  1960, 'Ol',        0.91, 0.67),
    (14.1, 2090, 'Wad',       0.80, 0.51),
    (17.7, 2200, 'Rw',        0.74, 0.50),
]

# ============================================================
# 核心：從 Pierru Table 3 反推 Tm_Fa 和 XL
#
# 在 solidus 上：XS = X26 = 0.75（bulk = solid at solidus）
# 兩個未知數：Tm_Fa, XL
# 兩個方程式：eq1, eq2
# ============================================================
def two_eqs(unknowns, Tsol, TmFo_P, W, DH_Fo, DH_Fa):
    Tm_Fa, XL = unknowns
    XS = X26
    XL = np.clip(XL, 1e-9, 1-1e-9)
    e1 = (2*np.log(XL/XS) + W/(R*Tsol)*(1-XS)**2
          - (DH_Fo/R)*(1/TmFo_P - 1/Tsol))
    e2 = (2*np.log((1-XL)/(1-XS)) + W/(R*Tsol)*XS**2
          - (DH_Fa/R)*(1/Tm_Fa - 1/Tsol))
    return [e1, e2]

# ============================================================
# 相圖 solver: 給定 T, Tm_Fo, Tm_Fa, W → XS, XL
# ============================================================
def solve_XS_XL(T, TmFo_P, Tm_Fa_P, W, DH_Fo, DH_Fa):
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        e1 = (2*np.log(XL/XS) + W/(R*T)*(1-XS)**2
              - (DH_Fo/R)*(1/TmFo_P - 1/T))
        e2 = (2*np.log((1-XL)/(1-XS)) + W/(R*T)*XS**2
              - (DH_Fa/R)*(1/Tm_Fa_P - 1/T))
        return [e1, e2]
    for x0 in [[0.85,0.65],[0.90,0.70],[0.80,0.60],[0.70,0.50],[0.95,0.85]]:
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        XS, XL = sol
        if (ier==1 and np.max(np.abs(info['fvec']))<1e-6
                and 0<XS<1 and 0<XL<1 and XS>XL):
            return float(np.clip(XS,0,1)), float(np.clip(XL,0,1))
    return np.nan, np.nan

# ============================================================
# 印出結果
# ============================================================
print("="*70)
print("Tm_Fa 和 XL 反推結果（Pierru 2026 Table 3 + Stixrude W）")
print("="*70)

for W in W_list:
    print(f"\nW = {W/1000:.1f} kJ/mol:")
    print(f"  {'P(GPa)':<8} {'T_sol(K)':<10} {'Tm_Fo(K)':<10} "
          f"{'Tm_Fa(K)':<10} {'XL(pred)':<10} {'residual'}")
    print("  " + "-"*60)
    guess = [1400, 0.40]
    for P, Tsol in sorted(pierru26_sol.items()):
        TmFo_P = Tm_Fo(P)
        DH_Fo, DH_Fa = get_DH(P)
        sol, info, ier, _ = fsolve(
            two_eqs, guess,
            args=(Tsol, TmFo_P, W, DH_Fo, DH_Fa),
            full_output=True)
        resid = np.max(np.abs(info['fvec']))
        print(f"  {P:<8.2f} {Tsol:<10.0f} {TmFo_P:<10.0f} "
              f"{sol[0]:<10.0f} {sol[1]:<10.3f} {resid:.2e}")
        guess = list(sol)

# ============================================================
# 畫圖
# ============================================================
target_P   = [7.95, 9.6, 14.1, 17.7]
phase_name = {7.95:'Olivine', 9.6:'Olivine',
              14.1:'Wadsleyite', 17.7:'Ringwoodite'}

colors_W = plt.cm.plasma(np.linspace(0.1, 0.85, len(W_list)))

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for ax, P_target in zip(axes, target_P):
    TmFo_P   = Tm_Fo(P_target)
    Tsol26   = pierru26_sol[P_target]
    DH_Fo, DH_Fa = get_DH(P_target)

    for W, col in zip(W_list, colors_W):
        # 反推 Tm_Fa 和 XL
        sol, _, ier, _ = fsolve(
            two_eqs, [1400, 0.40],
            args=(Tsol26, TmFo_P, W, DH_Fo, DH_Fa),
            full_output=True)
        if ier != 1:
            continue
        Tm_Fa_P, XL_pred = sol

        # 畫相圖曲線
        T_arr  = np.linspace(Tm_Fa_P*0.97, TmFo_P*1.02, 400)
        XS_arr = np.full(len(T_arr), np.nan)
        XL_arr = np.full(len(T_arr), np.nan)
        for i, T in enumerate(T_arr):
            if T >= TmFo_P:
                XS_arr[i]=1.0; XL_arr[i]=1.0
            elif T <= Tm_Fa_P:
                XS_arr[i]=0.0; XL_arr[i]=0.0
            else:
                XS_arr[i], XL_arr[i] = solve_XS_XL(
                    T, TmFo_P, Tm_Fa_P, W, DH_Fo, DH_Fa)

        valid = ~np.isnan(XS_arr)
        lw  = 2.5 if W == 0 else 1.8
        lbl = f'W={W/1000:.1f} kJ/mol  ($T_m^{{Fa}}$={Tm_Fa_P:.0f}K)'
        ax.plot(XS_arr[valid], T_arr[valid], '-',  color=col, lw=lw, label=lbl)
        ax.plot(XL_arr[valid], T_arr[valid], '--', color=col, lw=lw)

        # Fa 端元熔點（只標 W=0）
        if W == 0:
            ax.plot(0.0, Tm_Fa_P, 'v', color=col, ms=10, zorder=6)
            ax.text(0.02, Tm_Fa_P+30, f'{Tm_Fa_P:.0f}K',
                    fontsize=9, color=col)

        # Pierru T_sol 錨點（solidus 通過這個點）
        ax.scatter([X26], [Tsol26], marker='s', s=250,
                   color='limegreen', edgecolors='darkgreen',
                   zorder=11, lw=2)

    # Fo 端元熔點
    ax.plot(1.0, TmFo_P, 'b^', ms=10, zorder=6)
    ax.text(0.87, TmFo_P+30, f'{TmFo_P:.0f}K', fontsize=9, color='blue')

    # Pierru Table 4 tie lines
    for (Pp, Tp, mineral, XSp, XLp) in pierru_pre:
        if abs(Pp - P_target) < 1.5:
            ax.scatter([XSp], [Tp], marker='o', s=180,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=10, lw=2)
            ax.scatter([XLp], [Tp], marker='D', s=180,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=10, lw=2)
            ax.plot([XSp, XLp], [Tp, Tp],
                    color='darkgreen', lw=2, zorder=9)
            ax.text(XLp-0.02, Tp+30,
                    f'{mineral} ({XSp}→{XLp})',
                    fontsize=8, color='darkgreen', ha='right')

    # Mg# 垂直線
    ax.axvline(0.75, color='darkorange', ls='--', lw=1.5, alpha=0.8)
    ax.axvline(0.89, color='gray',       ls=':',  lw=1.0, alpha=0.5)
    ax.text(0.64, TmFo_P*0.99, 'Mars\n0.75',
            fontsize=8, color='darkorange', ha='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(1000, 3100)
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(
        f'P = {P_target} GPa  |  {phase_name[P_target]}\n'
        f'$T_m^{{Fo}}$={TmFo_P:.0f}K  |  '
        f'$T_{{sol}}^{{26}}$={Tsol26}K  |  '
        f'$\\Delta H_{{Fo}}$={DH_Fo/1000:.0f}, '
        f'$\\Delta H_{{Fa}}$={DH_Fa/1000:.0f} kJ/mol',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    ax.text(0.05, TmFo_P*1.01, 'Liquid',
            fontsize=10, color='darkred', fontweight='bold')
    ax.text(0.82, 1200, 'Solid$_{ss}$',
            fontsize=10, color='darkblue', fontweight='bold')

plt.suptitle(
    'Fo-Fa Phase Diagram — Mars Effective (Pierru 2026)\n'
    '$T_m^{Fo}$: Ohtani 1981  |  '
    '$T_m^{Fa}$ + $X_L$: solved from Pierru 2026 Table 3 ($X_S$=0.75)\n'
    'Solid=solidus, Dashed=liquidus  |  '
    '■=Pierru T$_{sol}$ anchor  |  ●─◆=Pierru Table 4 tie lines',
    fontsize=11)
plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/fo_fa_mars_W.png',
#             dpi=150, bbox_inches='tight')
# plt.close()
print("\nSaved: fo_fa_mars_W.png")