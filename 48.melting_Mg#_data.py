#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 23:53:54 2026

@author: chingchen
"""

#!/usr/bin/env python3
"""
Fo-Fa Phase Diagram — Mars Effective
======================================
策略：
  只用 P > 13 GPa 的 Pierru 點（Wad, Rw）同時 fit W 和 Tm_Fa
  因為這些點在純 Fo-Fa 框架下自洽（eq1 和 eq2 同時滿足）

  olivine 壓力點（8, 9.6 GPa）只作為參考，不參與 fit

eq1: 2*ln(XL/XS) + W/(RT)*(1-XS)^2 = (DH_Fo/R)*(1/Tm_Fo - 1/T)
eq2: 2*ln((1-XL)/(1-XS)) + W/(RT)*XS^2 = (DH_Fa/R)*(1/Tm_Fa - 1/T)

fit 參數：WH, WV（W = WH + P*WV）
從每個高壓點同時解 W 和 Tm_Fa → fit W(P)
"""

import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt

# ============================================================
# 常數
# ============================================================
R   = 8.314

def get_DH(P):
    if P < 14.0:   return 142000.0, 89000.0
    elif P < 20.0: return 116000.0, 73000.0
    else:          return 105000.0, 74000.0

def Tm_Fo(P):
    return 2163.0 * (P/65.67 + 1.0)**0.7809

# ============================================================
# Pierru 2026 Table 4
# ============================================================
pierru_all = [
    (8.0,  1890, 'Ol (cold)', 0.80, 0.53),
    (8.0,  1890, 'Ol (warm)', 0.85, 0.58),
    (9.6,  1960, 'Ol',        0.91, 0.67),
    (14.1, 2090, 'Wad',       0.80, 0.51),   # ← fit に使う
    (17.7, 2200, 'Rw',        0.74, 0.50),   # ← fit に使う
]

# P > 13 GPa のみ fit に使用
pierru_fit = [(P,T,m,XS,XL) for (P,T,m,XS,XL) in pierru_all if P > 13]

# ============================================================
# Step 1: 從每個高壓點同時解 W 和 Tm_Fa（eq1=0, eq2=0）
# ============================================================
print("="*65)
print("Step 1: 從高壓點同時解 W 和 Tm_Fa")
print("="*65)
print(f"{'P':<6} {'Mineral':<10} {'Tm_Fa(K)':<12} {'W(kJ/mol)':<12} {'eq1 res':<12} {'eq2 res'}")
print("-"*65)

W_solved   = []
P_solved   = []
TmFa_solved = []

for (P, T, mineral, XS, XL) in pierru_fit:
    DH_Fo, DH_Fa = get_DH(P)
    TmFo = Tm_Fo(P)

    def eqs(v):
        W, TmFa = v
        e1 = (2*np.log(XL/XS) + W/(R*T)*(1-XS)**2
              - (DH_Fo/R)*(1/TmFo - 1/T))
        e2 = (2*np.log((1-XL)/(1-XS)) + W/(R*T)*XS**2
              - (DH_Fa/R)*(1/TmFa - 1/T))
        return [e1, e2]

    best = None
    for g in [[0, 1500], [-50000, 1500], [50000, 1500],
              [-100000, 2000], [100000, 2000]]:
        sol, info, ier, _ = fsolve(eqs, g, full_output=True)
        resid = np.max(np.abs(info['fvec']))
        if ier == 1 and resid < 1e-8 and sol[1] > 500:
            if best is None or resid < best[2]:
                best = (sol[0], sol[1], resid)
    if best:
        W_sol, TmFa_sol, resid = best
        e1 = (2*np.log(XL/XS) + W_sol/(R*T)*(1-XS)**2
              - (DH_Fo/R)*(1/TmFo - 1/T))
        e2 = (2*np.log((1-XL)/(1-XS)) + W_sol/(R*T)*XS**2
              - (DH_Fa/R)*(1/TmFa_sol - 1/T))
        print(f"{P:<6.1f} {mineral:<10} {TmFa_sol:<12.0f} "
              f"{W_sol/1000:<12.2f} {e1:.2e}    {e2:.2e}")
        W_solved.append(W_sol)
        P_solved.append(P)
        TmFa_solved.append(TmFa_sol)

# ============================================================
# Step 2: Fit W(P) = WH + P*WV
# ============================================================
P_arr = np.array(P_solved)
W_arr = np.array(W_solved)
TmFa_arr = np.array(TmFa_solved)

coeffs = np.polyfit(P_arr, W_arr, 1)
WV_fit, WH_fit = coeffs

print()
print("="*65)
print("Step 2: W(P) = WH + P*WV fit")
print("="*65)
print(f"  WH = {WH_fit/1000:.2f} kJ/mol")
print(f"  WV = {WV_fit/1000:.3f} kJ/mol/GPa")
for P in [8, 9.6, 14.1, 17.7]:
    W = WH_fit + WV_fit*P
    print(f"  W({P:.1f} GPa) = {W/1000:.2f} kJ/mol")

def W_of_P(P):
    return WH_fit + WV_fit*P

# ============================================================
# Step 3: Tm_Fa(P) — Simon-Glatzel fit（只用高壓點）
# 加上 1 atm 錨點
# ============================================================
def sg(P, T0, a, c):
    return T0 * (P/a + 1.0)**(1.0/c)

# 用高壓的 Tm_Fa_solved 點 + 1 atm 錨點（Ohtani 1979）
P_sg = np.concatenate([[0.0001], P_arr])
T_sg = np.concatenate([[1478.0], TmFa_arr])

try:
    popt, _ = curve_fit(sg, P_sg, T_sg, p0=[1478, 10, 1.5], maxfev=10000)
    def Tm_Fa_fit(P):
        return sg(P, *popt)
    print(f"\nTm_Fa Simon-Glatzel: T0={popt[0]:.1f}, a={popt[1]:.3f}, c={popt[2]:.4f}")
except:
    # 如果 fit 失敗，用線性插值
    def Tm_Fa_fit(P):
        return float(np.interp(P, P_sg, T_sg))
    print("\nTm_Fa: linear interpolation")

# ============================================================
# Phase diagram solver
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
    for x0 in [[0.85,0.65],[0.80,0.55],[0.70,0.45],[0.90,0.70],[0.60,0.40]]:
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        XS, XL = sol
        if (ier==1 and np.max(np.abs(info['fvec']))<1e-6
                and 0<XS<1 and 0<XL<1 and XS>XL):
            return float(np.clip(XS,0,1)), float(np.clip(XL,0,1))
    return np.nan, np.nan

def build_curves(P, TmFa, W):
    TmFo = Tm_Fo(P)
    T_arr  = np.linspace(TmFa*0.97, TmFo*1.02, 400)
    XS_arr = np.full(len(T_arr), np.nan)
    XL_arr = np.full(len(T_arr), np.nan)
    for i, T in enumerate(T_arr):
        if T >= TmFo:   XS_arr[i]=1.0; XL_arr[i]=1.0
        elif T <= TmFa: XS_arr[i]=0.0; XL_arr[i]=0.0
        else:
            XS_arr[i], XL_arr[i] = solve_XS_XL(T, P, TmFa, W)
    valid = ~np.isnan(XS_arr)
    return T_arr[valid], XS_arr[valid], XL_arr[valid]

# ============================================================
# 畫圖
# ============================================================
target_P   = [8.0, 9.6, 14.1, 17.7]
phase_name = {8.0:'Olivine (ref only)',  9.6:'Olivine (ref only)',
              14.1:'Wadsleyite (fit)',   17.7:'Ringwoodite (fit)'}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, P in zip(axes, target_P):
    TmFo   = Tm_Fo(P)
    TmFa   = Tm_Fa_fit(P)
    W      = W_of_P(P)
    DH_Fo, DH_Fa = get_DH(P)
    is_fit = (P > 13)

    # ── fit W(P) の曲線 ──
    T_v, XS_v, XL_v = build_curves(P, TmFa, W)
    col_main = 'steelblue' if is_fit else 'gray'
    lw_main  = 2.5 if is_fit else 1.5
    ax.plot(XS_v, T_v, '-',  color=col_main, lw=lw_main,
            label=f'W(P)={W/1000:.1f} kJ/mol (fit)  $T_m^{{Fa}}$={TmFa:.0f}K')
    ax.plot(XL_v, T_v, '--', color=col_main, lw=lw_main)
    ax.fill_betweenx(T_v, XS_v, XL_v, alpha=0.10,
                     color='steelblue' if is_fit else 'gray')

    # ── W=0 の参考曲線 ──
    # W=0 のとき Tm_Fa は eq2 から解析的に求める
    # 高圧点のみ、W=0 の場合の Tm_Fa を计算
    def TmFa_W0(P, T, XS, XL):
        _, DH_Fa = get_DH(P)
        LHS2 = 2*np.log((1-XL)/(1-XS))
        return 1.0 / (1.0/T + (R/DH_Fa)*LHS2)

    # W=0 の Tm_Fa を各圧力で推定（fit 点のみ）
    if is_fit:
        ref_pts = [(P2,T2,XS2,XL2) for (P2,T2,_,XS2,XL2) in pierru_fit
                   if abs(P2-P) < 0.5]
        if ref_pts:
            TmFa0 = np.mean([TmFa_W0(p,t,xs,xl) for p,t,xs,xl in ref_pts])
            T_v0, XS_v0, XL_v0 = build_curves(P, TmFa0, 0)
            ax.plot(XS_v0, T_v0, '-',  color='tomato', lw=1.5,
                    alpha=0.7, label=f'W=0  $T_m^{{Fa}}$={TmFa0:.0f}K')
            ax.plot(XL_v0, T_v0, '--', color='tomato', lw=1.5, alpha=0.7)

    # ── 端元熔點 ──
    ax.plot(1.0, TmFo, 'b^', ms=10, zorder=6)
    ax.plot(0.0, TmFa, 'v',  color=col_main, ms=10, zorder=6)
    ax.text(0.87, TmFo+30, f'{TmFo:.0f}K', fontsize=9, color='blue')
    ax.text(0.02, TmFa+30, f'{TmFa:.0f}K', fontsize=9, color=col_main)

    # ── Pierru Table 4 tie lines ──
    for (Pp, Tp, mineral, XSp, XLp) in pierru_all:
        if abs(Pp - P) < 1.0:
            mcol = 'limegreen' if P > 13 else 'orange'
            ecol = 'darkgreen' if P > 13 else 'darkorange'
            ax.scatter([XSp], [Tp], marker='o', s=200,
                       color=mcol, edgecolors=ecol, zorder=10, lw=2)
            ax.scatter([XLp], [Tp], marker='D', s=200,
                       color=mcol, edgecolors=ecol, zorder=10, lw=2)
            ax.plot([XSp, XLp], [Tp, Tp],
                    color=ecol, lw=2, zorder=9)
            ax.text(XLp-0.02, Tp+30,
                    f'{mineral}\n({XSp}→{XLp})',
                    fontsize=8, color=ecol, ha='right')

    # ── 垂直線 ──
    ax.axvline(0.75, color='darkorange', ls='--', lw=1.5, alpha=0.7)
    ax.axvline(0.89, color='gray',       ls=':',  lw=1.0, alpha=0.5)
    ax.text(0.64, 1100, 'Mars 0.75', fontsize=8, color='darkorange')

    # ── 凡例・軸 ──
    fit_str = '★ fit' if is_fit else '(ref only, not fit)'
    ax.set_xlim(0, 1)
    ax.set_ylim(1000, 3200)
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(
        f'P = {P} GPa  |  {phase_name[P]}  {fit_str}\n'
        f'$T_m^{{Fo}}$={TmFo:.0f}K  |  '
        f'$T_m^{{Fa}}$={TmFa:.0f}K  |  '
        f'W={W/1000:.1f} kJ/mol  |  '
        f'$\\Delta H_{{Fo}}$={DH_Fo/1000:.0f}, '
        f'$\\Delta H_{{Fa}}$={DH_Fa/1000:.0f} kJ/mol',
        fontsize=9, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # 領域ラベル
    ax.text(0.05, TmFo*1.02, 'Liquid',
            fontsize=10, color='darkred', fontweight='bold')
    ax.text(0.82, 1150, 'Solid$_{ss}$',
            fontsize=10, color='darkblue', fontweight='bold')

plt.suptitle(
    'Fo-Fa Phase Diagram — Mars Effective\n'
    f'W(P) = {WH_fit/1000:.1f} + P × {WV_fit/1000:.2f} kJ/mol  '
    f'(fit from Wad+Rw points, P>13 GPa)\n'
    'Green tie lines = fit points ✅  |  Orange tie lines = reference only ⚠️\n'
    'Blue = fit W(P)  |  Red = W=0 reference  |  Solid=solidus, Dashed=liquidus',
    fontsize=11)
plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/fo_fa_W_highP_fit.png',
            # dpi=150, bbox_inches='tight')
# plt.close()
print("\nSaved: fo_fa_W_highP_fit.png")