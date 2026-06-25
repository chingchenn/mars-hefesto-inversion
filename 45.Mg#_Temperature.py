#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:32:52 2026

@author: chingchen
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

R       = 8.314
DH_Fo   = 142000.0
DH_Fa   = 89000.0
Tm_Fo_0 = 2163.0
Tm_Fa_0 = 1478.0
X_bulk  = 0.89
fontsize = 14

def T_sol(P):
    if P < 24.0:
        T0, a, c = 1622.7, 237.79, 0.33615
    else:
        T0, a, c = 349.82, 0.10937, 2.9646
    return T0 * (P / a + 1.0) ** (1.0 / c)

def T_liq(P):
    if P < 18.0:
        T0, a, c = 1931.2, 222.27, 0.48148
    else:
        T0, a, c = 1498.9, 10.760, 2.3668
    return T0 * (P / a + 1.0) ** (1.0 / c)

solidus_xray_P = np.array([5.1, 3.8, 10.2, 17.0, 12.8, 17.1,
                            23.7, 24.2, 24.8, 22.0, 23.0, 23.3,
                            25.8, 20.4, 19.5])
solidus_xray_T = np.array([1775, 1720, 1855, 1890, 1905, 1975,
                            2165, 2150, 2180, 2110, 2135, 2150,
                            2210, 2070, 2045])
solidus_ec_P = np.array([4.5, 9.0, 14.5, 6.0, 9.5, 12.0])
solidus_ec_T = np.array([1715, 1780, 1855, 1765, 1825, 1870])
viscous_P = np.array([5.6, 10.6, 15.5, 17.0, 11.2, 20.0,
                      24.2, 23.0, 20.4, 19.5])
viscous_T = np.array([1855, 1950, 2035, 2065, 1990, 2150,
                      2250, 2200, 2160, 2130])
liquidus_P = np.array([5.6, 13.6, 17.0, 22.0, 23.3])
liquidus_T = np.array([2040, 2175, 2255, 2375, 2420])

def T_sol_duncan(P):
    P = float(P)
    if P <= 10.0:
        T_C = -4.877*P**2 + 120.2*P + 1088.0
    elif P <= 23.0:
        T_C = -1.323*(P-10.0)**2 + 38.18*(P-10.0) + 1802.0
    else:
        T_C = 77.75*(P-23.0) + 2075.0
    return T_C + 273.15

def T_liq_ruedas(P):
    """Ruedas & Breuer 2017 liquidus, T in K, P in GPa"""
    return 2160.6 + 64.7109*P - 3.97463*P**2 + 0.0957894*P**3

# ============================================================
# Pierru preprint Table 4: solid and melt Mg# at each P, T
# ============================================================
pierru_pre = [
    # (P_GPa, T_K, solid_mineral, solid_Mg#, melt_Mg#)
    (8.0,  1890, 'Ol (cold)',  0.80, 0.53),
    (8.0,  1890, 'Ol (warm)',  0.85, 0.58),
    (9.6,  1960, 'Ol',         0.91, 0.67),
    (14.1, 2090, 'Wad',        0.80, 0.51),
    (17.7, 2200, 'Rw',         0.74, 0.50),
]

duncan_sup_P = np.array([8.1, 9.8, 10.2, 9.9, 16.5, 16.5, 20.0, 23.0, 25.0])
duncan_sup_T = np.array([1675, 1815, 1850, 1875, 2050, 2110, 2100, 2150, 2300]) + 273.15
duncan_sub_P = np.array([9.9, 16.5, 20.0])
duncan_sub_T = np.array([1775, 2000, 1975]) + 273.15

def four_eqs(unknowns, Tsol, Tliq):
    Tm_Fo, Tm_Fa, XL_sol, XS_liq = unknowns
    XL_sol = np.clip(XL_sol, 1e-9, 1-1e-9)
    XS_liq = np.clip(XS_liq, 1e-9, 1-1e-9)
    eq1 = 2*np.log(XL_sol/X_bulk)        - (DH_Fo/R)*(1/Tm_Fo - 1/Tsol)
    eq2 = 2*np.log((1-XL_sol)/(1-X_bulk))- (DH_Fa/R)*(1/Tm_Fa - 1/Tsol)
    eq3 = 2*np.log(X_bulk/XS_liq)        - (DH_Fo/R)*(1/Tm_Fo - 1/Tliq)
    eq4 = 2*np.log((1-X_bulk)/(1-XS_liq))- (DH_Fa/R)*(1/Tm_Fa - 1/Tliq)
    return [eq1, eq2, eq3, eq4]

P_scan    = np.linspace(0.01, 25.0, 50)
guess     = [Tm_Fo_0, Tm_Fa_0, 0.95, 0.75]
Tm_Fo_arr = np.zeros(len(P_scan))
Tm_Fa_arr = np.zeros(len(P_scan))

for i, P in enumerate(P_scan):
    sol, info, ier, _ = fsolve(four_eqs, guess,
                                args=(T_sol(P), T_liq(P)), full_output=True)
    resid = np.max(np.abs(info['fvec']))
    if ier == 1 and resid < 1e-6:
        Tm_Fo_arr[i], Tm_Fa_arr[i] = sol[0], sol[1]
        guess = sol
    else:
        Tm_Fo_arr[i] = Tm_Fo_arr[i-1] if i > 0 else Tm_Fo_0
        Tm_Fa_arr[i] = Tm_Fa_arr[i-1] if i > 0 else Tm_Fa_0

def solve_XS_XL(T, Tm_Fo, Tm_Fa):
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        e1 = 2*np.log(XL/XS)        - (DH_Fo/R)*(1/Tm_Fo - 1/T)
        e2 = 2*np.log((1-XL)/(1-XS))- (DH_Fa/R)*(1/Tm_Fa - 1/T)
        return [e1, e2]
    sol, info, ier, _ = fsolve(eqs, [0.92, 0.85], full_output=True)
    XS, XL = sol
    if ier==1 and np.max(np.abs(info['fvec']))<1e-6 and 0<XS<1 and 0<XL<1:
        return np.clip(XS,0,1), np.clip(XL,0,1)
    return np.nan, np.nan

# 從相圖反推：給定 Mg#，找對應 solidus 溫度（XS = target_Mg#）
def predict_solidus_at_MgNum(target_MgNum, Tm_Fo, Tm_Fa, n=500):
    T_arr = np.linspace(Tm_Fa*1.001, Tm_Fo*0.999, n)
    XS_list, T_list = [], []
    for T in T_arr:
        XS, XL = solve_XS_XL(T, Tm_Fo, Tm_Fa)
        if not np.isnan(XS):
            XS_list.append(XS)
            T_list.append(T)
    if len(XS_list) > 2:
        return float(np.interp(target_MgNum, XS_list[::-1], T_list[::-1]))
    return np.nan

target_P = [0.0001, 9.6, 14.1, 17.7]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

print(f"\n驗算：phase diagram 預測 Mg#=0.75 的 solidus vs Duncan 實驗值")
print(f"{'P(GPa)':<8} {'Duncan T_sol(K)':<18} {'PhaseDiag pred(K)':<20} {'ΔT(pred-dunc)'}")
print("-"*65)

for ax, P_target in zip(axes, target_P):
    idx     = np.argmin(np.abs(P_scan - P_target))
    Tm_Fo_P = Tm_Fo_arr[idx]
    Tm_Fa_P = Tm_Fa_arr[idx]
    Tsol_P  = T_sol(P_target)
    Tliq_P  = T_liq(P_target)
    T_dun   = T_sol_duncan(P_target)
    T_rue   = T_liq_ruedas(P_target)

    # 驗算
    T_pred  = predict_solidus_at_MgNum(0.75, Tm_Fo_P, Tm_Fa_P)
    dT_verify = T_pred - T_dun
    print(f"{P_target:<8.3g} {T_dun:<18.1f} {T_pred:<20.1f} {dT_verify:+.1f} K")

    # 相圖曲線
    T_arr  = np.linspace(Tm_Fa_P*0.98, Tm_Fo_P*1.02, 300)
    XS_arr = np.full(len(T_arr), np.nan)
    XL_arr = np.full(len(T_arr), np.nan)
    for i, T in enumerate(T_arr):
        if T >= Tm_Fo_P:
            XS_arr[i]=1.0; XL_arr[i]=1.0
        elif T <= Tm_Fa_P:
            XS_arr[i]=0.0; XL_arr[i]=0.0
        else:
            XS_arr[i], XL_arr[i] = solve_XS_XL(T, Tm_Fo_P, Tm_Fa_P)

    valid = ~np.isnan(XS_arr)
    T_v, XS_v, XL_v = T_arr[valid], XS_arr[valid], XL_arr[valid]

    ax.plot(XS_v, T_v, 'b-', linewidth=2.5, label='Solidus')
    ax.plot(XL_v, T_v, 'r-', linewidth=2.5, label='Liquidus')
    ax.fill_betweenx(T_v, XS_v, XL_v, alpha=0.12, color='green')

    # Pierru 錨點：藍點（T_sol）紅點（T_liq）+ 數字
    ax.scatter([X_bulk], [Tsol_P], marker='o', s=120,
               color='blue', zorder=9)
    ax.text(X_bulk+0.02, Tsol_P+8,
            f'T$_{{sol}}$={Tsol_P:.0f} K',
            fontsize=fontsize-3, color='blue', va='bottom')

    ax.scatter([X_bulk], [Tliq_P], marker='o', s=120,
               color='red', zorder=9)
    ax.text(X_bulk+0.02, Tliq_P+8,
            f'T$_{{liq}}$={Tliq_P:.0f} K',
            fontsize=fontsize-3, color='red', va='bottom')

    # Duncan solidus：點點線 + 橙點 + 數字
    ax.axhline(T_dun, color='darkorange', linestyle=':', linewidth=2.0)
    ax.scatter([0.75], [T_dun], marker='o', s=120,
               color='darkorange', zorder=9)
    ax.text(0.01, T_dun+8,
            f'Duncan T$_{{sol}}$={T_dun:.0f} K',
            fontsize=fontsize-3, color='darkorange', va='bottom')

    # Ruedas 2017 liquidus：點點線 + 橙色三角 + 數字
    ax.axhline(T_rue, color='darkorange', linestyle=':', linewidth=2.0)
    ax.scatter([0.75], [T_rue], marker='^', s=120,
               color='darkorange', zorder=9)
    ax.text(0.01, T_rue+8,
            f'Ruedas T$_{{liq}}$={T_rue:.0f} K',
            fontsize=fontsize-3, color='darkorange', va='bottom')

    # # 箭頭 + ΔT（不畫紫色點）
    # if not np.isnan(T_pred):
    #     ax.annotate('', xy=(0.50, T_pred), xytext=(0.50, T_dun),
    #                 arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    #     ax.text(0.51, (T_pred+T_dun)/2,
    #             f'ΔT={dT_verify:+.0f} K',
    #             fontsize=fontsize-3, color='purple', va='center')

    # Pierru Table S-2 數據點
    win = 2.5
    mask_s  = np.abs(solidus_xray_P - P_target) < win
    mask_ec = np.abs(solidus_ec_P   - P_target) < win
    mask_v  = np.abs(viscous_P      - P_target) < win
    mask_l  = np.abs(liquidus_P     - P_target) < win

    if mask_s.any():
        ax.scatter([X_bulk]*mask_s.sum(), solidus_xray_T[mask_s],
                   marker='o', s=60, color='pink', edgecolors='blue',
                   zorder=7, label='Pierru solidus X-ray')
    if mask_ec.any():
        ax.scatter([X_bulk]*mask_ec.sum(), solidus_ec_T[mask_ec],
                   marker='s', s=60, color='red', edgecolors='darkred',
                   zorder=7, label='Pierru solidus EC')
    if mask_v.any():
        ax.scatter([X_bulk]*mask_v.sum(), viscous_T[mask_v],
                   marker='D', s=60, color='gray', edgecolors='black',
                   zorder=7, label='Pierru viscous trans.')
    if mask_l.any():
        ax.scatter([X_bulk]*mask_l.sum(), liquidus_T[mask_l],
                   marker='^', s=80, color='navy', edgecolors='blue',
                   zorder=7, label='Pierru liquidus')

    # Duncan 實驗數據點
    mask_sup = np.abs(duncan_sup_P - P_target) < win
    mask_sub = np.abs(duncan_sub_P - P_target) < win
    if mask_sup.any():
        ax.scatter([0.75]*mask_sup.sum(), duncan_sup_T[mask_sup],
                   marker='*', s=200, color='darkorange', zorder=8,
                   label='Duncan supersolidus')
    if mask_sub.any():
        ax.scatter([0.75]*mask_sub.sum(), duncan_sub_T[mask_sub],
                   marker='v', s=120, color='darkorange', zorder=8,
                   facecolors='none', linewidths=2,
                   label='Duncan subsolidus')

    # Pierru preprint Table 4 數據點
    # 只畫在壓力±3 GPa 範圍內的點
    win_pre = 3.0
    for (P_pre, T_pre, mineral, XS_pre, XL_pre) in pierru_pre:
        if abs(P_pre - P_target) < win_pre:
            # 固相點（實心圓，綠色）
            ax.scatter([XS_pre], [T_pre], marker='o', s=150,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=10, linewidths=1.5)
            # 液相點（實心菱形，深綠色）
            ax.scatter([XL_pre], [T_pre], marker='D', s=150,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=10, linewidths=1.5)
            # 連線固液兩點
            ax.plot([XS_pre, XL_pre], [T_pre, T_pre],
                    color='darkgreen', linewidth=1.5,
                    linestyle='-', zorder=9)
            # 標注礦物名和溫度
            ax.text(XL_pre - 0.02, T_pre + 15,
                    f'{mineral}\nT={T_pre:.0f}K',
                    fontsize=fontsize-5, color='darkgreen',
                    ha='right', va='bottom')

    # Mg# 垂直線
    ax.axvline(X_bulk, color='gray',       linestyle=':', alpha=0.6)
    ax.axvline(0.75,   color='darkorange', linestyle=':', alpha=0.5)
    ax.text(X_bulk+0.01, 1030, f'Mg#={X_bulk}',
            fontsize=fontsize-4, color='gray')
    ax.text(0.62, 1030, 'Mg#=0.75',
            fontsize=fontsize-4, color='darkorange')

    # 端元熔點
    ax.plot(1.0, Tm_Fo_P, 'b^', markersize=10, zorder=6)
    ax.plot(0.0, Tm_Fa_P, 'rv', markersize=10, zorder=6)
    ax.text(0.88, Tm_Fo_P+15, f'{Tm_Fo_P:.0f}K',
            fontsize=fontsize-3, color='blue')
    ax.text(0.02, Tm_Fa_P+15, f'{Tm_Fa_P:.0f}K',
            fontsize=fontsize-3, color='red')

    T_mid    = (Tm_Fo_P + Tm_Fa_P) / 2
    dT_range = Tm_Fo_P - Tm_Fa_P
    ax.text(0.10, T_mid+dT_range*0.35, 'Liquid',
            fontsize=11, color='darkred',   fontweight='bold')
    ax.text(0.40, T_mid, 'Ol + L',
            fontsize=11, color='darkgreen', fontweight='bold')
    ax.text(0.78, T_mid-dT_range*0.35, 'Ol$_{ss}$',
            fontsize=11, color='darkblue',  fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(1000, 2700)
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=fontsize)
    ax.set_ylabel('Temperature (K)', fontsize=fontsize)
    ax.set_title(f'P = {P_target} GPa  |  '
                 f'$T_m^{{Fo}}$={Tm_Fo_P:.0f}K, $T_m^{{Fa}}$={Tm_Fa_P:.0f}K',
                 fontsize=fontsize, fontweight='bold')
    ax.legend(fontsize=fontsize-5, loc='lower right')
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'Fo-Fa Phase Diagram\n'
    'Blue/Red dot = Pierru 2022 anchor (Mg#=0.89) | '
    'Orange = Duncan solidus & Ruedas liquidus (Mg#=0.75)\n'
    'Green ●─◆ = Pierru preprint Table 4: solid (●) & melt (◆) Mg# at measured P,T',
    fontsize=fontsize)
plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/phase_diagram_v8.png',
#             dpi=150, bbox_inches='tight')
# print("\nSaved.")
# plt.close()