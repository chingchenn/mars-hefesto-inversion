#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 23:20:37 2026

@author: chingchen
"""

#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

R     = 8.314
DH_Fo = 142000.0  # J/mol, Ohtani 1981
DH_Fa = 89000.0   # J/mol
X26   = 0.75      # Pierru 2026 bulk Mg#
fontsize = 14

# ============================================================
# Tm_Fo(P)：Ohtani & Kumazawa 1981 Simon-Glatzel 擬合
# ============================================================
def Tm_Fo(P):
    T0, a, c = 2163.0, 65.67, 0.7809
    return T0 * (P / a + 1.0) ** (1.0 / c)

# ============================================================
# Pierru 2022 Simon-Glatzel（僅用來畫參考線）
# ============================================================
def T_sol_22(P):
    if P < 24.0:
        T0, a, c = 1622.7, 237.79, 0.33615
    else:
        T0, a, c = 349.82, 0.10937, 2.9646
    return T0 * (P / a + 1.0) ** (1.0 / c)

def T_liq_22(P):
    if P < 18.0:
        T0, a, c = 1931.2, 222.27, 0.48148
    else:
        T0, a, c = 1498.9, 10.760, 2.3668
    return T0 * (P / a + 1.0) ** (1.0 / c)

# ============================================================
# Pierru 2026 Table 3: solidus at Mg#=0.75
# ============================================================
pierru26_sol = {7.95:1890, 9.6:1960, 14.1:2090, 17.7:2200, 18.0:2223}

# Pierru 2026 Table 4: solid/melt Mg# pairs
pierru_pre = [
    (8.0,  1890, 'Ol (cold)', 0.80, 0.53),
    (8.0,  1890, 'Ol (warm)', 0.85, 0.58),
    (9.6,  1960, 'Ol',        0.91, 0.67),
    (14.1, 2090, 'Wad',       0.80, 0.51),
    (17.7, 2200, 'Rw',        0.74, 0.50),
]

# Pierru 2022 Table S-2
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

duncan_sup_P = np.array([8.1, 9.8, 10.2, 9.9, 16.5, 16.5, 20.0, 23.0, 25.0])
duncan_sup_T = np.array([1675, 1815, 1850, 1875, 2050, 2110, 2100, 2150, 2300]) + 273.15
duncan_sub_P = np.array([9.9, 16.5, 20.0])
duncan_sub_T = np.array([1775, 2000, 1975]) + 273.15

# ============================================================
# 聯立方程式：2 個方程式，2 個未知數
# 已知：Tm_Fo(P) [Ohtani], T_sol_26 [Pierru 2026], XS=0.75
# 未知：Tm_Fa(P), XL (melt Mg#)
# ============================================================
def two_eqs(unknowns, Tsol26, TmFo_P):
    Tm_Fa, XL = unknowns
    XL = np.clip(XL, 1e-9, 1-1e-9)
    e1 = 2*np.log(XL/X26)         - (DH_Fo/R)*(1/TmFo_P - 1/Tsol26)
    e2 = 2*np.log((1-XL)/(1-X26)) - (DH_Fa/R)*(1/Tm_Fa  - 1/Tsol26)
    return [e1, e2]

# 在 Pierru 2026 的 5 個壓力點解方程式
P_solved   = sorted(pierru26_sol.keys())
Tm_Fa_dict = {}
XL_dict    = {}

guess = [1478, 0.55]
print(f"{'P(GPa)':<8} {'Tm_Fo(K)':<10} {'T_sol26(K)':<12} {'Tm_Fa(K)':<10} {'XL(pred)':<10} {'resid'}")
print("-"*62)
for P in P_solved:
    T26    = pierru26_sol[P]
    TmFo_P = Tm_Fo(P)
    sol, info, ier, _ = fsolve(two_eqs, guess,
                                args=(T26, TmFo_P), full_output=True)
    resid = np.max(np.abs(info['fvec']))
    Tm_Fa_dict[P] = sol[0]
    XL_dict[P]    = sol[1]
    print(f"{P:<8.2f} {TmFo_P:<10.1f} {T26:<12.0f} {sol[0]:<10.1f} {sol[1]:<10.3f} {resid:.2e}")
    guess = list(sol)

# ============================================================
def solve_XS_XL(T, TmFo_P, Tm_Fa_P):
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        e1 = 2*np.log(XL/XS)        - (DH_Fo/R)*(1/TmFo_P - 1/T)
        e2 = 2*np.log((1-XL)/(1-XS))- (DH_Fa/R)*(1/Tm_Fa_P - 1/T)
        return [e1, e2]
    sol, info, ier, _ = fsolve(eqs, [0.80, 0.55], full_output=True)
    XS, XL = sol
    if ier==1 and np.max(np.abs(info['fvec']))<1e-6 and 0<XS<1 and 0<XL<1:
        return np.clip(XS,0,1), np.clip(XL,0,1)
    return np.nan, np.nan

# ============================================================
# 畫圖：4個 panel（只在有解的壓力點）
# ============================================================
target_P = [7.95, 9.6, 14.1, 17.7]

fig, axes = plt.subplots(2, 2, figsize=(13, 12))
axes = axes.flatten()

for ax, P_target in zip(axes, target_P):
    TmFo_P  = Tm_Fo(P_target)
    Tm_Fa_P = Tm_Fa_dict[P_target]
    T26     = pierru26_sol[P_target]
    T_dun   = T_sol_duncan(P_target)
    Tsol_22 = T_sol_22(P_target)
    Tliq_22 = T_liq_22(P_target)

    # 相圖曲線
    T_arr  = np.linspace(Tm_Fa_P*0.95, TmFo_P*1.02, 400)
    XS_arr = np.full(len(T_arr), np.nan)
    XL_arr = np.full(len(T_arr), np.nan)
    for i, T in enumerate(T_arr):
        if T >= TmFo_P:
            XS_arr[i]=1.0; XL_arr[i]=1.0
        elif T <= Tm_Fa_P:
            XS_arr[i]=0.0; XL_arr[i]=0.0
        else:
            XS_arr[i], XL_arr[i] = solve_XS_XL(T, TmFo_P, Tm_Fa_P)

    valid = ~np.isnan(XS_arr)
    T_v, XS_v, XL_v = T_arr[valid], XS_arr[valid], XL_arr[valid]

    ax.plot(XS_v, T_v, 'b-', linewidth=2.5, label='Solidus')
    ax.plot(XL_v, T_v, 'r-', linewidth=2.5, label='Liquidus')
    ax.fill_betweenx(T_v, XS_v, XL_v, alpha=0.12, color='green')

    # Ohtani 1981 Tm_Fo 端元熔點
    # ax.plot(1.0, TmFo_P, 'b^', markersize=12, zorder=8,
            # label=f'Ohtani $T_m^{{Fo}}$={TmFo_P:.0f}K')
    ax.text(0.87, TmFo_P+15, f'{TmFo_P:.0f}K',
            fontsize=fontsize-3, color='blue')

    # Tm_Fa 端元熔點
    # ax.plot(0.0, Tm_Fa_P, 'rv', markersize=12, zorder=8,
            # label=f'Solved $T_m^{{Fa}}$={Tm_Fa_P:.0f}K')
    ax.text(0.02, Tm_Fa_P+15, f'{Tm_Fa_P:.0f}K',
            fontsize=fontsize-3, color='red')

    # Pierru 2026 Table 3: 只畫對應壓力的 solidus 點（Mg#=0.75）
    for P26, T26_i in pierru26_sol.items():
        if abs(P26 - P_target) < 1.5:
            ax.scatter([X26], [T26_i], marker='s', s=200,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=11, linewidths=2,
                       label=f'Pierru 2026 T$_{{sol}}$={T26_i}K')
            ax.text(X26+0.02, T26_i+8, f'T$_{{sol26}}$={T26_i}K',
                    fontsize=fontsize-3, color='darkgreen', va='bottom')

    # # Pierru 2022 參考點（藍/紅點）
    # ax.scatter([0.89], [Tsol_22], marker='o', s=100,
    #            color='blue', zorder=9, alpha=0.5,
    #            label=f'Pierru 2022 T$_{{sol}}$={Tsol_22:.0f}K')
    # ax.scatter([0.89], [Tliq_22], marker='o', s=100,
    #            color='red', zorder=9, alpha=0.5,
    #            label=f'Pierru 2022 T$_{{liq}}$={Tliq_22:.0f}K')

    # Duncan solidus 點點線
    ax.axhline(T_dun, color='darkorange', linestyle=':', linewidth=2.0)
    ax.scatter([0.75], [T_dun], marker='o', s=120,
               color='darkorange', zorder=9)
    ax.text(0.01, T_dun+8, f'Duncan T$_{{sol}}$={T_dun:.0f}K',
            fontsize=fontsize-3, color='darkorange', va='bottom')

    # Duncan 實驗點
    win = 2.5
    mask_sup = np.abs(duncan_sup_P - P_target) < win
    mask_sub = np.abs(duncan_sub_P - P_target) < win
    if mask_sup.any():
        ax.scatter([0.75]*mask_sup.sum(), duncan_sup_T[mask_sup],
                   marker='*', s=200, color='darkorange', zorder=8,
                   label='Duncan supersolidus')
    if mask_sub.any():
        ax.scatter([0.75]*mask_sub.sum(), duncan_sub_T[mask_sub],
                   marker='v', s=120, color='darkorange', zorder=8,
                   facecolors='none', linewidths=2, label='Duncan subsolidus')

    # Pierru 2026 Table 4 固液兩相 Mg# 點
    for (P_pre, T_pre, mineral, XS_pre, XL_pre) in pierru_pre:
        if abs(P_pre - P_target) < 2.0:
            ax.scatter([XS_pre], [T_pre], marker='o', s=150,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=10, linewidths=1.5)
            ax.scatter([XL_pre], [T_pre], marker='D', s=150,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=10, linewidths=1.5)
            ax.plot([XS_pre, XL_pre], [T_pre, T_pre],
                    color='darkgreen', linewidth=1.5, zorder=9)
            ax.text(XL_pre-0.02, T_pre+15,
                    f'{mineral} T={T_pre:.0f}K',
                    fontsize=fontsize-5, color='darkgreen',
                    ha='right', va='bottom')

    # Mg# 垂直線
    ax.axvline(0.89, color='gray',      linestyle=':', alpha=0.6)
    ax.axvline(0.75, color='darkgreen', linestyle=':', alpha=0.5)
    ax.text(0.90, 1050, 'Mg#=0.89', fontsize=fontsize-4, color='gray')
    ax.text(0.64, 1050, 'Mg#=0.75', fontsize=fontsize-4, color='darkgreen')

    T_mid    = (TmFo_P + Tm_Fa_P) / 2
    dT_range = TmFo_P - Tm_Fa_P
    ax.text(0.05, T_mid+dT_range*0.30, 'Liquid',
            fontsize=11, color='darkred', fontweight='bold')
    ax.text(0.35, T_mid, 'Ol + L',
            fontsize=11, color='darkgreen', fontweight='bold')
    ax.text(0.78, T_mid-dT_range*0.30, 'Ol$_{ss}$',
            fontsize=11, color='darkblue', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(1000, 3000)
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=fontsize)
    ax.set_ylabel('Temperature (K)', fontsize=fontsize)
    ax.set_title(f'P = {P_target} GPa  |  '
                 f'$T_m^{{Fo}}$={TmFo_P:.0f}K [Ohtani], '
                 f'$T_m^{{Fa}}$={Tm_Fa_P:.0f}K [solved]',
                 fontsize=fontsize, fontweight='bold')
    ax.legend(fontsize=fontsize-5, loc='lower right')
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'Fo-Fa Phase Diagram (v11)\n'
    '$T_m^{Fo}(P)$: Ohtani & Kumazawa 1981 | '
    '$T_m^{Fa}(P)$: solved from Pierru 2026 solidus (Mg#=0.75)\n'
    'Green ■ = Pierru 2026 solidus anchor | Green ●─◆ = Table 4 solid/melt Mg#',
    fontsize=fontsize)
plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/phase_diagram_v11.png',
#             dpi=150, bbox_inches='tight')
# print("\nSaved.")
# plt.close()
# plt.close()