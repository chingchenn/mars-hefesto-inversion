#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:30:38 2026

@author: chingchen
"""

import numpy as np
import matplotlib.pyplot as plt

fontsize = 13

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

# Duncan 2018
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
# Pierru 2026 preprint solidus 數據點（Table 3）
# ============================================================
pierru26_sol_P = np.array([7.95, 9.6,  14.1, 17.7,  18.0])
pierru26_sol_T = np.array([1890, 1960, 2090, 2200, 2223])

# Pierru 2026 melt Mg# 數據（Table 4）
# 畫在 P-T 圖上標出固液兩相的溫度位置
pierru26_melt_P = np.array([8.0,  8.0,  9.6,  14.1, 17.7])
pierru26_melt_T = np.array([1890, 1890, 1960, 2090, 2200])

# Simon-Glatzel 曲線
P_curve    = np.linspace(0.01, 26.0, 200)
T_sol_curve = np.array([T_sol(P) for P in P_curve])
T_liq_curve = np.array([T_liq(P) for P in P_curve])
T_dun_curve = np.array([T_sol_duncan(P) for P in P_curve])

fig, ax = plt.subplots(figsize=(11, 7))

# Simon-Glatzel 曲線
ax.plot(P_curve, T_sol_curve, 'b-', linewidth=2.5,
        label='Pierru 2022 solidus (Simon-Glatzel)')
ax.plot(P_curve, T_liq_curve, 'r-', linewidth=2.5,
        label='Pierru 2022 liquidus (Simon-Glatzel)')
ax.fill_between(P_curve, T_sol_curve, T_liq_curve,
                alpha=0.10, color='green', label='Two-phase region')

# Duncan solidus 曲線
ax.plot(P_curve, T_dun_curve, color='darkorange', linewidth=2,
        linestyle='-.', label='Duncan 2018 solidus (Mg#=0.75)')

# Pierru 2022 Table S-2 數據點
ax.scatter(solidus_xray_P, solidus_xray_T,
           marker='o', s=80, color='pink', edgecolors='blue',
           zorder=7, label='Pierru 2022 solidus X-ray (Table S-2)')
ax.scatter(solidus_ec_P, solidus_ec_T,
           marker='s', s=80, color='red', edgecolors='darkred',
           zorder=7, label='Pierru 2022 solidus EC (Table S-2)')
ax.scatter(viscous_P, viscous_T,
           marker='D', s=70, color='gray', edgecolors='black',
           zorder=7, label='Pierru 2022 viscous transition (Table S-2)')
ax.scatter(liquidus_P, liquidus_T,
           marker='^', s=100, color='navy', edgecolors='blue',
           zorder=7, label='Pierru 2022 liquidus (Table S-2)')

# Duncan 數據點
ax.scatter(duncan_sup_P, duncan_sup_T,
           marker='*', s=200, color='darkorange', zorder=8,
           label='Duncan 2018 supersolidus (Mg#=0.75)')
ax.scatter(duncan_sub_P, duncan_sub_T,
           marker='v', s=120, color='darkorange', zorder=8,
           facecolors='none', linewidths=2,
           label='Duncan 2018 subsolidus (Mg#=0.75)')

# ── Pierru 2026 solidus 數據點 ──
ax.scatter(pierru26_sol_P, pierru26_sol_T,
           marker='P', s=180, color='limegreen', edgecolors='darkgreen',
           zorder=9, linewidths=2,
           label='Pierru 2026 solidus (Table 3)')

# ── Pierru 2026 melt 實驗點（有固液兩相數據的溫度）──
ax.scatter(pierru26_melt_P, pierru26_melt_T,
           marker='D', s=120, color='limegreen', edgecolors='darkgreen',
           zorder=9, linewidths=2,
           label='Pierru 2026 melt experiments (Table 4)')

# 標注 2026 的點
for P, T in zip(pierru26_sol_P, pierru26_sol_T):
    ax.text(P+0.2, T+10, f'{T}K', fontsize=fontsize-4,
            color='darkgreen', va='bottom')

ax.set_xlabel('Pressure (GPa)', fontsize=fontsize)
ax.set_ylabel('Temperature (K)', fontsize=fontsize)
ax.set_title('P-T Diagram: Pierru 2022 Simon-Glatzel vs Experimental Data\n'
             'Green + = Pierru 2026 preprint solidus & melt experiments',
             fontsize=fontsize)
ax.legend(fontsize=fontsize-3, loc='upper left')
ax.set_xlim(0, 27)
ax.set_ylim(1600, 2500)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=fontsize-1)

plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/verify_PT_v2.png',
            # dpi=150, bbox_inches='tight')
# print("Saved.")
# plt.close()