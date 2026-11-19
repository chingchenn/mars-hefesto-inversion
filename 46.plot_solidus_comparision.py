# 46.plot_solidus_comparision.py
# Mantle solidus / liquidus in P-T: Pierru 2022 Simon-Glatzel fits, Duncan 2018 solidus,
# and the experimental points of Pierru 2022, Duncan 2018 and Pierru 2026.
import numpy as np
import matplotlib.pyplot as plt

FS = 14
plt.rcParams['font.family'] = 'DejaVu Sans'
P = np.linspace(0.01, 26.0, 200)        # GPa

# ── Pierru 2022 Simon-Glatzel: T = T0 * (P/a + 1)^(1/c) ───────────────────────
# solidus : P < 24 GPa  T0, a, c = 1622.7, 237.79, 0.33615 ; else 349.82, 0.10937, 2.9646
# liquidus: P < 18 GPa  T0, a, c = 1931.2, 222.27, 0.48148 ; else 1498.9, 10.760, 2.3668
T_sol = np.where(P < 24.0, 1622.7 * (P / 237.79 + 1) ** (1 / 0.33615),
                           349.82 * (P / 0.10937 + 1) ** (1 / 2.9646))
T_liq = np.where(P < 18.0, 1931.2 * (P / 222.27 + 1) ** (1 / 0.48148),
                           1498.9 * (P / 10.760 + 1) ** (1 / 2.3668))

# ── Duncan 2018 solidus (deg C, Mg# = 0.75) ──────────────────────────────────
# P <= 10 : -4.877 P^2 + 120.2 P + 1088
# P <= 23 : -1.323 (P-10)^2 + 38.18 (P-10) + 1802
# P >  23 : 77.75 (P-23) + 2075
T_dun = np.where(P <= 10, -4.877 * P**2 + 120.2 * P + 1088.0,
        np.where(P <= 23, -1.323 * (P - 10)**2 + 38.18 * (P - 10) + 1802.0,
                          77.75 * (P - 23) + 2075.0)) + 273.15

# ── data points (P in GPa, T in K) ───────────────────────────────────────────
# Pierru 2022 Table S-2
p22_sol_xray = ([5.1, 3.8, 10.2, 17.0, 12.8, 17.1, 23.7, 24.2, 24.8, 22.0, 23.0, 23.3, 25.8, 20.4, 19.5],
                [1775, 1720, 1855, 1890, 1905, 1975, 2165, 2150, 2180, 2110, 2135, 2150, 2210, 2070, 2045])
p22_sol_ec   = ([4.5, 9.0, 14.5, 6.0, 9.5, 12.0],
                [1715, 1780, 1855, 1765, 1825, 1870])
p22_viscous  = ([5.6, 10.6, 15.5, 17.0, 11.2, 20.0, 24.2, 23.0, 20.4, 19.5],
                [1855, 1950, 2035, 2065, 1990, 2150, 2250, 2200, 2160, 2130])
p22_liq      = ([5.6, 13.6, 17.0, 22.0, 23.3],
                [2040, 2175, 2255, 2375, 2420])
# Duncan 2018 (deg C -> K)
dun_sup = ([8.1, 9.8, 10.2, 9.9, 16.5, 16.5, 20.0, 23.0, 25.0],
           np.array([1675, 1815, 1850, 1875, 2050, 2110, 2100, 2150, 2300]) + 273.15)
dun_sub = ([9.9, 16.5, 20.0],
           np.array([1775, 2000, 1975]) + 273.15)
# Pierru 2026 preprint: solidus (Table 3), melt experiments (Table 4)
p26_sol  = ([7.95, 9.6, 14.1, 17.7, 18.0], [1890, 1960, 2090, 2200, 2223])
p26_melt = ([8.0, 8.0, 9.6, 14.1, 17.7],   [1890, 1890, 1960, 2090, 2200])

print('Pierru 2026 solidus points:')
for p, t in zip(*p26_sol):
    print(f'  {p:5.2f} GPa  {t} K')

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
ax.plot(P, T_sol, 'b-', lw=2.5, label='Pierru 2022 solidus (Simon-Glatzel)')
ax.plot(P, T_liq, 'r-', lw=2.5, label='Pierru 2022 liquidus (Simon-Glatzel)')
ax.fill_between(P, T_sol, T_liq, color='green', alpha=0.10, label='Two-phase region')
ax.plot(P, T_dun, color='darkorange', lw=2, ls='-.', label='Duncan 2018 solidus (Mg#=0.75)')

ax.scatter(*p22_sol_xray, marker='o', s=80,  color='pink',  edgecolors='blue',    zorder=7, label='Pierru 2022 solidus X-ray')
ax.scatter(*p22_sol_ec,   marker='s', s=80,  color='red',   edgecolors='darkred', zorder=7, label='Pierru 2022 solidus EC')
ax.scatter(*p22_viscous,  marker='D', s=70,  color='gray',  edgecolors='black',   zorder=7, label='Pierru 2022 viscous transition')
ax.scatter(*p22_liq,      marker='^', s=100, color='navy',  edgecolors='blue',    zorder=7, label='Pierru 2022 liquidus')
ax.scatter(*dun_sup, marker='*', s=200, color='darkorange', zorder=8, label='Duncan 2018 supersolidus')
ax.scatter(*dun_sub, marker='v', s=120, facecolors='none', edgecolors='darkorange', lw=2, zorder=8,
           label='Duncan 2018 subsolidus')
ax.scatter(*p26_sol,  marker='P', s=180, color='limegreen', edgecolors='darkgreen', lw=2, zorder=9,
           label='Pierru 2026 solidus (Table 3)')
ax.scatter(*p26_melt, marker='D', s=120, color='limegreen', edgecolors='darkgreen', lw=2, zorder=9,
           label='Pierru 2026 melt experiments (Table 4)')

ax.set_xlabel('Pressure (GPa)', fontsize=FS)
ax.set_ylabel('Temperature (K)', fontsize=FS)
ax.set_xlim(0, 27)
ax.set_ylim(1600, 2500)
ax.grid(alpha=0.3)
ax.tick_params(labelsize=14)
ax.legend(fontsize=FS - 4, loc='lower right')
plt.tight_layout()
plt.show()
