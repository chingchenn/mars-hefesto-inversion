#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 23:41:14 2026

@author: chingchen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── INPUT ─────────────────────────────────────
# HEFESTO_1500 = '/Users/chingchen/Desktop/HeFESTo/ad_samuel_noBML/fort.66'
# FORT56        = '/Users/chingchen/Desktop/HeFESTo/ad_samuel_noBML/fort.56'

HEFESTO_1500 = '/Users/chingchen/Desktop/HeFESTo/01_YM_bulk_mantle_noBML_T/fort.66'
FORT56        = '/Users/chingchen/Desktop/HeFESTo/01_YM_bulk_mantle_noBML_T/fort.56'


g_mars = 3.72  # m/s²

# ── READ fort.66 (phase) ─────────────────────
df = pd.read_csv(HEFESTO_1500, delim_whitespace=True)

P = df["Pi"].values  # GPa

# phase (%)
ol   = df["ol"].values * 100
wa   = df["wa"].values * 100
ri   = df["ri"].values * 100
opx  = df["opx"].values * 100
cpx  = df["cpx"].values * 100
hpcpx = df["c2c"].values * 100
gt   = df["gt"].values * 100


il   = df["il"].values * 100 if "il" in df else np.zeros_like(P)
cpv  = df["cpv"].values * 100 if "cpv" in df else np.zeros_like(P)

# ── cumulative boundaries ─────────────────────
b1 = ol
b2 = b1 + wa
b3 = b2 + ri
b4 = b3 + cpx
b5 = b4 + hpcpx
b6 = b5 + opx
b7 = b6 + gt
b8 = b7 + il
b9 = b8 + cpv

# ── READ fort.56 (for depth) ─────────────────
with open(FORT56) as f:
    f.readline()
    cols = f.readline().split()

df56 = pd.read_csv(FORT56, sep=r'\s+', skiprows=2, names=cols)

hef_P   = df56['P(GPa)'].values
hef_rho = df56['rho(g/cm^3)'].values

# pressure → depth（你原本的方法）
dP      = np.diff(hef_P) * 1e9
rho_mid = (hef_rho[:-1] + hef_rho[1:]) / 2
dz      = dP / (rho_mid * 1000 * g_mars) / 1000  # km

depth = np.zeros(len(hef_P))
depth[1:] = np.cumsum(dz)

# interpolation：P → depth
depth_interp = np.interp(P, hef_P, depth)

# ── PLOT ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

# fill（白色只是保持風格）
for lower, upper in zip(
    [0, b1, b2, b3, b4, b5, b6, b7, b8],
    [b1, b2, b3, b4, b5, b6, b7, b8, b9]
):
    ax.fill_between(P, lower, upper, color="white")

# boundary lines
for b in [b1, b2, b3, b4, b5, b6, b7, b8, b9]:
    ax.plot(P, b, color="darkblue", lw=0.8)

# ── axes ────────────────────────────────────

ax.set_ylim(0, 100)
ax.set_ylabel("Vol. %")

# 上：pressure


# 下：depth
ax2 = ax.twiny()
ax.set_xlim(P.min(), P.max())
ax2.set_xlim(P.min(), P.max())
ax2.set_xlabel("Pressure (GPa)")

ax.set_xlabel("Depth (km)")

from scipy.interpolate import interp1d

# depth → pressure

# 固定 pressure ticks（上面）
p_ticks = np.array([4, 8, 12, 16, 20])


# 上軸：pressure
ax2.set_xticks(p_ticks)
ax2.set_xticklabels([f"{p:.0f}" for p in p_ticks])




d2p = interp1d(depth, hef_P, fill_value="extrapolate")
depth_ticks = np.array([450, 600, 900, 1200, 1500])
pressure_ticks = d2p(depth_ticks)
ax.set_xticks(pressure_ticks)
ax.set_xticklabels([str(d) for d in depth_ticks])

# ── labels（稍微補兩個） ────────────────────
# ax.text(2.5, 20, "Ol", fontsize=10)
# ax.text(10.5, 20, "Wads", fontsize=10)
# ax.text(15.0, 20, "Rw", fontsize=10)

# ax.text(2.0, 50, "Cpx", fontsize=9)
# ax.text(10.0, 55, "HpCpx", fontsize=9)
# ax.text(2.5, 75, "Opx", fontsize=9)
# ax.text(10.0, 90, "Gar", fontsize=9)

# 新 phase（小一點）
# if np.max(il) > 0.5:
#     ax.text(12, 65, "Il", fontsize=8)
# if np.max(cpv) > 0.5:
#     ax.text(16, 85, "CaPv", fontsize=8)

plt.tight_layout()
plt.show()