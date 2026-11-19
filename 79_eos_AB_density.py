#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:50:02 2026

@author: chingchen
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mars_config import RUN_BASE
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

# ── config ────────────────────────────────────────────────────────────────────
CORE_DIR  = HERE        # core_fes.py 的位置
CACHE_DIR = f'{RUN_BASE}/mcmcfigures'   # npz 快取的位置

sys.path.insert(0, CORE_DIR)
import core_fes as cf
cf.CACHE_DIR = CACHE_DIR      

WS_PRIOR   = (0.01, 0.30)     # MCMC 的 w_S 先驗
WS_GEOCHEM = 0.17             # Steenstra & van Westrenen 2018
P_LIST     = [20.0, 25.0, 30.0, 40.0]   # GPa,火星核 ~19-40
T_LIST     = [2000.0, 3000.0]           # K,solid 模態 / molten 模態
NW         = 61

ws = np.linspace(*WS_PRIOR, NW)

# ── 取密度 ────────────────────────────────────────────────────────────────────
# core_properties 內部已處理:A 的 3D mix 表、B 的端元相加 + Margules 過剩體積、
# V -> rho 的莫耳質量換算。這裡不重複實作,避免與 MCMC 用的版本漂移。
def rho_curve(P0, T0):
    P, T = np.array([P0]), np.array([T0])
    return np.array([cf.core_properties(P, T, w)['rho'][0] for w in ws])

RHO = {}
for rt in ('A', 'B'):
    cf.configure(route=rt, verbose=True)
    for T0 in T_LIST:
        for P0 in P_LIST:
            RHO[(rt, T0, P0)] = rho_curve(P0, T0)

# ── 數值表:w_S = 0.17 的 A - B ───────────────────────────────────────────────
print(f'\nrho at w_S = {WS_GEOCHEM:.2f}  (g/cm3)')
print(f'  {"T (K)":>7}{"P (GPa)":>9}{"A":>9}{"B":>9}{"A-B":>9}{"A-B %":>8}')
for T0 in T_LIST:
    for P0 in P_LIST:
        a = np.interp(WS_GEOCHEM, ws, RHO[('A', T0, P0)])
        b = np.interp(WS_GEOCHEM, ws, RHO[('B', T0, P0)])
        print(f'  {T0:7.0f}{P0:9.0f}{a:9.3f}{b:9.3f}{a-b:+9.3f}{100*(a-b)/b:+8.2f}')

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(T_LIST), figsize=(6.2*len(T_LIST), 5.4),
                         sharey=True)
axes = np.atleast_1d(axes)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(P_LIST)))

for ax, T0 in zip(axes, T_LIST):
    for c, P0 in zip(colors, P_LIST):
        ax.plot(ws, RHO[('A', T0, P0)], color=c, lw=2,
                label=f'{P0:.0f} GPa  A')
        ax.plot(ws, RHO[('B', T0, P0)], color=c, lw=2, ls='--',
                label=f'{P0:.0f} GPa  B')
    ax.axvline(WS_GEOCHEM, color='k', ls=':', lw=1.5)
    ax.set_xlabel('w$_S$ (wt fraction S)', fontsize=14)
    ax.set_title(f'T = {T0:.0f} K', fontsize=14)
    ax.tick_params(labelsize=14)

axes[0].set_ylabel(r'$\rho_{liq}$ (g/cm$^3$)', fontsize=14)
axes[-1].legend(fontsize=9, loc='lower right', ncol=2,
                title='solid = A (Liu Gamma)   dashed = B (Liu Fe + Sakai FeS)')
fig.tight_layout()
fig.savefig('79_eos_AB_density.png', dpi=150)
print('\nsaved 79_eos_AB_density.png')
plt.show()