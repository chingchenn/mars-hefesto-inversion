#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:03:08 2026

@author: chingchen
"""

import os
import types
import numpy as np
from scipy.optimize import brentq
import openpyxl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── config ────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
FORWARD_FILE = os.path.join(HERE, '26fall_main_mcmc.py')
XLSX_FILE    = os.path.join(HERE, 'Pierru 2026 data 2.xlsx')
SHEET        = 'Source Data Extended Data Fig.8'
P_MIN, P_MAX = 15.0, 21.5                # GPa, plotted range (BML ~19-22)
P_TAB        = [19.0, 21.0]              # pressures for the printed table
# melts: name, melt Mg# (= 1 - Fe#, Pierru SI Table 5), first column of each EoS block
#   Asimow block:          T, P, model_1, model_2           (headers correct)
#   Jing-Karato block:     P, T, rho   (sheet headers say T, P: swapped, checked from values)
#   Wakabayashi block:     P, T, rho
MELTS = [('S3631', 0.499, 27, 51, 69),
         ('S8295', 0.550, 43, 63, 81),
         ('H6050', 0.682, 35, 57, 75)]
SOLID_COL = 1                            # MA17 bulk Rw + Gt: T, P, rho
SOLID_T   = [1900, 2200]
# line styles: this work, Asimow model 1 (Fo-bearing), Asimow model 2 (Hd instead of Fo, Pierru's preferred),
#              Jing-Karato, Wakabayashi-Funamori
LS = {'this': '-', 'A1': '--', 'A2': '-.', 'JK': ':', 'WF': (0, (3, 1, 1, 1, 1, 1))}

# %12
# ── load ONLY the liquid EoS block of the forward model ───────────────────────
src = open(FORWARD_FILE).read()
i0 = src.index('# ── Liquid BML EoS')
i1 = src.index('# ── composition')
m = types.ModuleType('eos')
m.__dict__.update(np=np, brentq=brentq)
exec(src[i0:i1], m.__dict__)

# ── read the sheet into a float array (blank or text -> nan) ──────────────────
ws = openpyxl.load_workbook(XLSX_FILE, data_only=True, read_only=True)[SHEET]
rows = list(ws.iter_rows(values_only=True))[3:]
width = max(len(r) for r in rows)
D = np.full((len(rows), width), np.nan)
for i, r in enumerate(rows):
    for j, c in enumerate(r):
        if isinstance(c, (int, float)):
            D[i, j] = c

# ── figure: one panel per melt, x = density, y = pressure (increasing downward) ──
T_ALL = [2223, 2323, 2400, 2600, 2800, 3000]
cmap  = plt.cm.plasma
cnorm = plt.Normalize(2150, 3050)
P_mod = np.linspace(P_MIN, P_MAX, 27)

fig, ax = plt.subplots(1, 3, figsize=(17, 6.5), sharey=True)
print(f"{'melt':6s}{'T (K)':>7s}{'P':>6s}{'this work':>11s}{'Asimow 1':>10s}{'Asimow 2':>10s}"
      f"{'J-K':>8s}{'W-F':>8s}{'(this - A1)':>13s}{'(this - A2)':>13s}")
for k, (name, mg, cA, cJ, cW) in enumerate(MELTS):
    a = ax[k]
    temps = np.unique(D[np.isfinite(D[:, cA]), cA])

    # bulk mantle solid (Rw + Gt, MA17) as reference
    for Ts in SOLID_T:
        s = (D[:, SOLID_COL] == Ts) & (D[:, SOLID_COL + 1] >= P_MIN)
        a.plot(D[s, SOLID_COL + 2] / 1000, D[s, SOLID_COL + 1], color='0.6',
               lw=1.2, ls='-' if Ts == SOLID_T[0] else '--')

    for T in temps:
        c = cmap(cnorm(T))
        sA = (D[:, cA] == T) & (D[:, cA + 1] >= P_MIN)
        sJ = (D[:, cJ + 1] == T) & (D[:, cJ] >= P_MIN)
        sW = (D[:, cW + 1] == T) & (D[:, cW] >= P_MIN)
        a.plot(D[sA, cA + 2] / 1000, D[sA, cA + 1], ls=LS['A1'], color=c, lw=1.2)
        a.plot(D[sA, cA + 3] / 1000, D[sA, cA + 1], ls=LS['A2'], color=c, lw=1.2)
        a.plot(D[sJ, cJ + 2] / 1000, D[sJ, cJ],     ls=LS['JK'], color=c, lw=1.4)
        a.plot(D[sW, cW + 2] / 1000, D[sW, cW],     ls=LS['WF'], color=c, lw=1.2)
        # this work: Thomas 2012 Fa + Thomas & Asimow 2013 Fo, ideal volume mixing
        rho_mod = np.array([m.liquid_bml_properties(P, T, mg)[0] for P in P_mod])
        a.plot(rho_mod, P_mod, ls=LS['this'], color=c, lw=2.4)

        for Pt in P_TAB:
            r_m  = m.liquid_bml_properties(Pt, T, mg)[0]
            oA = np.argsort(D[sA, cA + 1]); oJ = np.argsort(D[sJ, cJ]); oW = np.argsort(D[sW, cW])
            rA1 = np.interp(Pt, D[sA, cA + 1][oA], D[sA, cA + 2][oA]) / 1000
            rA2 = np.interp(Pt, D[sA, cA + 1][oA], D[sA, cA + 3][oA]) / 1000
            rJ  = np.interp(Pt, D[sJ, cJ][oJ], D[sJ, cJ + 2][oJ]) / 1000
            rW  = np.interp(Pt, D[sW, cW][oW], D[sW, cW + 2][oW]) / 1000
            print(f"{name:6s}{T:7.0f}{Pt:6.1f}{r_m:11.3f}{rA1:10.3f}{rA2:10.3f}{rJ:8.3f}{rW:8.3f}"
                  f"{100*(r_m - rA1)/rA1:+12.1f}%{100*(r_m - rA2)/rA2:+12.1f}%")

    a.set_xlabel(r'density (g cm$^{-3}$)', fontsize=14)
    a.set_title(f'{name} melt, Mg# {mg:.3f}', fontsize=14)
    a.tick_params(labelsize=14)

ax[0].set_ylim(P_MAX, P_MIN)             # pressure increases downward (shared y)
ax[0].set_ylabel('P (GPa)', fontsize=14)
h = [Line2D([], [], color='k', lw=2.4, ls=LS['this'], label='this work (binary Fo-Fa)'),
     Line2D([], [], color='k', lw=1.2, ls=LS['A1'],   label='Pierru: Asimow model 1 (with Fo)'),
     Line2D([], [], color='k', lw=1.2, ls=LS['A2'],   label='Pierru: Asimow model 2 (Hd, preferred)'),
     Line2D([], [], color='k', lw=1.4, ls=LS['JK'],   label='Pierru: Jing-Karato'),
     Line2D([], [], color='k', lw=1.2, ls=LS['WF'],   label='Pierru: Wakabayashi-Funamori'),
     Line2D([], [], color='0.6', lw=1.2, ls='-',  label='bulk solid Rw+Gt, 1900 K'),
     Line2D([], [], color='0.6', lw=1.2, ls='--', label='bulk solid Rw+Gt, 2200 K')]
h += [Line2D([], [], color=cmap(cnorm(T)), lw=2.4, label=f'{T} K') for T in T_ALL]
ax[0].legend(handles=h, loc='lower left', fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig('139_liquid_density_pierru.png', dpi=200)
print("\nsaved 139_liquid_density_pierru.png")