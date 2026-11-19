#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:09:30 2026

@author: chingchen
"""

import os
import types
import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────────
FORWARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '26fall_main_mcmc.py')
P_GRID  = np.arange(15.0, 24.01, 0.25)            # GPa
MG_LIST = [0.30, 0.50, 0.70, 0.75, 0.80]
T_LOW   = 1000.0     # below every solidus: bml_phase_diagram returns T_sol / T_liq only
# old a52 (script 86) Fa effective melting nodes, alpha reference (K); constant above 20 GPa
OLD_P = [0.0, 3.0, 6.3, 12.0, 20.0]
OLD_T = [1490., 1650., 1793., 1659., 1432.]
# Pierru 2026: 17.7 GPa tie-line solidus; 19 GPa runs (nominal T)
PIERRU_SOL = (17.7, 2200.0)
PIERRU_19  = [(19.0, 2303.0, 'S8293'), (19.0, 2323.0, 'S8295'), (19.0, 2403.0, 'H6050')]

# ── load the phase-diagram block (a60 calibration done at load) ───────────────
src = open(FORWARD_FILE).read()
i0 = src.index('# ── physics: solidus')
i1 = src.index('# ── composition')
m = types.ModuleType('pd')
m.__dict__.update(np=np, brentq=brentq, fsolve=fsolve)
exec(src[i0:i1], m.__dict__)

# ── T_sol, T_liq, branch on the (Mg#, P) grid ─────────────────────────────────
TS = np.full((len(MG_LIST), len(P_GRID)), np.nan)
TL = np.full((len(MG_LIST), len(P_GRID)), np.nan)
for i, X in enumerate(MG_LIST):
    for j, P in enumerate(P_GRID):
        r = m.bml_phase_diagram(T_LOW, P, X)
        if r is not None:
            TS[i, j], TL[i, j] = r['T_solidus'], r['T_liquidus']
    print(f"Mg# {X:.2f}: T_sol {np.nanmin(TS[i]):.0f}-{np.nanmax(TS[i]):.0f} K  "
          f"T_liq {np.nanmin(TL[i]):.0f}-{np.nanmax(TL[i]):.0f} K  "
          f"failures {np.sum(~np.isfinite(TS[i]))}")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

# left: effective Fa melting temperature (alpha reference state)
Pf = np.linspace(0.0, 24.0, 241)
ax[0].plot(Pf, np.interp(Pf, OLD_P, OLD_T), '--', color='0.5', lw=1.8,
           label='a52 (old nodes, kink at 20 GPa)')
ax[0].plot(OLD_P, OLD_T, 'o', color='0.5', ms=6)
ax[0].plot(Pf, [m._Tm_Fa(p) for p in Pf], '-', color='C3', lw=2.0,
           label=f'a60 (slope {m._FA_SLOPE:.2f} K/GPa above 6.3 GPa)')
ax[0].axvline(17.7, color='k', ls=':', lw=1)
ax[0].set_xlabel('P (GPa)', fontsize=14)
ax[0].set_ylabel(r'effective $T_{m,Fa}$ ($\alpha$ ref., K)', fontsize=14)
ax[0].set_title('Fa end-member', fontsize=14)
ax[0].tick_params(labelsize=14)
ax[0].legend(loc='lower right', fontsize=10)

# right: a60 solidus / liquidus vs P for several Mg#
for i, X in enumerate(MG_LIST):
    c = f"C{i}"
    ax[1].plot(P_GRID, TS[i], '-',  color=c, lw=1.8, label=f'Mg# {X:.2f}')
    ax[1].plot(P_GRID, TL[i], '--', color=c, lw=1.4)
ax[1].plot(*PIERRU_SOL, 'D', color='limegreen', mec='k', ms=8, label='Pierru 17.7 GPa (calibration)')
for P, T, name in PIERRU_19:
    ax[1].errorbar(P, T, yerr=50.0, fmt='s', color='k', mfc='none', ms=7, capsize=3)
    ax[1].annotate(name, (P, T), textcoords='offset points', xytext=(6, 2), fontsize=10)
ax[1].set_xlabel('P (GPa)', fontsize=14)
ax[1].set_ylabel('T (K)', fontsize=14)
ax[1].set_title('a60 BML phase diagram (solid: T_sol, dashed: T_liq)', fontsize=14)
ax[1].tick_params(labelsize=14)
ax[1].legend(loc='lower right', fontsize=10)

fig.tight_layout()
fig.savefig('143_a60_phase_diagram_TP.png', dpi=200)
print("saved 143_a60_phase_diagram_TP.png")