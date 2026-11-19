#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 133_fa_melt_check.py
# Diagnose the Fa end-member melting curve used in the BML phase diagram.
#
#  (1) Melting temperature of each END-MEMBER IN THE GAMMA STRUCTURE, i.e. the
#      T at which mu_L = mu_gamma for pure Fo and pure Fa:
#        Fo:  dH_fus_Fo (1 - T/Tm_Fo(P)) = dG_Fo(alpha->gamma)(P, T)
#        Fa:  dH_fus_Fa (1 - T/Tm_Fa_eff(P)) = dG_Fa(alpha->gamma)(P, T)
#      A normal gamma+L loop (liquid more Fe-rich than solid, as in Pierru)
#      needs  T_gamma(Fa) < T_gamma(Fo).
#  (2) Same quantity for Fa from a Simon fit to Ohtani (1979) gamma-Fe2SiO4
#      melting data (metastable above ~13-17 GPa).
#  (3) At the Pierru tie-line (17.7 GPa, XS = 0.74, XL = 0.50, Tsol = 2200 K):
#      model XL and required rigid shift as a function of Tm_Fa_eff.
#
# The phase-diagram block is read from the MCMC source (from '_R_GAS' to the
# line before '# ── BML thermal state') and exec'd alone, so torch / nGibbs /
# TauP are never imported.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq, curve_fit

plt.rcParams['font.family'] = 'DejaVu Sans'

# ── config ───────────────────────────────────────────────────────────────
SRC = 'main_MCMC_TA.py'

# Ohtani (1979) gamma-Fe2SiO4 congruent melting, W-W25%Re, Table 1 brackets
# (CHECK AGAINST THE TABLE). First point = alpha-gamma-L triple point.
OHTANI_P = np.array([7.0,    8.2,    11.0,   13.0])
OHTANI_T = np.array([1883.0, 2053.0, 2373.0, 2523.0])
P_TRIPLE, T_TRIPLE = 7.0, 1883.0

P_TIE, XS_TIE, XL_TIE, TSOL_TIE = 17.7, 0.74, 0.50, 2200.0   # Pierru tie-line
TMFA_SCAN = np.arange(1300.0, 1951.0, 50.0)                    # K, Tm_Fa_eff at P_TIE
P_GRID    = np.arange(8.0, 24.01, 0.5)

fontsize = 14

# ── load phase-diagram block ─────────────────────────────────────────────
lines = open(SRC).read().splitlines()
i0 = [i for i, l in enumerate(lines) if l.startswith('_R_GAS')][0]
i1 = [i for i, l in enumerate(lines) if l.startswith('# ── BML thermal state')][0]
ns = {'np': np, 'fsolve': fsolve, 'brentq': brentq}
exec('\n'.join(lines[i0:i1]), ns)
print(f'phase-diagram block: lines {i0+1}-{i1} of {SRC}')
print(f'current DELTA_SHIFT = {ns["_DELTA_SHIFT"]:.1f} K   raw solidus = {ns["_RAW_SOL_REF"]:.1f} K'
      f'   model XL at tie = {ns["_XL_TIE_MODEL"]:.3f}')

# ── (1)+(2) end-member melting in the gamma structure ────────────────────
popt, _ = curve_fit(lambda p, a, c: T_TRIPLE*((p - P_TRIPLE)/a + 1.0)**(1.0/c),
                    OHTANI_P, OHTANI_T, p0=[5.0, 3.0])
SIMON_A, SIMON_C = popt
print(f'Simon fit to Ohtani 1979: a = {SIMON_A:.3f} GPa, c = {SIMON_C:.3f}')

Tg_Fo, Tg_Fa_code, Tg_Fa_ohtani = [], [], []
for P in P_GRID:
    TmFo, TmFa = ns['_Tm_Fo'](P), ns['_Tm_Fa'](P)
    Tg_Fo.append(brentq(lambda T: ns['_DH_FO_FUS']*(1 - T/TmFo) - ns['_dG_Fo_ag'](P, T), 800., 6000.))
    Tg_Fa_code.append(brentq(lambda T: ns['_DH_FA_FUS']*(1 - T/TmFa) - ns['_dG_Fa_ag'](P, T), 500., 6000.))
    Tg_Fa_ohtani.append(T_TRIPLE*((P - P_TRIPLE)/SIMON_A + 1.0)**(1.0/SIMON_C))
Tg_Fo, Tg_Fa_code, Tg_Fa_ohtani = map(np.array, (Tg_Fo, Tg_Fa_code, Tg_Fa_ohtani))

print('\nmelting T of pure end-members in the gamma structure (K)')
print('   P    gamma-Fo   gamma-Fa(code)   gamma-Fa(Ohtani fit)')
for k in range(0, len(P_GRID), 4):
    print(f'{P_GRID[k]:6.1f} {Tg_Fo[k]:9.0f} {Tg_Fa_code[k]:12.0f} {Tg_Fa_ohtani[k]:16.0f}')
inv = P_GRID[Tg_Fa_ohtani > Tg_Fo]
if inv.size:
    print(f'\nOhtani-based gamma-Fa melts ABOVE gamma-Fo for P = {inv.min():.1f}-{inv.max():.1f} GPa'
          f' -> inverted gamma+L loop, no solution with XL < XS')

# ── (3) Pierru tie-line vs Tm_Fa_eff at P_TIE ────────────────────────────
TmFo_tie = ns['_Tm_Fo'](P_TIE)
Ts = np.linspace(1500.0, 3200.0, 700)
xl_model, shift_req = [], []
for TmFa in TMFA_SCAN:
    XS = np.array([ns['_gamma_XS_XL'](t, P_TIE, TmFo_tie, TmFa)[0] for t in Ts])
    ok = np.isfinite(XS)
    if ok.sum() < 3 or not (XS[ok].min() <= XS_TIE <= XS[ok].max()):
        xl_model.append(np.nan); shift_req.append(np.nan); continue
    o = np.argsort(XS[ok])
    T_raw = float(np.interp(XS_TIE, XS[ok][o], Ts[ok][o]))
    xl_model.append(ns['_gamma_XS_XL'](T_raw, P_TIE, TmFo_tie, TmFa)[1])
    shift_req.append(TSOL_TIE - T_raw)
xl_model, shift_req = np.array(xl_model), np.array(shift_req)

print(f'\nPierru tie-line at {P_TIE} GPa (XS = {XS_TIE}, XL = {XL_TIE}, Tsol = {TSOL_TIE:.0f} K)')
print('  Tm_Fa_eff   model XL   required shift')
for t, x, s in zip(TMFA_SCAN, xl_model, shift_req):
    print(f'  {t:8.0f}   {x:8.3f}   {s:10.0f}')
ok = np.isfinite(xl_model)
TMFA_BEST = float(np.interp(XL_TIE, xl_model[ok], TMFA_SCAN[ok]))
print(f'  Tm_Fa_eff that reproduces XL = {XL_TIE}: {TMFA_BEST:.0f} K'
      f'   (code uses {ns["_Tm_Fa"](P_TIE):.0f} K)')

# ── plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.plot(P_GRID, Tg_Fo, color='#4E6E8E', lw=2, label=r'$\gamma$-Fo (code)')
ax.plot(P_GRID, Tg_Fa_code, color='#CD5C5C', lw=2, label=r'$\gamma$-Fa (code, effective)')
ax.plot(P_GRID, Tg_Fa_ohtani, color='#CD5C5C', lw=2, ls='--', label=r'$\gamma$-Fa (Ohtani 1979 fit)')
ax.scatter(OHTANI_P, OHTANI_T, color='k', zorder=5, label='Ohtani 1979')
ax.axvspan(19.5, 24.0, color='0.88', zorder=0, label=r'posterior $P_{CMB}$')
ax.set_xlabel('Pressure (GPa)', fontsize=fontsize)
ax.set_ylabel('end-member melting T in $\\gamma$ structure (K)', fontsize=fontsize)
ax.tick_params(labelsize=14)
ax.legend(fontsize=10, loc='lower right')

ax = axes[1]
ax.plot(TMFA_SCAN, xl_model, color='#414F67', lw=2, marker='o', label='model $X_L$ at tie-line')
ax.axhline(XL_TIE, color='k', ls='--', lw=1.5, label=f'Pierru $X_L$ = {XL_TIE}')
ax.axvline(ns['_Tm_Fa'](P_TIE), color='#CD5C5C', lw=1.5, label='code value')
ax.set_xlabel(r'$T_{m,Fa}^{eff}$ at 17.7 GPa (K)', fontsize=fontsize)
ax.set_ylabel('$X_L$', fontsize=fontsize)
ax.tick_params(labelsize=14)
ax.legend(fontsize=10, loc='lower right')

fig.tight_layout()
fig.savefig('133_fa_melt_check.png', dpi=150)
plt.show()
