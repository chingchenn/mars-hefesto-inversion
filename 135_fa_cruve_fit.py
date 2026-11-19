#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:24:48 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 135_fa_curve_fit.py
# Replace the two provisional Fa nodes (12 GPa 1990 K, 20 GPa 2200 K) with a smooth
# effective gamma-Fa melting curve fitted to the four Pierru (2026) tie-lines.
#
#   gamma-Fa effective melting:  T_g(P) = T_g0 [ (P - 6.3)/a + 1 ]^(1/c),  P >= 6.3 GPa
#   T_g0 is fixed so that the alpha-reference value at 6.3 GPa equals 1793 K
#   (continuous with the alpha-Fa nodes below 6.3 GPa); a and c are fitted.
#
#   Step 1 (shape):       a, c from the observed X_L at X_S = X_S,obs (X_L does not
#                         depend on the rigid shift)
#   Step 2 (temperature): shift_i = T_sol,i - T_raw,i ; constant shift = mean,
#                         residuals show whether a single shift is enough
#
# Branch used for each tie-line = the solid Pierru observed:
#   8.0, 9.6 GPa olivine (alpha, W = 2.5 kJ/mol, SLB24; calibration only, not in MCMC)
#   14.1 GPa wadsleyite (beta), 17.7 GPa ringwoodite (gamma)
#
# Reads the phase-diagram block of the MCMC code (as 133/134) and the plotting
# machinery of plot_phase_diagram_3D.py (everything before '# ── 影格 1').

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq, least_squares

plt.rcParams['font.family'] = 'DejaVu Sans'

# ── config ───────────────────────────────────────────────────────────────
SRC      = '86.change_traveltime_a52.py'
PLOT_SRC = 'plot_phase_diagram_3D.py'
W_ALPHA  = 2500.0                      # SLB24 Table C2, ol fo-fa
P_ANCHOR, T_ANCHOR = 6.3, 1793.0       # alpha-gamma-L triple point (A67)

# Pierru 2026: Table 5 (P, solidus T) and Table 4 (molar Mg#)
TIE = [dict(P=7.95, Tsol=1890., XS=0.804, XL=0.525, br='a', name='Ol'),
       dict(P=9.60, Tsol=1960., XS=0.910, XL=0.670, br='a', name='Ol'),
       dict(P=14.10, Tsol=2090., XS=0.797, XL=0.507, br='b', name='Wad'),
       dict(P=17.70, Tsol=2200., XS=0.740, XL=0.499, br='g', name='Rw')]
P_SECTIONS = [7.95, 9.60, 14.10, 17.70, 20.0, 23.0]
fontsize = 14

# ── load MCMC phase-diagram block ───────────────────────────────────────
lines = open(SRC).read().splitlines()
i0 = [i for i, l in enumerate(lines) if l.startswith('_R_GAS')][0]
i1 = [i for i, l in enumerate(lines) if l.startswith('# ── BML thermal state')][0]
ns = {'np': np, 'fsolve': fsolve, 'brentq': brentq}
exec('\n'.join(lines[i0:i1]), ns)
TmFa_old = ns['_Tm_Fa']
dG0 = lambda P, T: 0.0
BR = {'a': (W_ALPHA,         dG0,              dG0),
      'b': (ns['_W_BETA'],  ns['_dG_Fo_ab'],  ns['_dG_Fa_ab']),
      'g': (ns['_W_GAMMA'], ns['_dG_Fo_ag'],  ns['_dG_Fa_ag'])}
GUESS = ([0.97, 0.80], [0.92, 0.70], [0.85, 0.60], [0.80, 0.50], [0.75, 0.48],
         [0.70, 0.45], [0.60, 0.35], [0.45, 0.25], [0.30, 0.15])

def loop(T, P, br, TmFa):
    W, dFo, dFa = BR[br]
    return ns['_loop_XS_XL'](T, P, ns['_Tm_Fo'](P), TmFa, W, dFo, dFa, GUESS)

# ── effective Fa curve ──────────────────────────────────────────────────
# T_g0 such that the alpha-reference value at 6.3 GPa is 1793 K
T_G0 = brentq(lambda Tg: ns['_Tm_eff'](P_ANCHOR, Tg, 'g') - T_ANCHOR, 1500., 2500.)

def make_TmFa(a, c):
    def f(P):
        if P < P_ANCHOR:
            return TmFa_old(P)
        Tg = T_G0*((P - P_ANCHOR)/a + 1.0)**(1.0/c)
        return ns['_Tm_eff'](P, Tg, 'g')
    return f

def gamma_melt(P, TmFa_fun):
    TmFo, TmFa = ns['_Tm_Fo'](P), TmFa_fun(P)
    return brentq(lambda T: ns['_DH_FA_FUS']*(1 - T/TmFa) - ns['_dG_Fa_ag'](P, T), 500., 6000.)

def tie_eval(TmFa_fun):
    """model X_L and raw solidus T at X_S = X_S,obs for every tie-line"""
    out = []
    Ts = np.linspace(1200., 3400., 450)
    for t in TIE:
        TmFa = TmFa_fun(t['P'])
        XS = np.array([loop(T, t['P'], t['br'], TmFa)[0] for T in Ts])
        ok = np.isfinite(XS)
        if ok.sum() < 3 or not (XS[ok].min() <= t['XS'] <= XS[ok].max()):
            out.append((np.nan, np.nan)); continue
        o = np.argsort(XS[ok])
        Traw = float(np.interp(t['XS'], XS[ok][o], Ts[ok][o]))
        out.append((loop(Traw, t['P'], t['br'], TmFa)[1], Traw))
    return np.array(out)

# ── step 1: fit a, c to X_L ─────────────────────────────────────────────
def resid(p):
    r = tie_eval(make_TmFa(*p))[:, 0] - np.array([t['XL'] for t in TIE])
    return np.where(np.isfinite(r), r, 1.0)

fit = least_squares(resid, x0=[3.0, 3.7], bounds=([0.3, 1.0], [50.0, 20.0]))
A_FIT, C_FIT = fit.x
TmFa_new = make_TmFa(A_FIT, C_FIT)
print(f'T_g0 = {T_G0:.0f} K at {P_ANCHOR} GPa;  fitted a = {A_FIT:.3f} GPa, c = {C_FIT:.3f}')

# ── step 2: shift from solidus temperatures ─────────────────────────────
ev_old, ev_new = tie_eval(TmFa_old), tie_eval(TmFa_new)
sh_old = np.array([t['Tsol'] for t in TIE]) - ev_old[:, 1]
sh_new = np.array([t['Tsol'] for t in TIE]) - ev_new[:, 1]
SHIFT_NEW = float(np.nanmean(sh_new))
print(f'constant shift (mean of 4) = {SHIFT_NEW:.0f} K   (code: {ns["_DELTA_SHIFT"]:.0f} K, 17.7 GPa only)')
print('\n   P    solid   X_L obs  X_L old  X_L new   shift_old  shift_new  resid_new')
for t, eo, en, so, sn in zip(TIE, ev_old, ev_new, sh_old, sh_new):
    print(f'{t["P"]:6.2f}  {t["name"]:4s}  {t["XL"]:7.3f}  {eo[0]:7.3f}  {en[0]:7.3f}'
          f'  {so:9.0f}  {sn:9.0f}  {sn - SHIFT_NEW:+9.0f}')

P_grid = np.arange(6.3, 24.01, 0.25)
print('\n   P   gamma-Fa melt old / new (K)')
for P in [8, 12, 16, 20, 22, 24]:
    print(f'{P:5.1f}   {gamma_melt(P, TmFa_old):6.0f} / {gamma_melt(P, TmFa_new):6.0f}')

# ── figure 1: calibration summary ───────────────────────────────────────
OHT_P = np.array([7.0, 8.2, 11.0, 13.0]); OHT_T = np.array([1883., 2053., 2373., 2523.])
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
a0 = ax[0]
a0.plot(P_grid, [gamma_melt(P, TmFa_old) for P in P_grid], color='#CD5C5C', lw=2, ls='--', label='old (two nodes)')
a0.plot(P_grid, [gamma_melt(P, TmFa_new) for P in P_grid], color='#CD5C5C', lw=2.5, label='fitted (this script)')
a0.plot(P_grid, [brentq(lambda T: ns['_DH_FO_FUS']*(1 - T/ns['_Tm_Fo'](P)) - ns['_dG_Fo_ag'](P, T), 800., 6000.)
                 for P in P_grid], color='#4E6E8E', lw=2, label=r'$\gamma$-Fo')
a0.scatter(OHT_P, OHT_T, color='k', zorder=5, label='Ohtani 1979')
a0.set_xlabel('Pressure (GPa)', fontsize=fontsize)
a0.set_ylabel(r'end-member melting T in $\gamma$ structure (K)', fontsize=fontsize)
a0.legend(fontsize=11, loc='lower right')

a1 = ax[1]
Pt = [t['P'] for t in TIE]
a1.plot(Pt, [t['XL'] for t in TIE], 'ks', ms=9, label='Pierru $X_L$')
a1.plot(Pt, ev_old[:, 0], 'o', mfc='none', mec='#CD5C5C', ms=10, mew=2, label='old')
a1.plot(Pt, ev_new[:, 0], 'o', color='#CD5C5C', ms=8, label='fitted')
for t in TIE:
    a1.annotate(t['name'], (t['P'], t['XL']), textcoords='offset points', xytext=(6, -14), fontsize=11)
a1.set_xlabel('Pressure (GPa)', fontsize=fontsize)
a1.set_ylabel(r'$X_L$ at $X_S = X_{S,obs}$', fontsize=fontsize)
a1.legend(fontsize=11)

a2 = ax[2]
a2.plot(Pt, sh_old, 'o', mfc='none', mec='#414F67', ms=10, mew=2, label='old Fa curve')
a2.plot(Pt, sh_new, 'o', color='#414F67', ms=8, label='fitted Fa curve')
a2.axhline(SHIFT_NEW, color='#414F67', lw=1.5, label=f'mean = {SHIFT_NEW:.0f} K')
a2.axhline(ns['_DELTA_SHIFT'], color='0.5', ls=':', lw=1.5, label=f'code = {ns["_DELTA_SHIFT"]:.0f} K')
a2.set_xlabel('Pressure (GPa)', fontsize=fontsize)
a2.set_ylabel('required shift (K)', fontsize=fontsize)
a2.legend(fontsize=11)
for a in ax:
    a.tick_params(labelsize=14)
fig.tight_layout()
fig.savefig('135_fa_curve_fit.png', dpi=150)

# ── figure 2: isobaric sections with the new calibration ───────────────
plines = open(PLOT_SRC).read().splitlines()
j1 = [i for i, l in enumerate(plines) if l.startswith('# ── 影格 1')][0]
ps = {}
exec('\n'.join(plines[:j1]), ps)          # prints the plot script's own DELTA_SHIFT once
ps['Tm_Fa'] = TmFa_new
ps['DELTA_SHIFT'] = SHIFT_NEW
ps['_TOPO_CACHE'].clear()
ps['PIERRU'] = {t['P']: dict(Tsol=t['Tsol'], XS=t['XS'], XL=t['XL'], mineral=t['name']) for t in TIE}
wt = ps['wt_fa']

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for axi, P in zip(axes.ravel(), P_SECTIONS):
    sec = ps['section_TX'](P)
    for k in sec['SEQ']:
        Tg, XS, XL = sec['branches'][k]
        m = np.isfinite(XS) & np.isfinite(XL)
        Tp = Tg + SHIFT_NEW
        axi.fill_betweenx(Tp[m], wt(XL[m]), wt(XS[m]), color=ps['C_FILL'][k], alpha=0.16, lw=0)
        axi.plot(wt(XS[m]), Tp[m], '-', color=ps['PH'][k]['c'], lw=2.4, label=f"{ps['PH'][k]['sym']} + L")
        axi.plot(wt(XL[m]), Tp[m], '--', color=ps['PH'][k]['c'], lw=1.8)
    for l in sec['LINES']:
        if l['X1'] - l['X2'] < 0.01:
            continue
        Tl = l['T'] + SHIFT_NEW
        axi.plot([wt(l['XL']), wt(l['X1'])], [Tl, Tl], '-', color=ps['C_TIE'], lw=2.2)
        axi.plot([wt(l['XL']), wt(l['X2']), wt(l['X1'])], [Tl]*3, 'o', color=ps['C_TIE'], ms=6)
        axi.text(wt(l['X1']) - 1.5, Tl, f"{l['pair']}L {Tl:.0f} K", fontsize=11,
                 color=ps['C_TIE'], va='center', ha='right')
    for t in TIE:
        if abs(t['P'] - P) < 0.3:
            axi.plot([wt(t['XL']), wt(t['XS'])], [t['Tsol']]*2, '-', color=ps['C_PIERRU'], lw=3)
            axi.plot([wt(t['XS'])], [t['Tsol']], 'o', color=ps['C_PIERRU'], ms=10, mec='k')
            axi.plot([wt(t['XL'])], [t['Tsol']], 'D', color='white', mec=ps['C_PIERRU'], mew=2, ms=9)
            axi.text(wt(t['XL']) + 2, t['Tsol'], f"Pierru {t['name']}", fontsize=11,
                     color=ps['C_PIERRU'], va='center')
    axi.axvline(wt(0.75), color='k', ls=':', lw=1.2, alpha=0.7)
    axi.set_title(f'P = {P:.2f} GPa   ({"/".join(sec["SEQ"])})', fontsize=fontsize)
    axi.set_xlim(0, 100); axi.set_ylim(1300, 3150)
    axi.set_xlabel('Fa (wt %)', fontsize=fontsize); axi.set_ylabel('T (K)', fontsize=fontsize)
    axi.tick_params(labelsize=14)
    axi.legend(fontsize=11, loc='lower left')
fig.suptitle(f'Fitted Fa curve (a = {A_FIT:.2f} GPa, c = {C_FIT:.2f}), constant shift {SHIFT_NEW:.0f} K',
             fontsize=16)
fig.tight_layout()
fig.savefig('135_sections_fitted.png', dpi=130)
plt.show()