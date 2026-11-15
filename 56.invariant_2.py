#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 01:09:18 2026

@author: chingchen
"""

"""
Fo-Fa phase diagram pipeline 
==================================
Topology is decided by the CLOSURE TEMPERATURES at the two ends, not by
whether a particular three-phase line happens to solve:

    Fo end: whichever of alpha/beta/gamma has the HIGHEST closure temperature
            is the most stable solid there -> it owns the top layer.
    Fa end: same rule -> it owns the bottom layer.
    The three-phase lines are then whatever is needed to connect them.

@author: chingchen
"""

COLORS = ['#CD5C5C', '#35838D', '#849DAB', '#414F67', '#97795D', '#7B9E87',
          '#9B6B8A', '#4E6E8E', '#C47F3E', '#5C7A5C', '#8B6F6F', '#4A7C7C',
          '#7A6B9B', '#6B8E7A']

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve

R     = 8.314
T_fix = 2200.0     # fig1 (isothermal P-X)
P_fix = 20   # fig2/3/4 (isobaric T-X)

# ============================================================
# Thermodynamic data (Akaogi 1989 Table 3)
# ============================================================
dv298 = {('Fo','ab'): -3.16, ('Fo','bg'): -0.98, ('Fo','ag'): -4.14,
         ('Fa','ab'): -3.20, ('Fa','bg'): -1.04, ('Fa','ag'): -4.24}
transformation_point = {('Fo','ab'): (14.2, 1600.),   # Katsura et al. 2004b
                        ('Fo','bg'): (19.0, 1473.),   # Katsura & Ito 1989
                        ('Fa','ag'): (5.25, 1273.)}   # Yagi et al. 1987 eq.(1)
dS_Akaogi = {('Fo','ab'): -7.7,  ('Fo','bg'): -7.3, ('Fo','ag'): -15.0,
             ('Fa','ab'): -10.9, ('Fa','bg'): -3.1, ('Fa','ag'): -14.0}
dH_Akaogi = {('Fo','ab'): 29970, ('Fo','bg'): 9080,  ('Fo','ag'): 39050,
             ('Fa','ab'): 9620,  ('Fa','bg'): -5790, ('Fa','ag'): 3830}

dV = {}
for k in [('Fo','ab'), ('Fo','bg')]:
    Ptr, Ttr = transformation_point[k]
    dV[k] = -(dH_Akaogi[k] - Ttr*dS_Akaogi[k]) / (Ptr*1000.)
Ptr, Ttr = transformation_point[('Fa','ag')]
_DH = dH_Akaogi[('Fa','ab')] + dH_Akaogi[('Fa','bg')]
_DS = dS_Akaogi[('Fa','ab')] + dS_Akaogi[('Fa','bg')]
_DV = -(_DH - Ttr*_DS) / (Ptr*1000.)
for k in [('Fa','ab'), ('Fa','bg')]:
    dV[k] = _DV * dv298[k] / dv298[('Fa','ag')]

W_SETS = {
 'akaogi':    {'a': lambda P: 12600. + 260.*P, 'b': lambda P: 7800.,  'g': lambda P: 7800.},
 'slb2024':   {'a': lambda P: 2500.,           'b': lambda P: 12900., 'g': lambda P: 6200.},
 'chanyshev': {'a': lambda P: 4074.,           'b': lambda P: 12900., 'g': lambda P: 5394.},
}
W = W_SETS['slb2024']

DH_Fo_ab, DS_Fo_ab, DV_Fo_ab = dH_Akaogi[('Fo','ab')], dS_Akaogi[('Fo','ab')], dV[('Fo','ab')]
DH_Fo_bg, DS_Fo_bg, DV_Fo_bg = dH_Akaogi[('Fo','bg')], dS_Akaogi[('Fo','bg')], dV[('Fo','bg')]
DH_Fa_ab, DS_Fa_ab, DV_Fa_ab = dH_Akaogi[('Fa','ab')], dS_Akaogi[('Fa','ab')], dV[('Fa','ab')]
DH_Fa_bg, DS_Fa_bg, DV_Fa_bg = dH_Akaogi[('Fa','bg')], dS_Akaogi[('Fa','bg')], dV[('Fa','bg')]
DH_Fo_ag = DH_Fo_ab + DH_Fo_bg;  DS_Fo_ag = DS_Fo_ab + DS_Fo_bg;  DV_Fo_ag = DV_Fo_ab + DV_Fo_bg
DH_Fa_ag = DH_Fa_ab + DH_Fa_bg;  DS_Fa_ag = DS_Fa_ab + DS_Fa_bg;  DV_Fa_ag = DV_Fa_ab + DV_Fa_bg

def dG_Fo_ab(P,T): return DH_Fo_ab - T*DS_Fo_ab + DV_Fo_ab*1000.0*P
def dG_Fo_bg(P,T): return DH_Fo_bg - T*DS_Fo_bg + DV_Fo_bg*1000.0*P
def dG_Fa_ab(P,T): return DH_Fa_ab - T*DS_Fa_ab + DV_Fa_ab*1000.0*P
def dG_Fa_bg(P,T): return DH_Fa_bg - T*DS_Fa_bg + DV_Fa_bg*1000.0*P
def dG_Fo_ag(P,T): return dG_Fo_ab(P,T) + dG_Fo_bg(P,T)     # Hess
def dG_Fa_ag(P,T): return dG_Fa_ab(P,T) + dG_Fa_bg(P,T)
zero = lambda P, T: 0.0

# ============================================================
# Melting inputs
# ============================================================
DH_FO_fus = 142000.0    # Richet et al. 1993
DH_FA_fus = 89300.0     # Richet et al. 1993
W_LIQ     = 0.0         # liquid non-ideality (the ONLY free W in the melting equations)
                        # solids use their own W_phi from W_SETS -- see note below.
# NOTE: the old single 'W_LIQ' applied to every solid was inconsistent with the
# solid-solution W used in the solid-solid equations, so three-phase points did
# not land on the solid-solid loops. Each solid now carries its own W_phi.

def Tm_Fo_func(P):
    K0, Kp = 125.4, 5.33
    def P_of_x(x):
        f = ((1/x)**(2/3) - 1)/2
        return 3*K0*f*(1+2*f)**2.5 * (1 + 1.5*(Kp-4)*f) - P
    return 2163.0*(1 + 3.0*(1 - brentq(P_of_x, 0.5, 1.0)))

FA_MELT = {0.0:(1490.,'a'), 3.0:(1650.,'a'), 6.3:(1793.,'a'),
           12.0:(1990.,'g'), 20.0:(2200.,'g')}
def Tm_eff(P, Tm, ph):
    d = 0.0 if ph=='a' else dG_Fa_ag(P, Tm)
    return Tm/(1.0 - d/DH_FA_fus)

Tm_Fo_P = Tm_Fo_func(P_fix)
_Ps  = sorted(FA_MELT)
_Tms = [Tm_eff(p, *FA_MELT[p]) for p in _Ps]
Tm_Fa_P = np.interp(P_fix, _Ps, _Tms)

def find_closure(f, Tlo=300., Thi=4000., n=600):
    """scan for a sign change then brentq; NaN if no root exists at this P"""
    Ts = np.linspace(Tlo, Thi, n)
    fs = np.array([f(t) for t in Ts])
    for i in range(n-1):
        if np.isfinite(fs[i]) and np.isfinite(fs[i+1]) and fs[i]*fs[i+1] < 0:
            return brentq(f, Ts[i], Ts[i+1])
    return np.nan

# ============================================================
# Phase registry: shift terms + closure temperatures + plot styling
# ============================================================
PH = {
 'a': dict(sym=r'\alpha', dFo=zero,      dFa=zero,      W=W['a'](P_fix), c_s=COLORS[6],  c_l=COLORS[7]),
 'b': dict(sym=r'\beta',  dFo=dG_Fo_ab,  dFa=dG_Fa_ab,  W=W['b'](P_fix), c_s=COLORS[8],  c_l=COLORS[9]),
 'g': dict(sym=r'\gamma', dFo=dG_Fo_ag,  dFa=dG_Fa_ag,  W=W['g'](P_fix), c_s=COLORS[10], c_l=COLORS[11]),
}
PH['a']['Tc_Fo'] = Tm_Fo_P
PH['a']['Tc_Fa'] = Tm_Fa_P
for k in ['b', 'g']:
    PH[k]['Tc_Fo'] = find_closure(lambda T, k=k: DH_FO_fus*(1 - T/Tm_Fo_P) - PH[k]['dFo'](P_fix, T))
    PH[k]['Tc_Fa'] = find_closure(lambda T, k=k: DH_FA_fus*(1 - T/Tm_Fa_P) - PH[k]['dFa'](P_fix, T))

print(f'P_fix = {P_fix} GPa   Tm_Fo = {Tm_Fo_P:.1f} K   Tm_Fa(eff) = {Tm_Fa_P:.1f} K')
print('closure temperatures (the pure end-member melting point of each structure):')
print(f"  Fo end:  a={PH['a']['Tc_Fo']:7.1f}  b={PH['b']['Tc_Fo']:7.1f}  g={PH['g']['Tc_Fo']:7.1f}")
print(f"  Fa end:  a={PH['a']['Tc_Fa']:7.1f}  b={PH['b']['Tc_Fa']:7.1f}  g={PH['g']['Tc_Fa']:7.1f}")

# ============================================================
# Melting loop scanner
# ============================================================
def scan_melt(Tgrid, k, WL=W_LIQ):
    dFo, dFa = PH[k]['dFo'], PH[k]['dFa']
    Ws = PH[k]['W']                       # this solid's own regular-solution parameter
    out = []; X = [0.999, 0.995]
    for T in Tgrid:
        def eqs(v):
            Xs, Xl = v
            e1 = 2*np.log(Xl/Xs)         + (WL*(1-Xl)**2 - Ws*(1-Xs)**2)/(R*T) \
                 - DH_FO_fus/R*(1/Tm_Fo_P - 1/T) - dFo(P_fix,T)/(R*T)
            e2 = 2*np.log((1-Xl)/(1-Xs)) + (WL*Xl**2     - Ws*Xs**2)/(R*T) \
                 - DH_FA_fus/R*(1/Tm_Fa_P - 1/T) - dFa(P_fix,T)/(R*T)
            return [e1, e2]
        s, info, ok, _ = fsolve(eqs, X, full_output=True)
        if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
            out.append(s); X = list(s)
        else:
            out.append([np.nan, np.nan])
    return np.array(out)

# ============================================================
# Three-phase solver: multistart + three-gate check
# (a single guess fails at pressures where the line sits far from it)
# ============================================================
def solve_three(k1, k2, WL=W_LIQ):
    dFo1, dFa1, W1 = PH[k1]['dFo'], PH[k1]['dFa'], PH[k1]['W']
    dFo2, dFa2, W2 = PH[k2]['dFo'], PH[k2]['dFa'], PH[k2]['W']
    def eqs(v):
        X1, X2, Xl, T = v
        return [2*np.log(Xl/X1) + (WL*(1-Xl)**2 - W1*(1-X1)**2)/(R*T)
                - DH_FO_fus/R*(1/Tm_Fo_P - 1/T) - dFo1(P_fix,T)/(R*T),
                2*np.log((1-Xl)/(1-X1)) + (WL*Xl**2 - W1*X1**2)/(R*T)
                - DH_FA_fus/R*(1/Tm_Fa_P - 1/T) - dFa1(P_fix,T)/(R*T),
                2*np.log(Xl/X2) + (WL*(1-Xl)**2 - W2*(1-X2)**2)/(R*T)
                - DH_FO_fus/R*(1/Tm_Fo_P - 1/T) - dFo2(P_fix,T)/(R*T),
                2*np.log((1-Xl)/(1-X2)) + (WL*Xl**2 - W2*X2**2)/(R*T)
                - DH_FA_fus/R*(1/Tm_Fa_P - 1/T) - dFa2(P_fix,T)/(R*T)]
    for T0 in np.linspace(1400., 3000., 17):
        for (x1, x2, xl) in [(0.95,0.90,0.70), (0.85,0.75,0.45), (0.70,0.55,0.30),
                             (0.55,0.40,0.20), (0.35,0.25,0.12)]:
            s, info, ok, _ = fsolve(eqs, [x1, x2, xl, T0], full_output=True)
            if (ok==1 and max(abs(info['fvec']))<1e-8
                    and 0 < s[2] < s[1] < s[0] < 1 and 300 < s[3] < 4000):
                return dict(X1=s[0], X2=s[1], XL=s[2], T=s[3], has=True, pair=k1+k2)
    return dict(X1=np.nan, X2=np.nan, XL=np.nan, T=np.nan, has=False, pair=k1+k2)

# ============================================================
# TOPOLOGY: highest closure wins each end; connect them
# ============================================================
_fo = [(k, PH[k]['Tc_Fo']) for k in 'abg' if np.isfinite(PH[k]['Tc_Fo'])]
_fa = [(k, PH[k]['Tc_Fa']) for k in 'abg' if np.isfinite(PH[k]['Tc_Fa'])]
TOP = max(_fo, key=lambda x: x[1])[0]     # owns the top layer (Fo end)
BOT = max(_fa, key=lambda x: x[1])[0]     # owns the bottom layer (Fa end)

if TOP == BOT:
    SEQ, LINES = [TOP], []
elif TOP == 'a' and BOT == 'g':
    _ab = solve_three('a', 'b')
    _bg = solve_three('b', 'g')
    if _ab['has'] and _bg['has'] and _ab['T'] > _bg['T']:
        SEQ, LINES = ['a','b','g'], [_ab, _bg]           # beta has a melting window
    else:
        _ag = solve_three('a', 'g')                       # beta squeezed out
        SEQ, LINES = (['a','g'], [_ag]) if _ag['has'] else ([TOP], [])
else:
    _ln = solve_three(TOP, BOT)
    SEQ, LINES = ([TOP, BOT], [_ln]) if _ln['has'] else ([TOP], [])

TOPO = ' + L  /  '.join(PH[k]['sym'] for k in SEQ) + ' + L'
print(f"\ntopology at {P_fix} GPa:  Fo end -> '{TOP}'   Fa end -> '{BOT}'")
print(f"  stable sequence : {' / '.join(k+'+L' for k in SEQ)}")
for l in LINES:
    print(f"  three-phase line: {l['pair']}L  T={l['T']:.1f} K  "
          f"X1={l['X1']:.3f} X2={l['X2']:.3f} XL={l['XL']:.3f}")
if not LINES:
    print('  (no three-phase line: one solid owns the whole solidus)')

# ---- consistency check: a three-phase point must ALSO satisfy the
#      corresponding solid-solid equilibrium at that temperature ----
_SSPAIR = {'ab': (dG_Fo_ab, dG_Fa_ab), 'bg': (dG_Fo_bg, dG_Fa_bg),
           'ag': (dG_Fo_ag, dG_Fa_ag)}
for l in LINES:
    dGFo, dGFa = _SSPAIR[l['pair']]
    W1, W2 = PH[l['pair'][0]]['W'], PH[l['pair'][1]]['W']
    T = l['T']
    def _ss(v):
        X1, X2 = v
        return [2*np.log(X2/X1) + (W2*(1-X2)**2 - W1*(1-X1)**2)/(R*T) + dGFo(P_fix,T)/(R*T),
                2*np.log((1-X2)/(1-X1)) + (W2*X2**2 - W1*X1**2)/(R*T) + dGFa(P_fix,T)/(R*T)]
    s = fsolve(_ss, [l['X1'], l['X2']])
    print(f"  consistency {l['pair']}L: melting gives ({l['X1']:.4f}, {l['X2']:.4f}), "
          f"solid-solid gives ({s[0]:.4f}, {s[1]:.4f})  "
          f"-> max|diff| = {max(abs(l['X1']-s[0]), abs(l['X2']-s[1])):.2e}")

# ============================================================
# SUBSOLIDUS TOPOLOGY: the solid-solid three-phase line T3
#   T3 stable   -> ab, bg below T3 ; ag above T3
#   T3 metastable (above the melt) or absent -> only ab, bg ; ag never drawn
# ============================================================
def solve_T3():
    Wa, Wb, Wg = W['a'](P_fix), W['b'](P_fix), W['g'](P_fix)
    def eqs(v):
        Xa, Xb, Xg, T = v
        return [2*np.log(Xb/Xa) + (Wb*(1-Xb)**2 - Wa*(1-Xa)**2)/(R*T) + dG_Fo_ab(P_fix,T)/(R*T),
                2*np.log((1-Xb)/(1-Xa)) + (Wb*Xb**2 - Wa*Xa**2)/(R*T) + dG_Fa_ab(P_fix,T)/(R*T),
                2*np.log(Xg/Xb) + (Wg*(1-Xg)**2 - Wb*(1-Xb)**2)/(R*T) + dG_Fo_bg(P_fix,T)/(R*T),
                2*np.log((1-Xg)/(1-Xb)) + (Wg*Xg**2 - Wb*Xb**2)/(R*T) + dG_Fa_bg(P_fix,T)/(R*T)]
    for T0 in np.linspace(800., 3000., 23):
        s, i, o, _ = fsolve(eqs, [0.7, 0.55, 0.35, T0], full_output=True)
        if (o==1 and max(abs(i['fvec']))<1e-8 and 0 < s[2] < s[1] < s[0] < 1 and 300 < s[3] < 4000):
            return dict(Xa=s[0], Xb=s[1], Xg=s[2], T=s[3], has=True)
    return dict(Xa=np.nan, Xb=np.nan, Xg=np.nan, T=np.nan, has=False)

SS3 = solve_T3()
T_MELT_MIN = min([l['T'] for l in LINES]) if LINES else np.inf
T3_STABLE = SS3['has'] and SS3['T'] < T_MELT_MIN

if T3_STABLE:
    SS_WINDOW = {'ab': (-np.inf, SS3['T']),
                 'bg': (-np.inf, SS3['T']),
                 'ag': (SS3['T'], np.inf)}
else:
    SS_WINDOW = {'ab': (-np.inf, np.inf),
                 'bg': (-np.inf, np.inf),
                 'ag': None}
# ============================================================
# fig1 : isothermal P-X
# ============================================================
def equations(v):
    Xa, Xb, Xg, P = v
    Wa, Wb, Wg = W['a'](P), W['b'](P), W['g'](P)
    T = T_fix
    return [2*np.log(Xb/Xa) + (Wb*(1-Xb)**2 - Wa*(1-Xa)**2)/(R*T) + dG_Fo_ab(P,T)/(R*T),
            2*np.log((1-Xb)/(1-Xa)) + (Wb*Xb**2 - Wa*Xa**2)/(R*T) + dG_Fa_ab(P,T)/(R*T),
            2*np.log(Xg/Xb) + (Wg*(1-Xg)**2 - Wb*(1-Xb)**2)/(R*T) + dG_Fo_bg(P,T)/(R*T),
            2*np.log((1-Xg)/(1-Xb)) + (Wg*Xg**2 - Wb*Xb**2)/(R*T) + dG_Fa_bg(P,T)/(R*T)]
sol, info, ok, _ = fsolve(equations, [0.8, 0.8, 0.6, 13.9], full_output=True)
XaP3, XbP3, XgP3, P3 = sol
has_P3 = (ok==1 and max(abs(info['fvec']))<1e-8 and 0 < XgP3 < XbP3 < XaP3 < 1)

def scan_ss(grid, Wlo, Whi, dGFo, dGFa, isobar):
    """isobar=True -> grid is T at P_fix ; isobar=False -> grid is P at T_fix"""
    out = []; X = [0.999, 0.995]
    for v0 in grid:
        P, T = (P_fix, v0) if isobar else (v0, T_fix)
        Wl, Wh = Wlo(P), Whi(P)
        def eqs(v):
            X1, X2 = v
            return [2*np.log(X2/X1) + (Wh*(1-X2)**2 - Wl*(1-X1)**2)/(R*T) + dGFo(P,T)/(R*T),
                    2*np.log((1-X2)/(1-X1)) + (Wh*X2**2 - Wl*X1**2)/(R*T) + dGFa(P,T)/(R*T)]
        s, i, o, _ = fsolve(eqs, X, full_output=True)
        if o==1 and max(abs(i['fvec']))<1e-8 and 0<s[1]<s[0]<1:
            out.append(s); X = list(s)
        else:
            out.append([np.nan, np.nan])
    return np.array(out)

fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.set_xlim(0,1); ax1.set_ylim(4,22)
ax1.set_xlabel('Mg#  ($X_{Fo}$)', fontsize=14); ax1.set_ylabel('P (GPa)', fontsize=14)
ax1.set_title(f'fix T = {T_fix:.0f} K', fontsize=14)
ax1.tick_params(labelsize=14); ax1.grid(alpha=0.3)

Pg_ab = np.linspace(-(DH_Fo_ab - T_fix*DS_Fo_ab)/DV_Fo_ab/1000,
                    -(DH_Fa_ab - T_fix*DS_Fa_ab)/DV_Fa_ab/1000, 250)
Pg_bg = np.linspace(-(DH_Fo_bg - T_fix*DS_Fo_bg)/DV_Fo_bg/1000,
                    -(DH_Fa_bg - T_fix*DS_Fa_bg)/DV_Fa_bg/1000, 250)
Pg_ag = np.linspace(-(DH_Fo_ag - T_fix*DS_Fo_ag)/DV_Fo_ag/1000,
                    -(DH_Fa_ag - T_fix*DS_Fa_ag)/DV_Fa_ag/1000, 250)
abP = scan_ss(Pg_ab, W['a'], W['b'], dG_Fo_ab, dG_Fa_ab, False)
bgP = scan_ss(Pg_bg, W['b'], W['g'], dG_Fo_bg, dG_Fa_bg, False)
agP = scan_ss(Pg_ag, W['a'], W['g'], dG_Fo_ag, dG_Fa_ag, False)
mP_ab = Pg_ab > P3 if has_P3 else np.ones_like(Pg_ab, bool)
mP_bg = Pg_bg > P3 if has_P3 else np.ones_like(Pg_bg, bool)
mP_ag = Pg_ag < P3 if has_P3 else np.ones_like(Pg_ag, bool)
ax1.plot(abP[mP_ab,0], Pg_ab[mP_ab], '-',  color=COLORS[0], lw=2, label=r'$X^\alpha$ (a+b)')
ax1.plot(abP[mP_ab,1], Pg_ab[mP_ab], '--', color=COLORS[1], lw=2, label=r'$X^\beta$ (a+b)')
ax1.plot(bgP[mP_bg,0], Pg_bg[mP_bg], '-',  color=COLORS[2], lw=2, label=r'$X^\beta$ (b+g)')
ax1.plot(bgP[mP_bg,1], Pg_bg[mP_bg], '--', color=COLORS[3], lw=2, label=r'$X^\gamma$ (b+g)')
ax1.plot(agP[mP_ag,0], Pg_ag[mP_ag], '-',  color=COLORS[4], lw=2, label=r'$X^\alpha$ (a+g)')
ax1.plot(agP[mP_ag,1], Pg_ag[mP_ag], '--', color=COLORS[5], lw=2, label=r'$X^\gamma$ (a+g)')
if has_P3:
    ax1.plot([XgP3, XaP3], [P3]*2, 'k-', lw=3, zorder=5)
    ax1.plot([XaP3, XbP3, XgP3], [P3]*3, 'ko', ms=9, zorder=6)
ax1.legend(fontsize=10)

# ============================================================
# fig2/fig3 : isobaric T-X
# ============================================================
T_Fo_ab = (DH_Fo_ab + 1000*DV_Fo_ab*P_fix)/DS_Fo_ab
T_Fa_ab = (DH_Fa_ab + 1000*DV_Fa_ab*P_fix)/DS_Fa_ab
T_Fo_bg = (DH_Fo_bg + 1000*DV_Fo_bg*P_fix)/DS_Fo_bg
T_Fa_bg = (DH_Fa_bg + 1000*DV_Fa_bg*P_fix)/DS_Fa_bg
T_Fo_ag = (DH_Fo_ag + 1000*DV_Fo_ag*P_fix)/DS_Fo_ag
T_Fa_ag = (DH_Fa_ag + 1000*DV_Fa_ag*P_fix)/DS_Fa_ag
Tg_ss = {'ab': np.linspace(T_Fo_ab, T_Fa_ab, 250),
         'bg': np.linspace(T_Fo_bg, T_Fa_bg, 250),
         'ag': np.linspace(T_Fo_ag, T_Fa_ag, 250)}
SS = {'ab': scan_ss(Tg_ss['ab'], W['a'], W['b'], dG_Fo_ab, dG_Fa_ab, True),
      'bg': scan_ss(Tg_ss['bg'], W['b'], W['g'], dG_Fo_bg, dG_Fa_bg, True),
      'ag': scan_ss(Tg_ss['ag'], W['a'], W['g'], dG_Fo_ag, dG_Fa_ag, True)}
SS_STYLE = {'ab': (r'\alpha', r'\beta',  COLORS[0], COLORS[1]),
            'bg': (r'\beta',  r'\gamma', COLORS[2], COLORS[3]),
            'ag': (r'\alpha', r'\gamma', COLORS[4], COLORS[5])}

# melting loops: each scanned over its own two closure temperatures
Tg_m, ML = {}, {}
for k in 'abg':
    t1, t2 = PH[k]['Tc_Fo'], PH[k]['Tc_Fa']
    if np.isfinite(t1) and np.isfinite(t2):
        Tg_m[k] = np.linspace(t1*0.999, t2*1.001, 300)
        ML[k]   = scan_melt(Tg_m[k], k)

fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.set_xlim(0,1); ax2.set_ylim(1000,3000)
ax2.set_xlabel(r'Mg#  ($X_{Fo}$)', fontsize=14); ax2.set_ylabel('T (K)', fontsize=14)
ax2.set_title(f'fix P = {P_fix:.1f} GPa : all solutions (untruncated)', fontsize=14)
ax2.tick_params(labelsize=14); ax2.grid(alpha=0.3)
for key, arr in SS.items():
    s1, s2, c1, c2 = SS_STYLE[key]
    ax2.plot(arr[:,0], Tg_ss[key], '-',  color=c1, lw=2, label=f'$X^{{{s1}}}$ ({key})')
    ax2.plot(arr[:,1], Tg_ss[key], '--', color=c2, lw=2, label=f'$X^{{{s2}}}$ ({key})')
for k in ML:
    ax2.plot(ML[k][:,0], Tg_m[k], '-',  color=PH[k]['c_s'], lw=2, label=f"$X^{{{PH[k]['sym']}}}$ ({k}+L)")
    ax2.plot(ML[k][:,1], Tg_m[k], '--', color=PH[k]['c_l'], lw=2, label=f'$X^L$ ({k}+L)')
ax2.legend(fontsize=8, loc='lower left', ncol=2)

fig3, ax3 = plt.subplots(figsize=(8,8))
ax3.set_xlim(0,1); ax3.set_ylim(1000,3000)
ax3.set_xlabel(r'Mg#  ($X_{Fo}$)', fontsize=14); ax3.set_ylabel('T (K)', fontsize=14)
ax3.set_title(f'fix P = {P_fix:.1f} GPa : stable phase diagram\n${TOPO}$', fontsize=13)
ax3.tick_params(labelsize=14); ax3.grid(alpha=0.3)

# each loop in SEQ lives between its neighbouring three-phase lines
for i, k in enumerate(SEQ):
    hi = LINES[i-1]['T'] if i > 0 else np.inf     # line above it
    lo = LINES[i]['T']   if i < len(LINES) else -np.inf   # line below it
    m = (Tg_m[k] < hi) & (Tg_m[k] > lo)
    ax3.plot(ML[k][m,0], Tg_m[k][m], '-',  color=PH[k]['c_s'], lw=2,
             label=f"$X^{{{PH[k]['sym']}}}$ ({k}+L)")
    ax3.plot(ML[k][m,1], Tg_m[k][m], '--', color=PH[k]['c_l'], lw=2,
             label=f'$X^L$ ({k}+L)')

# three-phase tie lines
for l in LINES:
    ax3.plot([l['XL'], l['X1']], [l['T']]*2, 'k-', lw=3, zorder=5)
    ax3.plot([l['X1'], l['X2'], l['XL']], [l['T']]*3, 'ko', ms=8, zorder=6)
    ax3.text(1.01, l['T'], f"{l['T']:.0f}", fontsize=9, va='center')

# solid-solid loops: window from the subsolidus topology, then capped by its
# own melting three-phase line if one exists
for key, arr in SS.items():
    win = SS_WINDOW[key]
    if win is None:
        continue
    lo, hi = win
    own = next((l['T'] for l in LINES if set(l['pair']) == set(key)), None)
    hi = min(hi, own if own is not None else T_MELT_MIN)   
    m = (Tg_ss[key] > lo) & (Tg_ss[key] < hi)
    if m.sum() == 0:
        continue
    s1, s2, c1, c2 = SS_STYLE[key]
    ax3.plot(arr[m,0], Tg_ss[key][m], '-',  color=c1, lw=2, label=f'$X^{{{s1}}}$ ({key})')
    ax3.plot(arr[m,1], Tg_ss[key][m], '--', color=c2, lw=2, label=f'$X^{{{s2}}}$ ({key})')

# stable solid-solid tie line
if T3_STABLE:
    ax3.plot([SS3['Xg'], SS3['Xa']], [SS3['T']]*2, 'k-', lw=2.5, zorder=5)
    ax3.plot([SS3['Xa'], SS3['Xb'], SS3['Xg']], [SS3['T']]*3, 'ko', ms=7, zorder=6)
    ax3.text(1.01, SS3['T'], f"{SS3['T']:.0f}", fontsize=9, va='center')
ax3.legend(fontsize=8, loc='lower left', ncol=2)
# ---- Pierru 2026 overlay (multicomponent: our solidus should sit ABOVE these) ----
PIERRU_SOL = {7.95: 1890, 9.6: 1960, 14.1: 2090, 17.7: 2200}
PIERRU_TIE = [(8.0, 1890, 'Ol(cold)', 0.80, 0.53), (8.0, 1890, 'Ol(warm)', 0.85, 0.58),
              (9.6, 1960, 'Ol', 0.91, 0.67), (14.1, 2090, 'Wad', 0.80, 0.51),
              (17.7, 2200, 'Rw', 0.74, 0.50)]
for (Pp, Tp, mineral, XS_obs, XL_obs) in PIERRU_TIE:
    if abs(Pp - P_fix) < 1.0:
        ax3.plot([XS_obs, XL_obs], [Tp]*2, color='darkgreen', lw=2, zorder=9)
        ax3.scatter([XS_obs], [Tp], marker='o', s=140, color='limegreen',
                    edgecolors='darkgreen', zorder=12, lw=2)
        ax3.scatter([XL_obs], [Tp], marker='D', s=140, color='limegreen',
                    edgecolors='darkgreen', zorder=12, lw=2)
        ax3.text(XL_obs-0.02, Tp+28, f'{mineral}\n({XS_obs}$\\to${XL_obs})',
                 fontsize=8, color='darkgreen', ha='right')
_pk = min(PIERRU_SOL, key=lambda k: abs(k - P_fix))
if abs(_pk - P_fix) < 1.0:
    ax3.scatter([0.75], [PIERRU_SOL[_pk]], marker='s', s=200, color='limegreen',
                edgecolors='darkgreen', zorder=11, lw=2)

# ============================================================
# fig4 : pressure sweep of the topology
# ============================================================
Pg = np.linspace(6., 22., 33)
rows = []
for P in Pg:
    tmFo = Tm_Fo_func(P)
    tmFa = np.interp(P, _Ps, [Tm_eff(p, *FA_MELT[p]) for p in _Ps])
    tc = {'a': (tmFo, tmFa)}
    for k, dFo, dFa in [('b', dG_Fo_ab, dG_Fa_ab), ('g', dG_Fo_ag, dG_Fa_ag)]:
        tc[k] = (find_closure(lambda T: DH_FO_fus*(1 - T/tmFo) - dFo(P, T)),
                 find_closure(lambda T: DH_FA_fus*(1 - T/tmFa) - dFa(P, T)))
    fo = max([(k, v[0]) for k, v in tc.items() if np.isfinite(v[0])], key=lambda x: x[1])[0]
    fa = max([(k, v[1]) for k, v in tc.items() if np.isfinite(v[1])], key=lambda x: x[1])[0]
    rows.append((P, tc, fo, fa))

fig4, ax4 = plt.subplots(figsize=(9,7))
for k, c in [('a', COLORS[6]), ('b', COLORS[8]), ('g', COLORS[10])]:
    ax4.plot(Pg, [r[1][k][0] for r in rows], '-',  color=c, lw=2, label=f'{k} closure, Fo end')
    ax4.plot(Pg, [r[1][k][1] for r in rows], '--', color=c, lw=2, label=f'{k} closure, Fa end')
for r in rows:
    ax4.plot(r[0], r[1][r[2]][0], 'k.', ms=8, zorder=5)
    ax4.plot(r[0], r[1][r[3]][1], 'k.', ms=8, zorder=5)
ax4.axvline(P_fix, color='grey', ls=':', lw=1.5)
ax4.set_xlabel('P (GPa)', fontsize=14); ax4.set_ylabel('closure T (K)', fontsize=14)
ax4.set_title('Closure temperatures vs P\n(black dots = winner at each end = who owns that layer)',
              fontsize=13)
ax4.tick_params(labelsize=14); ax4.grid(alpha=0.3)
ax4.legend(fontsize=9, ncol=2)
plt.tight_layout()

print('\npressure sweep of the topology:')
prev = None
for (P, tc, fo, fa) in rows:
    tag = f'{fo}/{fa}'
    if tag != prev:
        print(f'  P >= {P:5.1f} GPa :  Fo end -> {fo} ,  Fa end -> {fa}')
        prev = tag


# ============================================================
# fig5 : W_LIQ sensitivity  (liquid non-ideality)
#   Solids now use their own W_phi, so the only remaining free W is the
#   liquid's. TOP/BOT are W-independent (closures have no W term).
# ============================================================
def build_topology(WM):
    """回傳 (SEQ, LINES) for this W_LIQ；TOP/BOT 沿用全域（與 W 無關）"""
    if TOP == BOT:
        return [TOP], []
    if TOP == 'a' and BOT == 'g':
        ab = solve_three('a', 'b', WM)
        bg = solve_three('b', 'g', WM)
        if ab['has'] and bg['has'] and ab['T'] > bg['T']:
            return ['a','b','g'], [ab, bg]
        ag = solve_three('a', 'g', WM)
        return (['a','g'], [ag]) if ag['has'] else ([TOP], [])
    ln = solve_three(TOP, BOT, WM)
    return ([TOP, BOT], [ln]) if ln['has'] else ([TOP], [])

W_LIST = [0., 2500., 8000., 15000.]   # candidate W_LIQ values
WCOL   = [COLORS[3], COLORS[6], COLORS[0], COLORS[8]]

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(15, 7))

# ---------- 左格：每個 W 的完整穩定結構 ----------
ax5a.set_xlim(0, 1)
ax5a.set_xlabel(r'Mg#  ($X_{Fo}$)', fontsize=14)
ax5a.set_ylabel('T (K)', fontsize=14)
ax5a.set_title(f'stable melting loops vs $W_{{MELT}}$   (P = {P_fix:.1f} GPa)', fontsize=14)
ax5a.tick_params(labelsize=14); ax5a.grid(alpha=0.3)

for WM, c in zip(W_LIST, WCOL):
    seqW, linesW = build_topology(WM)
    tag = '/'.join(seqW)
    for i, k in enumerate(seqW):
        if k not in Tg_m:
            continue
        loop = scan_melt(Tg_m[k], k, WM)
        hi = linesW[i-1]['T'] if i > 0 else np.inf
        lo = linesW[i]['T']   if i < len(linesW) else -np.inf
        m = (Tg_m[k] < hi) & (Tg_m[k] > lo)
        lbl = f'$W$={WM/1000:.1f} kJ  [{tag}]' if i == 0 else None
        ax5a.plot(loop[m,0], Tg_m[k][m], '-',  color=c, lw=2, label=lbl)
        ax5a.plot(loop[m,1], Tg_m[k][m], '--', color=c, lw=2)
    for l in linesW:
        ax5a.plot([l['XL'], l['X1']], [l['T']]*2, '-', color=c, lw=1.2, alpha=0.7)

# 收攏點（與 W 無關）
for k in 'abg':
    if k == TOP and np.isfinite(PH[k]['Tc_Fo']):
        ax5a.plot([1], [PH[k]['Tc_Fo']], 'k*', ms=14, zorder=6)
    if k == BOT and np.isfinite(PH[k]['Tc_Fa']):
        ax5a.plot([0], [PH[k]['Tc_Fa']], 'k*', ms=14, zorder=6)
ax5a.text(0.03, 0.97, 'solid = solidus,  dashed = liquidus\n'
                      'thin horizontals = three-phase lines\n'
                      'stars = closure (W-independent)',
          transform=ax5a.transAxes, fontsize=9, va='top')
ax5a.legend(fontsize=10, loc='lower right')

# ---------- 右格：三條三相線溫度 vs W（穩定實線、metastable 點線）----------
Wg = np.linspace(0., 20000., 50)
res = {'ab': [], 'bg': [], 'ag': []}
stable = {'ab': [], 'bg': [], 'ag': []}
for WM in Wg:
    seqW, linesW = build_topology(WM)
    active = {l['pair'] for l in linesW}
    for pair in ['ab', 'bg', 'ag']:
        l = solve_three(pair[0], pair[1], WM)
        res[pair].append(l['T'] if l['has'] else np.nan)
        stable[pair].append(pair in active)
for pair, c in [('ab', COLORS[0]), ('bg', COLORS[1]), ('ag', COLORS[4])]:
    T  = np.array(res[pair], dtype=float)
    st = np.array(stable[pair])
    has_stable = np.any(st & np.isfinite(T))
    has_meta   = np.any((~st) & np.isfinite(T))
    if has_stable:
        ax5b.plot(Wg/1000, np.where(st, T, np.nan), '-', color=c, lw=2.5,
                  label=rf'$T_{{{pair[0]}{pair[1]}L}}$ (stable)')
    if has_meta:
        ax5b.plot(Wg/1000, np.where(~st, T, np.nan), ':', color=c, lw=1.5,
                  label=rf'$T_{{{pair[0]}{pair[1]}L}}$ (metastable)')
ax5b.axvline(W_LIQ/1000, color='grey', ls=':', lw=1.5)

ax5b.set_xlabel(r'$W_{LIQ}$ (kJ/mol)', fontsize=14)
ax5b.set_ylabel('three-phase T (K)', fontsize=14)
ax5b.set_title(f'Three-phase temperatures vs $W_{{MELT}}$   (P = {P_fix:.1f} GPa)\n'
               'solid = stable branch, dotted = metastable', fontsize=13)
ax5b.tick_params(labelsize=14); ax5b.grid(alpha=0.3)
ax5b.legend(fontsize=11)
plt.tight_layout()

# ---------- 對照表 ----------
print(f'\nW_LIQ sensitivity at P = {P_fix} GPa   (TOP={TOP}, BOT={BOT}, W-independent)')
for WM in W_LIST:
    seqW, linesW = build_topology(WM)
    ln = '  '.join(f"{l['pair']}L={l['T']:.1f}" for l in linesW) or '(none)'
    print(f"  W={WM/1000:5.1f} kJ:  seq={'/'.join(seqW):8s}  {ln}")


# ---------- Pierru 實驗點疊在 fig5 左格 ----------
# ============================================================
# Pierru 2026 overlay on fig5 左格
#   比較基準：模型在 Mg#=0.75 的 solidus 溫度  vs  Pierru Table 3 的 Tsol
#   （多組分凝固點下降 => 模型應在綠點「上方」；這是方向檢查，不是擬合目標）
# ============================================================
PIERRU_SOL = {7.95: 1890, 9.6: 1960, 14.1: 2090, 17.7: 2200}
PIERRU_TIE = [(8.0, 1890, 'Ol(cold)', 0.80, 0.53), (8.0, 1890, 'Ol(warm)', 0.85, 0.58),
              (9.6, 1960, 'Ol', 0.91, 0.67), (14.1, 2090, 'Wad', 0.80, 0.51),
              (17.7, 2200, 'Rw', 0.74, 0.50)]

def solidus_T_at(X_target, WM):
    """沿著這個 W 的穩定 solidus，內插出 X_target 對應的溫度。
       回傳 (T, phase)；X_target 不在任何穩定 solidus 上 -> (nan, None)"""
    seqW, linesW = build_topology(WM)
    for i, k in enumerate(seqW):
        if k not in Tg_m:
            continue
        loop = scan_melt(Tg_m[k], k, WM)
        hi = linesW[i-1]['T'] if i > 0 else np.inf
        lo = linesW[i]['T']   if i < len(linesW) else -np.inf
        m = (Tg_m[k] < hi) & (Tg_m[k] > lo) & np.isfinite(loop[:,0])
        if m.sum() < 2:
            continue
        xs, tt = loop[m,0], Tg_m[k][m]
        if xs.min() <= X_target <= xs.max():
            o = np.argsort(xs)
            return float(np.interp(X_target, xs[o], tt[o])), k
    return np.nan, None

X_MARS = 0.75
ax5a.axvline(X_MARS, color='darkorange', ls='--', lw=1.2, alpha=0.6)
ax5a.text(X_MARS+0.005, ax5a.get_ylim()[0]+30, f'Mars Mg# {X_MARS}',
          fontsize=8, color='darkorange')

# --- Table 4 tie-line（只當視覺參考，不做預測比較）---
for (Pp, Tp, mineral, XS_obs, XL_obs) in PIERRU_TIE:
    if abs(Pp - P_fix) < 1.0:
        ax5a.plot([XS_obs, XL_obs], [Tp]*2, color='darkgreen', lw=2, zorder=9)
        ax5a.scatter([XS_obs], [Tp], marker='o', s=120, color='limegreen',
                     edgecolors='darkgreen', zorder=12, lw=2)
        ax5a.scatter([XL_obs], [Tp], marker='D', s=120, color='limegreen',
                     edgecolors='darkgreen', zorder=12, lw=2)
        ax5a.text(XL_obs-0.02, Tp+25, f'{mineral} ({XS_obs}$\\to${XL_obs})',
                  fontsize=8, color='darkgreen', ha='right')

# --- Table 3 solidus 錨點 + 各 W 的模型 solidus 溫度 ---
_pk = min(PIERRU_SOL, key=lambda k: abs(k - P_fix))
Tsol_obs = PIERRU_SOL[_pk] if abs(_pk - P_fix) < 1.0 else None

if Tsol_obs is not None:
    ax5a.scatter([X_MARS], [Tsol_obs], marker='s', s=200, color='limegreen',
                 edgecolors='darkgreen', zorder=11, lw=2, label='Pierru $T_{sol}$')
    print(f'\nmodel solidus at Mg#={X_MARS}  vs  Pierru Tsol = {Tsol_obs} K   (P = {P_fix} GPa)')
    for WM, c in zip(W_LIST, WCOL):
        T_model, ph = solidus_T_at(X_MARS, WM)
        if np.isnan(T_model):
            print(f'  W={WM/1000:5.1f} kJ:  Mg#={X_MARS} not on any stable solidus '
                  f'(swallowed by a three-phase line)')
            continue
        ax5a.scatter([X_MARS], [T_model], marker='_', s=300, color=c, lw=3, zorder=10)
        print(f'  W={WM/1000:5.1f} kJ:  model = {T_model:7.1f} K ({ph}+L)   '
              f'diff = {T_model-Tsol_obs:+7.1f} K')

ax5a.legend(fontsize=9, loc='lower right')