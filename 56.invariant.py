# 56.invariant.py
# Fo-Fa (Mg2SiO4-Fe2SiO4) binary with alpha (olivine), beta (wadsleyite), gamma (ringwoodite)
# and liquid, regular-solution model.
#   fig1: isothermal P-X section at T_fix (solid-solid loops + invariant point P3)
#   fig2: isobaric T-X section at P_fix, every two-phase loop untruncated
#   fig3: isobaric T-X section at P_fix, stable parts only
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve

plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = ['#CD5C5C', '#35838D', '#849DAB', '#414F67', '#97795D', '#7B9E87', '#9B6B8A',
          '#4E6E8E', '#C47F3E', '#5C7A5C', '#8B6F6F', '#4A7C7C', '#7A6B9B', '#6B8E7A']

R      = 8.314
T_fix  = 1500.0     # K, fig1
P_fix  = 16.0       # GPa, fig2 / fig3
W_MELT = 8000.0     # J/mol, solid non-ideality used in the melting equations

# ── thermodynamic data: Akaogi 1989 Table 3 ──────────────────────────────────
# dV(298 K, 1 bar), cm3/mol
dv298 = {('Fo','ab'): -3.16, ('Fo','bg'): -0.98, ('Fo','ag'): -4.14,
         ('Fa','ab'): -3.20, ('Fa','bg'): -1.04, ('Fa','ag'): -4.24}
dS_Akaogi = {('Fo','ab'): -7.7,  ('Fo','bg'): -7.3, ('Fo','ag'): -15.0,
             ('Fa','ab'): -10.9, ('Fa','bg'): -3.1, ('Fa','ag'): -14.0}
dH_Akaogi = {('Fo','ab'): 29970, ('Fo','bg'): 9080,  ('Fo','ag'): 39050,
             ('Fa','ab'): 9620,  ('Fa','bg'): -5790, ('Fa','ag'): 3830}

# experimental transition points (P GPa, T K)
#   Fo a->b: 14.2 GPa @ 1600 K   Katsura & Ito 2004
#   Fo b->g: 19.0 GPa @ 1473 K   Katsura & Ito 1989
#   Fa a->g: 5.25 GPa @ 1273 K   Yagi 1987 eq.(1): P = 2.75 + 2.5e-3 T(degC), +-0.1 GPa
#            (its slope dP/dT = 2.5 MPa/K can serve as a Clapeyron check)
transformation_point = {('Fo','ab'): (14.2, 1600.), ('Fo','bg'): (19.0, 1473.),
                        ('Fa','ag'): (5.25, 1273.)}

# dV from dG = 0 at the transition point: dV = -(dH - Ttr dS) / Ptr   (x1000: GPa -> J/cm3)
dV = {}
for k in [('Fo','ab'), ('Fo','bg')]:
    Ptr, Ttr = transformation_point[k]
    dV[k] = -(dH_Akaogi[k] - Ttr*dS_Akaogi[k]) / (Ptr*1000.)
# Fa: only a->g is measured (beta-Fa is metastable). Hess: dH_ag = dH_ab + dH_bg, same for dS.
# Then split dV_ag into ab and bg in the ratio of the 298 K volumes.
Ptr, Ttr = transformation_point[('Fa','ag')]
DH_ag = dH_Akaogi[('Fa','ab')] + dH_Akaogi[('Fa','bg')]
DS_ag = dS_Akaogi[('Fa','ab')] + dS_Akaogi[('Fa','bg')]
DV_ag = -(DH_ag - Ttr*DS_ag) / (Ptr*1000.)
for k in [('Fa','ab'), ('Fa','bg')]:
    dV[k] = DV_ag * dv298[k] / dv298[('Fa','ag')]

# solid-solution interaction parameters W(P), J/mol
W_SETS = {
 'akaogi':    {'a': lambda P: 12600. + 260.*P,
               'b': lambda P: 7800.,  'g': lambda P: 7800.},
 'slb2024':   {'a': lambda P: 2500.,               # Table C2: 2.5 +- 21 kJ
               'b': lambda P: 12900.,              # 12.9 +- 25 (V12 = 0 -> no P term)
               'g': lambda P: 6200.},              # 6.2 +- 17
 'chanyshev': {'a': lambda P: 4074.,               # 2.037 x 2 (half formula unit -> formula unit)
               'b': lambda P: 12900.,              # SLB
               'g': lambda P: 5394.},
}
W = W_SETS['slb2024']

DH_Fo_ab, DS_Fo_ab, DV_Fo_ab = dH_Akaogi[('Fo','ab')], dS_Akaogi[('Fo','ab')], dV[('Fo','ab')]
DH_Fo_bg, DS_Fo_bg, DV_Fo_bg = dH_Akaogi[('Fo','bg')], dS_Akaogi[('Fo','bg')], dV[('Fo','bg')]
DH_Fa_ab, DS_Fa_ab, DV_Fa_ab = dH_Akaogi[('Fa','ab')], dS_Akaogi[('Fa','ab')], dV[('Fa','ab')]
DH_Fa_bg, DS_Fa_bg, DV_Fa_bg = dH_Akaogi[('Fa','bg')], dS_Akaogi[('Fa','bg')], dV[('Fa','bg')]
DH_Fo_ag, DS_Fo_ag, DV_Fo_ag = DH_Fo_ab + DH_Fo_bg, DS_Fo_ab + DS_Fo_bg, DV_Fo_ab + DV_Fo_bg
DH_Fa_ag, DS_Fa_ag, DV_Fa_ag = DH_Fa_ab + DH_Fa_bg, DS_Fa_ab + DS_Fa_bg, DV_Fa_ab + DV_Fa_bg

# standard-state free energy change: dG = dH - T dS + dV P   (dV cm3/mol x 1000 -> J/(mol GPa))
def dG_Fo_ab(P, T): return DH_Fo_ab - T*DS_Fo_ab + DV_Fo_ab*1000.0*P
def dG_Fo_bg(P, T): return DH_Fo_bg - T*DS_Fo_bg + DV_Fo_bg*1000.0*P
def dG_Fa_ab(P, T): return DH_Fa_ab - T*DS_Fa_ab + DV_Fa_ab*1000.0*P
def dG_Fa_bg(P, T): return DH_Fa_bg - T*DS_Fa_bg + DV_Fa_bg*1000.0*P
def dG_Fo_ag(P, T): return dG_Fo_ab(P, T) + dG_Fo_bg(P, T)   # Hess
def dG_Fa_ag(P, T): return dG_Fa_ab(P, T) + dG_Fa_bg(P, T)

# Two-phase equilibrium phi1 -> phi2 (X = X_Fo; 2 Mg sites per formula unit):
#   Fo: 2 ln(X2/X1)         + [W2 (1-X2)^2 - W1 (1-X1)^2]/RT + dG_Fo/RT = 0
#   Fa: 2 ln((1-X2)/(1-X1)) + [W2 X2^2     - W1 X1^2    ]/RT + dG_Fa/RT = 0
# Each loop is traced by continuation along the grid (seed = last converged solution).
def trace(eqs, grid):
    out, X = [], [0.999, 0.995]
    for x in grid:
        s, info, ok, _ = fsolve(eqs, X, args=(x,), full_output=True)
        if ok == 1 and max(abs(info['fvec'])) < 1e-8 and 0 < s[1] < s[0] < 1:
            out.append(s); X = list(s)
        else:
            out.append([np.nan, np.nan])
    return np.array(out)

# ══ fig1: isothermal P-X at T_fix ══════════════════════════════════════════
# invariant point alpha + beta + gamma: 4 equations (ab Fo, ab Fa, bg Fo, bg Fa), unknowns Xa, Xb, Xg, P
def equations(v):
    T = T_fix
    Xa, Xb, Xg, P = v
    Wa, Wb, Wg = W['a'](P), W['b'](P), W['g'](P)
    return [2*np.log(Xb/Xa)         + Wb*(1-Xb)**2/(R*T) - Wa*(1-Xa)**2/(R*T) + dG_Fo_ab(P, T_fix)/(R*T),
            2*np.log((1-Xb)/(1-Xa)) + Wb*Xb**2/(R*T)     - Wa*Xa**2/(R*T)     + dG_Fa_ab(P, T_fix)/(R*T),
            2*np.log(Xg/Xb)         + Wg*(1-Xg)**2/(R*T) - Wb*(1-Xb)**2/(R*T) + dG_Fo_bg(P, T_fix)/(R*T),
            2*np.log((1-Xg)/(1-Xb)) + Wg*Xg**2/(R*T)     - Wb*Xb**2/(R*T)     + dG_Fa_bg(P, T_fix)/(R*T)]

Xa, Xb, Xg, P3 = fsolve(equations, [0.8, 0.8, 0.6, 13.9])
print(f'fig1: invariant point at T = {T_fix:.0f} K:  P3 = {P3:.3f} GPa  Xa = {Xa:.4f}  Xb = {Xb:.4f}  Xg = {Xg:.4f}')

# loops run between the pure end-member transition pressures: dG(P, T_fix) = 0 -> P = -(dH - T dS)/dV
Pg_ab = np.linspace(-(DH_Fo_ab - T_fix*DS_Fo_ab)/DV_Fo_ab/1000, -(DH_Fa_ab - T_fix*DS_Fa_ab)/DV_Fa_ab/1000, 250)
Pg_bg = np.linspace(-(DH_Fo_bg - T_fix*DS_Fo_bg)/DV_Fo_bg/1000, -(DH_Fa_bg - T_fix*DS_Fa_bg)/DV_Fa_bg/1000, 250)
Pg_ag = np.linspace(-(DH_Fo_ag - T_fix*DS_Fo_ag)/DV_Fo_ag/1000, -(DH_Fa_ag - T_fix*DS_Fa_ag)/DV_Fa_ag/1000, 250)

ab = trace(lambda v, P: [2*np.log(v[1]/v[0])         + (W['b'](P)*(1-v[1])**2 - W['a'](P)*(1-v[0])**2)/(R*T_fix) + dG_Fo_ab(P, T_fix)/(R*T_fix),
                         2*np.log((1-v[1])/(1-v[0])) + (W['b'](P)*v[1]**2     - W['a'](P)*v[0]**2)/(R*T_fix)     + dG_Fa_ab(P, T_fix)/(R*T_fix)], Pg_ab)
bg = trace(lambda v, P: [2*np.log(v[1]/v[0])         + (W['g'](P)*(1-v[1])**2 - W['b'](P)*(1-v[0])**2)/(R*T_fix) + dG_Fo_bg(P, T_fix)/(R*T_fix),
                         2*np.log((1-v[1])/(1-v[0])) + (W['g'](P)*v[1]**2     - W['b'](P)*v[0]**2)/(R*T_fix)     + dG_Fa_bg(P, T_fix)/(R*T_fix)], Pg_bg)
ag = trace(lambda v, P: [2*np.log(v[1]/v[0])         + (W['g'](P)*(1-v[1])**2 - W['a'](P)*(1-v[0])**2)/(R*T_fix) + dG_Fo_ag(P, T_fix)/(R*T_fix),
                         2*np.log((1-v[1])/(1-v[0])) + (W['g'](P)*v[1]**2     - W['a'](P)*v[0]**2)/(R*T_fix)     + dG_Fa_ag(P, T_fix)/(R*T_fix)], Pg_ag)
# stable parts: a+b and b+g above P3, a+g below P3
m_ab, m_bg, m_ag = Pg_ab > P3, Pg_bg > P3, Pg_ag < P3

fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.plot(ab[m_ab,0], Pg_ab[m_ab], '-',  color=COLORS[0], lw=2, label=r'$X^\alpha$ (a+b)')
ax1.plot(ab[m_ab,1], Pg_ab[m_ab], '--', color=COLORS[1], lw=2, label=r'$X^\beta$ (a+b)')
ax1.plot(bg[m_bg,0], Pg_bg[m_bg], '-',  color=COLORS[2], lw=2, label=r'$X^\beta$ (b+g)')
ax1.plot(bg[m_bg,1], Pg_bg[m_bg], '--', color=COLORS[3], lw=2, label=r'$X^\gamma$ (b+g)')
ax1.plot(ag[m_ag,0], Pg_ag[m_ag], '-',  color=COLORS[4], lw=2, label=r'$X^\alpha$ (a+g)')
ax1.plot(ag[m_ag,1], Pg_ag[m_ag], '--', color=COLORS[5], lw=2, label=r'$X^\gamma$ (a+g)')
ax1.legend(fontsize=10, loc='lower right')
ax1.plot([Xg, Xa], [P3, P3], 'k-', lw=3, zorder=5)
ax1.plot([Xa, Xb, Xg], [P3]*3, 'ko', ms=9, zorder=6)
ax1.set_xlim(0, 1); ax1.set_ylim(4, 22)
ax1.set_xlabel('Mg#  ($X_{Fo}$)', fontsize=14); ax1.set_ylabel('P (GPa)', fontsize=14)
ax1.set_title(f'fix T = {T_fix:.0f}  K', fontsize=14)
ax1.tick_params(labelsize=14); ax1.grid(alpha=0.3)

# ══ fig2: isobaric T-X at P_fix, all loops ═════════════════════════════════
# metastable solid-solid invariant point T3 at P_fix (unknowns Xa, Xb, Xg, T)
def equations_T(v):
    Xa, Xb, Xg, T = v
    Wa, Wb, Wg = W['a'](P_fix), W['b'](P_fix), W['g'](P_fix)
    return [2*np.log(Xb/Xa)         + (Wb*(1-Xb)**2 - Wa*(1-Xa)**2)/(R*T) + dG_Fo_ab(P_fix, T)/(R*T),
            2*np.log((1-Xb)/(1-Xa)) + (Wb*Xb**2     - Wa*Xa**2)/(R*T)     + dG_Fa_ab(P_fix, T)/(R*T),
            2*np.log(Xg/Xb)         + (Wg*(1-Xg)**2 - Wb*(1-Xb)**2)/(R*T) + dG_Fo_bg(P_fix, T)/(R*T),
            2*np.log((1-Xg)/(1-Xb)) + (Wg*Xg**2     - Wb*Xb**2)/(R*T)     + dG_Fa_bg(P_fix, T)/(R*T)]

Xa3, Xb3, Xg3, T3 = fsolve(equations_T, [0.7, 0.55, 0.35, 2400.])
print(f'fig2: solid-solid invariant at P = {P_fix:.0f} GPa:  T3 = {T3:.1f} K')

# end-member transition temperatures: dG(P_fix, T) = 0 -> T = (dH + dV P)/dS
T_Fo_ab = (DH_Fo_ab + 1000*DV_Fo_ab*P_fix)/DS_Fo_ab
T_Fa_ab = (DH_Fa_ab + 1000*DV_Fa_ab*P_fix)/DS_Fa_ab
T_Fo_bg = (DH_Fo_bg + 1000*DV_Fo_bg*P_fix)/DS_Fo_bg
T_Fa_bg = (DH_Fa_bg + 1000*DV_Fa_bg*P_fix)/DS_Fa_bg
T_Fo_ag = (DH_Fo_ag + 1000*DV_Fo_ag*P_fix)/DS_Fo_ag
T_Fa_ag = (DH_Fa_ag + 1000*DV_Fa_ag*P_fix)/DS_Fa_ag
print(f'  end-member T (Fo, Fa):  ab {T_Fo_ab:.1f} {T_Fa_ab:.1f}   bg {T_Fo_bg:.1f} {T_Fa_bg:.1f}   ag {T_Fo_ag:.1f} {T_Fa_ag:.1f}')
Tg_ab = np.linspace(T_Fo_ab, T_Fa_ab, 250)
Tg_bg = np.linspace(T_Fo_bg, T_Fa_bg, 250)
Tg_ag = np.linspace(T_Fo_ag, T_Fa_ag, 250)

Wa16, Wb16, Wg16 = W['a'](P_fix), W['b'](P_fix), W['g'](P_fix)
ab = trace(lambda v, T: [2*np.log(v[1]/v[0])         + (Wb16*(1-v[1])**2 - Wa16*(1-v[0])**2)/(R*T) + dG_Fo_ab(P_fix, T)/(R*T),
                         2*np.log((1-v[1])/(1-v[0])) + (Wb16*v[1]**2     - Wa16*v[0]**2)/(R*T)     + dG_Fa_ab(P_fix, T)/(R*T)], Tg_ab)
bg = trace(lambda v, T: [2*np.log(v[1]/v[0])         + (Wg16*(1-v[1])**2 - Wb16*(1-v[0])**2)/(R*T) + dG_Fo_bg(P_fix, T)/(R*T),
                         2*np.log((1-v[1])/(1-v[0])) + (Wg16*v[1]**2     - Wb16*v[0]**2)/(R*T)     + dG_Fa_bg(P_fix, T)/(R*T)], Tg_bg)
ag = trace(lambda v, T: [2*np.log(v[1]/v[0])         + (Wg16*(1-v[1])**2 - Wa16*(1-v[0])**2)/(R*T) + dG_Fo_ag(P_fix, T)/(R*T),
                         2*np.log((1-v[1])/(1-v[0])) + (Wg16*v[1]**2     - Wa16*v[0]**2)/(R*T)     + dG_Fa_ag(P_fix, T)/(R*T)], Tg_ag)

# ── melting ──────────────────────────────────────────────────────────────────
DH_FO_fus = 142000.0     # Richet et al. 1993, 142 +- 14 kJ/mol @ 2174 K
DH_FA_fus = 89300.0      # Richet et al. 1993, 89.3 +- 11 kJ/mol @ 1490 K

# Fo melting point: Birch-Murnaghan (K0 = 125.4 GPa, K' = 5.33) gives x = V/V0 at P,
#   f = ((1/x)^(2/3) - 1)/2,  P = 3 K0 f (1+2f)^2.5 (1 + 1.5 (K'-4) f)
#   Tm_Fo = 2163 (1 + 3 (1 - x))
K0, Kp = 125.4, 5.33
def P_of_x(x):
    f = ((1/x)**(2/3) - 1)/2
    return 3*K0*f*(1+2*f)**2.5 * (1 + 1.5*(Kp-4)*f) - P_fix
x16 = brentq(P_of_x, 0.5, 1.0)
Tm_Fo_16 = 2163.0*(1 + 3.0*(1 - x16))

# Fa melting data {P: (Tm, solid phase)}. A gamma melting point is converted to an effective
# alpha melting point (eq. 5.3): Tm_eff = Tm / (1 - dG_Fa_ag(P, Tm)/DH_FA_fus); alpha: Tm_eff = Tm
FA_MELT = {0.0: (1490., 'a'), 3.0: (1650., 'a'), 6.3: (1793., 'a'),
           12.0: (1990., 'g'), 20.0: (2200., 'g')}
Ps  = sorted(FA_MELT)
Tms = [FA_MELT[p][0] / (1.0 - (0.0 if FA_MELT[p][1] == 'a' else dG_Fa_ag(p, FA_MELT[p][0])) / DH_FA_fus)
       for p in Ps]
Tm_Fa_16 = np.interp(P_fix, Ps, Tms)      # P_fix = 16 lies between 12 and 20: interpolate
print(f'  Tm_Fo = {Tm_Fo_16:.1f} K   Tm_Fa(eff) = {Tm_Fa_16:.1f} K')

# solid phi + liquid (phi = alpha: no shift; beta, gamma: add -dG_(a->phi)/RT):
#   2 ln(XL/XS)         - W_MELT (1-XS)^2/RT - DH_Fo/R (1/Tm_Fo - 1/T) - dG_Fo_aphi/RT = 0
#   2 ln((1-XL)/(1-XS)) - W_MELT XS^2/RT     - DH_Fa/R (1/Tm_Fa - 1/T) - dG_Fa_aphi/RT = 0
Tg_aL = np.linspace(Tm_Fo_16*0.999, Tm_Fa_16*1.001, 300)
aL = trace(lambda v, T: [2*np.log(v[1]/v[0])         - W_MELT*(1-v[0])**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T),
                         2*np.log((1-v[1])/(1-v[0])) - W_MELT*v[0]**2/(R*T)     - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T)], Tg_aL)

# closure temperatures of the b+L and g+L loops: DH_fus (1 - T/Tm) = dG_(a->phi)(P_fix, T)
T_close_bL    = brentq(lambda T: DH_FO_fus*(1 - T/Tm_Fo_16) - dG_Fo_ab(P_fix, T), 1500., 2769.)
T_close_gL    = brentq(lambda T: DH_FO_fus*(1 - T/Tm_Fo_16) - dG_Fo_ag(P_fix, T), 1500., 2769.)
T_close_bL_Fa = brentq(lambda T: DH_FA_fus*(1 - T/Tm_Fa_16) - dG_Fa_ab(P_fix, T), 1200., 3000.)
T_close_gL_Fa = brentq(lambda T: DH_FA_fus*(1 - T/Tm_Fa_16) - dG_Fa_ag(P_fix, T), 1200., 3000.)
print(f'  closure T at Fo end: b+L {T_close_bL:.1f}  g+L {T_close_gL:.1f}')

Tg_bL = np.linspace(T_close_bL*0.999, T_close_bL_Fa*1.001, 300)
bL = trace(lambda v, T: [2*np.log(v[1]/v[0])         - W_MELT*(1-v[0])**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ab(P_fix, T)/(R*T),
                         2*np.log((1-v[1])/(1-v[0])) - W_MELT*v[0]**2/(R*T)     - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T) - dG_Fa_ab(P_fix, T)/(R*T)], Tg_bL)
Tg_gL = np.linspace(T_close_gL*0.999, T_close_gL_Fa*1.001, 300)
gL = trace(lambda v, T: [2*np.log(v[1]/v[0])         - W_MELT*(1-v[0])**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ag(P_fix, T)/(R*T),
                         2*np.log((1-v[1])/(1-v[0])) - W_MELT*v[0]**2/(R*T)     - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T) - dG_Fa_ag(P_fix, T)/(R*T)], Tg_gL)

fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.plot(ab[:,0], Tg_ab, '-',  color=COLORS[0],  lw=2, label=r'$X^\alpha$ (a+b)')
ax2.plot(ab[:,1], Tg_ab, '--', color=COLORS[1],  lw=2, label=r'$X^\beta$ (a+b)')
ax2.plot(bg[:,0], Tg_bg, '-',  color=COLORS[2],  lw=2, label=r'$X^\beta$ (b+g)')
ax2.plot(bg[:,1], Tg_bg, '--', color=COLORS[3],  lw=2, label=r'$X^\gamma$ (b+g)')
ax2.plot(ag[:,0], Tg_ag, '-',  color=COLORS[4],  lw=2, label=r'$X^\alpha$ (a+g)')
ax2.plot(ag[:,1], Tg_ag, '--', color=COLORS[5],  lw=2, label=r'$X^\gamma$ (a+g)')
ax2.plot(aL[:,0], Tg_aL, '-',  color=COLORS[6],  lw=2, label=r'$X^\alpha$ (a+L)')
ax2.plot(aL[:,1], Tg_aL, '--', color=COLORS[7],  lw=2, label=r'$X^L$ (a+L)')
ax2.plot(bL[:,0], Tg_bL, '-',  color=COLORS[8],  lw=2, label=r'$X^\beta$ (b+L)')
ax2.plot(bL[:,1], Tg_bL, '--', color=COLORS[9],  lw=2, label=r'$X^L$ (b+L)')
ax2.plot(gL[:,0], Tg_gL, '-',  color=COLORS[10], lw=2, label=r'$X^\gamma$ (g+L)')
ax2.plot(gL[:,1], Tg_gL, '--', color=COLORS[11], lw=2, label=r'$X^L$ (g+L)')
ax2.set_xlim(0, 1); ax2.set_ylim(1000, 3000)
ax2.set_xlabel(r'Mg#  ($X_{Fo}$)', fontsize=14); ax2.set_ylabel('T (K)', fontsize=14)
ax2.set_title(f'fix P = {P_fix:.0f} GPa : all loops', fontsize=14)
ax2.tick_params(labelsize=14); ax2.grid(alpha=0.3)
ax2.legend(fontsize=10, loc='lower right')

# ══ fig3: isobaric T-X at P_fix, stable diagram ════════════════════════════
# two melting three-phase points, each solved once (residual + physical ordering check)
# a+b+L: a+L pair and b+L pair share XL;  b+g+L: b+L pair and g+L pair share XL
def eqs4_abL(v):
    Xa_, Xb_, Xl_, T = v
    return [2*np.log(Xl_/Xa_) - W_MELT*(1-Xa_)**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T),
            2*np.log((1-Xl_)/(1-Xa_)) - W_MELT*Xa_**2/(R*T) - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T),
            2*np.log(Xl_/Xb_) - W_MELT*(1-Xb_)**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ab(P_fix, T)/(R*T),
            2*np.log((1-Xl_)/(1-Xb_)) - W_MELT*Xb_**2/(R*T) - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T) - dG_Fa_ab(P_fix, T)/(R*T)]

def eqs4_bgL(v):
    Xb_, Xg_, Xl_, T = v
    return [2*np.log(Xl_/Xb_) - W_MELT*(1-Xb_)**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ab(P_fix, T)/(R*T),
            2*np.log((1-Xl_)/(1-Xb_)) - W_MELT*Xb_**2/(R*T) - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T) - dG_Fa_ab(P_fix, T)/(R*T),
            2*np.log(Xl_/Xg_) - W_MELT*(1-Xg_)**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ag(P_fix, T)/(R*T),
            2*np.log((1-Xl_)/(1-Xg_)) - W_MELT*Xg_**2/(R*T) - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T) - dG_Fa_ag(P_fix, T)/(R*T)]

sol, info, ok, _ = fsolve(eqs4_abL, [0.95, 0.9, 0.7, 2500.], full_output=True)
Xa_abL, Xb_abL, Xl_abL, T_abL = sol
has_abL = ok == 1 and max(abs(info['fvec'])) < 1e-8 and 0 < Xl_abL < Xb_abL < Xa_abL < 1

sol, info, ok, _ = fsolve(eqs4_bgL, [0.85, 0.75, 0.45, 2300.], full_output=True)
Xb_bgL, Xg_bgL, Xl_bgL, T_bgL = sol
has_bgL = ok == 1 and max(abs(info['fvec'])) < 1e-8 and 0 < Xl_bgL < Xg_bgL < Xb_bgL < 1

print(f'fig3: has_abL = {has_abL} (T = {T_abL:.1f} K)   has_bgL = {has_bgL} (T = {T_bgL:.1f} K)')

fig3, ax3 = plt.subplots(figsize=(8, 8))
if has_abL and has_bgL:
    # high-P topology (e.g. 16 GPa): a+L / b+L / g+L stacked, separated by the two three-phase lines
    m_aL = Tg_aL > T_abL
    m_bL = (Tg_bL > T_bgL) & (Tg_bL < T_abL)
    m_gL = Tg_gL < T_bgL
    m_ab = Tg_ab < T_abL
    m_bg = Tg_bg < T_bgL
    ax3.plot(aL[m_aL,0], Tg_aL[m_aL], '-',  color=COLORS[6],  lw=2, label=r'$X^\alpha$ (a+L)')
    ax3.plot(aL[m_aL,1], Tg_aL[m_aL], '--', color=COLORS[7],  lw=2, label=r'$X^L$ (a+L)')
    ax3.plot(bL[m_bL,0], Tg_bL[m_bL], '-',  color=COLORS[8],  lw=2, label=r'$X^\beta$ (b+L)')
    ax3.plot(bL[m_bL,1], Tg_bL[m_bL], '--', color=COLORS[9],  lw=2, label=r'$X^L$ (b+L)')
    ax3.plot(gL[m_gL,0], Tg_gL[m_gL], '-',  color=COLORS[10], lw=2, label=r'$X^\gamma$ (g+L)')
    ax3.plot(gL[m_gL,1], Tg_gL[m_gL], '--', color=COLORS[11], lw=2, label=r'$X^L$ (g+L)')
    ax3.plot(ab[m_ab,0], Tg_ab[m_ab], '-',  color=COLORS[0],  lw=2, label=r'$X^\alpha$ (a+b)')
    ax3.plot(ab[m_ab,1], Tg_ab[m_ab], '--', color=COLORS[1],  lw=2, label=r'$X^\beta$ (a+b)')
    ax3.plot(bg[m_bg,0], Tg_bg[m_bg], '-',  color=COLORS[2],  lw=2, label=r'$X^\beta$ (b+g)')
    ax3.plot(bg[m_bg,1], Tg_bg[m_bg], '--', color=COLORS[3],  lw=2, label=r'$X^\gamma$ (b+g)')
    # the two three-phase tie lines
    ax3.plot([Xl_abL, Xa_abL], [T_abL]*2, 'k-', lw=3, zorder=5)
    ax3.plot([Xa_abL, Xb_abL, Xl_abL], [T_abL]*3, 'ko', ms=8, zorder=6)
    ax3.plot([Xl_bgL, Xb_bgL], [T_bgL]*2, 'k-', lw=3, zorder=5)
    ax3.plot([Xb_bgL, Xg_bgL, Xl_bgL], [T_bgL]*3, 'ko', ms=8, zorder=6)
    # metastable solid-solid T3 for reference
    ax3.plot([Xg3, Xa3], [T3]*2, ':', color='grey', lw=1.5, zorder=4)
else:
    # low-P topology (e.g. 1 atm): only a+L is stable, everything else is metastable
    ax3.plot(aL[:,0], Tg_aL, '-',  color=COLORS[6], lw=2, label=r'$X^\alpha$ (a+L)')
    ax3.plot(aL[:,1], Tg_aL, '--', color=COLORS[7], lw=2, label=r'$X^L$ (a+L)')
ax3.set_xlim(0, 1); ax3.set_ylim(1000, 3000)
ax3.set_xlabel(r'Mg#  ($X_{Fo}$)', fontsize=14); ax3.set_ylabel('T (K)', fontsize=14)
ax3.set_title(f'fix P = {P_fix:.0f} GPa : stable phase diagram', fontsize=14)
ax3.tick_params(labelsize=14); ax3.grid(alpha=0.3)
ax3.legend(fontsize=10, loc='lower right')
plt.show()
