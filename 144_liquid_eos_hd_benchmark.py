#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:53:20 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
144: Hd-type liquid EoS for the BML, benchmarked against Pierru et al. (2026).

Part 1  Reproduce Pierru Table 6 (S3631 melt) for Asimow model 1 (Di-An-Fa-Fo-En)
        and model 2 (Di-An-Fa-Hd-En, negative Hd).
Part 2  Binary (Mg,Fe)2SiO4 liquid: Fo-Fa (current code) vs Hd-type
        (Fo -> Fa + 2Di - 2Hd), density and Vp.

EoS: BM3 or BM4 isentrope + Mie-Grueneisen, thermal pressure referenced to the
isentrope temperature T_S(rho) (same as _rho_endmember in 26fall_main_mcmc.py).
Parameters
  Fo      Thomas & Asimow 2013b Table 3 (3BM/MG)
  Fa, Di  Thomas & Asimow 2013b Table 4 (= Thomas et al. 2012), 3BM/MG
  An, En  Thomas & Asimow 2013b Table 4, 4BM/MG
  Hd      Thomas & Asimow 2013a Table 3 (3BM/MG). Cv printed as 11,306.70 in the
          paper is a typo: Cv = Cp/(1 + alpha*gamma0*T0) = 1306.7 J/kg/K.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
PIERRU_XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'Pierru 2026 data.xlsx'

EM = {
    'Fo': dict(rho0=2.597, KS0=16.41, Ksp=7.37,  Kpp=None,  gam0=0.396, q=-2.02, Cv=1737.36, T0=2273., M=140.694),
    'Fa': dict(rho0=3.699, KS0=21.99, Ksp=7.28,  Kpp=None,  gam0=0.412, q=-0.95, Cv=1122.73, T0=1573., M=203.777),
    'Di': dict(rho0=2.643, KS0=24.57, Ksp=6.77,  Kpp=None,  gam0=0.493, q=-1.28, Cv=1506.21, T0=1673., M=216.55),
    'An': dict(rho0=2.560, KS0=19.77, Ksp=3.73,  Kpp=0.38,  gam0=0.174, q=-1.86, Cv=1511.28, T0=1932., M=278.21),
    'En': dict(rho0=2.617, KS0=24.66, Ksp=10.07, Kpp=-2.35, gam0=0.365, q=-0.88, Cv=1690.53, T0=1673., M=100.39),
    'Hd': dict(rho0=2.913, KS0=19.89, Ksp=6.22,  Kpp=None,  gam0=0.300, q=-1.93, Cv=1306.70, T0=1673., M=248.09),
}


def rho_em(e, P_GPa, T_K):
    """End-member density (g/cm3) at P, T."""
    def f(r):
        x = 0.5*((r/e['rho0'])**(2/3) - 1)
        br = 1 + 1.5*(e['Ksp'] - 4)*x
        if e['Kpp'] is not None:
            br += 1.5*(e['KS0']*e['Kpp'] + (e['Ksp'] - 4)*(e['Ksp'] - 3) + 35/9)*x**2
        PS = 3*e['KS0']*x*(1 + 2*x)**2.5*br*1e9
        gam = e['gam0']*(e['rho0']/r)**e['q']
        T_S = e['T0']*np.exp(e['gam0']/e['q']*(1 - (e['rho0']/r)**e['q']))
        return PS + gam*r*1e3*e['Cv']*(T_K - T_S) - P_GPa*1e9
    return brentq(f, e['rho0']*0.85, e['rho0']*2.2, xtol=1e-12)


def mix_props(c, P, T):
    """Ideal volume mixing of end-members with mole coefficients c (may be negative).
    Returns rho (kg/m3), Vp (km/s), K_S (GPa)."""
    M = sum(c[k]*EM[k]['M'] for k in c)
    Cv = sum(c[k]*EM[k]['M']*EM[k]['Cv'] for k in c)/M
    V = lambda p, t: sum(c[k]*EM[k]['M']/rho_em(EM[k], p, t) for k in c)
    v = V(P, T)
    dP, dT = 0.05, 5.0
    K_T = -v*2*dP/(V(P + dP, T) - V(P - dP, T))*1e9
    alpha = (V(P, T + dT) - V(P, T - dT))/(2*dT)/v
    rho = M/v*1e3
    gam = alpha*K_T/(rho*Cv)
    K_S = K_T*(1 + alpha*gam*T)
    return rho, np.sqrt(K_S/rho)/1e3, K_S/1e9


# ── Part 1: Pierru benchmark ──────────────────────────────────────────────────
OX_ORDER = ['SiO2', 'Al2O3', 'FeO', 'MgO', 'CaO']
OX_M = dict(SiO2=60.084, Al2O3=101.961, FeO=71.844, MgO=40.304, CaO=56.077)
EM_OX = dict(Di=[2, 0, 0, 1, 1], An=[2, 1, 0, 0, 1], Fa=[1, 0, 2, 0, 0],
             Hd=[2, 0, 1, 0, 1], En=[1, 0, 0, 1, 0], Fo=[1, 0, 0, 2, 0])
S3631 = dict(SiO2=37.37, Al2O3=1.13, FeO=37.63, MgO=21.07, CaO=2.8)   # Pierru Table 5, mass balance


def decompose(wt, ems):
    n = np.array([wt[k]/OX_M[k] for k in OX_ORDER])
    x = np.linalg.solve(np.array([EM_OX[e] for e in ems]).T, n)
    return dict(zip(ems, x/x.sum()))


def part1():
    c1 = decompose(S3631, ['Di', 'An', 'Fa', 'Fo', 'En'])
    c2 = decompose(S3631, ['Di', 'An', 'Fa', 'Hd', 'En'])
    print('model 1:', {k: round(v, 3) for k, v in c1.items()})
    print('model 2:', {k: round(v, 3) for k, v in c2.items()})
    if not PIERRU_XLSX.exists():
        print(f'{PIERRU_XLSX} not found; skipping comparison with Pierru Table 6')
        return
    import openpyxl
    ws = openpyxl.load_workbook(PIERRU_XLSX, data_only=True, read_only=True)['Table 6 Density']
    rows = [r for r in ws.iter_rows(values_only=True)
            if isinstance(r[0], (int, float)) and isinstance(r[3], (int, float))]
    print(f"{'P':>4} {'T':>7} | {'Pierru1':>7} {'here1':>6} {'d%':>6} | {'Pierru2':>7} {'here2':>6} {'d%':>6}")
    for r in rows:
        P, T, p1, p2 = r[0], r[1], r[3], r[5]
        m1 = mix_props(c1, P, T)[0]
        m2 = mix_props(c2, P, T)[0]
        print(f'{P:4.0f} {T:7.1f} | {p1:7.0f} {m1:6.0f} {100*(m1-p1)/p1:+6.2f} | '
              f'{p2:7.0f} {m2:6.0f} {100*(m2-p2)/p2:+6.2f}')


# ── Part 2: binary BML liquid ────────────────────────────────────────────────
def coeffs(XL, model):
    if model == 'binary':
        return {'Fo': XL, 'Fa': 1 - XL}
    return {'Fa': 1.0, 'Di': 2*XL, 'Hd': -2*XL}        # Fo -> Fa + 2Di - 2Hd


def part2():
    print(' P     T   XL | rho_bin rho_Hd  drho% | Vp_bin Vp_Hd   dVp% | KS_Hd')
    for P, T in [(19, 2300), (21, 2400), (23, 2500)]:
        for XL in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            rb, vb, _ = mix_props(coeffs(XL, 'binary'), P, T)
            rh, vh, kh = mix_props(coeffs(XL, 'hd'), P, T)
            print(f'{P:2d} {T:5d} {XL:4.1f} | {rb:7.0f} {rh:6.0f} {100*(rh-rb)/rb:+6.2f} | '
                  f'{vb:6.3f} {vh:5.3f} {100*(vh-vb)/vb:+6.2f} | {kh:5.1f}')


if __name__ == '__main__':
    part1()
    print()
    part2()