#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:59:33 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Liu & Asimow 2025 (JGR-SE) liquid-metal EoS at MARTIAN core conditions.

Test 1: pure Fe endmember vs Kuwayama 2020 fitted isotherms (Table SII), 20-60 GPa
Test 2: mixing model extrapolated to stoichiometric FeS (n_S = 0.5) vs
        Sakai 2026 DAC densities, 25-55 GPa / 2320-3260 K
        -> this is a DOUBLE extrapolation: pressure (FPMD data >= ~90 GPa) and
           composition (Fe-S calibrated only up to Fe90S18, S = 16.7 mol%)

Model (Liu eq 8):
  P(V,T) = 3 K0 f (1+2f)^{5/2} [1 + 1.5 (K0'-4) f] + Cv (a_g + b_g/V)(T - T0)
  f = 0.5 [ (V0/V)^{2/3} - 1 ],  T0 = 4000 K,  V per 108 atoms in A^3
  thermal term: Cv [J/K] * (a_g + b_g/V) [A^-3] * 1e21 -> GPa/K

Mixing (Liu eq 17, 22): P(n,V,T) = sum_i n_i Gamma_i P_i(V,T),
  Gamma_i = d_i + (1 - d_i) n_i,   d_Fe = 0.382, d_S = 1  (equal V, T mixing)

Table 1 ambiguity handled as two readings:
  Reading A: S row = fit of the BINARY Fe90S18 -> invert mixing to get the
             virtual pure-S endmember, then remix at n_S = 0.5
  Reading B: S row = the virtual pure-S ENDMEMBER itself -> mix directly
(one of the questions for Weiyi; both shown)
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib

import matplotlib.pyplot as plt
from mars_config import DATA_BASE

# ── config ────────────────────────────────────────────────────────────────────
KUW_XLS   = f'{DATA_BASE}/mars_paper/Kuwayama_PRL_TableSII_data.xls'
OUT_PNG   = '/mnt/user-data/outputs/90_liu_eos_validation.png'
P_MARS    = (19.0, 56.0)      # GPa, Martian core range
T0        = 4000.0            # K, Liu reference temperature
AMU       = 1.66054e-27       # kg
M_FE, M_S = 55.845, 32.06     # g/mol

# Liu Table 1 (per 108 atoms): V0 [A^3], K0 [GPa], K0', Cv [1e-21 J/K],
#                              a_gamma [1e-4 A^-3], d_Gamma
LIU = {
    'Fe': dict(V0=2690.0, K0=1.20, K0p=18.7, Cv=7.28e-21, ag=8.05e-4, d=0.382),
    'S' : dict(V0=3170.0, K0=0.26, K0p=40.9, Cv=6.85e-21, ag=9.60e-4, d=1.0),
}
BG = 2.0/3.0                  # b_gamma, Thomas-Fermi limit

# Sakai 2026 Table 1: run, P (GPa), sig_P, T_average (K), sig_T, rho (kg/m3), sig_rho
SAKAI = np.array([
    [1, 24.9, 0.7, 2320, 120, 5281,  51],
    [2, 35.1, 1.1, 3040, 300, 5775,  36],
    [3, 40.0, 1.2, 2850, 290, 6155, 146],
    [4, 48.8, 1.5, 2400, 120, 6501,  44],
    [5, 54.9, 1.6, 3260, 330, 6588, 137],
    [6, 53.3, 1.6, 2710, 140, 6799, 146],
])

# ── Liu endmember pressure ────────────────────────────────────────────────────
def p_liu(V, T, pr):
    # V in A^3 per 108 atoms
    f = 0.5*((pr['V0']/V)**(2.0/3.0) - 1.0)
    p_cold = 3.0*pr['K0']*f*(1.0+2.0*f)**2.5*(1.0+1.5*(pr['K0p']-4.0)*f)
    p_th   = pr['Cv']*(pr['ag'] + BG/V)*1e21*(T - T0)   # GPa
    return p_cold + p_th

def gamma(d, n):                          # Liu eq 22
    return d + (1.0-d)*n

# FeS mixing, Reading B: table S row IS the pure-S endmember
def p_fes_B(V, T, nS=0.5):
    nFe = 1.0-nS
    return (nFe*gamma(LIU['Fe']['d'], nFe)*p_liu(V, T, LIU['Fe'])
            + nS*gamma(LIU['S']['d'],  nS)*p_liu(V, T, LIU['S']))

# FeS mixing, Reading A: table S row is the Fe90S18 binary fit
#   P_bin(V,T) = nFe' G_Fe(nFe') P_Fe + nS' P_S_end   with nFe' = 90/108
#   -> P_S_end = [P_bin - nFe' G_Fe(nFe') P_Fe] / nS'
def p_s_end_A(V, T):
    nFe_b, nS_b = 90.0/108.0, 18.0/108.0
    p_bin = p_liu(V, T, LIU['S'])
    return (p_bin - nFe_b*gamma(LIU['Fe']['d'], nFe_b)*p_liu(V, T, LIU['Fe']))/nS_b

def p_fes_A(V, T, nS=0.5):
    nFe = 1.0-nS
    return nFe*gamma(LIU['Fe']['d'], nFe)*p_liu(V, T, LIU['Fe']) + nS*p_s_end_A(V, T)

# ── invert P -> rho ───────────────────────────────────────────────────────────
def rho_of_p(p_func, P, T, m_kg, vlo=300.0, vhi=8000.0):
    g = lambda V: p_func(V, T) - P
    # bracket: P monotonic decreasing in V within the physical branch
    Vs = np.linspace(vlo, vhi, 400)
    Ps = np.array([p_func(v, T) for v in Vs])
    sign = np.sign(Ps - P)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[-1]                                   # low-pressure (physical) root
    V = brentq(g, Vs[i], Vs[i+1], xtol=1e-8)
    return m_kg/(V*1e-30)                          # kg/m3

M108_FE  = 108*M_FE*1e-3/6.02214076e23             # kg per 108 atoms
M108_FES = 54*(M_FE+M_S)*1e-3/6.02214076e23

# ── Test 1: Fe vs Kuwayama isotherms ─────────────────────────────────────────
raw = pd.read_excel(KUW_XLS, header=None)
num = raw.iloc[5:].apply(pd.to_numeric, errors='coerce')
ok  = num[0].notna()
Pk  = num.loc[ok, 0].values
rhoK = {2000: num.loc[ok, 2].values,
        3000: num.loc[ok, 3].values,
        4000: num.loc[ok, 4].values}

print('Test 1: pure Fe, Liu model vs Kuwayama 2020 isotherms (Mars range %g-%g GPa)'
      % P_MARS)
t1 = {}
for T in (2000, 3000, 4000):
    mask = (Pk >= P_MARS[0]) & (Pk <= P_MARS[1]) & np.isfinite(rhoK[T])
    res = []
    for P, rk in zip(Pk[mask], rhoK[T][mask]):
        rl = rho_of_p(lambda V, TT: p_liu(V, TT, LIU['Fe']), P, T, M108_FE)/1000.0
        res.append((P, rk, rl, 100.0*(rl-rk)/rk))
        print(f'  T={T}K P={P:5.1f}  rho_Kuw={rk:6.3f}  rho_Liu={rl:6.3f}  '
              f'd={100*(rl-rk)/rk:+5.2f}%')
    t1[T] = np.array(res)

# ── Test 2: FeS vs Sakai ─────────────────────────────────────────────────────
print('\nTest 2: FeS (n_S=0.5), Liu mixing model vs Sakai 2026')
rows = []
for run, P, sP, T, sT, r_exp, s_r in SAKAI:
    rB = rho_of_p(p_fes_B, P, T, M108_FES)
    rA = rho_of_p(p_fes_A, P, T, M108_FES)
    # propagate experiment P,T errors through model (reading B) for fair sigma
    drdP = (rho_of_p(p_fes_B, P+sP, T, M108_FES) - rB)
    drdT = (rho_of_p(p_fes_B, P, T+sT, M108_FES) - rB)
    s_eff = np.sqrt(s_r**2 + drdP**2 + drdT**2)
    rows.append((run, P, T, r_exp, s_r, rA, rB, s_eff))
    print(f'  #{int(run)} P={P:5.1f} T={T:4.0f}  exp={r_exp:5.0f}+-{s_r:3.0f}  '
          f'Liu(A)={rA:6.0f} ({100*(rA-r_exp)/r_exp:+5.1f}%)  '
          f'Liu(B)={rB:6.0f} ({100*(rB-r_exp)/r_exp:+5.1f}%)  '
          f'[(B-exp)/sig_eff = {(rB-r_exp)/s_eff:+.1f}]')
rows = np.array(rows)

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

cols = {2000: 'tab:blue', 3000: 'tab:orange', 4000: 'tab:green'}
for T in (2000, 3000, 4000):
    a = t1[T]
    ax[0].plot(a[:, 0], a[:, 3], 'o-', color=cols[T], label=f'{T} K')
ax[0].axhspan(-1.1, 1.1, color='0.85', zorder=0)         # ~DAC 1sigma (1.1%)
ax[0].axhspan(-2.2, 2.2, color='0.93', zorder=-1)        # 2sigma
ax[0].axhline(0, color='k', lw=0.8)
ax[0].set_xlabel('P (GPa)', fontsize=15)
ax[0].set_ylabel(r'$(\rho_{\rm Liu}-\rho_{\rm Kuw})/\rho_{\rm Kuw}$ (%)', fontsize=15)
ax[0].set_title('Test 1: liquid Fe, 19–56 GPa', fontsize=15)
ax[0].legend(fontsize=13)

x = rows[:, 1]
ax[1].errorbar(x, 100*(rows[:, 6]-rows[:, 3])/rows[:, 3],
               yerr=100*rows[:, 7]/rows[:, 3], fmt='s', ms=8, color='tab:red',
               label='Reading B (S row = endmember)')
ax[1].plot(x, 100*(rows[:, 5]-rows[:, 3])/rows[:, 3], 'D', ms=7, mfc='none',
           color='tab:purple', label='Reading A (S row = Fe$_{90}$S$_{18}$ fit)')
ax[1].axhspan(-2, 2, color='0.85', zorder=0)             # Sakai 1-2% -> 2%
ax[1].axhspan(-4, 4, color='0.93', zorder=-1)
ax[1].axhline(0, color='k', lw=0.8)
for run, P in zip(rows[:, 0], x):
    ax[1].annotate(f'#{int(run)}', (P, 1), fontsize=11, ha='center')
ax[1].set_xlabel('P (GPa)', fontsize=15)
ax[1].set_ylabel(r'$(\rho_{\rm Liu}-\rho_{\rm Sakai})/\rho_{\rm Sakai}$ (%)', fontsize=15)
ax[1].set_title('Test 2: liquid FeS, 25–55 GPa', fontsize=15)
ax[1].legend(fontsize=12)

for a in ax:
    a.tick_params(labelsize=14)
fig.tight_layout()

