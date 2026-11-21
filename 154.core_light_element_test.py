#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 154: forward test before a70/a71 (Fe-S-C, Fe-S-H core). How much do C and H raise core Vp at fixed density?
#
#   python 154.core_carbon_test.py
#
# Take a64 posterior samples (P_cmb, T_core, w_S) and work at the core top (P_cmb, T_core).
# Route B as in core_fes.py (Liu Fe + Sakai FeS, Margules), plus liquid Fe3C from
# Samuel et al. 2023 SM Table 4 (Terasaki 2010, Morard 2017, Shimoyama 2016 data), mixed ideally.
#   Samuel eq 12 on all end-members:
#   V = N [ sum chi_i V_i + chi_Fe chi_FeS (chi_FeS W_Fe-FeS(P) + chi_Fe W_FeS-Fe(P)) ]
#   with C = 0 this is exactly core_fes._V_mix_arr (checked in step 1)
# For each sample and each w_C: solve for w_S' so that rho_cmb(w_S', w_C) = rho_cmb(w_S, 0),
#   i.e. keep the density that mass and MoI asked for, then dVp = Vp(w_S', w_C) - Vp(w_S, 0).
# H: liquid FeH_x from Tagawa, Helffrich, Hirose & Ohishi 2022 JGR-SE Table 2 (the EoS Irving 2023 and
#   Samuel 2023 cite for FeH). Their liquid model is a melting-curve fit (BM3, V_x = V_Fe + x V_H,
#   K, K', alpha additive per mol H, K(T) = K0 exp(-K' int alpha dT), reference 1 bar / 298 K).
#   Their pure liquid Fe is ~5-8% too dense (rho 8.59 at 20 GPa/2000 K vs Kuwayama 8.15), so only the
#   H contribution is taken:  V_FeH(P,T) = V_Fe,Liu(P,T) + dV_H(P,T),
#   dV_H = V_Tagawa(x=1) - V_Tagawa(x=0)   (~1.5-1.7 cm3/mol H at 20-40 GPa)
#   FeH mixes ideally with Fe (Irving/Samuel: Margules only for Fe-FeS, Fe-FeO). Cv_FeH = Cv_Fe + 3R.
#
# Literature core-top values for comparison (approximate, read from figures / project notes):
#   Irving 2023 (no BML, SKS):  Vp_cmb ~4.85-5.05 km/s at rho_cmb ~5.9-6.15 g/cm3 (Fig 3, 50% region)
#   Samuel 2023 (BML):          Vp_cmb ~5.0 km/s, rho_mean ~6.5 g/cm3
#   Khan 2023 (BML):            Vp_cmb ~5.3 km/s, rho_mean ~6.5-6.65 g/cm3
import os, json
import numpy as np
from scipy.optimize import brentq
from mars_config import MCMC_DIR
import core_fes as CF

PREFIX  = 'chain_a64_YM20'
CHAINS  = range(16)
BURN    = 1000
N_SAMP  = 200                     # posterior samples, evenly thinned
W_C     = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]
W_H     = [0.0025, 0.005, 0.0075, 0.01]
DP, DT  = 0.1, 10.0               # GPa, K for the numerical derivatives
SEED    = 1

M_FE, M_S, M_C, M_H = 55.845, 32.060, 12.011, 1.008
FE3C = dict(form='AG', M=3*M_FE + M_C, V0=26.5, T0=1723.0, K_T0=57.5, Kp0=15.0,
            a0=14.8e-5, d0=9.1, kappa=1.4, Cp=215.0, n=4.0)

CF.configure(route='B', verbose=False)
FE, FES = CF.FE(), CF.FES()
cv_fe   = FE['Cv']
cv_fes  = FES['Cp'] - FES['a0']**2*(FES['K_T0']*1e9)*(FES['V0']*1e-6)*FES['T0']
cv_fe3c = FE3C['Cp'] - FE3C['a0']**2*(FE3C['K_T0']*1e9)*(FE3C['V0']*1e-6)*FE3C['T0']
cv_feh  = cv_fe + 3*8.314462
print(f'Cv per mol: Fe {cv_fe:.1f}  FeS {cv_fes:.1f}  Fe3C {cv_fe3c:.1f}  FeH {cv_feh:.1f} J/K/mol')


def V_tagawa(P, T, x):
    # Tagawa 2022 Table 2, liquid FeH_x; V0 in J/bar -> x10 cm3/mol
    V0 = 7.0205 + 0.6437*x
    a  = 2.772e-5 + 7.385e-5*x
    K0 = 148.0 + 110.86*x
    Kp = 6.39 - 1.90*x
    VT = V0*np.exp(a*(T - 298.0))
    KT = K0*np.exp(-Kp*a*(T - 298.0))
    def f(V):
        e = 0.5*((VT/V)**(2.0/3.0) - 1.0)
        return 3*KT*e*(1 + 2*e)**2.5*(1 + 1.5*(Kp - 4)*e) - P
    return brentq(f, 0.3*VT, VT)

# ── posterior samples ───────────────────────────────────────────────────────
recs = []
for i in CHAINS:
    for line in open(os.path.join(MCMC_DIR, f'{PREFIX}_{i:02d}', 'chain.jsonl')):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break
        if r['step'] > BURN and r.get('P_cmb') is not None:
            recs.append(r)
rng = np.random.default_rng(SEED)
pick = rng.choice(len(recs), N_SAMP, replace=False)
S = [recs[j] for j in pick]
print(f'{PREFIX}: {len(recs)} records after burn-in, {N_SAMP} picked')


# ── mixture at one (P, T) from end-member volumes ───────────────────────────
# Vend[k] = (V_Fe, V_FeS, V_Fe3C, V_FeH) at the 5 stencil points: 0 (P,T), 1 (P+dP), 2 (P-dP), 3 (T+dT), 4 (T-dT)
def mix(Vend, P, T, w_S, w_C, w_H=0.0):
    nFeS, nFe3C, nFeH = 100*w_S/M_S, 100*w_C/M_C, 100*w_H/M_H
    nFe = 100*(1 - w_S - w_C - w_H)/M_FE - nFeS - 3*nFe3C - nFeH
    if nFe < 0:
        return None
    N = nFe + nFeS + nFe3C + nFeH
    x = np.array([nFe, nFeS, nFe3C, nFeH])/N
    Pk = np.array([P, P + DP, P - DP, P, P])
    V = np.empty(5)
    for k in range(5):
        V[k] = N*(x @ Vend[k] + x[0]*x[1]*(x[1]*CF.W_of_P(CF.W_FE_FES, Pk[k])
                                          + x[0]*CF.W_of_P(CF.W_FES_FE, Pk[k])))
    rho   = 100.0/V[0]                                   # g/cm3 (100 g of alloy)
    K_T   = -V[0]*2*DP/(V[1] - V[2])                     # GPa
    alpha = (V[3] - V[4])/(2*DT)/V[0]
    cv    = nFe*cv_fe + nFeS*cv_fes + nFe3C*cv_fe3c + nFeH*cv_feh   # J/K per 100 g
    gamma = alpha*(K_T*1e9)*(V[0]*1e-6)/cv
    K_S   = K_T*(1 + alpha*gamma*T)
    Vp    = np.sqrt(K_S*1e9/(rho*1e3))/1000.0
    return rho, Vp


out = []
for r in S:
    P, T, wS = r['P_cmb'], r['params']['T_core'], r['params']['w_S']
    stencil = [(P, T), (P + DP, T), (P - DP, T), (P, T + DT), (P, T - DT)]
    Vend = np.array([[CF.V_endmember_exact(p, t, em) for em in (FE, FES, FE3C)] for p, t in stencil])
    dVH  = np.array([V_tagawa(p, t, 1.0) - V_tagawa(p, t, 0.0) for p, t in stencil])
    Vend = np.column_stack([Vend, Vend[:, 0] + dVH])
    if not np.all(np.isfinite(Vend)):
        continue
    rho0, vp0 = mix(Vend, P, T, wS, 0.0)
    row = dict(P=P, T=T, wS=wS, rho0=rho0, vp0=vp0, vp_rec=r['Vp_core_cmb'],
               rho_mean=r['rho_core_mean'],
               state=('none' if r['params']['BML_thickness'] < 1 else 'solid' if r['h_liquid_km'] < 1
                      else 'molten' if r['h_solid_km'] < 1 else 'layered'))   # same as 153
    for wc in W_C:
        f = lambda s: mix(Vend, P, T, s, wc)[0] - rho0 if mix(Vend, P, T, s, wc) else np.nan
        # adding C lowers rho, so less S is needed; search w_S' in [0, w_S]
        # upper bound also limited by free Fe >= 0: (1-s-wc)/M_FE >= s/M_S + 3 wc/M_C
        s_max = (1 - wc - 3*wc*M_FE/M_C)/(1 + M_FE/M_S)
        lo, hi = 0.0, min(wS, s_max - 1e-6)
        if hi <= lo or f(lo)*f(hi) > 0:
            row[wc] = (np.nan, np.nan)
            continue
        s2 = brentq(f, lo, hi, xtol=1e-6)
        row[wc] = (s2, mix(Vend, P, T, s2, wc)[1] - vp0)
    row['dVH'] = dVH[0]
    for wh in W_H:
        g = lambda s: mix(Vend, P, T, s, 0.0, wh)[0] - rho0 if mix(Vend, P, T, s, 0.0, wh) else np.nan
        s_max = (1 - wh - wh*M_FE/M_H)/(1 + M_FE/M_S)
        lo, hi = 0.0, min(wS, s_max - 1e-6)
        if hi <= lo or g(lo)*g(hi) > 0:
            row[('H', wh)] = (np.nan, np.nan)
            continue
        s2 = brentq(g, lo, hi, xtol=1e-6)
        row[('H', wh)] = (s2, mix(Vend, P, T, s2, 0.0, wh)[1] - vp0)
    out.append(row)
print(f'{len(out)} samples evaluated')

# ── 1. check: C = 0 reproduces the run's own Vp_core_cmb ────────────────────
d = np.array([o['vp0'] - o['vp_rec'] for o in out])
print(f'\n1. check vs chain Vp_core_cmb (table + interpolation in the run): '
      f'diff median {np.median(d):+.4f}, max |diff| {np.max(np.abs(d)):.4f} km/s')

# ── 2. a64 core top as it is (Fe-S) ─────────────────────────────────────────
q = lambda x: np.percentile(x, [5, 50, 95])
for key, lab in (('vp0', 'Vp_cmb [km/s]'), ('rho0', 'rho_cmb [g/cm3]'), ('rho_mean', 'rho_mean [g/cm3]'),
                 ('wS', 'w_S'), ('P', 'P_cmb [GPa]'), ('T', 'T_core [K]')):
    a = q([o[key] for o in out])
    print(f'  {lab:>18}  5/50/95 = {a[0]:.3f} {a[1]:.3f} {a[2]:.3f}')
print('  targets: Irving ~4.85-5.05 (rho_cmb 5.9-6.15), Samuel ~5.0, Khan ~5.3 km/s')

# ── 3. adding C at fixed rho_cmb ────────────────────────────────────────────
print('\n3. add C, adjust S to keep rho_cmb:  dVp and resulting Vp_cmb (5/50/95), new w_S')
print(f"{'w_C':>6} {'n ok':>5} {'dVp 5/50/95 [km/s]':>24} {'Vp_cmb 5/50/95':>22} {'w_S median':>11} {'dS median':>10}")
for wc in W_C:
    ok = [o for o in out if np.isfinite(o[wc][1])]
    if not ok:
        print(f'{wc:6.3f}     0')
        continue
    dv = q([o[wc][1] for o in ok])
    vp = q([o['vp0'] + o[wc][1] for o in ok])
    s2 = np.median([o[wc][0] for o in ok])
    ds = np.median([o[wc][0] - o['wS'] for o in ok])
    print(f'{wc:6.3f} {len(ok):5d}   {dv[0]:+.3f} {dv[1]:+.3f} {dv[2]:+.3f}     '
          f'{vp[0]:.3f} {vp[1]:.3f} {vp[2]:.3f}   {s2:9.3f} {ds:+10.3f}')
print('  C solubility in Fe-S liquid is limited (Irving 2023 / Tsuno 2018: ~1-1.5 wt% at ~15 wt% S)')

# ── 3b. adding H at fixed rho_cmb ───────────────────────────────────────────
print(f'\n3b. add H (Tagawa 2022 dV_H, median {np.median([o["dVH"] for o in out]):.3f} cm3/mol H), adjust S to keep rho_cmb')
print(f"{'w_H':>6} {'n ok':>5} {'dVp 5/50/95 [km/s]':>24} {'Vp_cmb 5/50/95':>22} {'w_S median':>11} {'dS median':>10}")
for wh in W_H:
    ok = [o for o in out if np.isfinite(o[('H', wh)][1])]
    if not ok:
        print(f'{wh:6.4f}     0')
        continue
    dv = q([o[('H', wh)][1] for o in ok])
    vp = q([o['vp0'] + o[('H', wh)][1] for o in ok])
    s2 = np.median([o[('H', wh)][0] for o in ok])
    ds = np.median([o[('H', wh)][0] - o['wS'] for o in ok])
    print(f'{wh:6.4f} {len(ok):5d}   {dv[0]:+.3f} {dv[1]:+.3f} {dv[2]:+.3f}     '
          f'{vp[0]:.3f} {vp[1]:.3f} {vp[2]:.3f}   {s2:9.3f} {ds:+10.3f}')
print('  Irving/Khan composition models use ~0.5-2 wt% H; Samuel 2023 BML set up to 0.5 wt%')

# ── 4. by BML state ─────────────────────────────────────────────────────────
print('\n4. by BML state (median Vp_cmb with no C/H, with 1 wt% C, with 0.5 wt% H)')
for st in ('solid', 'layered', 'molten'):
    sel = [o for o in out if o['state'] == st]
    if sel:
        v1 = [o['vp0'] + o[0.01][1] for o in sel if np.isfinite(o[0.01][1])]
        vh = [o['vp0'] + o[('H', 0.005)][1] for o in sel if np.isfinite(o[('H', 0.005)][1])]
        print(f'  {st:>8} n={len(sel):3d}  Vp0 {np.median([o["vp0"] for o in sel]):.3f}  '
              f'rho_cmb {np.median([o["rho0"] for o in sel]):.3f}  '
              f'Vp(1%C) {np.median(v1) if v1 else np.nan:.3f}  Vp(0.5%H) {np.median(vh) if vh else np.nan:.3f}')
