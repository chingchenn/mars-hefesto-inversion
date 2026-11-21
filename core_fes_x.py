#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# core_fes_x.py : Fe-S core plus ONE extra light element (C or H), for 26fall_main_mcmc_a70.py
#
# core_fes.py is NOT modified (a63-a66 import it on beno3 at every 500-step restart).
# This module reuses its route-B tables (Liu Fe, Sakai FeS, Margules) and adds a third end-member:
#
#   'C'  liquid Fe3C, Samuel et al. 2023 SM Table 4 (= Irving et al. 2023 SM Table S5),
#        Anderson-Grueneisen form, data Terasaki 2010, Morard 2017, Shimoyama 2016 (<= 53 GPa)
#        Cv = Cp - alpha0^2 K_T0 V0 T0 at the reference state (same treatment as FeS)
#   'H'  liquid FeH. Tagawa, Helffrich, Hirose & Ohishi 2022 JGR-SE Table 2 (the FeH source cited
#        by Irving 2023 and Samuel 2023). Their liquid model is a melting-curve fit
#        (BM3; V_x = V_Fe + x V_H; K, K', alpha additive per mol H; K(T) = K0 exp(-K' int alpha dT);
#        reference 1 bar, 298 K). Their pure liquid Fe is ~5-8% too dense (rho 8.59 at 20 GPa,
#        2000 K vs Kuwayama 8.15), so only the H increment is used:
#            V_FeH(P,T) = V_Fe,Liu(P,T) + dV_H(P,T),  dV_H = V_Tag(x=1) - V_Tag(x=0)
#        (~1.5-1.7 cm3/mol H at 20-40 GPa).  Cv_FeH = Cv_Fe + 3R.
#
# Mixing (Samuel 2023 SM eq 12, all end-members, Margules only between Fe and FeS):
#   V = N [ sum_i chi_i V_i + chi_Fe chi_FeS (chi_FeS W_Fe-FeS(P) + chi_Fe W_FeS-Fe(P)) ]
#   per 100 g of alloy:  n_FeS = 100 w_S / M_S
#                        n_X   = 100 w_X / M_X        (X = C or H; one Fe3C per C, one FeH per H)
#                        n_Fe  = 100 (1 - w_S - w_X)/M_Fe - n_FeS - k n_X   (k = 3 for C, 1 for H)
#   n_Fe < 0 -> NaN (not enough Fe) -> sc_core returns None -> reject
# With w_X = 0 the call goes straight to core_fes.core_properties, so a70 with w_X = 0 is a63.
# Inner-core and liquidus (find_R_IC, gap_min, T_peritectic) stay Fe-S only.
import os
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import RegularGridInterpolator
import core_fes as CF

M_FE, M_S, M_C, M_H = CF.M_FE, CF.M_S, 12.011, 1.008
R_GAS = CF.R_GAS

FE3C = dict(form='AG', M=3*M_FE + M_C, V0=26.5, T0=1723.0, K_T0=57.5, Kp0=15.0,
            a0=14.8e-5, d0=9.1, kappa=1.4, Cp=215.0, n=4.0)

# Tagawa 2022 Table 2, liquid: V0 [J/bar] (x10 -> cm3/mol), alpha b1 [1/K], K [GPa], K'
TAG = dict(V0_Fe=0.70205, V0_H=0.06437, a_Fe=2.772e-5, a_H=7.385e-5,
           K_Fe=148.0, K_H=110.86, Kp_Fe=6.39, Kp_H=-1.90, T_ref=298.0)

LIGHT  = 'none'
_ITP_X = None


def V_tagawa(P, T, x):
    V0 = 10.0*(TAG['V0_Fe'] + x*TAG['V0_H'])
    a  = TAG['a_Fe'] + x*TAG['a_H']
    K0 = TAG['K_Fe'] + x*TAG['K_H']
    Kp = TAG['Kp_Fe'] + x*TAG['Kp_H']
    VT = V0*np.exp(a*(T - TAG['T_ref']))                # 1 bar volume at T
    KT = K0*np.exp(-Kp*a*(T - TAG['T_ref']))            # K(T) = K0 exp(-delta_T int alpha), delta_T = K'
    # BM3: P = 3 K f (1+2f)^(5/2) (1 + 3/2 (K'-4) f),  f = ((VT/V)^(2/3) - 1)/2
    f = lambda V: (3*KT*0.5*((VT/V)**(2/3) - 1)*((VT/V)**(2/3))**2.5
                   *(1 + 1.5*(Kp - 4)*0.5*((VT/V)**(2/3) - 1)) - P)
    return brentq(f, 0.3*VT, VT)


def configure(light, verbose=True):
    global LIGHT, _ITP_X
    LIGHT = light
    if light == 'none':
        _ITP_X = None
        return
    CF._ensure_tables()
    P_GRID, T_GRID = CF.P_GRID, CF.T_GRID
    cache = os.path.join(CF.CACHE_DIR, f'core_fes_x_tab_{CF.ROUTE}_{light}_{len(P_GRID)}x{len(T_GRID)}.npz')
    if os.path.exists(cache):
        tab = np.load(cache)['V']
        if verbose:
            print(f'core_fes_x: loaded {os.path.basename(cache)}', flush=True)
    else:
        if verbose:
            print(f'core_fes_x: building V table for light element {light} ...', flush=True)
        if light == 'C':
            # same as core_fes._build_table_AG, but an isobar whose ODE stops early (alpha runs away
            # at low P / very high T with delta_T = 9.1, far from any Mars core state) keeps NaN
            # there; those corners are filled along T below, as core_fes does
            from scipy.integrate import solve_ivp
            em  = FE3C
            tab = np.full((len(P_GRID), len(T_GRID)), np.nan)
            up, dn = T_GRID[T_GRID >= em['T0']], T_GRID[T_GRID < em['T0']][::-1]
            for i, P in enumerate(P_GRID):
                V_T0 = brentq(lambda V: CF._vinet(V, em['V0'], em['K_T0'], em['Kp0']) - P,
                              em['V0']*0.25, em['V0']*4.0, xtol=1e-13)
                rhs = lambda t, y: [CF._alpha_AG(y[0], em)*y[0]]
                s_up = solve_ivp(rhs, [em['T0'], up[-1]], [V_T0], t_eval=up, rtol=1e-10, atol=1e-12)
                s_dn = solve_ivp(rhs, [em['T0'], dn[-1]], [V_T0], t_eval=dn, rtol=1e-10, atol=1e-12)
                iu = np.flatnonzero(T_GRID >= em['T0'])
                idn = np.flatnonzero(T_GRID < em['T0'])[::-1]
                tab[i, iu[:s_up.y.shape[1]]]  = s_up.y[0]
                tab[i, idn[:s_dn.y.shape[1]]] = s_dn.y[0]
        else:
            PP, TT = np.meshgrid(P_GRID, T_GRID, indexing='ij')
            vfe = CF._ITP_FE(np.stack([PP.ravel(), TT.ravel()], axis=-1)).reshape(PP.shape)
            dvh = np.array([[V_tagawa(p, t, 1.0) - V_tagawa(p, t, 0.0) for t in T_GRID] for p in P_GRID])
            tab = vfe + dvh
        tab = CF._fill_nan_along_T(tab, f'V_{light}')
        np.savez_compressed(cache, V=tab)
    _ITP_X = RegularGridInterpolator((P_GRID, T_GRID), tab, method='cubic',
                                     bounds_error=False, fill_value=np.nan)
    if verbose:
        print(f'core_fes_x: light element {light} ready', flush=True)


def _V100(P, T, w_S, w_X):
    # volume of 100 g of alloy [cm3]; P, T same-shape arrays
    shp = np.shape(P)
    pts = np.stack([np.ravel(P), np.ravel(T)], axis=-1)
    vFe, vFeS, vX = (CF._ITP_FE(pts).reshape(shp), CF._ITP_FES(pts).reshape(shp),
                     _ITP_X(pts).reshape(shp))
    M_X, k = (M_C, 3.0) if LIGHT == 'C' else (M_H, 1.0)
    nFeS = 100*w_S/M_S
    nX   = 100*w_X/M_X
    nFe  = 100*(1 - w_S - w_X)/M_FE - nFeS - k*nX
    if nFe < 0:
        return np.full(shp, np.nan)
    N = nFe + nFeS + nX
    xFe, xFeS, xX = nFe/N, nFeS/N, nX/N
    V = xFe*vFe + xFeS*vFeS + xX*vX
    if not CF.IDEAL_MIXING:
        Pa = np.asarray(P, float)
        V = V + xFe*xFeS*(xFeS*CF.W_of_P(CF.W_FE_FES, Pa) + xFe*CF.W_of_P(CF.W_FES_FE, Pa))
    return N*V


def core_properties(P, T, w_S, w_X, dP=0.1, dT=10.0):
    if LIGHT == 'none' or w_X == 0.0:
        return CF.core_properties(P, T, w_S, dP=dP, dT=dT)
    P = np.asarray(P, float); T = np.asarray(T, float)
    V   = _V100(P,    T,    w_S, w_X)
    Vpp = _V100(P+dP, T,    w_S, w_X); Vpm = _V100(P-dP, T, w_S, w_X)
    Vtp = _V100(P,    T+dT, w_S, w_X); Vtm = _V100(P, T-dT, w_S, w_X)
    rho   = 100.0/V                                        # g/cm3
    K_T   = -V*(2*dP)/(Vpp - Vpm)                          # GPa
    alpha = (Vtp - Vtm)/(2*dT)/V
    # Cv per 100 g
    fe, fes = CF.FE(), CF.FES()
    cv_fe  = fe['Cv']
    cv_fes = fes['Cp'] - fes['a0']**2*(fes['K_T0']*1e9)*(fes['V0']*1e-6)*fes['T0']
    if LIGHT == 'C':
        cv_x, M_X, k = (FE3C['Cp'] - FE3C['a0']**2*(FE3C['K_T0']*1e9)*(FE3C['V0']*1e-6)*FE3C['T0'],
                        M_C, 3.0)
    else:
        cv_x, M_X, k = cv_fe + 3*R_GAS, M_H, 1.0
    nFeS = 100*w_S/M_S
    nX   = 100*w_X/M_X
    nFe  = 100*(1 - w_S - w_X)/M_FE - nFeS - k*nX
    cv   = nFe*cv_fe + nFeS*cv_fes + nX*cv_x               # J/K per 100 g
    gamma = alpha*(K_T*1e9)*(V*1e-6)/cv
    K_S   = K_T*(1.0 + alpha*gamma*T)
    Vp    = np.sqrt(np.clip(K_S, 0, None)*1e9/(rho*1e3))/1000.0
    return dict(rho=rho, Vp=Vp, K_T=K_T, K_S=K_S, alpha=alpha,
                gamma=gamma, dTdP=gamma*T/K_S, V=V)
