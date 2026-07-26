#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMC inversion for Mars interior structure using nGibbs (HeFESTo emulator) + BML physics
"""

import os
import re
import numpy as np
import json
import argparse
import glob
import sys
import torch
from pathlib import Path
from datetime import datetime
from scipy.optimize import fsolve, brentq
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

NGIBBS_ROOT = Path('/home/jcchen2/nGibbs/')
sys.path.insert(0, str(NGIBBS_ROOT / "src"))
sys.path.insert(0, str(NGIBBS_ROOT / "src" / "ngibbs"))

from ngibbs.engine.API import HeFESToMarsEmulatorCPU as EM
if torch.cuda.is_available():
    from ngibbs.engine.API import HeFESToMarsEmulatorGPU as EM

from mars_config import *

NG_T_HEADERS = ['P(GPa)(System_main)', 'T(K)(System_main)',
                'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
NG_S_HEADERS = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)',
                'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
NG_S_MIN     = 1.75
NG_S_MAX     = 2.90
NG_N_SHALLOW = 100
NG_N_DEEP    = 101

# per-proposal S_lit log (includes proposals rejected for being out of range)
_NG_S_LOG = {'last': np.nan, 'last_in_range': True, 'all': [], 'n_below': 0, 'n_above': 0}

# ── constants ─────────────────────────────────────────────────────────────────
MARS_RADIUS      = 3389.5
MARS_RADIUS_M    = MARS_RADIUS * 1000
T_SURF           = 220.0
MARS_MASS_OBS    = 6.4171e23
MARS_MASS_SIGMA  = MARS_MASS_OBS * 0.01
MOI_OBS          = 0.36379   # Konopliv et al. 2016 (used in Samuel 2021/2023)
MOI_SIGMA        = 0.0005    
MOI_BIAS         = 0.00136   # pipeline systematic: pure-Samuel PanelJ through this
                      # integrator gives MoI=0.36515 vs obs 0.36379 (budget test)
MCMC_TEMPERATURE = 1.0    
_DTDP_LIQ = 10.0      # K/GPa, liquid BML adiabat slope

TRUE_CMB_DEPTH = None   # km  true CMB

# ── MCMC parameters ──────────────────────────────────────────────────────────
_MG_FE_TOTAL = 5.16834   # Mg + Fe mol (Yoshida-Motoyama)
_COMP_FIXED  = {'Si': 4.01931, 'Ca': 0.27259,
                'Al': 0.37376, 'Na': 0.10105, 'Cr': 0.06146}

START_PARAMS = {
'T_lit': 1539.0,
'P_lit': 3.69,
'Mg#': 4.08235 / 5.16834,
'T_core': 2400.0, # CMB temperature = liquid BML temperature
'Mg#_bulk_bml': 0.62, # bulk BML composition (Samuel ~Fe-rich)
'BML_thickness': 168.7,
}

PRIOR = {
'T_lit': (1000.0, 2600.0),
'P_lit': (1.4, 9.0),
'Mg#': (0.50, 0.86),
'T_core': (1800.0, 3200.0),
'Mg#_bulk_bml': (0.30, 0.80),
'BML_thickness': (50.0, 400.0),
}

STEP = {
'T_lit': 50.0,
'P_lit': 0.7,
'Mg#': 0.025,
'T_core': 100.0,
'Mg#_bulk_bml': 0.06,
'BML_thickness': 25.0,
}

# ── Samuel 2023 paths ─────────────────────────────────────────────────────────
SAMUEL_DATA_DIR = ("/net/beno3/data1/jcchen2/Mars_Samuel_2023/"
                   "Nature_Samuel_s41586-023-06601-8/"
                   "METADATA_BML/DATA_FIG2/PANEL_B")

SAMUEL_RHO_PROFILE_PATH = ("/net/beno3/data1/jcchen2/Mars_Samuel_2023/"
                            "Nature_Samuel_s41586-023-06601-8/"
                            "METADATA_BML/DATA_FIG1/PANEL_J/rho_profile.dat")

SAMUEL_FIG1K_DIR = ("/net/beno3/data1/jcchen2/Mars_Samuel_2023/"
                    "Nature_Samuel_s41586-023-06601-8/"
                    "METADATA_BML/DATA_FIG1/PANEL_K")

# ── seismic data ──────────────────────────────────────────────────────────────
SAMUEL_DATA = {
    'S0154a': {
        'delta': 29.7,  'depth': 17.4,
        'SS-S':  (25.3, 5.0),
        'SSS-S': (35.0, 8.0),
    },
    'S0173a': {
        'delta': 30.9,  'depth': 28.4,
        'S-P':   (178.8,  5.0),
        'sP-P':  (9.43,   5.0),
        'PP-P':  (19.9,   8.0),
        'PPP-P': (34.4,  12.0),
        'sS-S':  (13.2,   5.0),
        'SS-S':  (24.4,   5.0),
        'SSS-S': (40.5,   8.0),
        'ScS-S': (345.2, 12.0),
    },
    'S0185a': {
        'delta': 54.8,  'depth': 17.4,
        'S-P':   (327.28,  5.0),
        'pP-P':  (4.0,     5.0),
        'PP-P':  (22.47,   8.0),
        'PPP-P': (49.3,   12.0),
        'sS-S':  (10.0,    5.0),
        'SS-S':  (30.9,    5.0),
        'SSS-S': (55.4,    8.0),
        'ScS-S': (152.3,  12.0),
    },
    'S0235b': {
        'delta': 30.5,  'depth': 26.1,
        'S-P':   (171.4,  5.0),
        'PP-P':  (18.6,   8.0),
        'PPP-P': (32.0,  12.0),
        'sS-S':  (9.2,    5.0),
        'SS-S':  (23.2,   5.0),
        'SSS-S': (33.3,   8.0),
        'ScS-S': (343.9, 12.0),
    },
    'S0325a': {
        'delta': 42.0,  'depth': 33.8,
        'S-P':   (229.3,  5.0),
        'pP-P':  (9.8,    5.0),
        'PP-P':  (21.1,   8.0),
        'PPP-P': (34.4,  12.0),
        'sS-S':  (13.8,   5.0),
        'SS-S':  (26.1,   5.0),
        'SSS-S': (50.3,   8.0),
        'ScS-S': (220.4, 12.0),
    },
    'S0407a': {
        'delta': 29.1,  'depth': 31.3,
        'S-P':   (170.7,  5.0),
        'pP-P':  (6.77,   5.0),
        'PP-P':  (23.38,  8.0),
        'sS-S':  (13.3,   5.0),
        'SS-S':  (21.1,   5.0),
        'SSS-S': (33.1,   8.0),
        'ScS-S': (370.0, 12.0),
    },
    'S0409d': {
        'delta': 30.6,  'depth': 26.1,
        'S-P':   (163.2,  5.0),
        'pP-P':  (8.3,    5.0),
        'PP-P':  (27.6,   8.0),
        'PPP-P': (36.94, 12.0),
        'sS-S':  (8.4,    5.0),
        'SS-S':  (20.9,   5.0),
        'SSS-S': (39.8,   8.0),
        'ScS-S': (320.1, 12.0),
    },
    'S0474a': {
        'delta': 20.7,  'depth': 30.7,
        'S-P':   (121.6,  5.0),
        'PP-P':  (13.4,   8.0),
        'sS-S':  (9.7,    5.0),
        'SS-S':  (16.9,   5.0),
        'SSS-S': (25.8,   8.0),
        'ScS-S': (455.0, 12.0),
    },
    'S0484b': {
        'delta': 26.0,  'depth': 30.1,
        'S-P':   (151.0,  5.0),
        'PP-P':  (16.0,   8.0),
        'PPP-P': (29.4,  12.0),
        'sS-S':  (11.6,   5.0),
        'SS-S':  (19.6,   5.0),
        'SSS-S': (33.4,   8.0),
        'ScS-S': (360.0, 12.0),
    },
    'S0820a': {
        'delta': 75.2,  'depth': 26.1,
        'S-P':   (437.5,  5.0),
        'SS-S':  (48.4,   5.0),
        'SSS-S': (82.3,   8.0),
        'SS-PP': (22.6,  10.0),
    },
    'S1000a': {
        'delta': 87.0,  'depth': 27.0,
        'S-P':         (490.4,   5.0),
        'SS-S':        (54.9,    5.0),
        'SSS-S':       (93.3,    8.0),
        'SKS-PP':      (87.0,   10.0),
        'PP-PbdiffPcP': (22.5,  10.0),
    },
    'S1222a': {
        'delta': 36.1,  'depth': 32.8,
        'S-P':   (216.0,  5.0),
        'ScS-S': (258.0, 12.0),
    },
}

# ── gravity and pressure profiles (computed once at import) ───────────────────
def _detect_metal_cmb_from_density(rho, r_km):
    depth = MARS_RADIUS - r_km
    o = np.argsort(depth)
    depth, rho = depth[o], rho[o]
    return float(depth[np.argmax(np.abs(np.diff(rho))) + 1])

def _build_gravity_pressure():
    G = 6.674e-11
    data = np.loadtxt(SAMUEL_RHO_PROFILE_PATH)
    rho, r_km = data[:, 0], data[:, 1]
    cmb_depth = _detect_metal_cmb_from_density(rho, r_km)

    idx = np.argsort(r_km)
    rho, r = rho[idx], r_km[idx] * 1000.0   # m, center→surface

    M = np.zeros(len(r))
    for i in range(1, len(r)):
        M[i] = M[i-1] + 4*np.pi * ((rho[i]+rho[i-1])/2) * ((r[i]+r[i-1])/2)**2 * (r[i]-r[i-1])
    g = np.zeros(len(r));  g[1:] = G * M[1:] / r[1:]**2

    depth = (r[-1] - r) / 1000.0   # km, surface=0
    idx2  = np.argsort(depth)
    depth, g, rho = depth[idx2], g[idx2], rho[idx2]

    P = np.zeros(len(depth))
    for i in range(1, len(depth)):
        P[i] = P[i-1] + ((rho[i]+rho[i-1])/2) * ((g[i]+g[i-1])/2) * (depth[i]-depth[i-1]) * 1e3 / 1e9

    # also return rho profile in g/cm³ for samuel median
    rho_gcm3 = rho / 1000.0
    print("DEBUG cmb_depth from density =", cmb_depth)
    return depth, g, depth.copy(), P, depth.copy(), rho_gcm3, cmb_depth  

_grav_depth, _grav_g, _pres_depth, _pres_gpa, _rho_depth, _rho_all, TRUE_CMB_DEPTH = _build_gravity_pressure()

def gravity_mars(depth_km):  return np.interp(depth_km, _grav_depth, _grav_g)
def pressure_mars(depth_km): return np.interp(depth_km, _pres_depth, _pres_gpa)

# ── Samuel 2023 median model ──────────────────────────────────────────────────
def _load_samuel_reference():
    vp_data = np.loadtxt(os.path.join(SAMUEL_FIG1K_DIR, 'vp_profile.dat'))
    vs_data = np.loadtxt(os.path.join(SAMUEL_FIG1K_DIR, 'vs_profile.dat'))
    depth = MARS_RADIUS - vp_data[:, 1]      # km
    vp    = vp_data[:, 0] / 1000.0           # km/s
    vs    = vs_data[:, 0] / 1000.0           # km/s
    o = np.argsort(depth)
    depth, vp, vs = depth[o], vp[o], vs[o]

    crust_z = np.linspace(0.0, 100.0, 200)
    core_z  = np.linspace(TRUE_CMB_DEPTH, MARS_RADIUS, 200)  

    solid = vs >= 0.1
    crust_vp = np.interp(crust_z, depth[solid], vp[solid])
    crust_vs = np.interp(crust_z, depth[solid], vs[solid])

    core_sel = depth >= TRUE_CMB_DEPTH
    core_vp  = np.interp(core_z, depth[core_sel], vp[core_sel])


    liquid = vs < 0.1
    bml_top_depth = float(depth[liquid].min()) if liquid.any() else np.nan

    crust_rho = np.interp(crust_z, _rho_depth, _rho_all)
    core_rho  = np.interp(core_z,  _rho_depth, _rho_all)

    print(f"Samuel Fig1 reference: metal CMB={TRUE_CMB_DEPTH:.1f} km  "
          f"apparent CMB / BML top (Vs->0)={bml_top_depth:.1f} km")
    return {
        'crust_z':        crust_z,
        'crust_vp':       crust_vp,
        'crust_vs':       crust_vs,
        'crust_rho':      crust_rho,
        'core_z':         core_z,
        'core_vp':        core_vp,
        'core_vs':        np.zeros(len(core_z)),
        'core_rho':       core_rho,
        'true_cmb_depth': TRUE_CMB_DEPTH,
        'bml_top_depth':  bml_top_depth,   
    }

_samuel_cache = _load_samuel_reference()

# ── physics: solidus / liquidus / melt fraction ───────────────────────────────
MG_DUNCAN_REF  = 0.75

# Kono 2025: the BML must be denser than the mantle above and lighter than the core
# below, else it overturns on a timescale << 4.5 Gyr. This is a hard rejection.

REJECT_GRAV_UNSTABLE = True
_GRAV_FAIL = {'upper': 0, 'lower': 0}

def solidus_duncan2018(P_GPa):
    P = float(P_GPa)
    if P <= 10.0:
        T_C = -4.877*P**2 + 120.2*P + 1088.0
    elif P <= 23.0:
        T_C = -1.323*(P-10.0)**2 + 38.18*(P-10.0) + 1802.0
    else:
        T_C = 77.75*(P-23.0) + 2075.0
    return T_C + 273.15


def solidus_duncan_Mg(P_GPa, Mg):
    # Duncan 2018 solidus + Samuel 2021 iron correction (-6 K per Fe%).
    Fe     = 100.0 * (1.0 - Mg)
    Fe_ref = 100.0 * (1.0 - MG_DUNCAN_REF)
    return solidus_duncan2018(P_GPa) + (-6.0 * (Fe - Fe_ref))


# ── BML phase diagram (Fo-Fa, Pierru 2026) ─────────────────────
_R_GAS = 8.314
 
# ── Akaogi 1989 solid-solid data ─────────────────────────────────────────────
_dv298 = {('Fo','ab'): -3.16, ('Fo','bg'): -0.98,
          ('Fa','ab'): -3.20, ('Fa','bg'): -1.04, ('Fa','ag'): -4.24}
_tp = {('Fo','ab'): (14.2, 1600.), ('Fo','bg'): (19.0, 1473.), ('Fa','ag'): (5.25, 1273.)}
_dS = {('Fo','ab'): -7.7, ('Fo','bg'): -7.3, ('Fa','ab'): -10.9, ('Fa','bg'): -3.1}
_dH = {('Fo','ab'): 29970, ('Fo','bg'): 9080, ('Fa','ab'): 9620, ('Fa','bg'): -5790}
_dV = {}
for _k in [('Fo','ab'), ('Fo','bg')]:
    _P0, _T0 = _tp[_k]; _dV[_k] = -(_dH[_k] - _T0*_dS[_k]) / (_P0*1000.)
_P0, _T0 = _tp[('Fa','ag')]
_DHa = _dH[('Fa','ab')] + _dH[('Fa','bg')]; _DSa = _dS[('Fa','ab')] + _dS[('Fa','bg')]
_DVa = -(_DHa - _T0*_DSa) / (_P0*1000.)
for _k in [('Fa','ab'), ('Fa','bg')]:
    _dV[_k] = _DVa * _dv298[_k] / _dv298[('Fa','ag')]
 
# alpha->gamma (Hess) and alpha->beta Gibbs shifts
_DH_Fo_ag = _dH[('Fo','ab')] + _dH[('Fo','bg')]
_DS_Fo_ag = _dS[('Fo','ab')] + _dS[('Fo','bg')]
_DV_Fo_ag = _dV[('Fo','ab')] + _dV[('Fo','bg')]
_DH_Fa_ag = _dH[('Fa','ab')] + _dH[('Fa','bg')]
_DS_Fa_ag = _dS[('Fa','ab')] + _dS[('Fa','bg')]
_DV_Fa_ag = _dV[('Fa','ab')] + _dV[('Fa','bg')]
 
def _dG_Fo_ab(P, T): return _dH[('Fo','ab')] - T*_dS[('Fo','ab')] + _dV[('Fo','ab')]*1000.0*P
def _dG_Fa_ab(P, T): return _dH[('Fa','ab')] - T*_dS[('Fa','ab')] + _dV[('Fa','ab')]*1000.0*P
def _dG_Fo_ag(P, T): return _DH_Fo_ag - T*_DS_Fo_ag + _DV_Fo_ag*1000.0*P
def _dG_Fa_ag(P, T): return _DH_Fa_ag - T*_DS_Fa_ag + _DV_Fa_ag*1000.0*P
 
_W_GAMMA   = 6200.0      # SLB2024 ringwoodite Margules
_W_BETA    = 12900.0     # SLB2024 wadsleyite Margules
_DH_FO_FUS = 142000.0
_DH_FA_FUS = 89300.0
_FA_MELT = {0.0:(1490.,'a'), 3.0:(1650.,'a'), 6.3:(1793.,'a'),
            12.0:(1990.,'g'), 20.0:(2200.,'g')}
 
def _Tm_Fo(P):
    K0, Kp = 125.4, 5.33
    def f(x):
        ff = ((1/x)**(2/3) - 1)/2
        return 3*K0*ff*(1+2*ff)**2.5 * (1 + 1.5*(Kp-4)*ff) - P
    return 2163.0 * (1 + 3.0*(1 - brentq(f, 0.5, 1.0)))
 
def _Tm_eff(P, Tm, ph):
    d = 0.0 if ph == 'a' else _dG_Fa_ag(P, Tm)
    return Tm / (1.0 - d/_DH_FA_FUS)
 
_FA_P = sorted(_FA_MELT)
def _Tm_Fa(P):
    return float(np.interp(P, _FA_P, [_Tm_eff(p, *_FA_MELT[p]) for p in _FA_P]))
 
# ── generic solid+L two-phase solver (robust multistart) ──────────────────────
def _loop_XS_XL(T, P, TmFo, TmFa, W, dGFo, dGFa, guesses):
    def eqs(v):
        Xs, Xl = np.clip(v, 1e-9, 1-1e-9)
        e1 = 2*np.log(Xl/Xs)         + (-W*(1-Xs)**2)/(_R_GAS*T) \
             - _DH_FO_FUS/_R_GAS*(1/TmFo - 1/T) - dGFo(P,T)/(_R_GAS*T)
        e2 = 2*np.log((1-Xl)/(1-Xs)) + (-W*Xs**2)/(_R_GAS*T) \
             - _DH_FA_FUS/_R_GAS*(1/TmFa - 1/T) - dGFa(P,T)/(_R_GAS*T)
        return [e1, e2]
    for x0 in guesses:
        s, info, ier, _ = fsolve(eqs, x0, full_output=True)
        Xs, Xl = s
        if ier == 1 and np.max(np.abs(info['fvec'])) < 1e-8 and 0 < Xl < Xs < 1:
            return float(Xs), float(Xl)
    return np.nan, np.nan
 
_GUESS_G = ([0.85,0.65],[0.70,0.45],[0.55,0.32],[0.40,0.22],
            [0.28,0.14],[0.18,0.08],[0.10,0.04],[0.92,0.78])
_GUESS_B = ([0.97,0.75],[0.93,0.60],[0.90,0.50],[0.88,0.45],[0.85,0.55],[0.80,0.40])
 
def _gamma_XS_XL(T, P, TmFo, TmFa):
    return _loop_XS_XL(T, P, TmFo, TmFa, _W_GAMMA, _dG_Fo_ag, _dG_Fa_ag, _GUESS_G)
def _beta_XS_XL(T, P, TmFo, TmFa):
    return _loop_XS_XL(T, P, TmFo, TmFa, _W_BETA,  _dG_Fo_ab, _dG_Fa_ab, _GUESS_B)
 
# ── beta-gamma-L three-phase invariant ────────────────────────────────────────
def _three_phase_bg(P, TmFo, TmFa):
    def eqs(v):
        Xb, Xg, Xl, T = v
        return [2*np.log(Xl/Xb) + (-_W_BETA*(1-Xb)**2)/(_R_GAS*T)
                - _DH_FO_FUS/_R_GAS*(1/TmFo-1/T) - _dG_Fo_ab(P,T)/(_R_GAS*T),
                2*np.log((1-Xl)/(1-Xb)) + (-_W_BETA*Xb**2)/(_R_GAS*T)
                - _DH_FA_FUS/_R_GAS*(1/TmFa-1/T) - _dG_Fa_ab(P,T)/(_R_GAS*T),
                2*np.log(Xl/Xg) + (-_W_GAMMA*(1-Xg)**2)/(_R_GAS*T)
                - _DH_FO_FUS/_R_GAS*(1/TmFo-1/T) - _dG_Fo_ag(P,T)/(_R_GAS*T),
                2*np.log((1-Xl)/(1-Xg)) + (-_W_GAMMA*Xg**2)/(_R_GAS*T)
                - _DH_FA_FUS/_R_GAS*(1/TmFa-1/T) - _dG_Fa_ag(P,T)/(_R_GAS*T)]
    for T0 in np.linspace(2000., 2900., 12):
        s, info, ier, _ = fsolve(eqs, [0.84, 0.71, 0.44, T0], full_output=True)
        if (ier == 1 and np.max(np.abs(info['fvec'])) < 1e-8
                and 0 < s[2] < s[1] < s[0] < 1 and 1500 < s[3] < 3500):
            return dict(Xb=float(s[0]), Xg=float(s[1]), XL=float(s[2]), T=float(s[3]))
    return None
 
# ── rigid multicomponent shift, anchored to Pierru gamma tie-line ─────────────
_PIERRU_TIE = dict(P=17.7, Tsol=2200., XS=0.74, XL=0.50)
 
def _raw_solidus_gamma(P, Xref):
    TmFo, TmFa = _Tm_Fo(P), _Tm_Fa(P)
    Ts = np.linspace(1900., 3200., 400)
    XS = np.array([_gamma_XS_XL(t, P, TmFo, TmFa)[0] for t in Ts])
    ok = np.isfinite(XS); Ts, XS = Ts[ok], XS[ok]
    if len(XS) < 2 or not (XS.min() <= Xref <= XS.max()):
        return np.nan
    o = np.argsort(XS)
    return float(np.interp(Xref, XS[o], Ts[o]))
 
def _compute_shift():
    raw = _raw_solidus_gamma(_PIERRU_TIE['P'], _PIERRU_TIE['XS'])
    shift = _PIERRU_TIE['Tsol'] - raw
    TmFo, TmFa = _Tm_Fo(_PIERRU_TIE['P']), _Tm_Fa(_PIERRU_TIE['P'])
    _, XL = _gamma_XS_XL(raw, _PIERRU_TIE['P'], TmFo, TmFa)
    return shift, raw, XL
 
_DELTA_SHIFT, _RAW_SOL_REF, _XL_TIE_MODEL = _compute_shift()
 
# ── diagnostics ───────────────────────────────────────────────────────────────
_PD_FAIL    = {'no_loop': 0, 'no_invariant': 0, 'solidus': 0}
_PHASE_USED = {'gamma': 0, 'three-phase': 0, 'beta': 0}
 
def _scan(loopf, P, TmFo, TmFa, Tlo=1600., Thi=3200., n=300):
    Tp = np.linspace(Tlo, Thi, n); raw = Tp - _DELTA_SHIFT
    XS = np.empty(n); XL = np.empty(n)
    for i, tr in enumerate(raw):
        XS[i], XL[i] = loopf(tr, P, TmFo, TmFa)
    ok = np.isfinite(XS) & np.isfinite(XL) & (XL < XS)
    return Tp[ok], XS[ok], XL[ok]
 
# ── main entry (same dict contract as old bml_phase_diagram, + 'phase') ───────
def bml_phase_diagram(T_interface, P_GPa, Mg_bulk):
    TmFo, TmFa = _Tm_Fo(P_GPa), _Tm_Fa(P_GPa)
    tp = _three_phase_bg(P_GPa, TmFo, TmFa)
 
    def _branches(Tsol, Tliq, loop_at, phase):
        if T_interface <= Tsol:
            return {'melting': False, 'XS': Mg_bulk, 'XL': np.nan,
                    'f_solid': 1.0, 'f_liquid': 0.0,
                    'T_solidus': Tsol, 'T_liquidus': Tliq, 'phase': phase}
        if T_interface >= Tliq:
            return {'melting': True, 'XS': np.nan, 'XL': Mg_bulk,
                    'f_solid': 0.0, 'f_liquid': 1.0,
                    'T_solidus': Tsol, 'T_liquidus': Tliq, 'phase': phase}
        XS_i, XL_i = loop_at(T_interface)
        if not (np.isfinite(XS_i) and np.isfinite(XL_i)) or abs(XS_i - XL_i) < 1e-6:
            return None
        f = float(np.clip((Mg_bulk - XS_i) / (XL_i - XS_i), 0.0, 1.0))
        return {'melting': True, 'XS': XS_i, 'XL': XL_i,
                'f_solid': 1.0 - f, 'f_liquid': f,
                'T_solidus': Tsol, 'T_liquidus': Tliq, 'phase': phase}
 
    def _sol_liq(loopf, loop_at, phase, Tsol_override=None):
        Tg, XS, XL = _scan(loopf, P_GPa, TmFo, TmFa)
        if len(XS) < 3:
            _PD_FAIL['no_loop'] += 1
            return None
        o = np.argsort(XS); o2 = np.argsort(XL)
        if Tsol_override is None:
            if not (XS.min() <= Mg_bulk <= XS.max()):
                _PD_FAIL['solidus'] += 1
                return None
            Tsol = float(np.interp(Mg_bulk, XS[o], Tg[o]))
        else:
            Tsol = Tsol_override
        if not (XL.min() <= Mg_bulk <= XL.max()):
            _PD_FAIL['solidus'] += 1
            return None
        Tliq = float(np.interp(Mg_bulk, XL[o2], Tg[o2]))
        return _branches(Tsol, max(Tliq, Tsol + 1.0), loop_at, phase)
 
    ga = lambda T: _gamma_XS_XL(T - _DELTA_SHIFT, P_GPa, TmFo, TmFa)
    ba = lambda T: _beta_XS_XL(T - _DELTA_SHIFT, P_GPa, TmFo, TmFa)
 
    if tp is None:
        # no invariant (single-phase topology at this P) -> gamma only
        _PD_FAIL['no_invariant'] += 1
        r = _sol_liq(_gamma_XS_XL, ga, 'gamma(no-inv)')
        if r: _PHASE_USED['gamma'] += 1
        return r
 
    Xg, Xb, T3 = tp['Xg'], tp['Xb'], tp['T'] + _DELTA_SHIFT
    if Mg_bulk <= Xg:
        r = _sol_liq(_gamma_XS_XL, ga, 'gamma')
        if r: _PHASE_USED['gamma'] += 1
        return r
    if Mg_bulk >= Xb:
        r = _sol_liq(_beta_XS_XL, ba, 'beta')
        if r: _PHASE_USED['beta'] += 1
        return r
    # three-phase region: solidus = T3 (invariant), above T3 -> beta+L residual solid
    r = _sol_liq(_beta_XS_XL, ba, 'three-phase(bg)->beta', Tsol_override=T3)
    if r: _PHASE_USED['three-phase'] += 1
    return r
 
 
def bml_melt_fraction(T_K, P_GPa, Mg_bulk):
    pd = bml_phase_diagram(T_K, P_GPa, Mg_bulk)
    return 0.0 if pd is None else float(pd['f_liquid'])
 

# ── BML thermal state: Ra → Case 2+3 (conductive) or Case 4 (convective) ──────
_ETA0    = 1e21      # Pa·s, reference viscosity
_ESTAR   = 3e5       # J/mol, activation energy
_T0_ETA  = 1600.0   # K, reference temperature
_ALPHA_S = 2.0e-5   # K⁻¹, thermal expansion (solid BML)
_RHO_S   = 3800.0   # kg/m³
_KAPPA_S = 1.0e-6   # m²/s, thermal diffusivity
_CP_S    = 1200.0   # J/kg/K
_VSTAR = 3.0e-6      # m³/mol  (Drilleau 2026 Table 1: V* 0–10 cm³/mol, inverted)
_P_REF = 3.0e9       # Pa      (Drilleau 2021/2026 reference pressure)
# Ra_cr. 1e3 is the classical CONSTANT-viscosity Rayleigh-Benard value (657.5
# free-free, 1707.8 rigid-rigid, plane layer) and does not apply here: with
# E* = 3e5 J/mol the viscosity contrast across the solid BML is ~4e3, and Ra_cr
# rises with E* in variable-viscosity spherical geometry (Yanagisawa 2016).
# Samuel 2021 SM S4 adopt 1e5 as a LOWER bound for exactly this reason.
# >>> This is a modelling CHOICE you must defend at the exam. Test the sensitivity.
_RA_C  = 1.0e5

def compute_bml_thermal_state(T_mantle_bottom, T_interface, h_solid_km, bml_top_km,
                              rho_solid=_RHO_S):
    """Ra of the solid sublayer. Boundary temps are FIXED by the neighbours
    (top: mantle bottom, bottom: solid/liquid interface). Ra selects the
    internal profile SHAPE only — it never modifies boundary temperatures."""
    z_mid = bml_top_km + h_solid_km / 2.0
    P_mid = float(pressure_mars(z_mid)) * 1e9
    T_mid = 0.5 * (T_mantle_bottom + T_interface)
    dT    = max(T_interface - T_mantle_bottom, 0.0)
    eta   = _ETA0 * np.exp((_ESTAR + P_mid*_VSTAR) / (_R_GAS * T_mid)
                           - (_ESTAR + _P_REF*_VSTAR) / (_R_GAS * _T0_ETA))
    h     = h_solid_km * 1000.0
    g     = gravity_mars(z_mid)
    Ra    = _ALPHA_S * rho_solid * g * dT * h**3 / (_KAPPA_S * eta)
    return Ra, ('conductive' if Ra < _RA_C else 'convective')

# ── Liquid BML EoS (Thomas 2012 Fa + Thomas 2013 Fo, linear mixing) ───────────
def liquid_bml_properties(P_GPa, T_K, Mg_liquid):
    # Fo end-member: Thomas & Asimow 2013, 3BM/MG, T0=2273K
    rho0_Fo=2.806; KS0_Fo=17.5; Ksp_Fo=5.3; gam0_Fo=0.62; q_Fo=-0.8; Cv_Fo=900.0; T0_Fo=2273.0
    # Fa end-member: Thomas et al. 2012, 3BM/MG, T0=1573K
    rho0_Fa=3.699; KS0_Fa=21.99; Ksp_Fa=7.28; gam0_Fa=0.412; q_Fa=-0.95; Cv_Fa=670.0; T0_Fa=1573.0

    Mg   = float(np.clip(Mg_liquid, 0.0, 1.0))
    rho0 = Mg*rho0_Fo + (1-Mg)*rho0_Fa
    KS0  = Mg*KS0_Fo  + (1-Mg)*KS0_Fa
    Ksp  = Mg*Ksp_Fo  + (1-Mg)*Ksp_Fa
    gam0 = Mg*gam0_Fo + (1-Mg)*gam0_Fa
    q    = Mg*q_Fo    + (1-Mg)*q_Fa
    Cv   = Mg*Cv_Fo   + (1-Mg)*Cv_Fa
    T0   = Mg*T0_Fo   + (1-Mg)*T0_Fa
    P_Pa = P_GPa * 1e9

    def P_of_rho(rho):
        f   = 0.5 * ((rho/rho0)**(2/3) - 1)
        PS  = 3*KS0*f*(1+2*f)**2.5 * (1 + 1.5*(Ksp-4)*f) * 1e9
        gam = gam0 * (rho0/rho)**q
        ES  = 4.5*KS0/rho0*(f**2 + (Ksp-4)*f**3) * 1e9   # J/kg
        Pth = gam * rho*1e3 * Cv * (T_K - T0)
        return PS + Pth - P_Pa

    try:
        rho = brentq(P_of_rho, rho0*0.5, rho0*3.0, xtol=1e-6)
    except Exception:
        return rho0, 4.0, 0.0

    dr  = rho * 0.001
    KS  = abs(rho * (P_of_rho(rho+dr) - P_of_rho(rho-dr)) / (2*dr)) / 1e9   # GPa
    Vp  = float(np.sqrt(KS * 1e9 / (rho * 1e3))) / 1000.0   # km/s
    return float(rho), Vp, 0.0

# ── composition ───────────────────────────────────────────────────────────────
def composition_from_params(params):
    Mg = params['Mg#'] * _MG_FE_TOTAL
    Fe = (1.0 - params['Mg#']) * _MG_FE_TOTAL
    return {**_COMP_FIXED, 'Mg': Mg, 'Fe': Fe,
            'T_lit': params['T_lit'], 'P_lit': params['P_lit']}


def compute_oxygen(p):
    return (2.0*p['Si'] + p['Mg'] + p['Fe'] + p['Ca'] +
            1.5*p['Al'] + 0.5*p['Na'] + 1.5*p['Cr'])

# ── nGibbs helpers ────────────────────────────────────────────────────────────
def _ng_props(component_moles, P, T, names):
    PT = np.stack([np.asarray(P, dtype=np.float64),
                   np.asarray(T, dtype=np.float64)], axis=1)
    return EM.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(np.asarray(component_moles), dtype=torch.float64),
        torch.tensor(PT, dtype=torch.float64),
        property_names=names,
    )


def _ng_forward(P, T_or_S, comp_values, headers, outputs):
    x = np.zeros((len(P), 2 + len(comp_values)), dtype=np.float32)
    x[:, 0]  = P
    x[:, 1]  = T_or_S
    x[:, 2:] = comp_values
    with torch.no_grad():
        return EM.ForwardMB(x, headers=headers, outputs=outputs)


def _comp_values(p, O):
    return [p['Si'], p['Mg'], p['Fe'], p['Ca'],
            p['Al'], p['Na'], p['Cr'], O]


def run_ngibbs(params, P_bml_top=None):
    p = composition_from_params(params)
    O = compute_oxygen(p)
    T_lit, P_lit = p['T_lit'], p['P_lit']
    comp_values = _comp_values(p, O)

    if P_bml_top is None:
        bml_top_km = TRUE_CMB_DEPTH - params.get('BML_thickness', 168.7)
        P_bml_top  = float(pressure_mars(bml_top_km))

    # step 1: S_lit at (P_lit, T_lit)
    out1  = _ng_forward([P_lit], [T_lit], comp_values,
                        NG_T_HEADERS, ['component_moles'])
    S_lit = float(_ng_props(out1['component_moles'],
                            [P_lit], [T_lit], ['S'])['S'][0])

    in_range = np.isfinite(S_lit) and NG_S_MIN <= S_lit <= NG_S_MAX
    _NG_S_LOG['last']          = S_lit
    _NG_S_LOG['last_in_range'] = in_range
    _NG_S_LOG['all'].append(S_lit)
    if np.isfinite(S_lit) and S_lit < NG_S_MIN: _NG_S_LOG['n_below'] += 1
    if np.isfinite(S_lit) and S_lit > NG_S_MAX: _NG_S_LOG['n_above'] += 1

    if not in_range:
        print(f"  S_lit={S_lit:.4f} outside nGibbs training range "
              f"[{NG_S_MIN}, {NG_S_MAX}]")
        return None
    print(f"  S_lit={S_lit:.6f}  T={T_lit:.1f}K  P={P_lit:.2f}GPa")

    # pressure grid: surface -> P_bml_top
    P = np.concatenate([np.linspace(0.01, 3.0, NG_N_SHALLOW),
                        np.linspace(3.0, P_bml_top, NG_N_DEEP)[1:]])
    n    = len(P)
    isen = P >= P_lit
    iso  = ~isen

    T   = np.zeros(n)
    rho = np.zeros(n)
    Vp  = np.zeros(n)
    Vs  = np.zeros(n)

    # step 2: isentropic asthenosphere (T(P) directly, no bisection)
    out2   = _ng_forward(P[isen], np.full(isen.sum(), S_lit), comp_values,
                         NG_S_HEADERS, ['component_moles', 'temperature'])
    T_isen = np.asarray(out2['temperature'].detach().cpu(), dtype=np.float64)
    T[isen] = T_isen
    pr = _ng_props(out2['component_moles'], P[isen], T_isen, ['rho', 'Vp', 'Vs'])
    rho[isen] = pr['rho']
    Vp[isen]  = pr['Vp']
    Vs[isen]  = pr['Vs']

    # step 3: conductive lithosphere, T linear in P
    T_iso  = T_SURF + (T_lit - T_SURF) * (P[iso] / P_lit)
    T[iso] = T_iso
    out3 = _ng_forward(P[iso], T_iso, comp_values,
                       NG_T_HEADERS, ['component_moles'])
    pr = _ng_props(out3['component_moles'], P[iso], T_iso, ['rho', 'Vp', 'Vs'])
    rho[iso] = pr['rho']
    Vp[iso]  = pr['Vp']
    Vs[iso]  = pr['Vs']

    if not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        print("  nGibbs returned invalid density")
        return None

    print(f"  Adiabat: T={T_isen[0]:.1f}K@{P[isen][0]:.2f}GPa -> "
          f"T={T[-1]:.1f}K@{P[-1]:.2f}GPa")

    if not (np.all(np.isfinite(Vp)) and np.all(np.isfinite(Vs)) and np.all(np.isfinite(rho))):
        print(f"  nGibbs 產生 NaN/Inf → reject  (T_lit={params.get('T_lit', -1):.0f} "
              f"T_core={params.get('T_core', -1):.0f} Mg#={params.get('Mg#', -1):.3f})")
        return None

    return {
        'depth_km':  np.interp(P, _pres_gpa, _pres_depth),
        'P_GPa':     P,
        'T_K':       T,
        'S':         np.full(n, S_lit),
        'Vp':        Vp,
        'Vs':        Vs,
        'rho':       rho,
        'P_profile': P,
        'T_profile': T,
    }


def run_ngibbs_bml(params, T_core, T_mantle_bottom, true_cmb_depth,
                   rho_mantle_bottom=_RHO_S, n_points=20):
    bml_thickness = params['BML_thickness']
    Mg_bulk       = params['Mg#_bulk_bml']
    bml_top_depth = true_cmb_depth - bml_thickness
    P_top         = max(float(pressure_mars(bml_top_depth)), 5.0)
    P_bottom      = float(pressure_mars(true_cmb_depth))
    P_mid         = (P_top + P_bottom) / 2.0

    # ── self-consistent iteration (pure analytical, < 0.01s) ─────────────────
    T_interface   = T_core       # initial guess: conductive → T_interface = T_core
    h_solid_km    = bml_thickness
    h_liquid_km   = 0.0
    Mg_solid      = Mg_bulk
    Mg_liquid     = np.nan
    Ra            = np.nan
    thermal_state = 'undefined'
    pd            = {'melting': False, 'T_solidus': np.nan}

    for _ in range(10):
        T_old = T_interface
        pd = bml_phase_diagram(T_interface, P_mid, Mg_bulk)
        if pd is None:
            return None
        if not pd['melting']:                       # 全固: 層底直接接核
            h_solid_km, h_liquid_km = bml_thickness, 0.0
            Mg_solid, Mg_liquid    = Mg_bulk, np.nan
            T_interface = T_core
            break
        elif pd['f_solid'] < 1e-3:                  # 全熔
            h_solid_km, h_liquid_km = 0.0, bml_thickness
            Mg_solid, Mg_liquid    = np.nan, Mg_bulk
            Ra, thermal_state      = np.nan, 'molten'
            T_interface = T_core - _DTDP_LIQ * (P_bottom - P_top)
            break
        else:                                       # 分層
            h_solid_km  = max(bml_thickness * pd['f_solid'],  0.0)
            h_liquid_km = max(bml_thickness * pd['f_liquid'], 0.0)
            Mg_solid, Mg_liquid = pd['XS'], pd['XL']
        if h_solid_km < 1.0:
            h_solid_km, h_liquid_km = 0.0, bml_thickness
            Ra, thermal_state       = np.nan, 'molten'
            break
        # 界面溫度 = T_core − 液體絕熱降 (Ra 不再參與)
        P_int = P_top + (P_bottom - P_top) * h_solid_km / bml_thickness
        T_interface = T_core - _DTDP_LIQ * (P_bottom - P_int)
        if abs(T_interface - T_old) < 5.0:
            break

    # Ra 事後診斷 (descriptor, not driver)
    if h_solid_km >= 1.0:
        Ra, thermal_state = compute_bml_thermal_state(
            T_mantle_bottom, T_interface, h_solid_km, bml_top_depth,
            rho_solid=rho_mantle_bottom)

    print(f"  BML: Ra={Ra:.1e} [{thermal_state}]  "
          f"h_sol={h_solid_km:.0f}km  h_liq={h_liquid_km:.0f}km  "
          f"T_int={T_interface:.0f}K  melt={pd['melting']}")

    # ── Step 3a: nGibbs solid BML ─────────────────────────────────────────────
    solid_data = None
    if h_solid_km >= 1.0:
        n_sol   = max(int(n_points * h_solid_km / bml_thickness), 3)
        P_sol   = np.linspace(P_top, P_top + (P_bottom-P_top)*h_solid_km/bml_thickness, n_sol)
        if thermal_state == 'convective':
            delta = min(max(h_solid_km * (_RA_C / Ra)**(1.0/3.0),
                        h_solid_km / n_sol),
                        h_solid_km / 2.0)
            T_i   = 0.5 * (T_mantle_bottom + T_interface)
            z_rel = np.linspace(0.0, h_solid_km, n_sol)
            T_sol = np.where(z_rel < delta,
                     T_mantle_bottom + (T_i - T_mantle_bottom)*z_rel/delta,
                     np.where(z_rel > h_solid_km - delta,
                       T_i + (T_interface - T_i)*(z_rel-(h_solid_km-delta))/delta,
                       T_i))
        else:
            T_sol = np.linspace(T_mantle_bottom, T_interface, n_sol)

        p_s = composition_from_params({'Mg#': Mg_solid, 'T_lit': T_mantle_bottom, 'P_lit': P_sol[0]})
        O_s = compute_oxygen(p_s)

        out_s = _ng_forward(P_sol, T_sol, _comp_values(p_s, O_s),
                            NG_T_HEADERS, ['component_moles'])
        pr     = _ng_props(out_s['component_moles'], P_sol, T_sol, ['rho', 'Vp', 'Vs'])
        rho_s  = np.asarray(pr['rho'])
        Vp_raw = np.asarray(pr['Vp'])
        Vs_raw = np.asarray(pr['Vs'])

        if not np.all(np.isfinite(rho_s)) or np.any(rho_s <= 0):
            return None

        depth_s = np.interp(P_sol, _pres_gpa, _pres_depth)
        solid_data = {'depth_km': depth_s, 'P_GPa': P_sol,
                      'T_K': T_sol, 'Vp': np.asarray(Vp_raw), 'Vs': np.asarray(Vs_raw),
                      'rho': rho_s, 'phi': np.zeros(n_sol)}

    # ── Step 3b: liquid BML (Thomas EoS, Vs=0, isothermal = T_core) ──────────
    liquid_data = None
    if h_liquid_km >= 1.0 and not np.isnan(Mg_liquid):
        n_liq   = max(int(n_points * h_liquid_km / bml_thickness), 3)
        P_liq   = np.linspace(P_top + (P_bottom-P_top)*h_solid_km/bml_thickness, P_bottom, n_liq)
        rho_liq = np.zeros(n_liq)
        Vp_liq  = np.zeros(n_liq)
        T_liq_arr = np.linspace(T_interface if h_solid_km >= 1.0
                                else T_core - _DTDP_LIQ*(P_bottom-P_top),
                                T_core, n_liq)
        for i in range(n_liq):
            rho_liq[i], Vp_liq[i], _ = liquid_bml_properties(P_liq[i], T_liq_arr[i], Mg_liquid)

        off = solid_data['depth_km'][-1] if solid_data is not None else np.interp(P_liq[0], _pres_gpa, _pres_depth)
        depth_liq = np.interp(P_liq, _pres_gpa, _pres_depth)
        depth_liq = np.maximum(depth_liq, off)   # keep monotonic across the interface

        liquid_data = {'depth_km': depth_liq, 'P_GPa': P_liq,
                       'T_K': T_liq_arr,
                       'Vp': Vp_liq, 'Vs': np.zeros(n_liq),
                       'rho': rho_liq, 'phi': np.ones(n_liq)}

    if solid_data is None and liquid_data is None:
        return None

    # ── merge solid + liquid ──────────────────────────────────────────────────
    parts = [d for d in [solid_data, liquid_data] if d is not None]
    def cat(key): return np.concatenate([d[key] for d in parts])
    depth = cat('depth_km'); P_GPa = cat('P_GPa'); T_K  = cat('T_K')
    Vp    = cat('Vp');       Vs    = cat('Vs');     rho  = cat('rho'); phi = cat('phi')

    oc_offset = depth[-1] - depth[0]
    for i in range(len(Vs)):
        if Vs[i] == 0.0:
            oc_offset = depth[i] - depth[0]; break

    if not (np.all(np.isfinite(Vp)) and np.all(np.isfinite(Vs)) and np.all(np.isfinite(rho))):
        print(f"  nGibbs 產生 NaN/Inf → reject  (T_lit={params.get('T_lit', -1):.0f} "
              f"T_core={params['T_core']:.0f} Mg#={params['Mg#']:.3f})")
        return None

    return {
        'depth_km':                depth,
        'P_GPa':                   P_GPa,
        'T_K':                     T_K,
        'Vp':                      Vp,
        'Vs':                      Vs,
        'rho':                     rho,
        'phi':                     phi,
        'outer_core_depth_offset': oc_offset,
        'Ra':                      Ra,
        'thermal_state':           thermal_state,
        'h_solid_km':              h_solid_km,
        'h_liquid_km':             h_liquid_km,
        'Mg_solid':                Mg_solid,
        'Mg_liquid':               Mg_liquid if not np.isnan(Mg_liquid) else -1.0,
        'melting':                 pd['melting'],
        'T_interface':             T_interface,
    }


# ── TauP ──────────────────────────────────────────────────────────────────────
def build_taup(fort56_data, model_name, samuel_cache, bml_data=None):
    os.makedirs(TAUP_WORK_DIR, exist_ok=True)
    model_name = model_name.replace(".npz", "")
    npz_path   = os.path.join(TAUP_WORK_DIR, f'{model_name}.npz')
    nd_path    = os.path.join(TAUP_WORK_DIR, f"{model_name}.nd")
    if os.path.exists(npz_path):
        return TauPyModel(model=npz_path)

    true_cmb_depth = samuel_cache['true_cmb_depth']
    bml_top_depth  = float(bml_data['depth_km'][0])

    hef_depth   = fort56_data['depth_km']
    mantle_mask = (hef_depth >= 100.0) & (hef_depth <= bml_top_depth)
    man_depth   = hef_depth[mantle_mask]
    man_Vp      = fort56_data['Vp'][mantle_mask]
    man_Vs      = fort56_data['Vs'][mantle_mask]
    man_rho     = fort56_data['rho'][mantle_mask]
    if len(man_depth) == 0:
        raise ValueError(f"Empty mantle depth range")

    with open(nd_path, 'w') as f:
        for d, vp, vs, r in zip(samuel_cache['crust_z'], samuel_cache['crust_vp'],
                                 samuel_cache['crust_vs'], samuel_cache['crust_rho']):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        f.write("mantle\n")
        for d, vp, vs, r in zip(man_depth, man_Vp, man_Vs, man_rho):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

        bml_depth = bml_data['depth_km']
        bml_vp    = bml_data['Vp']
        bml_vs    = bml_data['Vs']
        bml_rho   = bml_data['rho']
        bml_mask  = (bml_depth >= bml_top_depth) & (bml_depth <= true_cmb_depth)

        d_b   = bml_depth[bml_mask]
        vp_b  = bml_vp[bml_mask]
        vs_b  = bml_vs[bml_mask]
        rho_b = bml_rho[bml_mask]

        # --- NaN/Inf to TauP  ---
        def _bad(*arrs):
            for a in arrs:
                a = np.asarray(a, dtype=float)
                if a.size == 0 or not np.all(np.isfinite(a)):
                    return True
            return False
        if _bad(man_Vp, man_Vs, man_rho, vp_b, vs_b, rho_b):
            raise ValueError("profile has non-finite Vp/Vs/rho (nGibbs out of range)")
        if np.any(man_Vp <= 0) or np.any(vp_b <= 0) or np.any(man_rho <= 0):
            raise ValueError("profile has non-positive Vp/rho")

        # first index where Vs vanishes (melt correction can zero Vs inside the
        # nominally solid layer, so locate the fluid boundary from the data itself)
        zero = np.where(vs_b <= 1e-6)[0]
        i0   = int(zero[0]) if len(zero) else len(d_b)

        # solid part (Vs > 0)
        for d, vp, vs, r in zip(d_b[:i0], vp_b[:i0], vs_b[:i0], rho_b[:i0]):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

        if i0 < len(d_b):
            # fluid starts inside the BML: discontinuity at d_b[i0]
            d_oc = d_b[i0]
            if i0 > 0:
                vp_s, vs_s, rho_s = vp_b[i0-1], vs_b[i0-1], rho_b[i0-1]
            else:
                vp_s, vs_s, rho_s = man_Vp[-1], man_Vs[-1], man_rho[-1]
            f.write(f"{d_oc:.3f}  {vp_s:.4f}  {vs_s:.4f}  {rho_s:.4f}\n")
            f.write("outer-core\n")
            for d, vp, r in zip(d_b[i0:], vp_b[i0:], rho_b[i0:]):
                f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")
        else:
            # fully solid BML: fluid boundary is the CMB itself
            f.write(f"{true_cmb_depth:.3f}  {vp_b[-1]:.4f}  {vs_b[-1]:.4f}  {rho_b[-1]:.4f}\n")
            f.write("outer-core\n")

        core_z   = samuel_cache['core_z']
        core_vp  = samuel_cache['core_vp']
        core_rho = samuel_cache['core_rho']
        mask     = core_z >= true_cmb_depth
        f.write(f"{true_cmb_depth:.3f}  {core_vp[mask][0]:.4f}  0.0000  {core_rho[mask][0]:.4f}\n")
        for d, vp, r in zip(core_z[mask][1:], core_vp[mask][1:], core_rho[mask][1:]):
            f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")

    build_taup_model(nd_path, output_folder=TAUP_WORK_DIR)
    return TauPyModel(model=npz_path)

# ── mass and MoI ──────────────────────────────────────────────────────────────
def compute_mass_and_moi(fort56_data, samuel_cache, bml_data):
    R           = MARS_RADIUS_M
    true_cmb_km = samuel_cache['true_cmb_depth']
    bml_top_km  = float(bml_data['depth_km'][0])

    # crust: Samuel profile from surface to crust base (~100 km)
    crust_z      = samuel_cache['crust_z']
    crust_rho    = samuel_cache['crust_rho']
    crust_mask   = crust_z <= 100.0
    crust_r      = R - crust_z[crust_mask] * 1000
    crust_rho_si = crust_rho[crust_mask] * 1000

    # mantle: depths are absolute (Samuel P-z scale), 100 km -> BML top
    hef_depth    = fort56_data['depth_km']
    mantle_mask  = (hef_depth >= 100.0) & (hef_depth <= bml_top_km + 1.0)
    man_r        = R - hef_depth[mantle_mask] * 1000
    man_rho_si   = fort56_data['rho'][mantle_mask] * 1000

    # BML
    bml_depth  = bml_data['depth_km']
    bml_mask   = (bml_depth >= bml_top_km) & (bml_depth <= true_cmb_km)
    bml_r      = R - bml_depth[bml_mask] * 1000
    bml_rho_si = bml_data['rho'][bml_mask] * 1000

    # core: Samuel profile below CMB
    core_z      = samuel_cache['core_z']
    core_rho    = samuel_cache['core_rho']
    core_mask   = core_z >= true_cmb_km
    core_r      = R - core_z[core_mask] * 1000
    core_rho_si = core_rho[core_mask] * 1000

    all_r   = np.concatenate([core_r, bml_r, man_r, crust_r])
    all_rho = np.concatenate([core_rho_si, bml_rho_si, man_rho_si, crust_rho_si])
    idx     = np.argsort(all_r)
    all_r, all_rho = all_r[idx], all_rho[idx]

    M   = 4 * np.pi * np.trapezoid(all_rho * all_r**2, all_r)
    I   = (8*np.pi/3) * np.trapezoid(all_rho * all_r**4, all_r)
    return M, I / (M * R**2)

# ── solidus penalty ───────────────────────────────────────────────────────────
def compute_solidus_penalty(fort56_data, params):
    P_prof = fort56_data.get('P_profile')
    T_prof = fort56_data.get('T_profile')
    if P_prof is None or T_prof is None:
        return 0.0
    bml_top_km  = TRUE_CMB_DEPTH - params['BML_thickness']
    P_bml_top   = float(pressure_mars(bml_top_km))
    mask        = (P_prof >= params['P_lit']) & (P_prof <= P_bml_top)
    P_m, T_m   = P_prof[mask], T_prof[mask]
    if len(P_m) == 0:
        return 0.0
    # Mg#-dependent: more Fe lowers the solidus (Elkins-Tanton 2008), exactly as in
    # Drilleau 2026. The fixed Mg#=0.75 curve made this penalty blind to Mg#.
    T_sol_arr   = np.array([solidus_duncan_Mg(p, params['Mg#']) for p in P_m])
    excess  = np.clip(T_m - T_sol_arr, 0.0, None)
    penalty = float(np.trapezoid(excess, P_m)) / 16.0   # ∫excess dP (K·GPa), 網格無關
    if penalty > 0:
        print(f"  Solidus penalty = {penalty:.4f}")
    return penalty

# ── misfit ────────────────────────────────────────────────────────────────────
def compute_misfit(taup_model, obs_dataset, fort56_data, bml_data, params=None,
                   mass_sigma=0.0, moi_sigma=0.0):
    phases_std   = ['P', 'S', 'pP', 'sP', 'PP', 'PPP', 'SS', 'SSS', 'sS', 'ScS', 'SKS']
    phases_pdiff = phases_std + ['Pdiff']

    tt_total = 0.0      # Σ over events of each event's MEAN residual
    tt_n_ev  = 0        # events that contributed
    tt_n_ph  = 0        # total phases used (report only)
    tt_n_miss = 0       # phases that the model could not predict

    for event, obs in obs_dataset.items():
        delta = obs['delta']
        depth = obs.get('depth', 10.0)
        try:
            arrivals = taup_model.get_travel_times(
                source_depth_in_km=depth, distance_in_degree=delta,
                phase_list=phases_pdiff if event == 'S1000a' else phases_std)
        except Exception as e:
            print(f"  {event}: FAILED {e}"); continue

        times = {}
        for a in arrivals:
            if a.name not in times:
                times[a.name] = a.time

        pred = {
            'S-P':           times.get('S',   None) and times.get('P',    None) and times['S']   - times['P'],
            'pP-P':          times.get('pP',  None) and times.get('P',    None) and times['pP']  - times['P'],
            'sP-P':          times.get('sP',  None) and times.get('P',    None) and times['sP']  - times['P'],
            'PP-P':          times.get('PP',  None) and times.get('P',    None) and times['PP']  - times['P'],
            'PPP-P':         times.get('PPP', None) and times.get('P',    None) and times['PPP'] - times['P'],
            'sS-S':          times.get('sS',  None) and times.get('S',    None) and times['sS']  - times['S'],
            'SS-S':          times.get('SS',  None) and times.get('S',    None) and times['SS']  - times['S'],
            'SSS-S':         times.get('SSS', None) and times.get('S',    None) and times['SSS'] - times['S'],
            'ScS-S':         times.get('ScS', None) and times.get('S',    None) and times['ScS'] - times['S'],
            'SS-PP':         times.get('SS',  None) and times.get('PP',   None) and times['SS']  - times['PP'],
            'SKS-PP':        times.get('SKS', None) and times.get('PP',   None) and times['SKS'] - times['PP'],
            'PP-PbdiffPcP':  times.get('PP',  None) and times.get('Pdiff',None) and times['PP']  - times['Pdiff'],
        }

        ev_sum, ev_n, ev_miss = 0.0, 0, 0
        for phase, obs_val in obs.items():
            if phase in ('delta', 'depth') or not isinstance(obs_val, tuple):
                continue
            p_val = pred.get(phase)
            if p_val is None or p_val is False:
                ev_sum += 5.0; ev_n += 1; ev_miss += 1   # lack phases = 5σ fixed residual
                continue
            obs_t, sigma = obs_val
            if sigma <= 0 or not np.isfinite(obs_t) or not np.isfinite(p_val):
                ev_sum += 5.0; ev_n += 1; ev_miss += 1
                continue
            val = abs(obs_t - p_val) / sigma
            if np.isfinite(val):
                ev_sum += val; ev_n += 1
            else:
                ev_sum += 5.0; ev_n += 1; ev_miss += 1

        if ev_n > 0:
            tt_total  += ev_sum / ev_n
            tt_n_ev   += 1
            tt_n_ph   += ev_n - ev_miss    
            tt_n_miss += ev_miss
        

    tt_misfit = tt_total if tt_n_ev > 0 else 999.0
    if not np.isfinite(tt_misfit):
        tt_misfit = 999.0

    n_ev_expected = len(obs_dataset)
    if tt_n_ev < n_ev_expected:
        tt_misfit += 10.0 * (n_ev_expected - tt_n_ev)

    solidus_penalty = compute_solidus_penalty(fort56_data, params) if params else 0.0
    total_misfit = tt_misfit + solidus_penalty + mass_sigma + moi_sigma
    if not np.isfinite(total_misfit):
        total_misfit = 999.0

    print(f"  TT={tt_misfit:.4f}(events={tt_n_ev}/{n_ev_expected}, phases={tt_n_ph}, miss={tt_n_miss})  "
          f"solidus={solidus_penalty:.4f}  mass={mass_sigma:.2f}σ  "
          f"moi={moi_sigma:.2f}σ  total={total_misfit:.4f}")

    return total_misfit, tt_n_ph, {
        'tt': tt_misfit, 'solidus': solidus_penalty,
        'tt_n_ev': tt_n_ev, 'tt_n_ph': tt_n_ph, 'tt_n_miss': tt_n_miss,
    }

# ── forward model ─────────────────────────────────────────────────────────────
def forward(params, run_dir, model_name, samuel_cache):
    true_cmb_depth = samuel_cache['true_cmb_depth']
    bml_top_km     = true_cmb_depth - params['BML_thickness']
    P_bml_top      = float(pressure_mars(bml_top_km))

    fort56_data = run_ngibbs(params, P_bml_top=P_bml_top)
    if fort56_data is None:
        return None, None, None, None, None

    T_profile = fort56_data.get('T_profile')
    P_profile = fort56_data.get('P_profile')
    T_mantle_bottom = (float(np.interp(P_bml_top, P_profile, T_profile))
                       if T_profile is not None else float(fort56_data['T_K'][-1]))
    rho_mantle_bottom = float(fort56_data['rho'][-1])   # mantle bottom density → proxy for solid BML ρ
    print(f"  T_mantle_bottom={T_mantle_bottom:.1f}K  P_bml_top={P_bml_top:.2f}GPa  rho_bot={rho_mantle_bottom:.4f}")

    bml_raw = run_ngibbs_bml(params,
                             T_core=params['T_core'],
                             T_mantle_bottom=T_mantle_bottom,
                             true_cmb_depth=true_cmb_depth,
                             rho_mantle_bottom=rho_mantle_bottom)
    if bml_raw is None:
        print("  BML failed → reject"); return None, None, None, None, None

    bml_raw['depth_km'] = bml_raw['depth_km'] - bml_raw['depth_km'][0] + bml_top_km
    bml_raw['outer_core_depth_abs'] = bml_top_km + bml_raw['outer_core_depth_offset']

    rho_mantle_bot = float(fort56_data['rho'][-1])
    rho_bml_top    = float(bml_raw['rho'][0])
    rho_bml_bot    = float(bml_raw['rho'][-1])
    core_mask      = samuel_cache['core_z'] >= true_cmb_depth
    rho_core_top   = float(samuel_cache['core_rho'][core_mask][0]) if core_mask.any() else 6.4

    bml_raw['upper_contrast'] = rho_bml_top - rho_mantle_bot
    bml_raw['lower_contrast'] = rho_core_top - rho_bml_bot
    print(f"  density contrasts: upper={bml_raw['upper_contrast']:+.4f}  "
          f"lower={bml_raw['lower_contrast']:+.4f}")

    if REJECT_GRAV_UNSTABLE and (bml_raw['upper_contrast'] <= 0.0 or bml_raw['lower_contrast'] <= 0.0):
        if bml_raw['upper_contrast'] <= 0.0: _GRAV_FAIL['upper'] += 1
        if bml_raw['lower_contrast'] <= 0.0: _GRAV_FAIL['lower'] += 1
        print("  BML gravitationally unstable → reject (Kono 2025)")
        return None, None, None, None, None

    bml_data = bml_raw

    # mass and MoI as soft constraints (contribute to misfit, no hard rejection)
    M_pred, moi_pred = compute_mass_and_moi(fort56_data, _samuel_cache, bml_data)
    mass_sigma = abs(MARS_MASS_OBS - M_pred) / MARS_MASS_SIGMA
    moi_sigma = abs(MOI_OBS - (moi_pred - MOI_BIAS)) / MOI_SIGMA
    print(f"  mass={mass_sigma:.2f}σ  moi={moi_sigma:.2f}σ")

    try:
        taup_model = build_taup(fort56_data, model_name, samuel_cache, bml_data=bml_data)
    except Exception as e:
        print(f"  TauP failed: {e}"); return None, None, None, None, None

    misfit, n_data, components = compute_misfit(
        taup_model, SAMUEL_DATA, fort56_data, bml_data=bml_data, params=params,
        mass_sigma=mass_sigma, moi_sigma=moi_sigma)

    components['upper_contrast']  = bml_data.get('upper_contrast')
    components['lower_contrast']  = bml_data.get('lower_contrast')
    components['mass_sigma']      = mass_sigma
    components['moi_sigma']       = moi_sigma
    components['moi_pred']        = moi_pred      
    components['M_pred']          = M_pred       
    components['Ra']              = bml_data.get('Ra')
    components['thermal_state']   = bml_data.get('thermal_state')
    components['h_solid_km']      = bml_data.get('h_solid_km')
    components['h_liquid_km']     = bml_data.get('h_liquid_km')
    components['melting']         = bml_data.get('melting', False)
    components['Mg_solid']        = bml_data.get('Mg_solid')
    components['Mg_liquid']       = bml_data.get('Mg_liquid')
    components['T_mantle_bottom'] = T_mantle_bottom
    components['T_interface']     = bml_data.get('T_interface')
    

    return misfit, n_data, components, fort56_data, bml_data

# ── MCMC ──────────────────────────────────────────────────────────────────────
def propose(current, rng):
    proposed = {}
    for key in PRIOR:
        lo, hi = PRIOR[key]
        while True:
            val = current[key] + rng.normal(0, STEP[key])
            if lo <= val <= hi:
                proposed[key] = val
                break
    return proposed


def run_mcmc(chain_id, n_steps, start_params=None, prefix='chain'):
    chain_dir = os.path.join(MCMC_DIR, f"{prefix}_{chain_id:02d}")
    os.makedirs(chain_dir, exist_ok=True)

    samuel_cache = _samuel_cache

    rng     = np.random.default_rng(42 + chain_id)
    current = (start_params or START_PARAMS).copy()

    chain_file = os.path.join(chain_dir, "chain.json")
    chain      = []
    if os.path.exists(chain_file):
        with open(chain_file) as f:
            chain = json.load(f)
        if chain:
            current = chain[-1]['params']
            # migrate old parameter names → new names
            if 'T_bml' in current and 'T_core' not in current:
                current['T_core'] = current.pop('T_bml')
            if 'Mg#_bml' in current and 'Mg#_bulk_bml' not in current:
                current['Mg#_bulk_bml'] = current.pop('Mg#_bml')
            print(f"Chain {chain_id}: resuming from step {len(chain)}")

    step_start = len(chain)
    accept_count = 0
    current_components = {'tt': 999.0, 'solidus': 0.0,
                          'mass_sigma': None, 'moi_sigma': None,
                          'upper_contrast': None, 'lower_contrast': None}

    if chain:
        current_misfit = chain[-1]['misfit']
        accept_count   = sum(1 for s in chain if s.get('accepted', False))
        current_components = {
            'tt':             chain[-1].get('misfit_tt',      999.0),
            'solidus':        chain[-1].get('misfit_solidus', 0.0),
            'mass_sigma':     chain[-1].get('mass_sigma'),
            'moi_sigma':      chain[-1].get('moi_sigma'),
            'moi_pred':       chain[-1].get('moi_pred'),   
            'M_pred':         chain[-1].get('M_pred'),      
            'upper_contrast': chain[-1].get('upper_contrast'),
            'lower_contrast': chain[-1].get('lower_contrast'),
            'tt_n_ev':        chain[-1].get('tt_n_ev'),
        }
    else:
        run_dir    = os.path.join(chain_dir, "step_current")
        model_name = f"mcmc_{prefix}_c{chain_id:02d}_current"
        for attempt in range(20):
            trial = current.copy() if attempt == 0 else {
                k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PRIOR.items()}
            if attempt > 0:
                print(f"  Retry {attempt}: T_lit={trial['T_lit']:.1f}  "
                      f"Mg#={trial['Mg#']:.3f}  T_core={trial['T_core']:.1f}  Mg#_bulk_bml={trial['Mg#_bulk_bml']:.3f}")
            current_misfit, _, current_components, _, _ = forward(
                trial, run_dir, model_name, samuel_cache)
            if current_misfit is not None and np.isfinite(current_misfit):
                current = trial
                print(f"  Start OK (attempt {attempt+1}): misfit={current_misfit:.4f}")
                break
        else:
            print(f"  Starting point failed, giving up chain {chain_id}.")
            return

    print(f"\nChain {chain_id}  initial misfit={current_misfit:.4f}")

    for step in range(step_start, step_start + n_steps):
        t0           = datetime.now()
        proposed     = propose(current, rng)
        run_dir      = os.path.join(chain_dir, f"step_{step+1:05d}")
        model_name   = f"mcmc_{prefix}_c{chain_id:02d}_s{step+1:05d}"

        proposed_misfit, n_data, components, fort56_data, bml_data = forward(
            proposed, run_dir, model_name, samuel_cache)

        if proposed_misfit is None or proposed_misfit >= 990.0:
            accepted        = False
            proposed_misfit = 999.0
            components      = {'tt': 999.0, 'solidus': 0.0,
                               'mass_sigma': None, 'moi_sigma': None,
                               'upper_contrast': None, 'lower_contrast': None}
        else:
            delta = proposed_misfit - current_misfit
            accepted = delta <= 0 or np.log(rng.uniform()) < -delta / MCMC_TEMPERATURE

        if accepted:
            current            = proposed
            current_misfit     = proposed_misfit
            current_components = components
            accept_count      += 1

            if fort56_data is not None:
                npz_dict = {k: fort56_data[k]
                            for k in ('depth_km','Vp','Vs','rho','T_K','P_GPa')}
                if bml_data is not None:
                    npz_dict.update({
                        'bml_depth_km':             bml_data['depth_km'],
                        'bml_Vp':                   bml_data['Vp'],
                        'bml_Vs':                   bml_data['Vs'],
                        'bml_rho':                  bml_data['rho'],
                        'bml_T_K':                  bml_data['T_K'],
                        'bml_P_GPa':                bml_data['P_GPa'],
                        'bml_phi':                  bml_data['phi'],
                        'bml_outer_core_depth_abs': np.array([bml_data['outer_core_depth_abs']]),
                        'bml_Ra':                   np.array([bml_data.get('Ra', 0.0)]),
                        'bml_thermal_state':        np.array([bml_data.get('thermal_state', '')]),
                        'bml_h_solid_km':           np.array([bml_data.get('h_solid_km', 0.0)]),
                        'bml_h_liquid_km':          np.array([bml_data.get('h_liquid_km', 0.0)]),
                        'bml_Mg_solid':             np.array([bml_data.get('Mg_solid', -1.0)]),
                        'bml_Mg_liquid':            np.array([bml_data.get('Mg_liquid', -1.0)]),
                    })
                np.savez(os.path.join(chain_dir, f"profile_s{step+1:05d}.npz"), **npz_dict)

        elapsed     = (datetime.now() - t0).total_seconds()
        accept_rate = accept_count / (step + 1) * 100
        uc          = current_components.get('upper_contrast')
        print(f"  Step {step+1:4d}: misfit={current_misfit:.4f}  "
              f"{'ACCEPT' if accepted else 'reject'}  "
              f"rate={accept_rate:.1f}%  "
              f"uc={f'{uc:+.4f}' if uc is not None else 'N/A'}  "
              f"({elapsed:.0f}s)")

        chain.append({
            'step':           step + 1,
            'params':         current,
            'misfit':         current_misfit,
            'misfit_tt':      current_components.get('tt',      999.0),
            'misfit_solidus': current_components.get('solidus', 0.0),
            'mass_sigma':     current_components.get('mass_sigma'),
            'moi_sigma':      current_components.get('moi_sigma'),
            'moi_pred':       current_components.get('moi_pred'),   
            'M_pred':         current_components.get('M_pred'),     
            'upper_contrast': current_components.get('upper_contrast'),
            'lower_contrast': current_components.get('lower_contrast'),
            'Ra':             current_components.get('Ra'),
            'thermal_state':  current_components.get('thermal_state'),
            'h_solid_km':     current_components.get('h_solid_km'),
            'h_liquid_km':    current_components.get('h_liquid_km'),
            'melting':        current_components.get('melting'),
            'Mg_solid':       current_components.get('Mg_solid'),
            'Mg_liquid':      current_components.get('Mg_liquid'),
            'T_mantle_bottom':current_components.get('T_mantle_bottom'),
            'T_interface':    current_components.get('T_interface'),
            'tt_n_ev':        current_components.get('tt_n_ev'),
            'tt_n_ph':        current_components.get('tt_n_ph'),
            'tt_n_miss':      current_components.get('tt_n_miss'),
            'proposal_S_lit':      _NG_S_LOG['last'],
            'proposal_S_in_range': bool(_NG_S_LOG['last_in_range']),
            'accepted':       bool(accepted),
            'accept_rate':    accept_rate,
        })

        with open(chain_file, 'w') as f:
            json.dump(chain, f, indent=2)

    print(f"\nChain {chain_id} done  "
          f"accept_rate={accept_count/n_steps*100:.1f}%  "
          f"misfit={current_misfit:.4f}")

    print(f"  phase-diagram failures: no_loop={_PD_FAIL['no_loop']}  "
          f"no_invariant={_PD_FAIL['no_invariant']}  solidus={_PD_FAIL['solidus']}")
    print(f"  phase branch: gamma={_PHASE_USED['gamma']}  "
          f"three-phase={_PHASE_USED['three-phase']}  beta={_PHASE_USED['beta']}")
    
    print(f"  grav-unstable rejects: upper(BML too light)={_GRAV_FAIL['upper']}  "
      f"lower(BML too dense)={_GRAV_FAIL['lower']}")

    s_all = np.asarray(_NG_S_LOG['all'])
    s_fin = s_all[np.isfinite(s_all)]
    if len(s_fin):
        print(f"  S_lit proposals: n={len(s_fin)}  "
              f"range=[{s_fin.min():.4f}, {s_fin.max():.4f}]  "
              f"below {NG_S_MIN}: {_NG_S_LOG['n_below']} "
              f"({100*_NG_S_LOG['n_below']/len(s_fin):.1f}%)  "
              f"above {NG_S_MAX}: {_NG_S_LOG['n_above']} "
              f"({100*_NG_S_LOG['n_above']/len(s_fin):.1f}%)")

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain',        type=int,  default=0)
    parser.add_argument('--steps',        type=int,  default=100)
    parser.add_argument('--test',         action='store_true')
    parser.add_argument('--prefix',       type=str,  default='chain')
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--start',        type=str,  default=None)
    args = parser.parse_args()

    os.makedirs(MCMC_DIR, exist_ok=True)
    start_params = None

    if args.start is not None:
        try:
            start_params = json.loads(args.start)
        except json.JSONDecodeError:
            with open(args.start) as f:
                start_params = json.load(f)
        missing = [k for k in PRIOR if k not in start_params]
        if missing:
            raise ValueError(f"--start missing: {missing}")

    elif args.random_start:
        rng_init     = np.random.default_rng(args.chain)
        start_params = {k: float(rng_init.uniform(lo, hi)) for k, (lo, hi) in PRIOR.items()}
        print(f"random_start: {start_params}")

    if args.test:
        run_mcmc(0, 1, prefix=args.prefix, start_params=start_params)
    else:
        run_mcmc(args.chain, args.steps, prefix=args.prefix, start_params=start_params)
