#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMC inversion for Mars interior structure using nGibbs (HeFESTo emulator) + BML physics
a35: Fe-S EoS core in the forward model (core_fes.py). New free parameter w_S.
     T_core now feeds both the BML and the core adiabat; core rho/Vp respond to
     (T_core, w_S) via mass, MoI, ScS/SKS travel times, and Kono density contrast.
     R_ic from the static ICB thermometer is recorded every step; set
     USE_RIC_LIKELIHOOD=True to include Bi 2025 R_ic=612 km in the likelihood (Run B).
"""

import os
import numpy as np
import json
import argparse
import sys
import faulthandler
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
faulthandler.enable(file=sys.stderr, all_threads=True)
import torch
torch.set_num_threads(1)
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

# ── Fe-S core EoS (a26: core in forward model) ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_fes as CF     # kuwayama Fe + sakai FeS + Margules, 查表加速

USE_RIC_LIKELIHOOD = False   # Run A: False(只有 mass/MoI/TT 約束 T_core, w_S)
                             # Run B: True (加 Bi 2025 內核半徑進 likelihood)
BI_RIC_KM  = 612.0           # Bi 2025 refined inversion
BI_RIC_SIG = 112.0           # sqrt(50^2 + 100^2): 形式誤差 + OC 速度 5%→100 km 系統項
RIC_MAX_KM = 750.0           # InSight 上限(SKS 最大穿透深度, Irving 2023)

NG_T_HEADERS = ['P(GPa)(System_main)', 'T(K)(System_main)',
                'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
NG_S_HEADERS = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)',
                'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
NG_S_MIN     = 1.75
NG_S_MAX     = 2.90
NG_N_SHALLOW = 100
NG_N_DEEP    = 101

# ── travel-time misfit scale ──────────────────────────────────────────────────
# Ordering must be strict:  good fit  <  worst present phase (RES_CAP)
#                                     <  phase the model cannot predict (MISS_PENALTY)
# a32 violated this (missing = 5.0 while present residuals were unbounded to 43 sigma),
# which made dropping hard phases profitable and created an absorbing state at
# P_lit ~ 2.5 GPa.  Keep the inequalities whenever these numbers are touched.
RES_CAP       = 10.0    # sigma; covers the genuine 6.5 sigma ScS-S residuals,
                        # still crushes TauP branch-jumps (2662 s = 333 sigma)
MISS_PENALTY  = 12.0    # must be > RES_CAP
EVENT_FAIL    = 20.0    # must be > MISS_PENALTY (whole event lost, e.g. TauP raised)
SOLIDUS_SIGMA = 12.0    # K, temperature scale for ~1% melt fraction
SOLIDUS_CAP   = 25.0    # corresponds to 60 K excess

# per-proposal S_lit log (includes proposals rejected for being out of range)
_NG_S_LOG = {'last': np.nan, 'last_in_range': True, 'all': [], 'n_below': 0, 'n_above': 0}

# ── constants ─────────────────────────────────────────────────────────────────
MARS_RADIUS      = 3389.5
MARS_RADIUS_M    = MARS_RADIUS * 1000
T_SURF           = 220.0
MARS_MASS_OBS    = 6.4171e23
MARS_MASS_SIGMA  = MARS_MASS_OBS * 0.01
MOI_OBS          = 0.36379   # Bagheri 2019 / Konopliv 2016, C/MR^2 with
                             # R = mean radius 3389.5 km (Samuel 2021/2023 framework).
                             # Drilleau 2026 uses Konopliv 2020 = 0.3634 +- 0.00006,
                             # 1 sigma away and in the opposite direction -> discussion
                             # sensitivity, not the baseline.
MOI_SIGMA        = 0.0005
MOI_BIAS         = 0.00136   # pipeline systematic: pure-Samuel PanelJ through this
                             # integrator gives MoI=0.36515 vs obs 0.36379 (budget test)
MCMC_TEMPERATURE = 1.0

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
'w_S': 0.17,     # core S weight fraction (Fe-S binary, effective light element)
}

# a34 prior widening (a33 pressed three walls):
#   T_core       3200 -> 3600   Drilleau 2026 BML Tc = 3437 +- 201 / 3497 +- 272 K
#                               sat entirely outside the old wall
#   Mg#          0.86 -> 0.90   a33 posterior 95% upper 0.846, only 0.014 from the wall
#                               while MoI keeps pushing it up
#   Mg#_bulk_bml 0.30 -> 0.20   Ohtani 1979 Fe2SiO4 melting relations support it
# NOTE: changing the Mg#_bulk_bml axis INVALIDATES bml_phase_table_prior_*.npz.
#       Rebuild the table before any prior scan or Bayes factor.
PRIOR = {
'T_lit': (1000.0, 2600.0),
'P_lit': (1.4, 9.0),
'Mg#': (0.50, 0.90),
'T_core': (1550.0, 3600.0),   # a26: lower 1800->1550, ICB 溫度計臨界溫度在 1700-1820 K
'Mg#_bulk_bml': (0.20, 0.80),
'BML_thickness': (0.0, 400.0),
'w_S': (0.05, 0.30),          # 化學計量上限 0.365; 地球化學上限 ~0.17 不設進 prior,
}                             # 讓 posterior 與 17 wt% 的關係成為結果

STEP = {
'T_lit': 50.0,
'P_lit': 0.7,
'Mg#': 0.025,
'T_core': 100.0,
'Mg#_bulk_bml': 0.06,
'BML_thickness': 25.0,
'w_S': 0.02,
}

CHAIN_KEYS = (
    'misfit_tt', 'misfit_solidus', 'grav_sigma', 'mass_sigma', 'moi_sigma',
    'moi_pred', 'M_pred','upper_contrast', 'lower_contrast', 'Ra', 'thermal_state',
    'h_solid_km', 'h_liquid_km', 'melting', 'Mg_solid', 'Mg_liquid',
    'rho_solid_bml', 'rho_liquid_bml', 'rho_S_interface', 'P_interface',
    'T_mantle_bottom', 'T_interface', 'S_lit', 'tt_n_ev', 'tt_n_ph', 'tt_n_miss',
    'tt_n_capped', 'tt_miss_by_event',
    'R_ic_km', 'ric_sigma', 'rho_core_mean', 'Vp_core_cmb',
    'P_center_core', 'T_center_core',
)
BML_KEYS = ('Ra', 'thermal_state', 'h_solid_km', 'h_liquid_km', 'melting',
            'Mg_solid', 'Mg_liquid', 'T_interface',
            'rho_solid_bml', 'rho_liquid_bml', 'rho_S_interface', 'P_interface',
            'upper_contrast', 'lower_contrast')

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
# below, else it overturns on a timescale << 4.5 Gyr. Soft penalty, not a rejection:
# hard rejection biases against high-Mg# BML compositions.

REJECT_GRAV_UNSTABLE = False
GRAV_SCALE = 0.15        # g/cm3, ~ the Fo end-member EoS systematic (old-new = 0.18)
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

# _scan is the expensive part of the phase diagram (300 fsolve calls). It is called
# once per (branch, P) but _interface_state iterates 3x at the same P_int and brentq
# calls _interface_state ~10x, so without memoisation the same scan is repeated ~30x
# per step. Key rounded to 0.01 GPa (~0.8 km, T_sol changes < 1 K).
_SCAN_CACHE = {}

def _scan(loopf, P, TmFo, TmFa, Tlo=1600., Thi=3200., n=300):
    key = (loopf.__name__, round(P, 2))
    if key in _SCAN_CACHE:
        return _SCAN_CACHE[key]
    Tp = np.linspace(Tlo, Thi, n); raw = Tp - _DELTA_SHIFT
    XS = np.empty(n); XL = np.empty(n)
    for i, tr in enumerate(raw):
        XS[i], XL[i] = loopf(tr, P, TmFo, TmFa)
    ok = np.isfinite(XS) & np.isfinite(XL) & (XL < XS)
    _SCAN_CACHE[key] = (Tp[ok], XS[ok], XL[ok])
    return _SCAN_CACHE[key]

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
_T0_ETA  = 1600.0    # K, reference temperature
_ALPHA_S = 2.0e-5    # K^-1, thermal expansion (solid BML)
_CP_S    = 1200.0    # J/kg/K
_VSTAR   = 3.0e-6    # m^3/mol  (Drilleau 2026 Table 1: V* 0-10 cm3/mol, inverted)
_P_REF   = 3.0e9     # Pa       (Drilleau 2021/2026 reference pressure)

# Thermal conductivity of the solid BML sublayer. The literature range is wide:
# Drilleau 2026 invert kd = 1.6 +- 0.5 W/m/K, Samuel 2023 fix 4.0. This value is
# the SINGLE source of k for the whole pipeline -- the heat-flux post-processing
# (68.heat_flux.py) must import or match it, otherwise the chain's Ra and the
# post-hoc q use different conductivities.
#   Ra ~ 1/K_SOLID exactly, so any other k can be recovered afterwards:
#       Ra(k) = Ra_stored * K_SOLID / k
#   k enters the forward model ONLY through thermal_state (Ra < _RA_C). With
#   Ra <= ~2e4 << _RA_C = 1e5 the branch never flips for any plausible k, so the
#   posterior is insensitive to this choice and the sensitivity test is pure
#   post-processing. (NOT true for the Ra_c = 1e3 test -- that one needs a rerun.)
#   Planned upgrade: k = k(ringwoodite, XS, P, T), Fe as phonon scatterer.
K_SOLID  = 4.0       # W/m/K

# Ra_cr. 1e3 is the classical CONSTANT-viscosity Rayleigh-Benard value (657.5
# free-free, 1707.8 rigid-rigid, plane layer) and does not apply here: with
# E* = 3e5 J/mol the viscosity contrast across the solid BML is ~4e3, and Ra_cr
# rises with E* in variable-viscosity spherical geometry (Yanagisawa 2016).
# Samuel 2021 SM S4 adopt 1e5 as a LOWER bound for exactly this reason.
# >>> This is a modelling CHOICE you must defend at the exam. Test the sensitivity.
_RA_C  = 1.0e5

def compute_bml_thermal_state(T_mantle_bottom, T_interface, h_solid_km, bml_top_km,
                              rho_solid):
    """Ra of the solid sublayer. Boundary temps are FIXED by the neighbours
    (top: mantle bottom, bottom: solid/liquid interface). Ra selects the
    internal profile SHAPE only — it never modifies boundary temperatures.
    rho_solid in kg/m3, required (no default: a silent 3800 proxy caused the
    a31 Ra error of three orders of magnitude)."""
    z_mid = bml_top_km + h_solid_km / 2.0
    P_mid = float(pressure_mars(z_mid)) * 1e9
    T_mid = 0.5 * (T_mantle_bottom + T_interface)
    dT    = max(T_interface - T_mantle_bottom, 0.0)
    eta   = _ETA0 * np.exp((_ESTAR + P_mid*_VSTAR) / (_R_GAS * T_mid)
                           - (_ESTAR + _P_REF*_VSTAR) / (_R_GAS * _T0_ETA))
    h     = h_solid_km * 1000.0
    g     = gravity_mars(z_mid)
    kappa = K_SOLID / (rho_solid * _CP_S)
    Ra    = _ALPHA_S * rho_solid * g * dT * h**3 / (kappa * eta)
    return Ra, ('conductive' if Ra < _RA_C else 'convective')

# ── Liquid BML EoS (Thomas 2012 Fa + Thomas 2013 Fo, volume mixing) ───────────

_RHO_S_CONST = 4.17   # g/cm3. INITIAL GUESS ONLY for the rho_S self-consistency
                      # iteration; the value actually used every step is computed
                      # live from nGibbs at (XS, P_interface, T_interface).
_M_FO = 140.694
_M_FA = 203.777
def mole_to_volume_fraction(f_solid, XS, XL, rho_S, rho_L):
    # lever rule gives MOLE fractions; thickness needs VOLUME fractions.
    # M_S != M_L because XS != XL, and rho_S != rho_L, so the two differ.
    M_S = XS * _M_FO + (1.0 - XS) * _M_FA
    M_L = XL * _M_FO + (1.0 - XL) * _M_FA
    V_S = f_solid         * M_S / rho_S
    V_L = (1.0 - f_solid) * M_L / rho_L
    return V_S / (V_S + V_L)

_FO = dict(rho0=2.597, KS0=16.41, Ksp=7.37, gam0=0.396, q=-2.02, Cv=1737.36, T0=2273.0, M=140.694)
_FA = dict(rho0=3.699, KS0=21.99, Ksp=7.28, gam0=0.412, q=-0.95, Cv=1122.73, T0=1573.0, M=203.777)

def _rho_endmember(em, P_Pa, T_K):
    # 3BM is NON-monotonic on strong expansion: P(rho) has a minimum near
    # rho ~ 0.75*rho0 and turns positive again below it. Bracketing from
    # 0.4*rho0 therefore fails whenever P_Pa < P(0.4*rho0) ~ 0.3-0.9 GPa.
    # 0.85*rho0 sits to the right of that minimum for T = 1000-4000 K and
    # P(0.85*rho0) < 0 there, so f(a) < 0 is guaranteed for any P_Pa >= 0.
    def f(rho):
        x   = 0.5 * ((rho/em['rho0'])**(2/3) - 1)
        PS  = 3*em['KS0']*x*(1+2*x)**2.5 * (1 + 1.5*(em['Ksp']-4)*x) * 1e9
        gam = em['gam0'] * (em['rho0']/rho)**em['q']
        return PS + gam * rho*1e3 * em['Cv'] * (T_K - em['T0']) - P_Pa
    a, b = em['rho0']*0.85, em['rho0']*4.0
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa*fb > 0:
        print(f"  _rho_endmember bracket failed: P={P_Pa/1e9:.3f} GPa  T={T_K:.1f} K  "
              f"f(a)={fa/1e9:+.3f}  f(b)={fb/1e9:+.3f}")
        return np.nan
    return brentq(f, a, b, xtol=1e-10)

def _V_mix(P_Pa, T_K, Mg):
    rho_Fo = _rho_endmember(_FO, P_Pa, T_K)
    rho_Fa = _rho_endmember(_FA, P_Pa, T_K)
    if not (np.isfinite(rho_Fo) and np.isfinite(rho_Fa)):
        return np.nan
    return Mg*_FO['M']/rho_Fo + (1.0-Mg)*_FA['M']/rho_Fa

def liquid_bml_properties(P_GPa, T_K, Mg_liquid):
    Mg     = float(np.clip(Mg_liquid, 0.0, 1.0))
    x_fa   = 1.0 - Mg
    P_Pa   = P_GPa * 1e9
    M_mix  = Mg*_FO['M'] + x_fa*_FA['M']
    Cv_mix = (Mg*_FO['M']*_FO['Cv'] + x_fa*_FA['M']*_FA['Cv']) / M_mix   # J/kg/K
    dP, dT = 0.05e9, 5.0

    try:
        V    = _V_mix(P_Pa,      T_K,      Mg)
        V_Pp = _V_mix(P_Pa + dP, T_K,      Mg)
        V_Pm = _V_mix(P_Pa - dP, T_K,      Mg)
        V_Tp = _V_mix(P_Pa,      T_K + dT, Mg)
        V_Tm = _V_mix(P_Pa,      T_K - dT, Mg)
        if not all(np.isfinite([V, V_Pp, V_Pm, V_Tp, V_Tm])):
            print(f"  liquid EoS failed  P={P_GPa:.2f}GPa T={T_K:.0f}K Mg_L={Mg:.3f}")
            return np.nan, np.nan, 0.0
    except Exception as e:
        print(f"  liquid EoS failed  P={P_GPa:.2f}GPa T={T_K:.0f}K Mg_L={Mg:.3f}: {e}")
        return np.nan, np.nan, 0.0

    rho    = M_mix / V          # g/cm3
    rho_si = rho * 1e3          # kg/m3
    K_T    = -V * (2*dP) / (V_Pp - V_Pm)          # Pa, isothermal
    alpha  = (V_Tp - V_Tm) / (2*dT) / V           # 1/K
    gam    = alpha * K_T / (rho_si * Cv_mix)      # Grueneisen of the mixture
    K_S    = K_T * (1.0 + alpha * gam * T_K)      # adiabatic: seismic waves need this

    if not (np.isfinite(rho) and rho > 0 and np.isfinite(K_S) and K_S > 0):
        print(f"  liquid EoS unphysical  rho={rho} K_S={K_S}")
        return np.nan, np.nan, 0.0

    Vp = float(np.sqrt(K_S / rho_si)) / 1000.0    # km/s
    return float(rho), Vp, 0.0

def liquid_adiabat_dTdP(P_GPa, T_K, Mg_liquid):
    # dT/dP|_S = gamma * T / K_S, in K/GPa -- replaces the hardwired _DTDP_LIQ
    if not (np.isfinite(P_GPa) and np.isfinite(T_K) and 0.5 < P_GPa < 100.0):
        print(f"  liquid_adiabat_dTdP bad args: P={P_GPa} T={T_K} Mg={Mg_liquid}")
        return np.nan
    Mg     = float(np.clip(Mg_liquid, 0.0, 1.0))
    P_Pa   = P_GPa * 1e9
    M_mix  = Mg*_FO['M'] + (1.0-Mg)*_FA['M']
    Cv_mix = (Mg*_FO['M']*_FO['Cv'] + (1.0-Mg)*_FA['M']*_FA['Cv']) / M_mix
    dP, dT = 0.05e9, 5.0
    V      = _V_mix(P_Pa, T_K, Mg)
    V_Pp   = _V_mix(P_Pa + dP, T_K, Mg)
    V_Pm   = _V_mix(P_Pa - dP, T_K, Mg)
    V_Tp   = _V_mix(P_Pa, T_K + dT, Mg)
    V_Tm   = _V_mix(P_Pa, T_K - dT, Mg)
    if not all(np.isfinite([V, V_Pp, V_Pm, V_Tp, V_Tm])):
        return np.nan
    rho_si = M_mix / V * 1e3
    K_T    = -V * (2*dP) / (V_Pp - V_Pm)
    alpha  = (V_Tp - V_Tm) / (2*dT) / V
    gam    = alpha * K_T / (rho_si * Cv_mix)
    K_S    = K_T * (1.0 + alpha * gam * T_K)
    return gam * T_K / K_S * 1e9

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


def rho_solid_ngibbs(XS, P_GPa, T_K):
    # Solid BML density at the interface, g/cm3. Replaces the frozen _RHO_S_CONST
    # in mole_to_volume_fraction and in Ra.
    p   = composition_from_params({'Mg#': XS, 'T_lit': T_K, 'P_lit': P_GPa})
    O   = compute_oxygen(p)
    out = _ng_forward([P_GPa], [T_K], _comp_values(p, O),
                      NG_T_HEADERS, ['component_moles'])
    return float(_ng_props(out['component_moles'], [P_GPa], [T_K], ['rho'])['rho'][0])


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

    # pressure grid: surface -> P_bml_top.
    # P_lit and P_lit-eps are inserted explicitly: the fixed 3.0 GPa breakpoint means
    # the shallow grid is ~0.03 GPa and the deep grid ~0.18 GPa, so without these two
    # nodes the conductive/adiabatic kink (and therefore the LVZ sharpness, and
    # therefore the P-wave shadow zone and tt_n_miss) would be smeared by an amount
    # that depends systematically on whether P_lit falls below or above 3.0 GPa.
    P = np.unique(np.concatenate([np.linspace(0.01, 3.0, NG_N_SHALLOW),
                                  np.linspace(3.0, P_bml_top, NG_N_DEEP)[1:],
                                  [P_lit - 1e-4, P_lit]]))
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


class _PDFail(Exception):
    """相圖/EoS 在求根過程中失敗 → 外層 reject 該樣本"""
    pass


def _interface_state(h_s, T_core, P_top, P_bottom, Mg_bulk, bml_thickness, rho_S):
    """給定試探 h_solid,回傳 (h_pred, pd, T_interface, P_int, rho_L_int) 或 None。
    內層 3 次小迭代處理 dTdP 對 XL 的弱依賴(收斂極快)。
    rho_S (g/cm3) 由外層的自洽迴圈提供。"""
    P_int = P_top + (P_bottom - P_top) * h_s / bml_thickness
    XL_g  = Mg_bulk
    pd    = None
    T_int = T_core
    for _ in range(3):
        dTdP = liquid_adiabat_dTdP(P_bottom, T_core, XL_g)
        if not np.isfinite(dTdP):
            return None
        T_int = T_core - dTdP * (P_bottom - P_int)
        pd = bml_phase_diagram(T_int, P_int, Mg_bulk)
        if pd is None:
            return None
        if pd['melting'] and np.isfinite(pd.get('XL', np.nan)):
            XL_g = pd['XL']
    if not pd['melting']:                      # 全固:界面收在 CMB
        return bml_thickness, pd, T_int, P_int, np.nan
    if pd['f_solid'] < 1e-3:                   # 全熔
        return 0.0, pd, T_int, P_int, np.nan
    rho_L_int, _, _ = liquid_bml_properties(P_int, T_int, pd['XL'])
    if not np.isfinite(rho_L_int):
        return None
    phi_S = mole_to_volume_fraction(pd['f_solid'], pd['XS'], pd['XL'],
                                    rho_S, rho_L_int)
    return bml_thickness * phi_S, pd, T_int, P_int, rho_L_int


def run_ngibbs_bml(params, T_core, T_mantle_bottom, true_cmb_depth, n_points=20):
    bml_thickness = params['BML_thickness']
    Mg_bulk       = params['Mg#_bulk_bml']
    bml_top_depth = true_cmb_depth - bml_thickness
    P_top         = max(float(pressure_mars(bml_top_depth)), 5.0)
    P_bottom      = float(pressure_mars(true_cmb_depth))

    Ra            = np.nan
    thermal_state = 'undefined'

    # ── 自洽界面 + 自洽 rho_S ────────────────────────────────────────────────
    # 內層:未知數 h_solid,殘差 g(h) = h − h_pred(h)。
    #        g(0) = −h_pred(0) ≤ 0、g(D) = D − h_pred(D) ≥ 0 恆成立 → bracket 保證存在。
    # 外層:rho_S 由 nGibbs 在解出的 (XS, P_int, T_int) 現算,最多兩趟。
    #        不在 brentq 內部呼叫 nGibbs —— 那會慢十倍。
    rho_S = _RHO_S_CONST
    for _pass in range(2):
        def _resid(h_s):
            st = _interface_state(h_s, T_core, P_top, P_bottom, Mg_bulk,
                                  bml_thickness, rho_S)
            if st is None:
                raise _PDFail()
            return h_s - st[0]
        try:
            g0 = _resid(0.0)
            gD = _resid(bml_thickness)
            if gD <= 0.0:                # h_pred(D) >= D → 全固
                h_solid_km = bml_thickness
            elif g0 >= 0.0:              # h_pred(0) <= 0 → 全熔
                h_solid_km = 0.0
            else:
                h_solid_km = brentq(_resid, 0.0, bml_thickness, xtol=0.5)  # 0.5 km
        except _PDFail:
            return None

        st = _interface_state(h_solid_km, T_core, P_top, P_bottom, Mg_bulk,
                              bml_thickness, rho_S)
        if st is None:
            return None
        _, pd, T_interface, P_int, rho_L_int = st

        # 全固 / 全熔 用不到 rho_S(沒有兩相體積換算)→ 不必第二趟
        if not (pd['melting'] and 1.0 <= h_solid_km < bml_thickness - 1e-9):
            break
        rho_new = rho_solid_ngibbs(pd['XS'], P_int, T_interface)
        if not np.isfinite(rho_new) or rho_new <= 0:
            return None
        converged = abs(rho_new - rho_S) < 0.01
        rho_S = rho_new          # 先更新,再決定是否停 → rho_S 與 h_solid_km 同步
        if converged:
            break

    # ── 依解出的 h_solid 分三態,明確設定所有狀態變數 ─────────────────────────
    if h_solid_km < 1.0:
        # 全熔(含 brentq 解出 <1 km 殘餘固體的 snap)
        h_solid_km, h_liquid_km = 0.0, bml_thickness
        Mg_solid, Mg_liquid     = np.nan, Mg_bulk
        Ra, thermal_state       = np.nan, 'molten'
        dTdP = liquid_adiabat_dTdP(P_bottom, T_core, Mg_bulk)
        if not np.isfinite(dTdP):
            return None
        T_interface = T_core - dTdP * (P_bottom - P_top)
        P_int       = P_top
    elif (not pd['melting']) or h_solid_km >= bml_thickness - 1e-9:
        # 全固
        h_solid_km, h_liquid_km = bml_thickness, 0.0
        Mg_solid, Mg_liquid     = Mg_bulk, np.nan
        T_interface, P_int      = T_core, P_bottom
    else:
        # 分層:成分從相圖取回
        h_liquid_km         = bml_thickness - h_solid_km
        Mg_solid, Mg_liquid = pd['XS'], pd['XL']

    # Ra 事後診斷 (descriptor, not driver)
    if h_solid_km >= 1.0:
        Ra, thermal_state = compute_bml_thermal_state(
            T_mantle_bottom, T_interface, h_solid_km, bml_top_depth,
            rho_solid=rho_S * 1000.0)

    print(f"  BML: Ra={Ra:.1e} [{thermal_state}]  "
          f"h_sol={h_solid_km:.0f}km  h_liq={h_liquid_km:.0f}km  "
          f"T_int={T_interface:.0f}K  rho_S={rho_S:.3f}  melt={pd['melting']}")

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

    # ── Step 3b: liquid BML (Thomas EoS, Vs=0) ───────────────────────────────
    liquid_data = None
    if h_liquid_km >= 1.0 and not np.isnan(Mg_liquid):
        n_liq   = max(int(n_points * h_liquid_km / bml_thickness), 3)
        P_liq   = np.linspace(P_top + (P_bottom-P_top)*h_solid_km/bml_thickness, P_bottom, n_liq)
        rho_liq = np.zeros(n_liq)
        Vp_liq  = np.zeros(n_liq)
        T_liq_arr = np.linspace(T_interface, T_core, n_liq)
        for i in range(n_liq):
            rho_liq[i], Vp_liq[i], _ = liquid_bml_properties(P_liq[i], T_liq_arr[i], Mg_liquid)

        off = solid_data['depth_km'][-1] if solid_data is not None else np.interp(P_liq[0], _pres_gpa, _pres_depth)
        depth_liq = np.interp(P_liq, _pres_gpa, _pres_depth)
        clash = depth_liq <= off
        if clash.any():
            depth_liq[clash] = off + 0.002 * (np.arange(clash.sum()) + 1)

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
        'Mg_solid':                Mg_solid if not np.isnan(Mg_solid) else -1.0,
        'Mg_liquid':               Mg_liquid if not np.isnan(Mg_liquid) else -1.0,
        'melting':                 pd['melting'],
        'T_interface':             T_interface,
        'rho_solid_bml':           float(np.mean(solid_data['rho']))  if solid_data  is not None else -1.0,
        'rho_liquid_bml':          float(np.mean(liquid_data['rho'])) if liquid_data is not None else -1.0,
        'rho_S_interface':         float(rho_S),
        'P_interface':             float(P_int),
        'interface_solver':        'brentq',
    }


# ── TauP ──────────────────────────────────────────────────────────────────────
def _validate_nd_depths(*depth_arrays):
    """驗證即將寫進 .nd 的深度(以 %.3f 捨入後):禁倒退、禁三連同值。
    成對重複(不連續面)合法。"""
    d  = np.round(np.concatenate([np.asarray(a, float) for a in depth_arrays]), 3)
    dd = np.diff(d)
    if np.any(dd < 0):
        raise ValueError("nd profile: depth not monotonic (after rounding)")
    same = (dd == 0)
    if same.size > 1 and np.any(same[:-1] & same[1:]):
        raise ValueError("nd profile: >=3 consecutive identical depths (after rounding)")

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

        _validate_nd_depths(man_depth, d_b)
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
def compute_solidus_penalty(fort56_data, params, true_cmb_depth):
    P_prof = fort56_data.get('P_profile')
    T_prof = fort56_data.get('T_profile')
    if P_prof is None or T_prof is None:
        return 0.0
    bml_top_km  = true_cmb_depth - params['BML_thickness']
    P_bml_top   = float(pressure_mars(bml_top_km))
    mask        = (P_prof >= params['P_lit']) & (P_prof <= P_bml_top)
    P_m, T_m   = P_prof[mask], T_prof[mask]
    if len(P_m) == 0:
        return 0.0
    # Mg#-dependent: more Fe lowers the solidus (Elkins-Tanton 2008), exactly as in
    # Drilleau 2026. The fixed Mg#=0.75 curve made this penalty blind to Mg#.
    T_sol_arr   = np.array([solidus_duncan_Mg(p, params['Mg#']) for p in P_m])
    excess  = np.clip(T_m - T_sol_arr, 0.0, None)
    penalty = min((np.max(excess) / SOLIDUS_SIGMA)**2, SOLIDUS_CAP) if len(excess) else 0.0
    if penalty > 0:
        print(f"  Solidus penalty = {penalty:.4f}")
    return penalty

# ── misfit ────────────────────────────────────────────────────────────────────
def compute_misfit(taup_model, obs_dataset, fort56_data, bml_data, params=None,
                   true_cmb_depth=None,
                   mass_sigma=0.0, moi_sigma=0.0, grav_sigma=0.0, ric_sigma=0.0):
    phases_std   = ['P', 'S', 'pP', 'sP', 'PP', 'PPP', 'SS', 'SSS', 'sS', 'ScS', 'SKS']
    phases_pdiff = phases_std + ['Pdiff']

    tt_total    = 0.0   # Σ over events of each event's MEAN residual
    tt_n_ev     = 0     # events that contributed
    tt_n_ph     = 0     # total phases used (report only)
    tt_n_miss   = 0     # phases that the model could not predict
    tt_n_capped = 0     # present phases whose residual hit RES_CAP
    miss_by_event = {}  # per-event miss count (the penalty is diluted by ev_n,
                        # so its strength varies ~4x between 2-phase and 8-phase events)

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
        def _tdiff(times, a, b):
            ta, tb = times.get(a), times.get(b)
            return None if (ta is None or tb is None) else ta - tb

        pred = {
            'S-P':              _tdiff(times, 'S',   'P'),
            'pP-P':             _tdiff(times, 'pP',  'P'),
            'sP-P':             _tdiff(times, 'sP',   'P'),
            'PP-P':             _tdiff(times, 'PP',   'P'),
            'PPP-P':            _tdiff(times, 'PPP',  'P'),
            'sS-S':             _tdiff(times, 'sS',   'S'),
            'SS-S':             _tdiff(times, 'SS',   'S'),
            'SSS-S':            _tdiff(times, 'SSS',  'S'),
            'ScS-S':            _tdiff(times, 'ScS',  'S'),
            'SS-PP':            _tdiff(times, 'SS',   'PP'),
            'SKS-PP':           _tdiff(times, 'SKS',  'PP'),
            'PP-PbdiffPcP':     _tdiff(times, 'PP',   'Pdiff'),
        }

        ev_sum, ev_n, ev_miss = 0.0, 0, 0
        for phase, obs_val in obs.items():
            if phase in ('delta', 'depth') or not isinstance(obs_val, tuple):
                continue
            p_val = pred.get(phase)
            if p_val is None:
                # model cannot predict this phase (shadow zone / caustic)
                ev_sum += MISS_PENALTY; ev_n += 1; ev_miss += 1
                continue
            obs_t, sigma = obs_val
            if sigma <= 0 or not np.isfinite(obs_t):
                continue                 # data problem: do not penalise the model
            res_sigma = abs(obs_t - p_val) / sigma
            if not np.isfinite(res_sigma):
                ev_sum += MISS_PENALTY; ev_n += 1; ev_miss += 1
                continue
            if res_sigma >= RES_CAP:
                tt_n_capped += 1
            ev_sum += min(res_sigma, RES_CAP); ev_n += 1

        if ev_n > 0:
            tt_total  += ev_sum / ev_n
            tt_n_ev   += 1
            tt_n_ph   += ev_n - ev_miss
            tt_n_miss += ev_miss
            miss_by_event[event] = ev_miss

    tt_misfit = tt_total if tt_n_ev > 0 else 999.0
    if not np.isfinite(tt_misfit):
        tt_misfit = 999.0

    n_ev_expected = len(obs_dataset)
    if tt_n_ev < n_ev_expected:
        tt_misfit += EVENT_FAIL * (n_ev_expected - tt_n_ev)

    solidus_penalty = (compute_solidus_penalty(fort56_data, params, true_cmb_depth)
                       if params else 0.0)
    total_misfit = tt_misfit + solidus_penalty + mass_sigma + moi_sigma + grav_sigma + ric_sigma
    if not np.isfinite(total_misfit):
        total_misfit = 999.0

    print(f"  TT={tt_misfit:.4f}(events={tt_n_ev}/{n_ev_expected}, phases={tt_n_ph}, "
          f"miss={tt_n_miss}, capped={tt_n_capped})  "
          f"solidus={solidus_penalty:.4f}  mass={mass_sigma:.2f}σ  "
          f"moi={moi_sigma:.2f}σ  grav={grav_sigma:.2f}σ  ric={ric_sigma:.2f}σ  total={total_misfit:.4f}")

    return total_misfit, tt_n_ph, {
        'misfit_tt': tt_misfit, 'misfit_solidus': solidus_penalty,
        'grav_sigma': grav_sigma, 'ric_sigma': ric_sigma,
        'tt_n_ev': tt_n_ev, 'tt_n_ph': tt_n_ph, 'tt_n_miss': tt_n_miss,
        'tt_n_capped': tt_n_capped, 'tt_miss_by_event': miss_by_event,
    }

# ── forward model ─────────────────────────────────────────────────────────────
def forward(params, run_dir, model_name, samuel_cache):
    true_cmb_depth = samuel_cache['true_cmb_depth']
    bml_top_km     = true_cmb_depth - params['BML_thickness']
    P_bml_top      = float(pressure_mars(bml_top_km))

    # ── a26: Fe-S EoS core profile (replaces fixed Samuel core) ──────────────
    # P_cmb 來自 Samuel 壓力場(非完全自洽,第一版近似;w_S 掃 0.05-0.30 時
    # 漂移 < 0.3 GPa,可接受)。R_cmb 固定為 Samuel 的 metal CMB。
    R_cmb_km = MARS_RADIUS - true_cmb_depth
    P_cmb    = float(pressure_mars(true_cmb_depth))
    core_prof = CF.build_core_profile(params['T_core'], params['w_S'],
                                      R_cmb_km=R_cmb_km, P_cmb=P_cmb)
    if core_prof is None:
        print("  core EoS failed (out of table range) → reject")
        return None, None, None, None, None

    core_z_eos   = np.append(MARS_RADIUS - core_prof['r']/1e3, MARS_RADIUS)
    core_vp_eos  = np.append(core_prof['Vp'],      core_prof['Vp'][-1])
    core_rho_eos = np.append(core_prof['rho']/1e3, core_prof['rho'][-1]/1e3)
    cache_eff = {**samuel_cache,
                 'core_z':   core_z_eos,
                 'core_vp':  core_vp_eos,
                 'core_vs':  np.zeros_like(core_z_eos),
                 'core_rho': core_rho_eos}

    R_ic = CF.find_R_IC(core_prof)      # 靜態 ICB 溫度計;固核密度未回饋(見 notes)
    ric_sigma = 0.0
    if USE_RIC_LIKELIHOOD:
        ric_sigma = abs(BI_RIC_KM - R_ic) / BI_RIC_SIG
        ric_sigma += max(0.0, (R_ic - RIC_MAX_KM) / 50.0)   # InSight R_ic<750 km 軟上限
    print(f"  core: w_S={params['w_S']:.3f}  P_c={core_prof['P_center']:.1f}GPa  "
          f"T_c={core_prof['T_center']:.0f}K  rho_mean={core_prof['rho_mean']/1e3:.3f}  "
          f"Vp_cmb={core_prof['Vp'][0]:.2f}  R_ic={R_ic:.0f}km")

    fort56_data = run_ngibbs(params, P_bml_top=P_bml_top)
    if fort56_data is None:
        return None, None, None, None, None

    T_profile = fort56_data.get('T_profile')
    P_profile = fort56_data.get('P_profile')
    T_mantle_bottom = (float(np.interp(P_bml_top, P_profile, T_profile))
                       if T_profile is not None else float(fort56_data['T_K'][-1]))
    rho_mantle_bot = float(fort56_data['rho'][-1])
    print(f"  T_mantle_bottom={T_mantle_bottom:.1f}K  P_bml_top={P_bml_top:.2f}GPa  "
          f"rho_bot={rho_mantle_bot:.4f}")

    bml_raw = run_ngibbs_bml(params,
                             T_core=params['T_core'],
                             T_mantle_bottom=T_mantle_bottom,
                             true_cmb_depth=true_cmb_depth)
    if bml_raw is None:
        print("  BML failed → reject"); return None, None, None, None, None

    bml_raw['depth_km'] = bml_raw['depth_km'] - bml_raw['depth_km'][0] + bml_top_km
    bml_raw['outer_core_depth_abs'] = bml_top_km + bml_raw['outer_core_depth_offset']

    rho_bml_top    = float(bml_raw['rho'][0])
    rho_bml_bot    = float(bml_raw['rho'][-1])
    rho_core_top   = float(cache_eff['core_rho'][0])   # a26: EoS 核頂密度 (Kono 穩定性)

    bml_raw['upper_contrast'] = rho_bml_top - rho_mantle_bot
    bml_raw['lower_contrast'] = rho_core_top - rho_bml_bot
    print(f"  density contrasts: upper={bml_raw['upper_contrast']:+.4f}  "
          f"lower={bml_raw['lower_contrast']:+.4f}")

    grav_sigma = (max(0.0, -bml_raw['upper_contrast']) +
                  max(0.0, -bml_raw['lower_contrast'])) / GRAV_SCALE
    if REJECT_GRAV_UNSTABLE and grav_sigma > 0:
        if bml_raw['upper_contrast'] <= 0.0: _GRAV_FAIL['upper'] += 1
        if bml_raw['lower_contrast'] <= 0.0: _GRAV_FAIL['lower'] += 1
        print("  BML gravitationally unstable → reject (Kono 2025)")
        return None, None, None, None, None

    bml_data = bml_raw

    # mass and MoI as soft constraints (contribute to misfit, no hard rejection)
    M_pred, moi_pred = compute_mass_and_moi(fort56_data, cache_eff, bml_data)
    mass_sigma = abs(MARS_MASS_OBS - M_pred) / MARS_MASS_SIGMA
    moi_sigma = abs(MOI_OBS - (moi_pred - MOI_BIAS)) / MOI_SIGMA
    print(f"  mass={mass_sigma:.2f}σ  moi={moi_sigma:.2f}σ")

    try:
        taup_model = build_taup(fort56_data, model_name, cache_eff, bml_data=bml_data)
    except Exception as e:
        print(f"  TauP failed: {e}"); return None, None, None, None, None

    misfit, n_data, components = compute_misfit(
        taup_model, SAMUEL_DATA, fort56_data, bml_data=bml_data, params=params,
        true_cmb_depth=true_cmb_depth,
        mass_sigma=mass_sigma, moi_sigma=moi_sigma, grav_sigma=grav_sigma,
        ric_sigma=ric_sigma)

    components.update({k: bml_data.get(k) for k in BML_KEYS})
    components['S_lit'] = float(fort56_data['S'][0])
    components['mass_sigma']      = mass_sigma
    components['moi_sigma']       = moi_sigma
    components['moi_pred']        = moi_pred
    components['M_pred']          = M_pred
    components['T_mantle_bottom'] = T_mantle_bottom
    components['R_ic_km']        = R_ic
    components['ric_sigma']      = ric_sigma
    components['rho_core_mean']  = core_prof['rho_mean']/1e3
    components['Vp_core_cmb']    = float(core_prof['Vp'][0])
    components['P_center_core']  = core_prof['P_center']
    components['T_center_core']  = core_prof['T_center']
    # 讓 run_mcmc 的 npz 能存核剖面(bml_data 只是載體,不進 json)
    bml_data['core_depth_km'] = core_z_eos[:-1]
    bml_data['core_Vp']       = core_prof['Vp']
    bml_data['core_rho']      = core_prof['rho']/1e3
    bml_data['core_T_K']      = core_prof['T']
    bml_data['core_P_GPa']    = core_prof['P']
    bml_data['core_R_ic_km']  = R_ic

    return misfit, n_data, components, fort56_data, bml_data

# ── MCMC ──────────────────────────────────────────────────────────────────────
def propose(current, rng):
    # No truncation: out-of-prior proposals are counted as rejections in run_mcmc.
    # resample-until-in-bounds would be an unnormalised truncated Gaussian,
    # q(x'|x) != q(x|x'), breaking detailed balance near the walls.
    return {k: current[k] + rng.normal(0, STEP[k]) for k in PRIOR}


def run_mcmc(chain_id, n_steps, start_params=None, prefix='chain'):
    chain_dir = os.path.join(MCMC_DIR, f"{prefix}_{chain_id:02d}")
    os.makedirs(chain_dir, exist_ok=True)

    samuel_cache = _samuel_cache

    rng     = np.random.default_rng(42 + chain_id)
    current = (start_params or START_PARAMS).copy()

    chain_file = os.path.join(chain_dir, "chain.jsonl")
    chain = []
    if os.path.exists(chain_file):
        with open(chain_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chain.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        if chain:
            current = chain[-1]['params']
            print(f"Chain {chain_id}: resuming from step {len(chain)}")

    step_start = len(chain)
    accept_count = 0
    current_components = {k: None for k in CHAIN_KEYS}          # 佔位

    if chain:
        current_misfit = chain[-1]['misfit']
        accept_count   = sum(1 for s in chain if s.get('accepted', False))
        current_components = {k: chain[-1].get(k) for k in CHAIN_KEYS}
    else:
        run_dir    = os.path.join(chain_dir, "step_current")
        model_name = f"mcmc_{prefix}_c{chain_id:02d}_current"
        for attempt in range(20):
            trial = current.copy() if attempt == 0 else {
                k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PRIOR.items()}
            if attempt > 0:
                print(f"  Retry {attempt}: T_lit={trial['T_lit']:.1f}  "
                      f"Mg#={trial['Mg#']:.3f}  T_core={trial['T_core']:.1f}  "
                      f"Mg#_bulk_bml={trial['Mg#_bulk_bml']:.3f}  w_S={trial['w_S']:.3f}")
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

        out_of_prior = any(not (PRIOR[k][0] <= proposed[k] <= PRIOR[k][1])
                           for k in PRIOR)

        if out_of_prior:
            accepted        = False
            proposed_misfit = 999.0
            fort56_data     = None
            bml_data        = None
        else:
            proposed_misfit, n_data, components, fort56_data, bml_data = forward(
                proposed, run_dir, model_name, samuel_cache)

            if proposed_misfit is None or proposed_misfit >= 990.0:
                accepted        = False
                proposed_misfit = 999.0
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
                    for _ck in ('core_depth_km', 'core_Vp', 'core_rho',
                                'core_T_K', 'core_P_GPa'):
                        if _ck in bml_data:
                            npz_dict[_ck] = bml_data[_ck]
                    npz_dict['core_R_ic_km'] = np.array(
                        [bml_data.get('core_R_ic_km', 0.0)])
                np.savez(os.path.join(chain_dir, f"profile_s{step+1:05d}.npz"), **npz_dict)

        elapsed     = (datetime.now() - t0).total_seconds()
        accept_rate = accept_count / (step + 1) * 100
        uc          = current_components.get('upper_contrast')
        print(f"  Step {step+1:4d}: misfit={current_misfit:.4f}  "
              f"{'ACCEPT' if accepted else 'reject'}  "
              f"rate={accept_rate:.1f}%  "
              f"uc={f'{uc:+.4f}' if uc is not None else 'N/A'}  "
              f"({elapsed:.0f}s)")

        record = {
            'step':        step + 1,
            'params':      current,
            'misfit':      current_misfit,
            'accepted':    bool(accepted),
            'accept_rate': accept_rate,
            'proposal_S_lit':      _NG_S_LOG['last'],
            'proposal_S_in_range': bool(_NG_S_LOG['last_in_range']),
        }
        record.update({k: current_components.get(k) for k in CHAIN_KEYS})
        chain.append(record)

        with open(chain_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    print(f"\nChain {chain_id} done  "
          f"accept_rate={accept_count/n_steps*100:.1f}%  "
          f"misfit={current_misfit:.4f}")

    print(f"  phase-diagram failures: no_loop={_PD_FAIL['no_loop']}  "
          f"no_invariant={_PD_FAIL['no_invariant']}  solidus={_PD_FAIL['solidus']}")
    print(f"  phase branch: gamma={_PHASE_USED['gamma']}  "
          f"three-phase={_PHASE_USED['three-phase']}  beta={_PHASE_USED['beta']}")
    print(f"  scan cache entries: {len(_SCAN_CACHE)}")

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
        # resume runs AFTER start_params is set, so a leftover chain.jsonl would
        # silently override the random start.
        _cf = os.path.join(MCMC_DIR, f"{args.prefix}_{args.chain:02d}", "chain.jsonl")
        if os.path.exists(_cf) and os.path.getsize(_cf) > 0:
            raise SystemExit(f"{_cf} exists: --random_start would be overridden by "
                             f"resume. Remove it or use a different --prefix.")
        rng_init     = np.random.default_rng(args.chain)
        start_params = {k: float(rng_init.uniform(lo, hi)) for k, (lo, hi) in PRIOR.items()}
        print(f"random_start: {start_params}")

    if args.test:
        run_mcmc(0, 1, prefix=args.prefix, start_params=start_params)
    else:
        run_mcmc(args.chain, args.steps, prefix=args.prefix, start_params=start_params)