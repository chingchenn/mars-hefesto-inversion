#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMC inversion for Mars interior structure using nGibbs (HeFESTo emulator) + BML physics
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

# ── Fe-S core EoS (a36: core in forward model) ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_fes as CF     



BI_RIC_KM  = 612.0           # Bi 2025 refined inversion
BI_RIC_SIG = 112.0           # sqrt(50^2 + 100^2): 形式誤差 + OC 速度 5%→100 km 系統項
RIC_MAX_KM = 750.0           # InSight 上限(SKS 最大穿透深度, Irving 2023)

NG_T_HEADERS = ['P(GPa)(System_main)', 'T(K)(System_main)',
                'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
NG_S_HEADERS = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)',
                'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
NG_S_MIN     = 1.75
NG_S_MAX     = 2.90


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
MOI_BIAS         = 0.0  # pipeline systematic: pure-Samuel PanelJ through this
                             # integrator gives MoI=0.36515 vs obs 0.36379 (budget test)
MCMC_TEMPERATURE = 1.0

SC_N_FIXED  = 40      # 節點數
SC_Z_FIXED  = 100.0   # km,外借區段底部
SC_N_MAN   = 250       # 地函節點
SC_N_BML   = 40        # BML 節點
SC_N_CORE  = 60        # 核節點
SC_N_ITER  = 8         # 鬆弛上限
SC_TOL_P   = 0.01      # GPa,收斂判準 max|P_new - P_old|
SC_G_GRAV  = 6.674e-11


GAP_ON      = False
GAP_SIGMA   = 150.0    # K
RIC_MAX_KM  = 750.0
RIC_MAX_SIG = 50.0     # km

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
'R_cmb':1650, #km 
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
                           # 讓 posterior 與 17 wt% 的關係成為結果
'R_cmb': (1400, 1900),
}  

STEP = {
'T_lit': 50.0,
'P_lit': 0.7,
'Mg#': 0.025,
'T_core': 100.0,
'Mg#_bulk_bml': 0.06,
'BML_thickness': 25.0,
'w_S': 0.02,
'R_cmb':25.0,
}

CHAIN_KEYS = (
    'misfit_tt', 'misfit_solidus', 'grav_sigma', 'mass_sigma', 'moi_sigma',
    'moi_pred', 'M_pred','upper_contrast', 'lower_contrast', 'Ra', 'thermal_state',
    'h_solid_km', 'h_liquid_km', 'melting', 'Mg_solid', 'Mg_liquid',
    'rho_solid_bml', 'rho_liquid_bml',  'P_interface',
    'T_mantle_bottom', 'T_interface', 'S_lit', 'tt_n_ev', 'tt_n_ph', 'tt_n_miss',
    'tt_n_capped', 'tt_miss_by_event',
    'R_ic_km', 'ric_sigma', 'rho_core_mean', 'Vp_core_cmb',
    'P_center_core', 'T_center_core',
    'gap_min', 'P_cmb', 'P_bml_top', 'z_lit_km',
    'sc_n_iter', 'sc_dP', 'sc_dz_lit', 'bml_n_pass',
)

PROP_SCALE  = 2.38**2 / 8.0
PARAM_ORDER = ('T_lit', 'P_lit', 'Mg#', 'T_core',
               'Mg#_bulk_bml', 'BML_thickness', 'w_S', 'R_cmb')

def _load_prop_chol(route):
    path = f'prop_cov_a37{route}.npz'
    if not os.path.exists(path):
        return None
    d = np.load(path)
    assert tuple(d['param_order']) == PARAM_ORDER
    print(f"proposal: cov loaded from {path}", flush=True)
    return np.linalg.cholesky(PROP_SCALE * d['cov'])

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
    G = SC_G_GRAV
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

# ── mass and MoI ──────────────────────────────────────────────────────────────
def sc_build_grid(params, z_lit=None):
    """幾何由 R_cmb 與 BML_thickness 決定,所以深度是已知的自變數。
    z_lit 給定時在地函網格插入該深度兩側的節點(保住傳導/絕熱轉折)。"""
    z_cmb     = MARS_RADIUS - params['R_cmb']
    z_bml_top = z_cmb - params['BML_thickness']
    if z_bml_top <= SC_Z_FIXED:
        return None, None, None

    z_fx  = np.linspace(0.0, SC_Z_FIXED, SC_N_FIXED)
    z_sh_bot = min(max(SC_Z_FIXED + 400.0,
                       (z_lit + 100.0) if z_lit is not None else 0.0),
                   z_bml_top)
    n_sh = int(SC_N_MAN*0.6)
    z_sh = np.linspace(SC_Z_FIXED, z_sh_bot, n_sh)
    if z_sh_bot < z_bml_top:
        z_dp  = np.linspace(z_sh_bot, z_bml_top, SC_N_MAN - n_sh + 1)[1:]
        z_man = np.concatenate([z_sh, z_dp])
    else:
        z_man = z_sh
    if z_lit is not None and SC_Z_FIXED < z_lit < z_bml_top:
        z_man = np.unique(np.concatenate([z_man, [z_lit - 0.02, z_lit]]))
    z_bml = np.linspace(z_bml_top, z_cmb,     SC_N_BML)
    z_cor = np.linspace(z_cmb,     MARS_RADIUS, SC_N_CORE)

    z   = np.concatenate([z_fx, z_man, z_bml, z_cor])
    lay = np.concatenate([np.zeros(len(z_fx), int), np.ones(len(z_man), int),
                          2*np.ones(len(z_bml), int), 3*np.ones(len(z_cor), int)])
    return z, lay, dict(z_bml_top=z_bml_top, z_cmb=z_cmb)


def sc_gravity(z_km, rho_si):
    r = (MARS_RADIUS - z_km) * 1e3                 # m,由大到小
    o = np.argsort(r)                              # 由中心往外
    ra, rhoa = r[o], rho_si[o]
    M = np.zeros(len(ra))
    rhm = 0.5*(rhoa[1:] + rhoa[:-1])
    M[1:] = np.cumsum(4*np.pi/3*rhm*(ra[1:]**3 - ra[:-1]**3))
    g = np.zeros(len(ra))
    ok = ra > 1.0
    g[ok] = SC_G_GRAV*M[ok]/ra[ok]**2
    inv = np.empty(len(ra), int); inv[o] = np.arange(len(ra))
    return g[inv], M[inv]

def sc_integrate_P(z_km, rho_si, g):
    """
        dP/dz = rho(z) g(z) ,  P(0) = 0
        P_i = P_{i-1} + 0.5*[(rho g)_i + (rho g)_{i-1}] * (z_i - z_{i-1})
    """
    w  = np.asarray(rho_si, float)*np.asarray(g, float)   # Pa/m
    dz = np.diff(z_km)*1e3                                # m
    P  = np.zeros(len(z_km))
    P[1:] = np.cumsum(0.5*(w[1:] + w[:-1])*dz)/1e9
    return P


def sc_fixed_layer(z_km, samuel_cache):
    rho = np.interp(z_km, samuel_cache['crust_z'], samuel_cache['crust_rho'])
    vp  = np.interp(z_km, samuel_cache['crust_z'], samuel_cache['crust_vp'])
    vs  = np.interp(z_km, samuel_cache['crust_z'], samuel_cache['crust_vs'])
    T   = T_SURF + np.zeros_like(z_km)      # T 由 sc_update_T 覆寫
    return rho, vp, vs, T


def sc_mantle(params, P_man):
    """
    P < P_lit linear 
    P >= P_lit S = S_lit 
    """
    p = composition_from_params(params)
    O = compute_oxygen(p)
    T_lit, P_lit = p['T_lit'], p['P_lit']
    cv = _comp_values(p, O)
 
    out1  = _ng_forward([P_lit], [T_lit], cv, NG_T_HEADERS, ['component_moles'])
    S_lit = float(_ng_props(out1['component_moles'], [P_lit], [T_lit], ['S'])['S'][0])
 
    in_range = np.isfinite(S_lit) and NG_S_MIN <= S_lit <= NG_S_MAX
    _NG_S_LOG['last'] = S_lit
    _NG_S_LOG['last_in_range'] = in_range
    _NG_S_LOG['all'].append(S_lit)
    if np.isfinite(S_lit) and S_lit < NG_S_MIN: _NG_S_LOG['n_below'] += 1
    if np.isfinite(S_lit) and S_lit > NG_S_MAX: _NG_S_LOG['n_above'] += 1
    if not in_range:
        print(f"  S_lit={S_lit:.4f} outside nGibbs range [{NG_S_MIN}, {NG_S_MAX}]")
        return None
 
    P = np.asarray(P_man, float)
    if np.any(P <= 0) or not np.all(np.isfinite(P)):
        print(f"  mantle: non-physical P in field (min={np.nanmin(P):.3f} GPa) -> reject")
        return None
    isen = P >= P_lit
    iso  = ~isen
    n = len(P)
    T = np.zeros(n); rho = np.zeros(n); Vp = np.zeros(n); Vs = np.zeros(n)
 
    if isen.any():
        o2 = _ng_forward(P[isen], np.full(int(isen.sum()), S_lit), cv,
                         NG_S_HEADERS, ['component_moles', 'temperature'])
        T_is = np.asarray(o2['temperature'].detach().cpu(), dtype=np.float64)
        T[isen] = T_is
        pr = _ng_props(o2['component_moles'], P[isen], T_is, ['rho', 'Vp', 'Vs'])
        rho[isen], Vp[isen], Vs[isen] = pr['rho'], pr['Vp'], pr['Vs']
 
    if iso.any():
        T_is = T_SURF + (T_lit - T_SURF)*(P[iso]/P_lit)
        T[iso] = T_is
        o3 = _ng_forward(P[iso], T_is, cv, NG_T_HEADERS, ['component_moles'])
        pr = _ng_props(o3['component_moles'], P[iso], T_is, ['rho', 'Vp', 'Vs'])
        rho[iso], Vp[iso], Vs[iso] = pr['rho'], pr['Vp'], pr['Vs']
 
    if not (np.all(np.isfinite(rho)) and np.all(rho > 0)
            and np.all(np.isfinite(Vp)) and np.all(np.isfinite(Vs))):
        print("  mantle nGibbs NaN/Inf -> reject")
        return None
    return dict(rho=rho, Vp=Vp, Vs=Vs, T=T, S_lit=S_lit)

 
def sc_core(params, z_cor, P_cmb, g_cor):
    """ 
    dP = rho g dz ;  dT = (gamma T / K_S) dP
    不再呼叫 CF.build_core_profile —— 那是獨立的自重積分,
    與全行星那一條會重複且不一致。
    """
    n   = len(z_cor)
    w_S = params['w_S']
    P = np.full(n, P_cmb); T = np.full(n, params['T_core'])
    rho = np.full(n, 6800.0)
    for _ in range(12):
        pr = CF.core_properties(P, T, w_S)
        rho_new = pr['rho']*1e3
        if not np.all(np.isfinite(rho_new)):
            return None
        dTdP = np.where(np.isfinite(pr['dTdP']), pr['dTdP'], 0.0)
        dz = np.diff(z_cor)*1e3
        P_new = np.empty(n); T_new = np.empty(n)
        P_new[0], T_new[0] = P_cmb, params['T_core']
        w = rho_new*g_cor
        for i in range(1, n):
            P_new[i] = P_new[i-1] + 0.5*(w[i] + w[i-1])*dz[i-1]/1e9
            T_new[i] = T_new[i-1] + dTdP[i-1]*(P_new[i] - P_new[i-1])
        dmax = float(np.max(np.abs(P_new - P)))
        P, T, rho = P_new, T_new, rho_new
        if dmax < 0.005:
            break
    pr = CF.core_properties(P, T, w_S)
    if not np.all(np.isfinite(pr['Vp'])):
        return None
    return dict(r=(MARS_RADIUS - z_cor)*1e3,        # m,由 CMB 到中心
               P=P, T=T, rho=pr['rho']*1e3, Vp=pr['Vp'],
               Vs=np.zeros(n), dTdP=pr['dTdP'])


# ── Samuel 2023 median model ──────────────────────────────────────────────────
def _load_samuel_reference():
    vp_data = np.loadtxt(os.path.join(SAMUEL_FIG1K_DIR, 'vp_profile.dat'))
    vs_data = np.loadtxt(os.path.join(SAMUEL_FIG1K_DIR, 'vs_profile.dat'))
    depth = MARS_RADIUS - vp_data[:, 1]      # km
    vp    = vp_data[:, 0] / 1000.0           # km/s
    vs    = vs_data[:, 0] / 1000.0           # km/s
    o = np.argsort(depth)
    depth, vp, vs = depth[o], vp[o], vs[o]

    crust_z = np.linspace(0.0, SC_Z_FIXED, 200)

    solid = vs >= 0.1
    crust_vp = np.interp(crust_z, depth[solid], vp[solid])
    crust_vs = np.interp(crust_z, depth[solid], vs[solid])
    crust_rho = np.interp(crust_z, _rho_depth, _rho_all)

    return {
        'crust_z':        crust_z,
        'crust_vp':       crust_vp,
        'crust_vs':       crust_vs,
        'crust_rho':      crust_rho,
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

def compute_bml_thermal_state(T_mantle_bottom, T_interface, h_solid_km,
                                    rho_solid, P_mid_Pa, g_mid):
    T_mid = 0.5*(T_mantle_bottom + T_interface)
    dT    = max(T_interface - T_mantle_bottom, 0.0)
    eta   = _ETA0*np.exp((_ESTAR + P_mid_Pa*_VSTAR)/(_R_GAS*T_mid)
                        - (_ESTAR + _P_REF*_VSTAR)/(_R_GAS*_T0_ETA))
    h     = h_solid_km*1000.0
    kappa = K_SOLID/(rho_solid*_CP_S)
    Ra    = _ALPHA_S*rho_solid*g_mid*dT*h**3/(kappa*eta)
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
class _NGFail(Exception):
    # nGibbs 輸入非有限/界外,或 polish_masses 內部發散 → 外層 reject 該樣本
    pass

def _ng_forward(P, T_or_S, comp_values, headers, outputs):
    x = np.zeros((len(P), 2 + len(comp_values)), dtype=np.float32)
    x[:, 0]  = P
    x[:, 1]  = T_or_S
    x[:, 2:] = comp_values
    # 負莫耳數會讓 polish_masses 的 componentAtomMoles 發散成 NaN,
    # pinv 再丟 _LinAlgError 殺掉整個 process
    if not np.all(np.isfinite(x)):
        raise _NGFail(f"nonfinite input: P={P[:3]} T_or_S={T_or_S[:3]} comp={comp_values}")
    if np.any(x[:, 2:] < 0.0):
        raise _NGFail(f"negative moles: comp={comp_values}")
    with torch.no_grad():
        try:
            return EM.ForwardMB(x, headers=headers, outputs=outputs)
        except torch._C._LinAlgError as exc:
            raise _NGFail(f"polish_masses diverged: P={P[:3]} T_or_S={T_or_S[:3]} "
                          f"comp={comp_values}") from exc
def _ng_props(component_moles, P, T, names):
    cm = np.asarray(component_moles.detach().cpu() if torch.is_tensor(component_moles)
                    else component_moles, dtype=np.float64)
    Pa = np.asarray(P, dtype=np.float64)
    Ta = np.asarray(T, dtype=np.float64)
    # pinv(componentAtomMoles) 在含 NaN 時會丟 _LinAlgError 並殺掉整個 process,
    # 所以必須在進 torch 之前攔下來
    if not (np.all(np.isfinite(cm)) and np.all(np.isfinite(Pa)) and np.all(np.isfinite(Ta))):
        raise _NGFail(f"nonfinite: cm={np.sum(~np.isfinite(cm))} "
                      f"P=[{np.nanmin(Pa):.3f},{np.nanmax(Pa):.3f}] "
                      f"T=[{np.nanmin(Ta):.1f},{np.nanmax(Ta):.1f}]")
    PT = np.stack([Pa, Ta], axis=1)
    out = EM.get_property_hefesto_vectorized_from_assemblage(
        torch.tensor(cm, dtype=torch.float64),
        torch.tensor(PT, dtype=torch.float64),
        property_names=names,
    )
    # nGibbs 在界外條件下由 dos_tables/therm_props 產生 NaN 而不報錯;
    # 帶 NaN 的 Vp/Vs/rho 進 TauP 會讓 slowness 溢位並寫壞 heap(malloc/munmap),
    # 那是 SIGABRT,Python 的 try/except 攔不到
    for k in names:
        v = out[k]
        v = np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, dtype=np.float64)
        if not np.all(np.isfinite(v)):
            raise _NGFail(f"nGibbs output {k}: {np.sum(~np.isfinite(v))}/{v.size} nonfinite")
    return out

def _comp_values(p, O):
    return [p['Si'], p['Mg'], p['Fe'], p['Ca'],
            p['Al'], p['Na'], p['Cr'], O]


def rho_solid_ngibbs(XS, P_GPa, T_K):
    if not (np.isfinite(XS) and 0.0 <= XS <= 1.0
            and np.isfinite(P_GPa) and P_GPa > 0.0 and np.isfinite(T_K) and T_K > 0.0):
        raise _NGFail(f"rho_solid_ngibbs bad args: XS={XS} P={P_GPa} T={T_K}")
    # Solid BML density at the interface, g/cm3. Replaces the frozen _RHO_S_CONST
    # in mole_to_volume_fraction and in Ra.
    p   = composition_from_params({'Mg#': XS, 'T_lit': T_K, 'P_lit': P_GPa})
    O   = compute_oxygen(p)
    out = _ng_forward([P_GPa], [T_K], _comp_values(p, O),
                      NG_T_HEADERS, ['component_moles'])
    return float(_ng_props(out['component_moles'], [P_GPa], [T_K], ['rho'])['rho'][0])

class _PDFail(Exception):
    """相圖/EoS 在求根過程中失敗 → 外層 reject 該樣本"""
    pass

def _sc_interface_state(h_s, T_core, z_bml, P_bml, Mg_bulk, D, rho_S, n_seg=5):
    z_int = z_bml[0] + h_s
    P_int = float(np.interp(z_int, z_bml, P_bml))
    P_bot = float(P_bml[-1])
    if n_seg is None:
        P_path = np.concatenate([P_bml[z_bml > z_int][::-1], [P_int]])
    else:
        P_path = np.linspace(P_bot, P_int, n_seg + 1)[1:]

    XL_g, pd, T_int = Mg_bulk, None, T_core
    for _ in range(3):
        T_p, P_p = T_core, P_bot
        for Pk in P_path:
            dTdP = liquid_adiabat_dTdP(0.5*(P_p + float(Pk)), T_p, XL_g)
            if not np.isfinite(dTdP): return None
            T_p += dTdP*(float(Pk) - P_p); P_p = float(Pk)
        T_int = T_p
        pd = bml_phase_diagram(T_int, P_int, Mg_bulk)
        if pd is None: return None
        if pd['melting'] and np.isfinite(pd.get('XL', np.nan)):
            XL_g = pd['XL']

    if not pd['melting']:
        return D, pd, T_int, P_int, np.nan
    if pd['f_solid'] < 1e-3:
        return 0.0, pd, T_int, P_int, np.nan
    rho_L, _, _ = liquid_bml_properties(P_int, T_int, pd['XL'])
    if not np.isfinite(rho_L):
        return None
    phi_S = mole_to_volume_fraction(pd['f_solid'], pd['XS'], pd['XL'], rho_S, rho_L)
    return D*phi_S, pd, T_int, P_int, rho_L
 
 
def sc_bml(params, z_bml, P_bml_init, g_bml, T_mantle_bottom, n_pass=3):
    """回傳 BML 在 z_bml 上的 rho, Vp, Vs, T, P 以及狀態量。
    P 在層內自積:dP = rho g dz,從 P_bml_init[0](地函底壓力)起算。"""
    D       = params['BML_thickness']
    Mg_bulk = params['Mg#_bulk_bml']
    T_core  = params['T_core']
    P_top   = float(P_bml_init[0])
    P_bml   = np.array(P_bml_init, float)
    rho_S   = _RHO_S_CONST
    out     = None
 
    for _p in range(n_pass):
        # ── 內層:解 h_solid ────────────────────────────────────────────────
        n_seg = None if _p == n_pass - 1 else 5
        def _resid(h_s):
            st = _sc_interface_state(h_s, T_core, z_bml, P_bml, Mg_bulk, D,
                                     rho_S, n_seg=n_seg)
            if st is None:
                raise _PDFail()
            return h_s - st[0]
        try:
            g0, gD = _resid(0.0), _resid(D)
            if gD <= 0.0:
                h_solid = D
            elif g0 >= 0.0:
                h_solid = 0.0
            else:
                h_solid = brentq(_resid, 0.0, D, xtol=0.5)
            st = _sc_interface_state(h_solid, T_core, z_bml, P_bml, Mg_bulk, D,
                                     rho_S, n_seg=n_seg)
            if st is None:
                print(f"  sc_bml pass {_p}: interface state None at "
                      f"h_solid={h_solid:.1f} km")
                return None
        except _PDFail:
            print(f"  sc_bml pass {_p}: phase/EoS failed  "
                  f"P_top={P_top:.2f} P_bot={float(P_bml[-1]):.2f} "
                  f"Mg_bulk={Mg_bulk:.3f} T_core={T_core:.0f} D={D:.0f}")
            return None
        _, pd, T_int, P_int, rho_L_int = st
        
        # ── 中層:rho_S 自洽 ────────────────────────────────────────────────
        # brentq(xtol=0.5) 求得的 h_solid 與重算的 pd 在全熔邊界上可能不一致:
        # h_solid >= 1.0 但 pd 落在 f_solid < 1e-3 那一支,XS 為 nan
        if pd['melting'] and np.isfinite(pd['XS']) and 1.0 <= h_solid < D - 1e-9:
            try:
                rho_new = rho_solid_ngibbs(pd['XS'], P_int, T_int)
            except _NGFail as exc:
                raise _NGFail(f"{exc} | h_solid={h_solid:.2f} f_solid={pd['f_solid']:.5f} "
                              f"phase={pd['phase']} melting={pd['melting']}") from exc
            if not np.isfinite(rho_new) or rho_new <= 0:
                return None
            rho_S = rho_new
 
        # ── 三態與成分 ─────────────────────────────────────────────────────
        Ra, state = np.nan, 'undefined'
        if h_solid < 1.0:
            h_solid, h_liquid = 0.0, D
            Mg_sol, Mg_liq = np.nan, Mg_bulk
            state = 'molten'
            P_seg = np.linspace(float(P_bml[-1]), P_top, 6)[1:]
            T_p, P_p = T_core, float(P_bml[-1])
            for Pk in P_seg:
                dTdP = liquid_adiabat_dTdP(0.5*(P_p + float(Pk)), T_p, Mg_bulk)
                if not np.isfinite(dTdP): return None
                T_p += dTdP*(float(Pk) - P_p); P_p = float(Pk)
            T_int, P_int = T_p, P_top
            P_int = P_top
            pd = bml_phase_diagram(T_int, P_int, Mg_bulk)
        elif (not pd['melting']) or h_solid >= D - 1e-9:
            h_solid, h_liquid = D, 0.0
            Mg_sol, Mg_liq = Mg_bulk, np.nan
            T_int, P_int = T_core, float(P_bml[-1])
        else:
            h_liquid = D - h_solid
            Mg_sol, Mg_liq = pd['XS'], pd['XL']
 
        if h_solid >= 1.0:
            z_mid = z_bml[0] + h_solid/2.0
            Ra, state = compute_bml_thermal_state(
                T_mantle_bottom, T_int, h_solid, rho_S*1000.0,
                float(np.interp(z_mid, z_bml, P_bml))*1e9,
                float(np.interp(z_mid, z_bml, g_bml)))       
 
        # ── 在 z_bml 上組出 T(z) ────────────────────────────────────────────
        z_rel  = z_bml - z_bml[0]
        is_sol = z_rel <= h_solid if h_solid >= 1.0 else np.zeros(len(z_bml), bool)
        T = np.empty(len(z_bml))
        if h_solid >= 1.0:
            zs = z_rel[is_sol]
            if state == 'convective' and Ra > 0:
                delta = min(max(h_solid*(_RA_C/Ra)**(1.0/3.0), h_solid/len(z_bml)),
                            h_solid/2.0)
                T_i = 0.5*(T_mantle_bottom + T_int)
                T[is_sol] = np.where(
                    zs < delta, T_mantle_bottom + (T_i-T_mantle_bottom)*zs/delta,
                    np.where(zs > h_solid-delta,
                             T_i + (T_int-T_i)*(zs-(h_solid-delta))/delta, T_i))
            else:
                T[is_sol] = np.interp(zs, [0.0, max(h_solid, 1e-9)],
                                      [T_mantle_bottom, T_int])
        if (~is_sol).any():
            # 液相段:絕熱線,以界面為起點往下積(dT/dP 隨 P 變,逐節點)
            idx = np.flatnonzero(~is_sol)
            T_prev, P_prev = T_int, P_int
            for k in idx:
                Mg_l = Mg_liq if np.isfinite(Mg_liq) else Mg_bulk
                dTdP = liquid_adiabat_dTdP(0.5*(P_prev + float(P_bml[k])),
                                           T_prev, Mg_l)
                if not np.isfinite(dTdP):
                    return None
                T[k] = T_prev + dTdP*(P_bml[k] - P_prev)
                T_prev, P_prev = T[k], P_bml[k]
 
        # ── 在 z_bml 上組出 rho, Vp, Vs ────────────────────────────────────
        rho = np.empty(len(z_bml)); Vp = np.empty(len(z_bml)); Vs = np.zeros(len(z_bml))
        if is_sol.any():
            Ps, Ts = P_bml[is_sol], T[is_sol]
            p_s = composition_from_params({'Mg#': Mg_sol if np.isfinite(Mg_sol) else Mg_bulk,
                                           'T_lit': T_mantle_bottom, 'P_lit': float(Ps[0])})
            O_s = compute_oxygen(p_s)
            o = _ng_forward(Ps, Ts, _comp_values(p_s, O_s),
                            NG_T_HEADERS, ['component_moles'])
            pr = _ng_props(o['component_moles'], Ps, Ts, ['rho', 'Vp', 'Vs'])
            rho[is_sol] = pr['rho']; Vp[is_sol] = pr['Vp']; Vs[is_sol] = pr['Vs']
        if (~is_sol).any():
            for k in np.flatnonzero(~is_sol):
                r_, v_, _ = liquid_bml_properties(
                    float(P_bml[k]), float(T[k]),
                    Mg_liq if np.isfinite(Mg_liq) else Mg_bulk)
                rho[k], Vp[k], Vs[k] = r_, v_, 0.0
        if not (np.all(np.isfinite(rho)) and np.all(rho > 0)
                and np.all(np.isfinite(Vp))):
            return None
 
        # ── 外層:層內壓力自積 ──────────────────────────────────────────────
        dz  = np.diff(z_bml)*1e3
        w = rho*1e3*g_bml
        P_new = np.empty(len(z_bml)); P_new[0] = P_top
        P_new[1:] = P_top + np.cumsum(0.5*(w[1:] + w[:-1])*dz)/1e9
        dP = float(np.max(np.abs(P_new - P_bml)))
        P_bml = P_new
        out = dict(rho=rho, Vp=Vp, Vs=Vs, T=T, P=P_bml,
                   h_solid_km=h_solid, h_liquid_km=h_liquid,
                   Mg_solid=Mg_sol, Mg_liquid=Mg_liq, melting=pd['melting'],
                   T_interface=T_int, P_interface=P_int,
                   rho_S_interface=float(rho_S), Ra=Ra, thermal_state=state,
                   n_pass=_p+1, dP_bml=dP)
        if dP < SC_TOL_P:
            break
    return out

def sc_build_profile(params, samuel_cache, verbose=False):
    # 初值:z_lit 與 P 都先用 Samuel 場猜,只影響收斂速度不影響收斂點
    z_lit = float(np.interp(params['P_lit'], _pres_gpa, _pres_depth))
    z, lay, geo = sc_build_grid(params, z_lit)
    if z is None:
        print("  BML top above fixed layer -> reject"); return None
    P = np.asarray(pressure_mars(z), float)
    g = np.asarray(gravity_mars(z), float)
    print(f"  init: P range {P.min():.3f}-{P.max():.3f} GPa, "
          f"g range {g.min():.3f}-{g.max():.3f}, z range {z.min():.1f}-{z.max():.1f}")
    hist = []
    def _remap(z_old, lay_old, val_old, z_new, lay_new):
        """把場從舊網格搬到新網格。層界有重複深度,不能直接 np.interp,
        所以逐層做:每層內部深度嚴格遞增。"""
        out = np.empty(len(z_new))
        for L in (0, 1, 2, 3):
            mo, mn = (lay_old == L), (lay_new == L)
            if not mn.any():
                continue
            if not mo.any():
                out[mn] = val_old[0]
                continue
            out[mn] = np.interp(z_new[mn], z_old[mo], val_old[mo])
        return out
    for it in range(SC_N_ITER):
        z_new, lay_new, geo = sc_build_grid(params, z_lit)
        if z_new is None:
            return None
        if it > 0 or len(z_new) != len(z):
            P = _remap(z, lay, P, z_new, lay_new)
            g = _remap(z, lay, g, z_new, lay_new)
        z, lay = z_new, lay_new
        m_fx  = (lay == 0); m_man = (lay == 1)
        m_bml = (lay == 2); m_cor = (lay == 3)
        n = len(z)
        rho = np.zeros(n); T = np.zeros(n); Vp = np.zeros(n); Vs = np.zeros(n)
        # ── 地殼 ────────────────────────────────────────────────────────────
        r_cr, vp_cr, vs_cr, _ = sc_fixed_layer(z[m_fx], samuel_cache)
        rho[m_fx], Vp[m_fx], Vs[m_fx] = r_cr*1e3, vp_cr, vs_cr
        T[m_fx] = T_SURF + (params['T_lit'] - T_SURF)*np.clip(
            P[m_fx]/params['P_lit'], 0.0, 1.0)
 
        # ── 地函 ────────────────────────────────────────────────────────────
        man = sc_mantle(params, P[m_man])
        if man is None:
            return None
        rho[m_man], Vp[m_man], Vs[m_man], T[m_man] = man['rho']*1e3, man['Vp'], man['Vs'], man['T']
        S_lit = man['S_lit']
        T_mantle_bottom = float(T[m_man][-1])
 
        # ── BML ─────────────────────────────────────────────────────────────
        bml = sc_bml(params, z[m_bml], P[m_bml], g[m_bml], T_mantle_bottom)
        if bml is None:
            print("  BML failed -> reject")
            return None
        rho[m_bml], Vp[m_bml], Vs[m_bml] = bml['rho']*1e3, bml['Vp'], bml['Vs']
        T[m_bml] = bml['T']
        P[m_bml] = bml['P']
 
        # ── 核 ──────────────────────────────────────────────────────────────
        cor = sc_core(params, z[m_cor], float(P[m_bml][-1]), g[m_cor])
        if cor is None:
            print("  core EoS out of table range -> reject")
            return None
        rho[m_cor], Vp[m_cor], Vs[m_cor] = cor['rho'], cor['Vp'], cor['Vs']
        T[m_cor], P[m_cor] = cor['T'], cor['P']
 
        # ── 更新 g,重積 P,檢查收斂 ───────────────────────────────────────
        g, M_r = sc_gravity(z, rho)
        P_new  = sc_integrate_P(z, rho, g)
        dP     = float(np.max(np.abs(P_new - P)))
        P      = P_new
        z_lit_new = float(np.interp(params['P_lit'], P[m_man], z[m_man]))
        dz_lit    = abs(z_lit_new - z_lit)
        z_lit     = z_lit_new
        hist.append((dP, dz_lit))
        if verbose:
            print(f"    sc iter {it+1}: max|dP|={dP:7.4f} GPa  "
                  f"dz_lit={dz_lit:6.2f} km  z_lit={z_lit:7.1f}  "
                  f"P_bml_top={P[m_man][-1]:6.3f}  P_cmb={P[m_bml][-1]:6.3f}")
        if dP < SC_TOL_P and dz_lit < 0.1:
            break
    else:
        print(f"  self-consistent field did not converge "
              f"(dP={hist[-1][0]:.3f} GPa, dz_lit={hist[-1][1]:.2f} km)")
        return None
 
    # ── 質量與慣量矩:同一條剖面,不再分層拼接 ────────────────────────────
    g, M_r = sc_gravity(z, rho)
    r = (MARS_RADIUS - z)*1e3
    o = np.argsort(r)
    ra, rhoa = r[o], rho[o]
    rhm = 0.5*(rhoa[1:] + rhoa[:-1])
    M   = float(np.sum(4*np.pi/3 *rhm*(ra[1:]**3 - ra[:-1]**3)))
    I   = float(np.sum(8*np.pi/15*rhm*(ra[1:]**5 - ra[:-1]**5)))
    moi = I/(M*MARS_RADIUS_M**2)
    r_cor = (MARS_RADIUS - z[m_cor])[::-1]        # km,由中心往外
    rho_c = (rho[m_cor]/1e3)[::-1]
    rho_core_mean = float(3*np.trapezoid(rho_c*r_cor**2, r_cor)/r_cor[-1]**3)
 
    return dict(
        z_km=z, layer=lay, P_GPa=P, T_K=T, rho=rho/1e3, Vp=Vp, Vs=Vs, g=g,
        z_lit=z_lit,
        m_crust=m_fx, m_mantle=m_man, m_bml=m_bml, m_core=m_cor,
        z_bml_top=geo['z_bml_top'], z_cmb=geo['z_cmb'],
        P_bml_top=float(P[m_man][-1]), P_cmb=float(P[m_bml][-1]),
        P_center=float(P[-1]), T_center=float(T[-1]),
        T_mantle_bottom=float(T[m_man][-1]),
        rho_mantle_bot=float(rho[m_man][-1]/1e3),
        rho_bml_top=float(rho[m_bml][0]/1e3),
        rho_bml_bot=float(rho[m_bml][-1]/1e3),
        rho_core_top=float(rho[m_cor][0]/1e3),
        rho_core_mean=rho_core_mean,
        Vp_core_cmb=float(Vp[m_cor][0]),
        core=cor,
        M_pred=float(M), moi_pred=float(moi), S_lit=S_lit,
        sc_n_iter=len(hist), sc_dP=hist[-1], bml=bml)
 

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


def build_taup(prof, model_name, samuel_cache):
    """.nd 檔的四段全部來自同一條自洽剖面。
 
    固定層(0-100 km)用 Samuel 的高解析度剖面(200 點),因為 TauP 的
    地殼段需要比自洽網格更密的節點來解 pP/sP 的深度相位。
    其餘三段(地函、BML、核)直接切自洽剖面。
 
    流體邊界由 Vs 決定,不由層標籤決定:BML 可能整層固、整層液或分層,
    所以要從資料本身找 Vs -> 0 的位置。
    """
    os.makedirs(TAUP_WORK_DIR, exist_ok=True)
    model_name = model_name.replace(".npz", "")
    npz_path   = os.path.join(TAUP_WORK_DIR, f'{model_name}.npz')
    nd_path    = os.path.join(TAUP_WORK_DIR, f"{model_name}.nd")
    if os.path.exists(npz_path):
        return TauPyModel(model=npz_path)
 
    z, Vp, Vs, rho = prof['z_km'], prof['Vp'], prof['Vs'], prof['rho']
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(Vp))
            and np.all(np.isfinite(Vs)) and np.all(np.isfinite(rho))
            and np.all(Vp > 0) and np.all(rho > 0) and np.all(Vs >= 0)):
        raise _NGFail(f"taup input nonfinite/nonpositive: "
                      f"Vp={np.sum(~np.isfinite(Vp))} Vs={np.sum(~np.isfinite(Vs))} "
                      f"rho={np.sum(~np.isfinite(rho))} Vp_min={np.nanmin(Vp):.3f}")
    m_man, m_bml, m_cor = prof['m_mantle'], prof['m_bml'], prof['m_core']
    z_cmb = prof['z_cmb']
 
    def _bad(*arrs):
        for a in arrs:
            a = np.asarray(a, float)
            if a.size == 0 or not np.all(np.isfinite(a)):
                return True
        return False
    if _bad(Vp[m_man], Vs[m_man], rho[m_man], Vp[m_bml], Vs[m_bml], rho[m_bml],
            Vp[m_cor], rho[m_cor]):
        raise ValueError("profile has non-finite Vp/Vs/rho")
    if np.any(Vp[m_man] <= 0) or np.any(Vp[m_bml] <= 0) or np.any(rho[m_man] <= 0):
        raise ValueError("profile has non-positive Vp/rho")
 
    _validate_nd_depths(z[m_man], z[m_bml])
    H_MIN_LAYER = 1.0   # km, 薄於此值的層不寫進 .nd
    with open(nd_path, 'w') as f:
        # ── 固定層:Samuel 高解析度 ────────────────────────────────────────
        for d, vp, vs, r in zip(samuel_cache['crust_z'], samuel_cache['crust_vp'],
                                samuel_cache['crust_vs'], samuel_cache['crust_rho']):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        f.write("mantle\n")
 
        # ── 地函 ──────────────────────────────────────────────────────────
        for d, vp, vs, r in zip(z[m_man], Vp[m_man], Vs[m_man], rho[m_man]):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
 
        # ── BML:固相段照寫,遇到 Vs -> 0 就開 outer-core ────────────────
        d_b, vp_b, vs_b, rho_b = z[m_bml], Vp[m_bml], Vs[m_bml], rho[m_bml]
        zero = np.where(vs_b <= 1e-6)[0]
        i0   = int(zero[0]) if len(zero) else len(d_b)
 
        # 液相段厚度 = z_cmb - 液相起點;太薄就當作整層固相
        # (否則 z_cmb 會被寫三次: 界面上側 / 零厚度液相 / 核心頂,TauP segfault)
        if i0 < len(d_b) and (z_cmb - d_b[i0]) < H_MIN_LAYER:
            d_b, vp_b, vs_b, rho_b = d_b[:i0], vp_b[:i0], vs_b[:i0], rho_b[:i0]
            i0 = len(d_b)

        # 固相段厚度太薄就當作整層液相,液相從地函底直接開始
        if i0 > 0 and (d_b[i0-1] - d_b[0]) < H_MIN_LAYER:
            d_b, vp_b, vs_b, rho_b = d_b[i0:], vp_b[i0:], vs_b[i0:], rho_b[i0:]
            i0 = 0

        for d, vp, vs, r in zip(d_b[:i0], vp_b[:i0], vs_b[:i0], rho_b[:i0]):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

        if i0 < len(d_b):
            # i0 == 0(BML 全液):地函段最後一點已經是 z_bml_top，那就是界面上側，
            # 這裡再寫一次會讓同一深度出現三次 -> nd has depth repeated >2x
            if i0 > 0:
                vp_s, vs_s, rho_s = vp_b[i0-1], vs_b[i0-1], rho_b[i0-1]
                f.write(f"{d_b[i0]:.3f}  {vp_s:.4f}  {vs_s:.4f}  {rho_s:.4f}\n")
            f.write("outer-core\n")
            for d, vp, r in zip(d_b[i0:], vp_b[i0:], rho_b[i0:]):
                f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")
        else:
            # BML 全固:流體邊界就是 CMB
            # d_b[-1] 可能已經等於 z_cmb,再寫一次就是零厚度層
            if z_cmb - d_b[-1] > 1e-6:
                f.write(f"{z_cmb:.3f}  {vp_b[-1]:.4f}  {vs_b[-1]:.4f}  {rho_b[-1]:.4f}\n")
            f.write("outer-core\n")
        # ── 核 ────────────────────────────────────────────────────────────
        d_c, vp_c, rho_c = z[m_cor], Vp[m_cor], rho[m_cor]
        f.write(f"{z_cmb:.3f}  {vp_c[0]:.4f}  0.0000  {rho_c[0]:.4f}\n")
        for d, vp, r in zip(d_c[1:], vp_c[1:], rho_c[1:]):
            f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")
        
    zz = np.array([float(l.split()[0]) for l in open(nd_path)
                   if l.strip() and not l[0].isalpha()])
    _, cnt = np.unique(zz, return_counts=True)
    if (cnt > 2).any():
        raise ValueError(f"nd has depth repeated >2x: {np.unique(zz)[cnt > 2]}")

    build_taup_model(nd_path, output_folder=TAUP_WORK_DIR)
    return TauPyModel(model=npz_path)

# ── solidus penalty ───────────────────────────────────────────────────────────
def compute_solidus_penalty(prof, params):
    """地函段沿自洽地溫線是否超過固相線。
    罰的是全球 1-D 平均剖面出現可觀的部分熔融(SOLIDUS_SIGMA = 12 K 約對應 1%),
    不排除局部熔融 —— 火星的零星火山活動來自局部上湧或揮發分降低固相線,
    那不是 1-D 平均剖面該表現的東西。"""
    m   = prof['m_mantle']
    P_m = prof['P_GPa'][m]
    T_m = prof['T_K'][m]
    sel = P_m >= params['P_lit']
    if not sel.any():
        return 0.0
    P_m, T_m = P_m[sel], T_m[sel]
    T_sol  = np.array([solidus_duncan_Mg(p, params['Mg#']) for p in P_m])
    excess = np.clip(T_m - T_sol, 0.0, None)
    penalty = min((float(np.max(excess))/SOLIDUS_SIGMA)**2, SOLIDUS_CAP)
    if penalty > 0:
        i = int(np.argmax(excess))
        print(f"  Solidus penalty = {penalty:.4f}  "
              f"(max excess {excess[i]:.1f} K at P={P_m[i]:.2f} GPa)")
    return penalty

# ── misfit ────────────────────────────────────────────────────────────────────

def compute_misfit(taup_model, obs_dataset, prof, params=None,
                   mass_sigma=0.0, moi_sigma=0.0, grav_sigma=0.0, ric_sigma=0.0):
    phases_std   = ['P', 'S', 'pP', 'sP', 'PP', 'PPP', 'SS', 'SSS', 'sS', 'ScS', 'SKS']
    phases_pdiff = phases_std + ['Pdiff']
 
    tt_total, tt_n_ev, tt_n_ph, tt_n_miss, tt_n_capped = 0.0, 0, 0, 0, 0
    miss_by_event = {}
 
    for event, obs in obs_dataset.items():
        try:
            arrivals = taup_model.get_travel_times(
                source_depth_in_km=obs.get('depth', 10.0),
                distance_in_degree=obs['delta'],
                phase_list=phases_pdiff if event == 'S1000a' else phases_std)
        except Exception as e:
            print(f"  {event}: FAILED {e}"); continue
 
        times = {}
        for a in arrivals:
            if a.name not in times:
                times[a.name] = a.time
        def _tdiff(a, b):
            ta, tb = times.get(a), times.get(b)
            return None if (ta is None or tb is None) else ta - tb
 
        pred = {
            'S-P':          _tdiff('S',   'P'),
            'pP-P':         _tdiff('pP',  'P'),
            'sP-P':         _tdiff('sP',  'P'),
            'PP-P':         _tdiff('PP',  'P'),
            'PPP-P':        _tdiff('PPP', 'P'),
            'sS-S':         _tdiff('sS',  'S'),
            'SS-S':         _tdiff('SS',  'S'),
            'SSS-S':        _tdiff('SSS', 'S'),
            'ScS-S':        _tdiff('ScS', 'S'),
            'SS-PP':        _tdiff('SS',  'PP'),
            'SKS-PP':       _tdiff('SKS', 'PP'),
            'PP-PbdiffPcP': _tdiff('PP',  'Pdiff'),
        }
 
        ev_sum, ev_n, ev_miss = 0.0, 0, 0
        for phase, obs_val in obs.items():
            if phase in ('delta', 'depth') or not isinstance(obs_val, tuple):
                continue
            p_val = pred.get(phase)
            if p_val is None:
                ev_sum += MISS_PENALTY; ev_n += 1; ev_miss += 1
                continue
            obs_t, sigma = obs_val
            if sigma <= 0 or not np.isfinite(obs_t):
                continue
            res_sigma = abs(obs_t - p_val)/sigma
            if not np.isfinite(res_sigma):
                ev_sum += MISS_PENALTY; ev_n += 1; ev_miss += 1
                continue
            if res_sigma >= RES_CAP:
                tt_n_capped += 1
            ev_sum += min(res_sigma, RES_CAP); ev_n += 1
 
        if ev_n > 0:
            tt_total  += ev_sum/ev_n
            tt_n_ev   += 1
            tt_n_ph   += ev_n - ev_miss
            tt_n_miss += ev_miss
            miss_by_event[event] = ev_miss
 
    tt_misfit = tt_total if tt_n_ev > 0 else 999.0
    if not np.isfinite(tt_misfit):
        tt_misfit = 999.0
    n_ev_expected = len(obs_dataset)
    if tt_n_ev < n_ev_expected:
        tt_misfit += EVENT_FAIL*(n_ev_expected - tt_n_ev)
 
    solidus_penalty = compute_solidus_penalty(prof, params) if params else 0.0
    total = tt_misfit + solidus_penalty + mass_sigma + moi_sigma + grav_sigma + ric_sigma
    if not np.isfinite(total):
        total = 999.0
 
    print(f"  TT={tt_misfit:.4f}(events={tt_n_ev}/{n_ev_expected}, phases={tt_n_ph}, "
          f"miss={tt_n_miss}, capped={tt_n_capped})  solidus={solidus_penalty:.4f}  "
          f"mass={mass_sigma:.2f}s  moi={moi_sigma:.2f}s  grav={grav_sigma:.2f}s  "
          f"ric={ric_sigma:.2f}s  total={total:.4f}")
 
    return total, tt_n_ph, {
        'misfit_tt': tt_misfit, 'misfit_solidus': solidus_penalty,
        'grav_sigma': grav_sigma, 'ric_sigma': ric_sigma,
        'tt_n_ev': tt_n_ev, 'tt_n_ph': tt_n_ph, 'tt_n_miss': tt_n_miss,
        'tt_n_capped': tt_n_capped, 'tt_miss_by_event': miss_by_event,
    }

# ── forward model ─────────────────────────────────────────────────────────────
def forward(params, run_dir, model_name, samuel_cache):
    try:
        return _forward_impl(params, run_dir, model_name, samuel_cache)
    except (_NGFail, _PDFail, torch._C._LinAlgError) as exc:
        print(f"  [NGFAIL] {exc}  R_cmb={params['R_cmb']:.1f} T_lit={params['T_lit']:.1f} "
              f"P_lit={params['P_lit']:.3f} Mg#={params['Mg#']:.4f} "
              f"T_core={params['T_core']:.1f} Mg#bml={params['Mg#_bulk_bml']:.4f} "
              f"D={params['BML_thickness']:.1f} w_S={params['w_S']:.4f}", flush=True)
        return None, None, None, None, None

def _forward_impl(params, run_dir, model_name, samuel_cache):
    """a36:一條自洽剖面決定一切。
 
    回傳 (misfit, n_data, components, prof, None)
    第四個位置從 fort56_data 換成 prof(整條剖面),第五個保留 None
    以維持呼叫端的 5-tuple 介面。
    """
    prof = sc_build_profile(params, samuel_cache, verbose=True)
    if prof is None:
        return None, None, None, None, None
 
    bml = prof['bml']
    print(f"  sc: {prof['sc_n_iter']} iters  dP={prof['sc_dP'][0]:.4f} GPa  "
          f"z_lit={prof['z_lit']:.1f}km  P_bml_top={prof['P_bml_top']:.2f}  "
          f"P_cmb={prof['P_cmb']:.2f}  P_c={prof['P_center']:.1f}GPa")
    print(f"  BML: Ra={bml['Ra']:.1e} [{bml['thermal_state']}]  "
          f"h_sol={bml['h_solid_km']:.0f}km  h_liq={bml['h_liquid_km']:.0f}km  "
          f"T_int={bml['T_interface']:.0f}K  rho_S={bml['rho_S_interface']:.3f}  "
          f"melt={bml['melting']}")
 
    # ── 內核:靜態 ICB 溫度計,由自洽核剖面判定 ──────────────────────────
    cor  = prof['core']
    R_ic = CF.find_R_IC(cor)
    gap  = CF.gap_min(cor)
    ric_sigma = 0.0
    if GAP_ON:
        # gap <= 0 已滿足內核存在,不罰;gap > 0 線性罰
        ric_sigma  = max(0.0, gap)/GAP_SIGMA if np.isfinite(gap) else 0.0
        ric_sigma += max(0.0, (R_ic - RIC_MAX_KM))/RIC_MAX_SIG
    print(f"  core: w_S={params['w_S']:.3f}  T_c={prof['T_center']:.0f}K  "
          f"rho_mean={prof['rho_core_mean']:.3f}  Vp_cmb={prof['Vp_core_cmb']:.2f}  "
          f"R_ic={R_ic:.0f}km  gap={gap:+.0f}K")
 
    # ── Kono 密度對比:BML 必須比上方地函重、比下方核輕 ────────────────
    upper_contrast = prof['rho_bml_top'] - prof['rho_mantle_bot']
    lower_contrast = prof['rho_core_top'] - prof['rho_bml_bot']
    print(f"  density contrasts: upper={upper_contrast:+.4f}  "
          f"lower={lower_contrast:+.4f}")
 
    grav_sigma = (max(0.0, -upper_contrast) + max(0.0, -lower_contrast))/GRAV_SCALE
    if REJECT_GRAV_UNSTABLE and grav_sigma > 0:
        if upper_contrast <= 0.0: _GRAV_FAIL['upper'] += 1
        if lower_contrast <= 0.0: _GRAV_FAIL['lower'] += 1
        print("  BML gravitationally unstable -> reject (Kono 2025)")
        return None, None, None, None, None
 
    # ── 質量與慣量矩:同一條積分算出來的,不再事後拼接 ────────────────
    M_pred, moi_pred = prof['M_pred'], prof['moi_pred']
    mass_sigma = abs(MARS_MASS_OBS - M_pred)/MARS_MASS_SIGMA
    moi_sigma  = abs(MOI_OBS - (moi_pred - MOI_BIAS))/MOI_SIGMA
    print(f"  mass={mass_sigma:.2f}s  moi={moi_sigma:.2f}s")
 
    try:
        taup_model = build_taup(prof, model_name, samuel_cache)
    except Exception as e:
        print(f"  TauP failed: {e}")
        return None, None, None, None, None
 
    misfit, n_data, components = compute_misfit(
        taup_model, SAMUEL_DATA, prof, params=params,
        mass_sigma=mass_sigma, moi_sigma=moi_sigma,
        grav_sigma=grav_sigma, ric_sigma=ric_sigma)
 
    components.update({
        'Ra':              bml['Ra'],
        'thermal_state':   bml['thermal_state'],
        'h_solid_km':      bml['h_solid_km'],
        'h_liquid_km':     bml['h_liquid_km'],
        'melting':         bml['melting'],
        'Mg_solid':        bml['Mg_solid']  if np.isfinite(bml['Mg_solid'])  else -1.0,
        'Mg_liquid':       bml['Mg_liquid'] if np.isfinite(bml['Mg_liquid']) else -1.0,
        'T_interface':     bml['T_interface'],
        'P_interface':     bml['P_interface'],
        'rho_S_interface': bml['rho_S_interface'],
        'rho_solid_bml':   float(np.mean(prof['rho'][prof['m_bml']][
                               prof['Vs'][prof['m_bml']] > 1e-6]))
                           if np.any(prof['Vs'][prof['m_bml']] > 1e-6) else -1.0,
        'rho_liquid_bml':  float(np.mean(prof['rho'][prof['m_bml']][
                               prof['Vs'][prof['m_bml']] <= 1e-6]))
                           if np.any(prof['Vs'][prof['m_bml']] <= 1e-6) else -1.0,
        'upper_contrast':  upper_contrast,
        'lower_contrast':  lower_contrast,
        'S_lit':           prof['S_lit'],
        'mass_sigma':      mass_sigma,
        'moi_sigma':       moi_sigma,
        'moi_pred':        moi_pred,
        'M_pred':          M_pred,
        'T_mantle_bottom': prof['T_mantle_bottom'],
        'R_ic_km':         R_ic,
        'gap_min':         gap,
        'ric_sigma':       ric_sigma,
        'rho_core_mean':   prof['rho_core_mean'],
        'Vp_core_cmb':     prof['Vp_core_cmb'],
        'P_center_core':   prof['P_center'],
        'T_center_core':   prof['T_center'],
        # 自洽場的診斷:收斂趟數與殘差。rejection 若與參數位置相關,
        # 等於在 likelihood 裡偷偷加了不連續,必須事後檢查。
        'P_cmb':           prof['P_cmb'],
        'P_bml_top':       prof['P_bml_top'],
        'z_lit_km':        prof['z_lit'],
        'sc_n_iter':       prof['sc_n_iter'],
        'sc_dP':           prof['sc_dP'][0],
        'sc_dz_lit':       prof['sc_dP'][1],
        'bml_n_pass':      bml['n_pass'],
    })
 
    return misfit, n_data, components, prof, None
# ── MCMC ──────────────────────────────────────────────────────────────────────
def propose(current, rng):
    if _PROP_CHOL is None:
        return {k: current[k] + rng.normal(0, STEP[k]) for k in PRIOR}
    dx = _PROP_CHOL @ rng.standard_normal(len(PARAM_ORDER))
    return {k: current[k] + dx[i] for i, k in enumerate(PARAM_ORDER)}

def run_mcmc(chain_id, n_steps, start_params=None, prefix='chain'):
    chain_dir = os.path.join(MCMC_DIR, f"{prefix}_{chain_id:02d}")
    os.makedirs(chain_dir, exist_ok=True)

    samuel_cache = _samuel_cache

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
    rng = np.random.default_rng([42, chain_id, step_start, os.getpid()])  
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
            prof = None
        else:
            proposed_misfit, n_data, components, prof, _= forward(
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

            if prof is not None:
                np.savez(os.path.join(chain_dir, f"profile_s{step+1:05d}.npz"),
                         z_km=prof['z_km'], layer=prof['layer'],
                         P_GPa=prof['P_GPa'], T_K=prof['T_K'],
                         rho=prof['rho'], Vp=prof['Vp'], Vs=prof['Vs'],
                         g=prof['g'],
                         z_bml_top=np.array([prof['z_bml_top']]),
                         z_cmb=np.array([prof['z_cmb']]),
                         h_solid_km=np.array([prof['bml']['h_solid_km']]),
                         h_liquid_km=np.array([prof['bml']['h_liquid_km']]),
                         Mg_solid=np.array([prof['bml']['Mg_solid']]),
                         Mg_liquid=np.array([prof['bml']['Mg_liquid']]),
                         thermal_state=np.array([prof['bml']['thermal_state']]),
                         R_ic_km=np.array([current_components.get('R_ic_km', 0.0)]))

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
    parser.add_argument('--scan_R', action='store_true')
    parser.add_argument('--route', default='B', choices=['A', 'B'],
                        help='core EoS route: A = all-Liu (Gamma mixing), '
                             'B = Liu Fe + Sakai FeS (Margules)')
    args = parser.parse_args()

    os.makedirs(MCMC_DIR, exist_ok=True)
    start_params = None

    CORE_EOS_ROUTE = args.route
    CF.configure(route=CORE_EOS_ROUTE, verbose=True)
    START_PARAMS['R_cmb'] = {'A': 1500.0, 'B': 1650.0}[CORE_EOS_ROUTE]

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

    _PROP_CHOL = _load_prop_chol(args.route)
    
    if args.scan_R:
        run_dir = os.path.join(MCMC_DIR, 'scan_R')
        os.makedirs(run_dir, exist_ok=True)
        print(f"{'R_cmb':>8}{'moi_pred':>11}{'moi_sig':>9}{'M_pred':>13}"
              f"{'mass_sig':>10}{'P_cmb':>8}{'rho_core':>10}{'R_ic':>7}")
        for R in np.arange(1400., 1901., 25.):
            p = {**START_PARAMS, 'R_cmb': float(R)}
            out = forward(p, run_dir, f'scanR{R:.0f}', _samuel_cache)
            if out[2] is None:
                print(f"{R:8.0f}   reject"); continue
            c = out[2]
            print(f"{R:8.0f}{c['moi_pred']:11.5f}"
                  f"{abs(MOI_OBS-c['moi_pred'])/MOI_SIGMA:9.2f}"
                  f"{c['M_pred']:13.4e}"
                  f"{abs(MARS_MASS_OBS-c['M_pred'])/MARS_MASS_SIGMA:10.2f}"
                  f"{float(c['P_cmb']):8.2f}"
                  f"{c['rho_core_mean']:10.3f}{c['R_ic_km']:7.0f}")
    elif args.test:
        run_mcmc(0, 1, prefix=args.prefix, start_params=start_params)
    else:
        run_mcmc(args.chain, args.steps, prefix=args.prefix, start_params=start_params)
