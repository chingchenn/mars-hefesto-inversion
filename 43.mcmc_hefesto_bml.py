#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:09:57 2026

@author: chingchen

Major updates from 21_mcmc_NPS.py:

1. run_hefesto() restructured into three steps
   Step 1: NPT @ (P_lit, T_lit)  → read S_lit
   Step 2: NPT scan (pressure points up to P_bml_top, two NPT runs each,
           interpolate to find T where S=S_lit) → build adiabatic T(P)
   Step 3: iensem=-1 with ad.in → full Vp/Vs/rho profile
   P_bml_top is now dynamic (= pressure at BML top = TRUE_CMB_DEPTH - BML_thickness)

2. BML (Basal Mantle Layer) physics
   - BML defined from iron core upward: top = TRUE_CMB_DEPTH - BML_thickness
   - BML_thickness is a free MCMC parameter (100–400 km)
   - BML temperature: linear from T_mantle_bottom (top, from mantle isentrope)
     to T_bml (bottom, free MCMC parameter)
   - Melt fraction φ computed per-point via solidus/liquidus (Duncan 2018 +
     Ruedas 2017) with Mg# correction (Elkins-Tanton Fe correction)
   - Pierru 2026 velocity correction applied per-point based on φ
   - outer-core marker placed at first depth where φ ≥ 0.30
   - BML solidus penalty removed: data decides whether BML exists

3. Samuel 2023 model (replaces Khan 2023)
   - compute_samuel_median() reads all vp*.dat + vs*.dat from SAMUEL_DATA_DIR
   - LSL_TOP_DEPTH = 1574.6 km, TRUE_CMB_DEPTH = 1743.3 km (hardcoded from
     rho_profile.dat density jumps)
   - Core rho from rho_profile.dat 
   - Gravity and pressure profiles both integrated from rho_profile.dat
   - pressure_mars(depth_km) function added
   - SAMUEL_DATA used for travel time misfit (replaces KHAN_DATA)
   - PP-PbdiffPcP constraint added for S1000a (Samuel 2023 Table 1)

4. MCMC mechanics
   - forward() returns 4 values: (misfit, n_data, components, fort56_data)
   - Accepted steps save profile_s{step}.npz
   - chain.json saved every step
   - --random_start and --start CLI arguments added
   - START_PARAMS updated from Samuel Tprofile.dat:
     T_lit=1539 K, P_lit=3.69 GPa (lithosphere/adiabat transition ~300 km)

5. Naming and cleanup
   - LSL → BML throughout (lsl_top_depth → bml_top_depth)
   - P_BML_TOP, P_MAX_GPA, P_BML_BOTTOM, BML_THICKNESS fixed constants removed
   - KHAN_DATA removed
   - TauP .nd discontinuity format fixed (two lines same depth + keyword)
"""

import os
import shutil
import subprocess
import numpy as np
from config import *
import json
import argparse
from datetime import datetime
import glob
import pandas as pd

from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

# ============================================================
# parameters
# ============================================================

MARS_RADIUS = 3389.5
T_SURF      = 220.0
GAMMA       = 1.1
MARS_MASS_OBS   = 6.4171e23
MARS_MASS_SIGMA = MARS_MASS_OBS * 0.01
MOI_OBS         = 0.3634
MOI_SIGMA       = 0.0006
MARS_RADIUS_M   = MARS_RADIUS * 1000
MCMC_TEMPERATURE = 1.0

# ============================================================
# gravity and pressure profiles from SAMUEL_RHO_PROFILE_PATH
# ============================================================

_GRAVITY_DEPTH  = None
_GRAVITY_G      = None
_PRESSURE_DEPTH = None
_PRESSURE_GPA   = None

def load_gravity_profile():
    global _GRAVITY_DEPTH, _GRAVITY_G, _PRESSURE_DEPTH, _PRESSURE_GPA
    if _GRAVITY_DEPTH is not None:
        return

    rho_path = SAMUEL_RHO_PROFILE_PATH
    try:
        data    = np.loadtxt(rho_path)
        rho_all = data[:, 0]   # kg/m³
        r_all   = data[:, 1]   # km radius
    except Exception as e:
        print(f" no rho_profile.dat ({e}), using g=3.72 m/s²")
        _GRAVITY_DEPTH  = np.array([0.0, MARS_RADIUS])
        _GRAVITY_G      = np.array([3.72, 3.72])
        _PRESSURE_DEPTH = np.array([0.0, MARS_RADIUS])
        _PRESSURE_GPA   = np.array([0.0, 25.0])
        return

    # sort by radius ascending (center → surface)
    idx = np.argsort(r_all)
    rho = rho_all[idx]        # kg/m³
    r   = r_all[idx] * 1000   # m

    G_CONST = 6.674e-11

    # integrate enclosed mass → gravity
    M_enc = np.zeros(len(r))
    for i in range(1, len(r)):
        dr       = r[i] - r[i-1]
        rho_mid  = (rho[i] + rho[i-1]) / 2
        r_mid    = (r[i]   + r[i-1])   / 2
        M_enc[i] = M_enc[i-1] + 4 * np.pi * rho_mid * r_mid**2 * dr

    g = np.zeros(len(r))
    g[1:] = G_CONST * M_enc[1:] / r[1:]**2

    # convert to depth (surface = 0)
    depth_km = (r[-1] - r) / 1000.0
    sort_d   = np.argsort(depth_km)
    _GRAVITY_DEPTH = depth_km[sort_d]
    _GRAVITY_G     = g[sort_d]

    # integrate pressure from surface downward
    P_GPa = np.zeros(len(_GRAVITY_DEPTH))
    rho_depth = rho[sort_d[::-1]][::-1]   # rho sorted by depth ascending
    for i in range(1, len(_GRAVITY_DEPTH)):
        dz        = (_GRAVITY_DEPTH[i] - _GRAVITY_DEPTH[i-1]) * 1000.0  # m
        g_mid     = (_GRAVITY_G[i] + _GRAVITY_G[i-1]) / 2
        rho_mid   = (rho_depth[i] + rho_depth[i-1]) / 2
        P_GPa[i]  = P_GPa[i-1] + rho_mid * g_mid * dz / 1e9

    _PRESSURE_DEPTH = _GRAVITY_DEPTH.copy()
    _PRESSURE_GPA   = P_GPa

    print(f"  Gravity profile loaded: surface g={_GRAVITY_G[0]:.3f} m/s²  "
          f"P at {TRUE_CMB_DEPTH:.0f} km = "
          f"{float(np.interp(TRUE_CMB_DEPTH, _PRESSURE_DEPTH, _PRESSURE_GPA)):.2f} GPa")


def gravity_mars(depth_km):
    if _GRAVITY_DEPTH is None:
        load_gravity_profile()
    return np.interp(depth_km, _GRAVITY_DEPTH, _GRAVITY_G)


def pressure_mars(depth_km):
    """Return pressure in GPa at given depth(s) in km."""
    if _PRESSURE_DEPTH is None:
        load_gravity_profile()
    return np.interp(depth_km, _PRESSURE_DEPTH, _PRESSURE_GPA)

# ============================================================
# composition
# ============================================================

YM_BASE = {
    'Si': 4.01931, 'Mg': 4.08235, 'Fe': 1.08599,
    'Ca': 0.27259, 'Al': 0.37376, 'Na': 0.10105, 'Cr': 0.06146,
}
MGFE_TOTAL = YM_BASE['Mg'] + YM_BASE['Fe']

START_PARAMS = {
    'T_lit':         1539.0,   # K @ ~300 km, from Samuel Tprofile.dat
    'P_lit':         3.69,     # GPa @ ~300 km, lithosphere/adiabat transition
    'Mg#':           YM_BASE['Mg'] / (YM_BASE['Mg'] + YM_BASE['Fe']),
    'T_bml':         2200.0,   # K, BML bottom temperature
    'Mg#_bml':       0.7,
    'BML_thickness': 168.7,    # km = TRUE_CMB_DEPTH - LSL_TOP_DEPTH (Samuel median)
}

FIXED_PARAMS = {
    'Si': YM_BASE['Si'], 'Ca': YM_BASE['Ca'], 'Al': YM_BASE['Al'],
    'Na': YM_BASE['Na'], 'Cr': YM_BASE['Cr'],
}

PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (1.5,    9.0),
    'Mg#':           (0.50,   0.86),
    'T_bml':         (1800.0, 3500.0),
    'Mg#_bml':       (0.50,   0.80),
    'BML_thickness': (50.0,  400.0),  
}

STEP = {
    'T_lit':         60.0,
    'P_lit':         0.3,
    'Mg#':           0.015,
    'T_bml':         60.0,
    'Mg#_bml':       0.015,
    'BML_thickness': 20.0,    
}

SIGMA = {
    'S-P': 10.0, 'pP-P': 3.0, 'sP-P': 5.0, 'PP-P': 8.0, 'PPP-P': 12.0,
    'sS-S': 5.0, 'SS-S': 5.0, 'SSS-S': 8.0, 'ScS-S': 12.0,
    'SS-PP': 10.0, 'SKS-PP': 10.0,
    'PP-PbdiffPcP': 10.0,   
}

def s(a, sa, b, sb):
    return np.sqrt(sa**2 + sb**2)


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
        'PPP-P': (24.8,  12.0),
        'SS-S':  (15.8,   5.0),
        'SSS-S': (32.4,   8.0),
    },
    'S0484b': {
        'delta': 31.3,  'depth': 24.9,
        'S-P':   (173.1,  5.0),
        'pP-P':  (5.5,    5.0),
        'PP-P':  (19.73,  8.0),
        'sS-S':  (13.0,   5.0),
        'SS-S':  (17.4,   5.0),
        'ScS-S': (322.3, 12.0),
    },
    'S0784a': {
        'delta': 30.2,  'depth': 16.8,
        'S-P':   (179.3,  5.0),
        'pP-P':  (6.5,    5.0),
        'PP-P':  (13.7,   8.0),
        'PPP-P': (22.4,  12.0),
        'sS-S':  (7.2,    5.0),
        'SS-S':  (19.6,   5.0),
        'SSS-S': (28.0,   8.0),
    },
    'S0802a': {
        'delta': 30.0,  'depth': 20.4,
        'S-P':   (180.3,  5.0),
        'pP-P':  (4.0,    5.0),
        'PP-P':  (25.6,   8.0),
        'PPP-P': (33.9,  12.0),
        'sS-S':  (9.3,    5.0),
        'SS-S':  (22.4,   5.0),
        'SSS-S': (36.5,   8.0),
        'ScS-S': (387.6, 12.0),
    },
    'S0809a': {
        'delta': 30.7,  'depth': 16.0,
        'S-P':   (191.95,  5.0),
        'pP-P':  (4.5,     5.0),
        'PP-P':  (16.25,   8.0),
        'PPP-P': (29.65,  12.0),
        'sS-S':  (8.1,     5.0),
        'SS-S':  (23.8,    5.0),
        'SSS-S': (39.3,    8.0),
        'ScS-S': (373.5,  12.0),
    },
    'S0820a': {
        'delta': 28.1,  'depth': 18.7,
        'S-P':   (174.1,  5.0),
        'PP-P':  (21.9,   8.0),
        'PPP-P': (32.1,  12.0),
        'sS-S':  (8.5,    5.0),
    },
    'S0861a': {
        'delta': 54.5,  'depth': 15.5,
        'S-P':   (319.3,  5.0),
        'PP-P':  (19.6,   8.0),
        'PPP-P': (47.6,  12.0),
        'SS-S':  (41.1,   5.0),
    },
    'S0864a': {
        'delta': 29.0,  'depth': 25.0,
        'S-P':   (171.4,  5.0),
        'PP-P':  (18.0,   8.0),
        'PPP-P': (27.9,  12.0),
        'sS-S':  (17.3,   5.0),
        'SS-S':  (26.4,   5.0),
    },
    'S0916d': {
        'delta': 30.2,  'depth': 16.3,
        'S-P':   (170.8,  5.0),
        'pP-P':  (3.9,    5.0),
        'PP-P':  (19.3,   8.0),
        'PPP-P': (36.1,  12.0),
        'SS-S':  (19.0,   5.0),
        'SSS-S': (42.9,   8.0),
        'ScS-S': (342.8, 12.0),
    },
    'S0918a': {
        'delta': 16.6,  'depth': 22.3,
        'S-P':   (102.4,  5.0),
        'PP-P':  (12.8,   8.0),
        'PPP-P': (22.5,  12.0),
        'SS-S':  (21.2,   5.0),
        'SSS-S': (35.0,   8.0),
    },
    'S0976a': {
        'delta': 144.0, 'depth': 30.0,
        'SS-PP':  (854.4, 10.0),
        'SKS-PP': (303.9, 10.0),
    },
    'S1000a': {
        'delta': 125.9, 'depth': 0.0,
        'SS-PP':        (749.0,   10.0),
        'SKS-PP':       (339.3,   10.0),
        'PP-PbdiffPcP': (180.65,  10.0),  
    },
    'S1094b': {
        'delta': 58.5,  'depth': 0.0,
        'S-P':  (343.0, 5.0),
    },
    'S1222a': {
        'delta': 36.1,  'depth': 32.8,
        'S-P':   (216.0,  5.0),
        'ScS-S': (258.0, 12.0),
    },
}


def solidus_duncan2018(P_GPa):
    scalar_input = np.ndim(P_GPa) == 0
    P = np.atleast_1d(np.asarray(P_GPa, dtype=float))
    T_C = np.where(
        P <= 10.0,
        -4.877 * P**2 + 120.2 * P + 1088.0,
        np.where(
            P <= 23.0,
            -1.323 * (P - 10.0)**2 + 38.18 * (P - 10.0) + 1802.0,
            77.75 * (P - 23.0) + 2075.0
        )
    )
    result = T_C + 273.15
    return float(result[0]) if scalar_input else result

# ============================================================
# ★ STEP 1: solidus / liquidus with Mg# correction
# ============================================================

MG_DUNCAN_REF = 0.75   # reference Mg# for Duncan 2018
PHI_OUTER_CORE = 0.30  # melt fraction threshold for outer-core marker (Pierru 2026)


def liquidus_ruedas2017(P_GPa):
    """Ruedas & Breuer 2017 liquidus, returns K"""
    return (2160.6 + 64.7109 * P_GPa
            - 3.97463 * P_GPa**2
            + 0.0957894 * P_GPa**3)


def fe_correction_K(Mg_number, Mg_ref=MG_DUNCAN_REF):
    """
    Elkins-Tanton 2008: dT = -6 * (Fe - Fe_ref)
    Fe# = 100 * (1 - Mg#)
    """
    Fe     = 100.0 * (1.0 - Mg_number)
    Fe_ref = 100.0 * (1.0 - Mg_ref)
    return -6.0 * (Fe - Fe_ref)


def solidus_bml(P_GPa, Mg_number_bml):
    """BML solidus = Duncan 2018 + Fe correction"""
    return solidus_duncan2018(P_GPa) + fe_correction_K(Mg_number_bml)


def liquidus_bml(P_GPa, Mg_number_bml):
    """BML liquidus = Ruedas 2017 + Fe correction"""
    return liquidus_ruedas2017(P_GPa) + fe_correction_K(Mg_number_bml)


# ============================================================
# ★ STEP 2: melt fraction and outer-core depth scan
# ============================================================

def compute_melt_fraction(T_K, P_GPa, Mg_number_bml):
    """
    φ = (T - T_sol) / (T_liq - T_sol), clipped to [0, 1]
    """
    T_sol = solidus_bml(P_GPa, Mg_number_bml)
    T_liq = liquidus_bml(P_GPa, Mg_number_bml)
    if T_liq <= T_sol:
        T_liq = T_sol + 200.0
    phi = (T_K - T_sol) / (T_liq - T_sol)
    return float(np.clip(phi, 0.0, 1.0))


def find_outer_core_depth(bml_P, bml_depth, T_bml, Mg_number_bml,
                           phi_threshold=PHI_OUTER_CORE):
    """
    Scan BML from top to bottom.
    Return depth (km) where φ first >= phi_threshold.
    If never reached, return bottom depth.
    """
    for P, d in zip(bml_P, bml_depth):
        phi = compute_melt_fraction(T_bml, P, Mg_number_bml)
        if phi >= phi_threshold:
            return d
    return bml_depth[-1]


# ============================================================
# ★ STEP 3: Pierru 2026 velocity correction
# ============================================================

def apply_melt_correction_pierru(Vp, Vs, phi):
    """
    Pierru 2026 Fig S9 (seismic-frequency equivalent):
      phi=0.00: dVp=0%,   dVs=0%
      phi=0.10: dVp=10%,  dVs=20%
      phi=0.20: dVp=17.5%,dVs=40%
      phi>=0.30:dVp=20%,  dVs=100% (Vs=0)
    """
    if phi <= 0.0:
        return Vp, Vs

    phi_nodes = np.array([0.0,  0.10, 0.20, 0.30, 1.0])
    dVp_nodes = np.array([0.0,  0.10, 0.175, 0.20, 0.20])
    dVs_nodes = np.array([0.0,  0.20, 0.40,  1.00, 1.00])

    dVp = float(np.interp(phi, phi_nodes, dVp_nodes))
    dVs = float(np.interp(phi, phi_nodes, dVs_nodes))

    return Vp * (1.0 - dVp), max(Vs * (1.0 - dVs), 0.0)

# ============================================================
# Samuel 2023 median model
# ============================================================

SAMUEL_DATA_DIR = ("/net/beno3/data1/jcchen2/Mars_Samuel_2023/"
                   "Nature_Samuel_s41586-023-06601-8/"
                   "METADATA_BML/DATA_FIG2/PANEL_B")

SAMUEL_RHO_PROFILE_PATH = ("/net/beno3/data1/jcchen2/Mars_Samuel_2023/"
                            "Nature_Samuel_s41586-023-06601-8/"
                            "METADATA_BML/DATA_FIG1/PANEL_J/rho_profile.dat")

# ★ hardcoded from rho_profile.dat density jumps
LSL_TOP_DEPTH  = 1574.6   # km  BML top  (solid mantle / BML interface)
TRUE_CMB_DEPTH = 1743.3   # km  true CMB (BML / iron core interface)

CMB_VS_THRESHOLD = 0.1    # km/s

_SAMUEL_CACHE = None

def compute_samuel_median():
    """
    Read all Samuel 2023 vp*.dat + vs*.dat models, compute median
    crust, core profiles and lsl_top / true_cmb depths.

    Format: col0 = velocity (m/s), col1 = radius (km)
    Vs last 2 lines forced to 0 (inner-core placeholder artefact).

    BML definition: from true_cmb_depth upward by BML_thickness km.
    """
    global _SAMUEL_CACHE
    if _SAMUEL_CACHE is not None:
        return _SAMUEL_CACHE

    vp_files = sorted(glob.glob(os.path.join(SAMUEL_DATA_DIR, 'vp*.dat')))
    print(f"Loading {len(vp_files)} Samuel models...")

    crust_z = np.linspace(0, 100, 200)
    core_z  = np.linspace(1500, MARS_RADIUS, 200)

    crust_vp_all = []; crust_vs_all = []; crust_rho_all = []
    core_vp_all  = []
    bml_top_depths  = []
    true_cmb_depths = []

    import re
    for vp_path in vp_files:
        # match vs file with same number
        m = re.search(r'vp(\d+)\.dat', os.path.basename(vp_path))
        if not m:
            continue
        sid = m.group(1)
        vs_path = os.path.join(SAMUEL_DATA_DIR, f'vs{sid}.dat')
        if not os.path.exists(vs_path):
            continue

        try:
            vp_data = np.loadtxt(vp_path)
            vs_data = np.loadtxt(vs_path)

            vp_ms  = vp_data[:, 0].copy()
            vs_ms  = vs_data[:, 0].copy()
            radius = vp_data[:, 1]

            # ★ force last 2 lines Vs=0 (inner-core placeholder)
            vs_ms[-2:] = 0.0

            vp    = vp_ms / 1000.0   # m/s → km/s
            vs    = vs_ms / 1000.0
            depth = MARS_RADIUS - radius

            # sort by depth ascending
            sort_idx = np.argsort(depth)
            depth = depth[sort_idx]
            vp    = vp[sort_idx]
            vs    = vs[sort_idx]

            # Birch's law density estimate (no rho file)
            rho = 0.32 * vp + 0.77

            liquid_mask = vs < CMB_VS_THRESHOLD
            solid_mask  = ~liquid_mask

            if solid_mask.sum() < 5 or liquid_mask.sum() < 5:
                continue

            # lsl_top: first liquid point from surface downward
            bml_top = depth[liquid_mask][0]
            bml_top_depths.append(bml_top)

            # true_cmb: biggest Vp jump within liquid region
            liq_depth = depth[liquid_mask]
            liq_vp    = vp[liquid_mask]
            dvp       = np.abs(np.diff(liq_vp))
            true_cmb  = liq_depth[np.argmax(dvp) + 1]
            true_cmb_depths.append(true_cmb)

            # crust: solid points above 100 km
            sd = depth[solid_mask]
            svp = vp[solid_mask]
            svs = vs[solid_mask]
            sr  = rho[solid_mask]
            if sd.max() > crust_z.max():
                crust_vp_all.append(np.interp(crust_z, sd, svp))
                crust_vs_all.append(np.interp(crust_z, sd, svs))
                crust_rho_all.append(np.interp(crust_z, sd, sr))

            # core: liquid points below true_cmb
            core_liq_mask = liquid_mask & (depth >= true_cmb)
            cd  = depth[core_liq_mask]
            cvp = vp[core_liq_mask]
            cr  = rho[core_liq_mask]
            if len(cd) > 0 and cd.max() >= core_z.max() * 0.9:
                core_vp_all.append(np.interp(core_z, cd, cvp))

        except Exception as e:
            print(f"  Warning: failed to load {sid}: {e}")
            continue

    print(f"  LSL top  (hardcoded): {LSL_TOP_DEPTH:.1f} km")
    print(f"  True CMB (hardcoded): {TRUE_CMB_DEPTH:.1f} km")
    print(f"  Samuel BML thickness: {TRUE_CMB_DEPTH - LSL_TOP_DEPTH:.1f} km")
    print(f"  Crust models: {len(crust_vp_all)}  Core models: {len(core_vp_all)}")

    crust_vp_med  = np.nanmedian(crust_vp_all, axis=0)
    crust_vs_med  = np.nanmedian(crust_vs_all, axis=0)
    crust_rho_med = np.nanmedian(crust_rho_all, axis=0)
    core_vp_med   = np.nanmedian(core_vp_all, axis=0)

    # core rho from rho_profile.dat
    try:
        rho_data   = np.loadtxt(SAMUEL_RHO_PROFILE_PATH)
        rho_all    = rho_data[:, 0]          # kg/m³
        r_all      = rho_data[:, 1]          # km radius
        depth_rho  = MARS_RADIUS - r_all
        sort_r     = np.argsort(depth_rho)
        depth_rho  = depth_rho[sort_r]
        rho_si     = rho_all[sort_r] / 1000.0   # kg/m³ → g/cm³
        core_rho_med = np.interp(core_z, depth_rho, rho_si)
    except Exception as e:
        print(f"  Warning: rho_profile.dat failed ({e}), using Birch's law")
        core_rho_med = 0.32 * core_vp_med + 0.77

    _SAMUEL_CACHE = {
        'crust_z':        crust_z,
        'crust_vp':       crust_vp_med,
        'crust_vs':       crust_vs_med,
        'crust_rho':      crust_rho_med,
        'core_z':         core_z,
        'core_vp':        core_vp_med,
        'core_vs':        np.zeros(len(core_z)),
        'core_rho':       core_rho_med,
        'bml_top_depth':  LSL_TOP_DEPTH,
        'true_cmb_depth': TRUE_CMB_DEPTH,
    }
    return _SAMUEL_CACHE

# ============================================================
# composition
# ============================================================

def composition_from_params(params):
    mgnum = params['Mg#']
    Mg    = mgnum * MGFE_TOTAL
    Fe    = (1.0 - mgnum) * MGFE_TOTAL
    return {
        'Si': FIXED_PARAMS['Si'], 'Mg': Mg, 'Fe': Fe,
        'Ca': FIXED_PARAMS['Ca'], 'Al': FIXED_PARAMS['Al'],
        'Na': FIXED_PARAMS['Na'], 'Cr': FIXED_PARAMS['Cr'],
        'T_lit': params['T_lit'], 'P_lit': params['P_lit'],
    }

def compute_oxygen(p):
    return (2.0 * p['Si'] + p['Mg'] + p['Fe'] + p['Ca'] +
            1.5 * p['Al'] + 0.5 * p.get('Na', FIXED_PARAMS['Na']) +
            1.5 * p.get('Cr', FIXED_PARAMS['Cr']))

# ============================================================
# HeFESTo
# ============================================================

CONTROL_PHASES = """\
phase plg
0
an
ab
phase sp
1
sp
hc
smag
picr
phase opx
0
en
fs
mgts
odi
phase c2c
0
mgc2
fec2
phase cpx
1
di
he
cen
cats
jd
acm
phase wo
1
wo
phase pwo
1
pwo
phase gt
0
py
al
gr
mgmj
namj
andr
knor
phase cpv
0
capv
phase ol
1
fo
fa
phase wa
0
mgwa
fewa
phase ri
0
mgri
feri
phase il
0
mgil
feil
co
hem
esk
phase pv
0
mgpv
fepv
alpv
hepv
hlpv
fapv
crpv
phase ppv
0
mppv
fppv
appv
hppv
cppv
phase cf
0
mgcf
fecf
nacf
hmag
crcf
phase nal
0
mnal
fnal
nnal
phase mw
0
pe
wu
wuls
anao
mag
phase qtz
0
qtz
phase coes
0
coes
phase st
0
st
phase apbo
0
apbo
phase ky
0
ky
phase neph
0
neph
phase fea
0
fea
phase feg
0
feg
phase fee
0
fee
"""

def make_control_lines(p, O, line1):
    return [
        line1, "8,2,4,2", "oxides",
        f"Si      {p['Si']:.5f}     {p['Si']:.5f}    0",
        f"Mg      {p['Mg']:.5f}     {p['Mg']:.5f}    0",
        f"Fe      {p['Fe']:.5f}     {p['Fe']:.5f}    0",
        f"Ca      {p['Ca']:.5f}     {p['Ca']:.5f}    0",
        f"Al      {p['Al']:.5f}     {p['Al']:.5f}    0",
        f"Na      {p.get('Na', FIXED_PARAMS['Na']):.5f}     {p.get('Na', FIXED_PARAMS['Na']):.5f}    0",
        f"Cr      {p.get('Cr', FIXED_PARAMS['Cr']):.5f}     {p.get('Cr', FIXED_PARAMS['Cr']):.5f}    0",
        f" O      {O:.5f}     {O:.5f}    0",
        "1,1,1,1", PAR_DIR, "73", CONTROL_PHASES,
    ]


def _cleanup_keep_key_files(step_dir):
    keep = {'fort.56', 'ad.in', 'control'}
    if not os.path.isdir(step_dir):
        return
    for fname in os.listdir(step_dir):
        if fname not in keep:
            fpath = os.path.join(step_dir, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath, ignore_errors=True)
            except Exception:
                pass


def run_hefesto_single(run_dir, control_lines, ad_in_content=None, timeout=1600):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "control"), 'w') as f:
        f.write("\n".join(control_lines))
    if ad_in_content is not None:
        with open(os.path.join(run_dir, "ad.in"), 'w') as f:
            f.write(ad_in_content)
    main_dst = os.path.join(run_dir, "main")
    if not os.path.exists(main_dst):
        shutil.copy2(HEFESTO_MAIN, main_dst)
        os.chmod(main_dst, 0o755)
    log_path = os.path.join(run_dir, "hefesto.log")
    try:
        with open(log_path, 'w') as log:
            subprocess.run(["./main"], cwd=run_dir,
                           stdout=log, stderr=log, timeout=timeout)
    except (subprocess.TimeoutExpired, Exception):
        return None
    fort56 = os.path.join(run_dir, "fort.56")
    if not os.path.exists(fort56) or os.path.getsize(fort56) == 0:
        return None
    return fort56


def read_fort56_full(fort56_path):
    try:
        with open(fort56_path) as f:
            f.readline()
            cols = f.readline().split()
        if not cols:
            return None

        clean_lines = []
        with open(fort56_path) as f:
            for i, line in enumerate(f):
                if i < 2:
                    clean_lines.append(line)
                    continue
                import re
                suspect = re.search(r'\d-\d', line)
                if suspect:
                    continue
                clean_lines.append(line)

        if len(clean_lines) <= 2:
            return None

        import io
        df = pd.read_csv(io.StringIO(''.join(clean_lines)),
                         sep=r'\s+', skiprows=2, names=cols)
        if df.empty:
            return None

    except Exception:
        return None

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    key_cols = ['P(GPa)', 'T(K)', 'S(J/g/K)', 'rho(g/cm^3)',
                'VS(km/s)', 'VP(km/s)']
    for kc in key_cols:
        if kc in df.columns and df[kc].isna().any():
            df = df.dropna(subset=[kc])

    if len(df) < 1:
        return None

    P_GPa = df['P(GPa)'].values
    T_K   = df['T(K)'].values
    S     = df['S(J/g/K)'].values
    rho   = df['rho(g/cm^3)'].values
    Vs    = df['VS(km/s)'].values
    Vp    = df['VP(km/s)'].values

    dP    = np.diff(P_GPa) * 1e9
    rho_m = (rho[:-1] + rho[1:]) / 2
    depth = np.zeros(len(P_GPa))
    for i in range(len(dP)):
        g_i        = gravity_mars(depth[i])
        rho_si     = rho_m[i] * 1000
        dz_i       = dP[i] / (rho_si * g_i) / 1000
        depth[i+1] = depth[i] + dz_i

    return {
        'depth_km': depth,
        'P_GPa':    P_GPa,
        'T_K':      T_K,
        'S':        S,
        'Vp':       Vp,
        'Vs':       Vs,
        'rho':      rho,
    }


def read_fort56(fort56_path):
    return read_fort56_full(fort56_path)


def run_hefesto(params, run_dir, P_bml_top=None):
    """
    Three-step HeFESTo run:
    Step 1: NPT @ (P_lit, T_lit)  → read S_lit
    Step 2: NPT scan → build adiabatic T(P) up to P_bml_top
    Step 3: iensem=-1 with ad.in  → full Vp/Vs/rho profile

    P_bml_top: pressure at BML top (GPa), computed dynamically from
               pressure_mars(TRUE_CMB_DEPTH - BML_thickness).
               Defines the bottom of the mantle HeFESTo run.
    Returns: (fort56_path, fort56_data)
    """
    p     = composition_from_params(params)
    O     = compute_oxygen(p)
    T_lit = p['T_lit']
    P_lit = p['P_lit']

    # compute P_bml_top if not provided
    if P_bml_top is None:
        bml_thickness = params.get('BML_thickness', 168.7)
        bml_top_km    = TRUE_CMB_DEPTH - bml_thickness
        P_bml_top     = float(pressure_mars(bml_top_km))
    print(f"    Mantle runs to P_bml_top = {P_bml_top:.2f} GPa")
    dir1  = os.path.join(run_dir, "s1_npt")
    line1 = f"{P_lit:.4f},{P_lit:.4f},1,{T_lit:.2f},{T_lit:.2f},0,0,0,0"
    fort56_1 = run_hefesto_single(dir1, make_control_lines(p, O, line1))
    if fort56_1 is None:
        print("    Step 1 failed")
        _cleanup_keep_key_files(dir1)
        return None, None
 
    data1 = read_fort56_full(fort56_1)
    if data1 is None:
        print("    Step 1 read failed")
        _cleanup_keep_key_files(dir1)
        return None, None
 
    S_lit = float(data1['S'][0])
    if not np.isfinite(S_lit) or S_lit <= 0 or S_lit > 10:
        print(f"    Step 1: invalid S_lit={S_lit}, skip")
        _cleanup_keep_key_files(dir1)
        return None, None
    print(f"    S_lit = {S_lit:.6f} J/g/K  "
          f"@ P={P_lit:.2f} GPa, T={T_lit:.1f} K")
    _cleanup_keep_key_files(dir1)
 
    # ── Step 2: NPT scan to build adiabatic temperature profile ────────────────────
    print(f"    Step 2: NPT scan...")
    # scan points up to P_bml_top (dynamic BML top pressure)
    P_scan_fixed = np.array([9.0, 12.0, 15.0, 17.0])
    P_scan_fixed = P_scan_fixed[P_scan_fixed < P_bml_top]
    P_scan_pts   = np.concatenate([[P_lit], P_scan_fixed, [P_bml_top]])
    T_guess        = T_lit
    P_adiab_list   = [P_lit]
    T_adiab_list   = [T_lit]
    n_npt_calls    = 0
    n_interpolated = 0
 
    for i_p, P_target in enumerate(P_scan_pts[1:]):
        d_lo, d_hi, T_lo, T_hi = None, None, None, None
 
        for dT in [20.0, 10.0, 5.0, 2.0]:
            T_lo = T_guess - dT
            T_hi = T_guess + dT
 
            dir_lo  = os.path.join(run_dir, f"s2_scan_{i_p}_lo")
            dir_hi  = os.path.join(run_dir, f"s2_scan_{i_p}_hi")
            line_lo = (f"{P_target:.4f},{P_target:.4f},1,"
                       f"{T_lo:.2f},{T_lo:.2f},0,0,0,0")
            line_hi = (f"{P_target:.4f},{P_target:.4f},1,"
                       f"{T_hi:.2f},{T_hi:.2f},0,0,0,0")
 
            f_lo = run_hefesto_single(
                dir_lo, make_control_lines(p, O, line_lo))
            f_hi = run_hefesto_single(
                dir_hi, make_control_lines(p, O, line_hi))
            n_npt_calls += 2
 
            d_lo = read_fort56_full(f_lo) if f_lo else None
            d_hi = read_fort56_full(f_hi) if f_hi else None
            _cleanup_keep_key_files(dir_lo)
            _cleanup_keep_key_files(dir_hi)
 
            if d_lo is not None and d_hi is not None:
                break
 
        if d_lo is None or d_hi is None:
            # all dT attempts failed → extrapolate from previous point gradient
            if len(T_adiab_list) >= 2:
                dT_dP     = ((T_adiab_list[-1] - T_adiab_list[-2]) /
                             (P_adiab_list[-1] - P_adiab_list[-2]))
                T_adiab_p = (T_adiab_list[-1] +
                             dT_dP * (P_target - P_adiab_list[-1]))
            else:
                T_adiab_p = T_guess
            T_adiab_p = float(np.clip(T_adiab_p, 800, 4000))
            print(f"    P={P_target:.1f} GPa: interpolated T={T_adiab_p:.1f}K")
            n_interpolated += 1
        else:
            S_lo = float(d_lo['S'][0])
            S_hi = float(d_hi['S'][0])
            if abs(S_hi - S_lo) < 1e-8:
                T_adiab_p = T_guess
            else:
                T_adiab_p = (T_lo + (S_lit - S_lo) *
                             (T_hi - T_lo) / (S_hi - S_lo))
            T_adiab_p = float(np.clip(T_adiab_p, 800, 4000))
 
        P_adiab_list.append(P_target)
        T_adiab_list.append(T_adiab_p)
        T_guess = T_adiab_p
 
    P_adiab = np.array(P_adiab_list)
    T_adiab = np.array(T_adiab_list)
    print(f"    Adiabatic (NPT scan, {n_npt_calls} calls, "
          f"{n_interpolated} interp): "
          f"T={T_adiab[0]:.1f}K @ P={P_adiab[0]:.2f}GPa"
          f"  →  T={T_adiab[-1]:.1f}K @ P={P_adiab[-1]:.2f}GPa")
 
    # ── Merge conductive + adiabatic → ad.in ─────────────────
    P_cond = np.linspace(1.04, P_lit, 100)
    T_cond = T_SURF + (T_lit - T_SURF) * (P_cond / P_lit)
 
    P_full = np.concatenate([P_cond, P_adiab])
    T_full = np.concatenate([T_cond, T_adiab])
    sort_idx       = np.argsort(P_full)
    P_full, T_full = P_full[sort_idx], T_full[sort_idx]
    _, uniq        = np.unique(P_full, return_index=True)
    P_full, T_full = P_full[uniq], T_full[uniq]
 
    ad_in = "".join(f"{P:.6f} 0.000000 {T:.6f}\n"
                    for P, T in zip(P_full, T_full))
 
    # ── Step 3: iensem=-1, run full calculation with ad.in ──────────────
    dir3  = os.path.join(run_dir, "s3_final")
    line3 = f"0,{P_bml_top:.4f},50,0,0,0,-1,0,0"
    fort56_3 = run_hefesto_single(dir3, make_control_lines(p, O, line3),
                                  ad_in_content=ad_in)
    _cleanup_keep_key_files(dir3)
 
    if fort56_3 is None:
        print("    Step 3 failed")
        return None, None
 
    data3 = read_fort56_full(fort56_3)
    if data3 is None:
        print("    Step 3 read failed")
        return None, None
 
    print("    Step 3 OK  (adiabat method=NPT_scan)")
    data3['P_profile'] = P_full
    data3['T_profile'] = T_full
 
    return fort56_3, data3


def run_hefesto_bml(params, run_dir, T_bml, T_mantle_bottom,
                    true_cmb_depth, n_points=20):
    """
    BML defined from core upward:
      BML bottom = true_cmb_depth (fixed, iron core top)
      BML top    = true_cmb_depth - BML_thickness
    Temperature: linear from T_mantle_bottom (top) to T_bml (bottom).
    """
    bml_thickness = params.get('BML_thickness', 168.7)   # km
    Mg_number_bml = params['Mg#_bml']

    # BML from core upward:
    #   bottom = true_cmb_depth, top = true_cmb_depth - bml_thickness
    bml_top_depth = true_cmb_depth - bml_thickness
    P_bottom = float(pressure_mars(true_cmb_depth))
    P_top    = float(pressure_mars(bml_top_depth))
    P_top    = max(P_top, 5.0)   # safety floor

    p = composition_from_params({
        'Mg#':   Mg_number_bml,
        'T_lit': T_mantle_bottom,   # use top temperature as reference
        'P_lit': P_top,
    })
    O = compute_oxygen(p)

    P_range = np.linspace(P_top, P_bottom, n_points)

    # ★ linear temperature profile: top = T_mantle_bottom, bottom = T_bml
    T_range = np.linspace(T_mantle_bottom, T_bml, n_points)

    ad_in = "".join(
        f"{P:.6f} 0.000000 {T:.6f}\n"
        for P, T in zip(P_range, T_range)
    )

    dir_bml = os.path.join(run_dir, "bml")
    line    = f"{P_top:.4f},{P_bottom:.4f},{n_points},0,0,0,-1,0,0"
    fort56  = run_hefesto_single(dir_bml,
                                 make_control_lines(p, O, line),
                                 ad_in_content=ad_in)
    _cleanup_keep_key_files(dir_bml)

    if fort56 is None:
        return None

    bml_raw = read_fort56_full(fort56)
    if bml_raw is None:
        return None

    # compute depth from pressure
    rho   = bml_raw['rho']
    P_GPa = bml_raw['P_GPa']
    dP    = np.diff(P_GPa) * 1e9
    depth = np.zeros(len(P_GPa))
    for i in range(len(dP)):
        g_i        = gravity_mars(depth[i])
        rho_si     = ((rho[i] + rho[i+1]) / 2) * 1000
        depth[i+1] = depth[i] + dP[i] / (rho_si * g_i) / 1000.0

    # ★ apply Pierru correction using each point's own temperature
    Vp_solid = bml_raw['Vp']
    Vs_solid = bml_raw['Vs']
    T_actual = bml_raw['T_K']   # HeFESTo output temperature at each point
    phi_arr  = np.zeros(len(P_GPa))
    Vp_corr  = np.zeros(len(P_GPa))
    Vs_corr  = np.zeros(len(P_GPa))
    outer_core_depth_offset = depth[-1]   # default: bottom

    for i in range(len(P_GPa)):
        # use actual temperature from HeFESTo output
        phi = compute_melt_fraction(float(T_actual[i]), P_GPa[i], Mg_number_bml)
        phi_arr[i] = phi
        Vp_corr[i], Vs_corr[i] = apply_melt_correction_pierru(
            float(Vp_solid[i]), float(Vs_solid[i]), phi
        )
        if phi >= PHI_OUTER_CORE and outer_core_depth_offset == depth[-1]:
            outer_core_depth_offset = depth[i]

    print(f"    BML: thickness={bml_thickness:.0f} km  "
          f"T_top={T_mantle_bottom:.0f}K  T_bot={T_bml:.0f}K  "
          f"dT={T_bml - T_mantle_bottom:.0f}K  "
          f"phi_top={phi_arr[0]:.3f}  phi_bot={phi_arr[-1]:.3f}  "
          f"outer_core_offset={outer_core_depth_offset:.1f} km")

    return {
        'depth_km':                  depth,
        'P_GPa':                     P_GPa,
        'T_K':                       T_actual,
        'Vp':                        Vp_corr,
        'Vs':                        Vs_corr,
        'rho':                       rho,
        'phi':                       phi_arr,
        'outer_core_depth_offset':   outer_core_depth_offset,
    }

# ============================================================
# TauP
# ============================================================

def build_taup(fort56_data, model_name, samuel_cache, bml_data=None):
    """
    ★ STEP 5: outer-core marker written dynamically at phi=0.30 depth.
    Above that depth: partial melt (Vs > 0, Pierru-corrected).
    Below: fully molten (Vs = 0) + iron core merged as one outer-core.
    """
    os.makedirs(TAUP_WORK_DIR, exist_ok=True)
    model_name = model_name.replace(".npz", "")
    npz_path   = os.path.join(TAUP_WORK_DIR, f'{model_name}.npz')
    nd_path    = os.path.join(TAUP_WORK_DIR, f"{model_name}.nd")
    if os.path.exists(npz_path):
        return TauPyModel(model=npz_path)

    true_cmb_depth = samuel_cache['true_cmb_depth']

    # BML top is dynamic: from bml_data if available, else from samuel lsl_top
    if bml_data is not None:
        bml_top_depth = float(bml_data['depth_km'][0])
    else:
        bml_top_depth = samuel_cache['bml_top_depth']
    mantle_bottom = bml_top_depth

    hef_depth = fort56_data['depth_km']
    hef_Vp    = fort56_data['Vp']
    hef_Vs    = fort56_data['Vs']
    hef_rho   = fort56_data['rho']

    mantle_mask = (hef_depth >= 100.0) & (hef_depth <= mantle_bottom)
    man_depth   = hef_depth[mantle_mask]
    man_Vp      = hef_Vp[mantle_mask]
    man_Vs      = hef_Vs[mantle_mask]
    man_rho     = hef_rho[mantle_mask]
    if len(man_depth) == 0:
        raise ValueError(f"Mantle depth range insufficient: "
                     f"hef_depth={hef_depth[0]:.1f}–{hef_depth[-1]:.1f} km, "
                     f"mantle_bottom={mantle_bottom:.1f} km")

    with open(nd_path, 'w') as f:
        # crust
        for d, vp, vs, r in zip(samuel_cache['crust_z'], samuel_cache['crust_vp'],
                                 samuel_cache['crust_vs'], samuel_cache['crust_rho']):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        f.write("mantle\n")

        # solid mantle
        for d, vp, vs, r in zip(man_depth, man_Vp, man_Vs, man_rho):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

        if bml_data is not None:
            bml_depth = bml_data['depth_km']   # absolute depths
            bml_vp    = bml_data['Vp']
            bml_vs    = bml_data['Vs']
            bml_rho   = bml_data['rho']

            # ★ outer_core_depth in absolute coordinates
            outer_core_abs = bml_data.get('outer_core_depth_abs', bml_depth[-1])

            bml_mask = (bml_depth >= mantle_bottom) & (bml_depth <= true_cmb_depth)

            # partial melt region (Vs > 0, above outer_core_abs)
            partial_mask = bml_mask & (bml_depth < outer_core_abs)
            for d, vp, vs, r in zip(bml_depth[partial_mask],
                                     bml_vp[partial_mask],
                                     bml_vs[partial_mask],
                                     bml_rho[partial_mask]):
                f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

            # ★ outer-core discontinuity at outer_core_abs
            idx_oc = np.searchsorted(bml_depth, outer_core_abs)
            idx_oc = min(idx_oc, len(bml_depth) - 1)

            # discontinuity: same depth two lines + outer-core keyword (TauP .nd format)
            f.write(f"{outer_core_abs:.3f}  "
                    f"{bml_vp[idx_oc]:.4f}  "
                    f"{bml_vs[idx_oc]:.4f}  "
                    f"{bml_rho[idx_oc]:.4f}\n")
            f.write("outer-core\n")
            f.write(f"{outer_core_abs:.3f}  "
                    f"{bml_vp[idx_oc]:.4f}  "
                    f"0.0000  "
                    f"{bml_rho[idx_oc]:.4f}\n")

            # fully molten BML below outer_core_abs
            fully_mask = bml_mask & (bml_depth > outer_core_abs)
            for d, vp, r in zip(bml_depth[fully_mask],
                                  bml_vp[fully_mask],
                                  bml_rho[fully_mask]):
                f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")

            # iron core
            core_z   = samuel_cache['core_z']
            core_vp  = samuel_cache['core_vp']
            core_rho = samuel_cache['core_rho']
            mask     = core_z >= true_cmb_depth
            f.write(f"{true_cmb_depth:.3f}  "
                    f"{core_vp[mask][0]:.4f}  0.0000  "
                    f"{core_rho[mask][0]:.4f}\n")
            for d, vp, r in zip(core_z[mask][1:],
                                 core_vp[mask][1:],
                                 core_rho[mask][1:]):
                f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")

        else:
            # no BML: CMB at true_cmb_depth
            core_z   = samuel_cache['core_z']
            core_vp  = samuel_cache['core_vp']
            core_rho = samuel_cache['core_rho']
            mask     = core_z >= true_cmb_depth

            f.write(f"{man_depth[-1]:.3f}  "
                    f"{man_Vp[-1]:.4f}  {man_Vs[-1]:.4f}  {man_rho[-1]:.4f}\n")
            f.write(f"{true_cmb_depth:.3f}  "
                    f"{core_vp[mask][0]:.4f}  0.0000  {core_rho[mask][0]:.4f}\n")
            f.write("outer-core\n")
            for d, vp, r in zip(core_z[mask][1:],
                                 core_vp[mask][1:],
                                 core_rho[mask][1:]):
                f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")

    build_taup_model(nd_path, output_folder=TAUP_WORK_DIR)
    return TauPyModel(model=npz_path)

# ============================================================
# Moment of Inertia
# ============================================================

def compute_mass_and_moi(fort56_data, samuel_cache, bml_data=None):
    R            = MARS_RADIUS_M
    true_cmb_km  = samuel_cache['true_cmb_depth']

    # BML top is dynamic: from bml_data if available, else from samuel lsl_top
    if bml_data is not None:
        bml_top_km = float(bml_data['depth_km'][0])
    else:
        bml_top_km = samuel_cache['bml_top_depth']
    mantle_bottom_km = bml_top_km

    crust_z      = samuel_cache['crust_z']
    crust_rho    = samuel_cache['crust_rho']
    crust_mask   = crust_z <= 100.0
    crust_r      = (R - crust_z[crust_mask] * 1000)
    crust_rho_si = crust_rho[crust_mask] * 1000

    hef_depth   = fort56_data['depth_km']
    hef_rho     = fort56_data['rho']
    mantle_mask = (hef_depth >= 100.0) & (hef_depth <= mantle_bottom_km)
    man_r       = (R - hef_depth[mantle_mask] * 1000)
    man_rho_si  = hef_rho[mantle_mask] * 1000

    if bml_data is not None:
        bml_depth  = bml_data['depth_km']
        bml_rho    = bml_data['rho']
        bml_mask   = ((bml_depth >= mantle_bottom_km) &
                      (bml_depth <= true_cmb_km))
        bml_r      = (R - bml_depth[bml_mask] * 1000)
        bml_rho_si = bml_rho[bml_mask] * 1000
    else:
        bml_r      = np.array([])
        bml_rho_si = np.array([])

    core_z      = samuel_cache['core_z']
    core_rho    = samuel_cache['core_rho']
    core_mask   = core_z >= true_cmb_km
    core_r      = (R - core_z[core_mask] * 1000)
    core_rho_si = core_rho[core_mask] * 1000

    all_r   = np.concatenate([core_r, bml_r, man_r, crust_r])
    all_rho = np.concatenate([core_rho_si, bml_rho_si,
                               man_rho_si, crust_rho_si])
    sort_idx = np.argsort(all_r)
    all_r    = all_r[sort_idx]
    all_rho  = all_rho[sort_idx]

    M   = 4 * np.pi * np.trapezoid(all_rho * all_r**2, all_r)
    I   = (8 * np.pi / 3) * np.trapezoid(all_rho * all_r**4, all_r)
    moi = I / (M * R**2)

    return M, moi

# ============================================================
# Solidus penalty
# ============================================================

def compute_solidus_penalty(fort56_data, params):
    penalty = 0.0

    P_prof = fort56_data.get('P_profile', None)
    T_prof = fort56_data.get('T_profile', None)

    if P_prof is not None and T_prof is not None:
        P_lit         = params['P_lit']
        bml_thickness = params.get('BML_thickness', 168.7)
        bml_top_km    = TRUE_CMB_DEPTH - bml_thickness
        # ★ use pressure_mars for accurate P at BML top
        P_bml_top = float(pressure_mars(bml_top_km))
        # P_prof and T_prof are both in pressure space (P_full, T_full)
        mantle_mask = (P_prof >= P_lit) & (P_prof <= P_bml_top)
        P_m = P_prof[mantle_mask]
        T_m = T_prof[mantle_mask]

        if len(P_m) > 0:
            T_sol_arr     = solidus_duncan2018(P_m)
            excess        = T_m - T_sol_arr
            mantle_excess = float(np.sum(excess[excess > 0])) / 100.0
            if mantle_excess > 0:
                print(f"    Mantle solidus penalty = {mantle_excess:.4f}")
                penalty += mantle_excess

    # BML solidus penalty removed:
    # we do not force BML to be molten — the data decides whether it exists.

    return penalty

# ============================================================
# Misfit
# ============================================================

def _diff(times, a, b):
    return times[a] - times[b] if a in times and b in times else None


def compute_misfit(taup_model, obs_dataset, fort56_data,
                   samuel_cache, bml_data=None, params=None):

    tt_total = 0.0
    tt_n     = 0

    # ★ add Pdiff for S1000a (PP-PbdiffPcP constraint)
    phases_standard   = ['P', 'S', 'pP', 'sP', 'PP', 'PPP',
                         'SS', 'SSS', 'sS', 'ScS', 'SKS']
    phases_with_pdiff = phases_standard + ['Pdiff']

    for event, obs in obs_dataset.items():
        delta = obs['delta']
        depth = obs.get('depth', 10.0)

        phase_list = phases_with_pdiff if event == 'S1000a' else phases_standard

        try:
            arrivals = taup_model.get_travel_times(
                source_depth_in_km=depth,
                distance_in_degree=delta,
                phase_list=phase_list)
            print(f"    {event}: delta={delta:.1f} arrivals={len(arrivals)}")
        except Exception as e:
            print(f"    {event}: FAILED {e}")
            continue

        times = {}
        for a in arrivals:
            if a.name not in times:
                times[a.name] = a.time

        pred = {
            'S-P':           _diff(times, 'S',    'P'),
            'pP-P':          _diff(times, 'pP',   'P'),
            'sP-P':          _diff(times, 'sP',   'P'),
            'PP-P':          _diff(times, 'PP',   'P'),
            'PPP-P':         _diff(times, 'PPP',  'P'),
            'sS-S':          _diff(times, 'sS',   'S'),
            'SS-S':          _diff(times, 'SS',   'S'),
            'SSS-S':         _diff(times, 'SSS',  'S'),
            'ScS-S':         _diff(times, 'ScS',  'S'),
            'SS-PP':         _diff(times, 'SS',   'PP'),
            'SKS-PP':        _diff(times, 'SKS',  'PP'),
            # ★ PP - PbdiffPcP: TauPy Pdiff = PbdiffPcP in BML models
            'PP-PbdiffPcP':  _diff(times, 'PP',   'Pdiff'),
        }

        for phase, obs_val in obs.items():
            if phase in ('delta', 'depth'):
                continue
            if not isinstance(obs_val, tuple):
                continue
            if phase not in pred or pred[phase] is None:
                continue
            obs_t, sigma = obs_val
            if sigma <= 0:
                continue
            if not np.isfinite(obs_t) or not np.isfinite(pred[phase]):
                continue
            val = abs(obs_t - pred[phase]) / sigma
            if not np.isfinite(val):
                print(f"    WARNING: non-finite TT at {event} {phase}: "
                      f"obs={obs_t:.2f} pred={pred[phase]:.2f} "
                      f"sigma={sigma:.2f} val={val}")
                continue
            tt_total += val
            tt_n     += 1

    tt_misfit = tt_total / tt_n if tt_n > 0 else 999.0
    if not np.isfinite(tt_misfit):
        tt_misfit = 999.0

    M_pred, moi_pred = compute_mass_and_moi(fort56_data, samuel_cache,
                                             bml_data=bml_data)
    mass_misfit = abs(MARS_MASS_OBS - M_pred) / MARS_MASS_SIGMA
    moi_misfit  = abs(MOI_OBS - moi_pred)     / MOI_SIGMA

    if not np.isfinite(mass_misfit): mass_misfit = 999.0
    if not np.isfinite(moi_misfit):  moi_misfit  = 999.0

    solidus_penalty = compute_solidus_penalty(fort56_data, params)
    if not np.isfinite(solidus_penalty): solidus_penalty = 0.0

    total_n      = tt_n + 2
    total_misfit = (tt_total + mass_misfit + moi_misfit
                    + solidus_penalty) / total_n

    if not np.isfinite(total_misfit):
        total_misfit = 999.0

    print(f"    TT misfit   = {tt_misfit:.4f}  (n={tt_n})")
    print(f"    Mass misfit = {mass_misfit:.4f}  "
          f"(pred={M_pred:.4e}, obs={MARS_MASS_OBS:.4e})")
    print(f"    MoI  misfit = {moi_misfit:.4f}  "
          f"(pred={moi_pred:.5f}, obs={MOI_OBS:.5f})")
    if solidus_penalty > 0:
        print(f"    Solidus pen = {solidus_penalty:.4f}")
    print(f"    Total misfit= {total_misfit:.4f}  (n={total_n})")

    components = {
        'tt':      tt_misfit,
        'mass':    mass_misfit,
        'moi':     moi_misfit,
        'solidus': solidus_penalty,
    }
    return total_misfit, total_n, components

# ============================================================
# forward model
# ============================================================

def forward(params, run_dir, model_name, samuel_cache,
            skip_bml_density_check=False):

    # ★ compute BML top depth and pressure before running HeFESTo
    true_cmb_depth = samuel_cache['true_cmb_depth']
    bml_thickness  = params.get('BML_thickness', 168.7)
    bml_top_km     = true_cmb_depth - bml_thickness
    P_bml_top      = float(pressure_mars(bml_top_km))
    print(f"    BML top={bml_top_km:.0f} km  P_bml_top={P_bml_top:.2f} GPa")

    fort56, fort56_data = run_hefesto(params, run_dir, P_bml_top=P_bml_top)
    if fort56 is None or fort56_data is None:
        return None, None, None, None, None

    # ★ extract T_mantle_bottom from isentropic profile at BML top
    # T_profile and P_profile are both in pressure space (from run_hefesto)
    T_profile = fort56_data.get('T_profile', None)
    P_profile = fort56_data.get('P_profile', None)
    if T_profile is not None and P_profile is not None:
        # P_bml_top already computed above — interpolate in pressure space
        T_mantle_bottom = float(np.interp(P_bml_top, P_profile, T_profile))
    else:
        T_mantle_bottom = float(fort56_data['T_K'][-1])
    print(f"    T_mantle_bottom = {T_mantle_bottom:.1f} K @ P={P_bml_top:.2f} GPa")

    # ★ run BML with temperature gradient from T_mantle_bottom to T_bml
    bml_raw = run_hefesto_bml(
        params          = params,
        run_dir         = run_dir,
        T_bml           = params['T_bml'],
        T_mantle_bottom = T_mantle_bottom,
        true_cmb_depth  = true_cmb_depth,
    )

    bml_data = None
    if bml_raw is not None:
        # bml_top_km already computed above as true_cmb_depth - bml_thickness

        # convert relative depth to absolute
        bml_raw['depth_km'] = (bml_raw['depth_km']
                               - bml_raw['depth_km'][0] + bml_top_km)

        # ★ compute absolute outer_core_depth
        outer_core_offset = bml_raw['outer_core_depth_offset']
        bml_raw['outer_core_depth_abs'] = bml_top_km + outer_core_offset

        rho_mantle_bottom = float(fort56_data['rho'][-1])
        rho_bml_top       = float(bml_raw['rho'][0])
        rho_bml_bottom    = float(bml_raw['rho'][-1])

        core_z    = samuel_cache['core_z']
        core_rho  = samuel_cache['core_rho']
        true_cmb  = samuel_cache['true_cmb_depth']
        core_mask = core_z >= true_cmb
        rho_core_top = float(core_rho[core_mask][0]) if core_mask.any() else 6.4

        upper_contrast = rho_bml_top  - rho_mantle_bottom
        lower_contrast = rho_core_top - rho_bml_bottom

        print(f"    BML density: mantle_bot={rho_mantle_bottom:.4f}  "
              f"bml_top={rho_bml_top:.4f}  "
              f"bml_bot={rho_bml_bottom:.4f}  "
              f"core_top={rho_core_top:.4f}")
        print(f"    Upper contrast={upper_contrast:+.4f}  "
              f"Lower contrast={lower_contrast:+.4f}")
        print(f"    outer_core_depth = {bml_raw['outer_core_depth_abs']:.1f} km")

        if upper_contrast <= 0 and not skip_bml_density_check:
            print(f"    BML REJECTED: upper interface unstable")
            return 999.0, 1, {
                'tt': 999.0, 'mass': 999.0, 'moi': 999.0,
                'solidus': 0.0,
                'upper_contrast': upper_contrast,
                'lower_contrast': lower_contrast,
            }, None

        if lower_contrast <= 0 and not skip_bml_density_check:
            print(f"    BML REJECTED: lower interface unstable")
            return 999.0, 1, {
                'tt': 999.0, 'mass': 999.0, 'moi': 999.0,
                'solidus': 0.0,
                'upper_contrast': upper_contrast,
                'lower_contrast': lower_contrast,
            }, None

        bml_data = bml_raw
        bml_data['upper_contrast'] = upper_contrast
        bml_data['lower_contrast'] = lower_contrast

    try:
        taup_model = build_taup(fort56_data, model_name,
                                samuel_cache, bml_data=bml_data)
    except Exception as e:
        print(f"    TauP failed: {e}")
        return None, None, None, None, None

    # ★ use SAMUEL_DATA instead of KHAN_DATA
    misfit, n_data, components = compute_misfit(
        taup_model, SAMUEL_DATA, fort56_data, samuel_cache,
        bml_data=bml_data, params=params)

    if components is not None and bml_data is not None:
        components['upper_contrast'] = bml_data.get('upper_contrast')
        components['lower_contrast'] = bml_data.get('lower_contrast')

    # ★ fix 1+2: return fort56_data as 4th value (same as nGibbs version)
    return misfit, n_data, components, fort56_data, bml_data

# ============================================================
# MCMC
# ============================================================

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

    load_gravity_profile()
    samuel_cache = compute_samuel_median()

    rng     = np.random.default_rng(42 + chain_id)
    current = start_params.copy() if start_params else START_PARAMS.copy()

    # ★ backward compatibility: ensure BML_thickness exists
    if 'BML_thickness' not in current:
        current['BML_thickness'] = 150.0

    chain_file = os.path.join(chain_dir, "chain.json")
    chain      = []
    if os.path.exists(chain_file):
        with open(chain_file) as f:
            chain = json.load(f)
        if chain:
            current = chain[-1]['params']
            # ★ backward compatibility
            if 'BML_thickness' not in current:
                current['BML_thickness'] = 150.0
            print(f"Chain {chain_id}: resuming from step {len(chain)}")

    step_start   = len(chain)
    accept_count = 0
    current_components = {
        'tt': 999.0, 'mass': 999.0, 'moi': 999.0,
        'solidus': 0.0, 'upper_contrast': None, 'lower_contrast': None,
    }

    print(f"\nChain {chain_id} starting")
    print(f"  Target steps: {n_steps}")
    print("=" * 60)

    if chain:
        current_misfit     = chain[-1]['misfit']
        accept_count       = sum(1 for s in chain if s.get('accepted', False))
        current_components = {
            'tt':             chain[-1].get('misfit_tt',      999.0),
            'mass':           chain[-1].get('misfit_mass',    999.0),
            'moi':            chain[-1].get('misfit_moi',     999.0),
            'solidus':        chain[-1].get('misfit_solidus', 0.0),
            'upper_contrast': chain[-1].get('upper_contrast'),
            'lower_contrast': chain[-1].get('lower_contrast'),
        }
    else:
        run_dir    = os.path.join(chain_dir, "step_current")
        model_name = f"mcmc_{prefix}_c{chain_id:02d}_current"

        MAX_RETRIES = 20
        for attempt in range(MAX_RETRIES):
            if attempt == 0:
                trial = current.copy()
            else:
                trial = {}
                for key in PRIOR:
                    lo, hi = PRIOR[key]
                    trial[key] = float(rng.uniform(lo, hi))
                print(f"  Retry {attempt}/{MAX_RETRIES}: "
                      f"T_lit={trial['T_lit']:.1f}  "
                      f"P_lit={trial['P_lit']:.2f}  "
                      f"Mg#={trial['Mg#']:.3f}  "
                      f"T_bml={trial['T_bml']:.1f}  "
                      f"Mg#_bml={trial['Mg#_bml']:.3f}")

            current_misfit, _, current_components, _, _ = forward(
                trial, run_dir, model_name, samuel_cache,
                skip_bml_density_check=True)

            if current_misfit is not None and np.isfinite(current_misfit):
                current = trial
                print(f"  Starting point OK "
                      f"(attempt {attempt+1}): misfit={current_misfit:.4f}")
                break
        else:
            print(f"  Starting point failed after {MAX_RETRIES} retries, "
                  f"giving up chain {chain_id}.")
            return

        if current_components is None:
            current_components = {
                'tt': 999.0, 'mass': 999.0, 'moi': 999.0,
                'solidus': 0.0, 'upper_contrast': None, 'lower_contrast': None,
            }

    print(f"  Initial misfit/datum = {current_misfit:.4f}")

    for step in range(step_start, step_start + n_steps):
        t0         = datetime.now()
        proposed   = propose(current, rng)
        run_dir    = os.path.join(chain_dir, f"step_{step+1:05d}")
        model_name = f"mcmc_{prefix}_c{chain_id:02d}_s{step+1:05d}"

        # ★ fix 1+2: receive 4 values including fort56_data
        proposed_misfit, n_data, components, fort56_data, bml_data  = forward(
            proposed, run_dir, model_name, samuel_cache)

        if proposed_misfit is None or proposed_misfit >= 990.0:
            accepted        = False
            proposed_misfit = 999.0
            fort56_data     = None
            bml_data        = None
            components      = {
                'tt': 999.0, 'mass': 999.0,
                'moi': 999.0, 'solidus': 0.0,
                'upper_contrast': components.get('upper_contrast') if components else None,
                'lower_contrast': components.get('lower_contrast') if components else None,
            }
        else:
            delta_misfit = proposed_misfit - current_misfit
            if delta_misfit <= 0:
                accepted = True
            else:
                log_alpha = -delta_misfit / MCMC_TEMPERATURE
                accepted  = np.log(rng.uniform()) < log_alpha

        if accepted:
            current            = proposed
            current_misfit     = proposed_misfit
            current_components = components
            accept_count      += 1

            # ★ fix 3: save npz when accepted (same as nGibbs version)
            if fort56_data is not None:
                npz_dict = dict(
                    # ── mantle (s3_final, no Pierru correction needed) ──
                    depth_km = fort56_data['depth_km'],
                    Vp       = fort56_data['Vp'],
                    Vs       = fort56_data['Vs'],
                    rho      = fort56_data['rho'],
                    T_K      = fort56_data['T_K'],
                    P_GPa    = fort56_data['P_GPa'],
                )
                if bml_data is not None:
                    # ── BML (Pierru-corrected Vp/Vs, phi included) ──
                    npz_dict.update(dict(
                        bml_depth_km             = bml_data['depth_km'],
                        bml_Vp                   = bml_data['Vp'],    # Pierru-corrected
                        bml_Vs                   = bml_data['Vs'],    # Pierru-corrected, 0 where phi>=0.30
                        bml_rho                  = bml_data['rho'],
                        bml_T_K                  = bml_data['T_K'],
                        bml_P_GPa                = bml_data['P_GPa'],
                        bml_phi                  = bml_data['phi'],   # melt fraction per point
                        bml_outer_core_depth_abs = np.array(
                            [bml_data['outer_core_depth_abs']]),      # scalar → 1-element array
                    ))
                np.savez(
                    os.path.join(chain_dir, f"profile_s{step+1:05d}.npz"),
                    **npz_dict,
                )

        elapsed     = (datetime.now() - t0).total_seconds()
        accept_rate = accept_count / (step - step_start + 1) * 100

        uc     = current_components.get('upper_contrast')
        uc_str = f"{uc:+.4f}" if uc is not None else "N/A"
        print(f"  Step {step+1:4d}: misfit={current_misfit:.4f}  "
              f"{'ACCEPT' if accepted else 'reject'}  "
              f"rate={accept_rate:.1f}%  "
              f"upper_contrast={uc_str}  ({elapsed:.0f}s)")

        chain.append({
            'step':           step + 1,
            'params':         current,
            'misfit':         current_misfit,
            'misfit_tt':      current_components.get('tt',      999.0),
            'misfit_mass':    current_components.get('mass',    999.0),
            'misfit_moi':     current_components.get('moi',     999.0),
            'misfit_solidus': current_components.get('solidus', 0.0),
            'upper_contrast': current_components.get('upper_contrast'),
            'lower_contrast': current_components.get('lower_contrast'),
            'accepted':       bool(accepted),
            'accept_rate':    accept_rate,
        })

        # ★ fix 4: save chain.json every step
        with open(chain_file, 'w') as f:
            json.dump(chain, f, indent=2)
        print(f"  [Saved {prefix}_{chain_id:02d}, "
              f"total {len(chain)} steps]")

    print(f"\nChain {chain_id} complete!")
    print(f"  Total steps:       {step_start + n_steps}")
    print(f"  Final accept rate: {accept_count/n_steps*100:.1f}%")
    print(f"  Final misfit:      {current_misfit:.4f}")

# ============================================================
# entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain',        type=int,  default=0)
    parser.add_argument('--steps',        type=int,  default=100)
    parser.add_argument('--test',         action='store_true')
    parser.add_argument('--prefix',       type=str,  default='chain')
    parser.add_argument('--random_start', action='store_true',
                        help='Start from a random point within the prior')
    parser.add_argument('--start',        type=str,  default=None,
                        help='JSON string or path to JSON file with start params')
    args = parser.parse_args()

    os.makedirs(MCMC_DIR, exist_ok=True)

    # ★ fix 5: determine starting parameters
    start_params = None

    if args.start is not None:
        try:
            start_params = json.loads(args.start)
        except json.JSONDecodeError:
            with open(args.start) as _f:
                start_params = json.load(_f)
        missing = [k for k in PRIOR if k not in start_params]
        if missing:
            raise ValueError(f"--start is missing parameters: {missing}")
        print(f"[start] Using specified start params: {start_params}")

    elif args.random_start:
        rng_init = np.random.default_rng(args.chain)
        start_params = {
            k: float(rng_init.uniform(lo, hi))
            for k, (lo, hi) in PRIOR.items()
        }
        print(f"[start] random_start chain={args.chain}: {start_params}")

    else:
        print(f"[start] Using default START_PARAMS: {START_PARAMS}")

    if args.test:
        print("Test mode: running 1 step")
        run_mcmc(chain_id=0, n_steps=1, prefix=args.prefix,
                 start_params=start_params)
    else:
        run_mcmc(chain_id=args.chain, n_steps=args.steps,
                 prefix=args.prefix, start_params=start_params)
