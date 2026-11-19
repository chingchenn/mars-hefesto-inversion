#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 01:24:35 2026

@author: chingchen
"""

"""
Modified MCMC with:
1. solidus/liquidus + Mg# correction → melt fraction φ
2. Scan BML for φ=0.30 → dynamic outer-core depth
3. Pierru 2026 velocity correction based on φ
4. build_taup: outer-core written at φ=0.30 depth
5. BML_thickness as free MCMC parameter
6. PbdiffPcP misfit for S1000a
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
import sys
from pathlib import Path
import torch

from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model


repo_root = Path('/home/jcchen2/nGibbs/')
src_root  = repo_root / "src"
module_root = src_root / "module"
sys.path.insert(0, str(src_root))
sys.path.insert(0, str(module_root))

from module.utils.math_utils import IDX_2D_Lithosphere
from module.engine.API import HeFESToEmulatorCPU
if torch.cuda.is_available():
    from module.engine.API import HeFESToEmulatorGPU

# ============================================================
# parameters
# ============================================================

MARS_RADIUS = 3389.5
T_SURF      = 220.0
P_MAX_GPA   = 19.0
GAMMA       = 1.1
MARS_MASS_OBS   = 6.4171e23
MARS_MASS_SIGMA = MARS_MASS_OBS * 0.01
MOI_OBS         = 0.3634
MOI_SIGMA       = 0.0006
MARS_RADIUS_M   = MARS_RADIUS * 1000

P_BML_TOP        = 19.0
MCMC_TEMPERATURE = 1.0

# Mg# reference for Duncan 2018 solidus
MG_DUNCAN_REF = 0.75

# Melt fraction threshold for outer-core marker (Pierru 2026)
PHI_OUTER_CORE = 0.30

# ============================================================
# ★ STEP 1: solidus / liquidus with Mg# correction
# ============================================================

def solidus_duncan2018(P_GPa):
    """Duncan 2018 solidus, reference Mg# = 0.75, returns K"""
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
    """BML solidus with Fe correction"""
    return solidus_duncan2018(P_GPa) + fe_correction_K(Mg_number_bml)


def liquidus_bml(P_GPa, Mg_number_bml):
    """BML liquidus with Fe correction"""
    return liquidus_ruedas2017(P_GPa) + fe_correction_K(Mg_number_bml)


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


# ============================================================
# ★ STEP 3: Pierru 2026 velocity correction
# ============================================================

def apply_melt_correction_pierru(Vp, Vs, phi):
    """
    Pierru 2026 Fig S9 experimental data (seismic-frequency equivalent):
    phi < 0.10 : Vs -20%, Vp -10%  (linear)
    phi < 0.20 : Vs -40%, Vp -17.5% (linear)
    phi >= 0.30: Vs = 0,  Vp -20%
    """
    if phi <= 0.0:
        return Vp, Vs

    # interpolation nodes: phi, dVp_frac, dVs_frac
    phi_nodes = np.array([0.0,  0.10, 0.20, 0.30, 1.0])
    dVp_nodes = np.array([0.0,  0.10, 0.175, 0.20, 0.20])
    dVs_nodes = np.array([0.0,  0.20, 0.40,  1.00, 1.00])

    dVp = float(np.interp(phi, phi_nodes, dVp_nodes))
    dVs = float(np.interp(phi, phi_nodes, dVs_nodes))

    return Vp * (1.0 - dVp), max(Vs * (1.0 - dVs), 0.0)


# ============================================================
# ★ STEP 2: scan BML for φ=0.30 depth
# ============================================================

def find_outer_core_depth(bml_P, bml_depth, T_bml, Mg_number_bml,
                           phi_threshold=PHI_OUTER_CORE):
    """
    Scan BML points from top to bottom.
    Return depth (km) where φ first exceeds phi_threshold.
    If never reached, return the bottom depth (entire BML is partial melt).
    """
    for P, d in zip(bml_P, bml_depth):
        phi = compute_melt_fraction(T_bml, P, Mg_number_bml)
        if phi >= phi_threshold:
            return d
    return bml_depth[-1]


# ============================================================
# Khan median
# ============================================================

KHAN_MEDIAN_PATH = "/net/beno3/data1/jcchen2/Mars_Khan_2023/LSL_Models/khan_median.npz"

_KHAN_CACHE = None

def compute_khan_median():
    global _KHAN_CACHE
    if _KHAN_CACHE is not None:
        return _KHAN_CACHE

    data = np.load(KHAN_MEDIAN_PATH)
    _KHAN_CACHE = {k: data[k] for k in data.files}
    _KHAN_CACHE['lsl_top_depth']  = float(data['lsl_top_depth'])
    _KHAN_CACHE['true_cmb_depth'] = float(data['true_cmb_depth'])
    _KHAN_CACHE['cmb_depth']      = _KHAN_CACHE['lsl_top_depth']
    print(f"  Khan median loaded from {KHAN_MEDIAN_PATH}")
    print(f"  LSL top:  {_KHAN_CACHE['lsl_top_depth']:.1f} km")
    print(f"  True CMB: {_KHAN_CACHE['true_cmb_depth']:.1f} km")
    return _KHAN_CACHE


# ============================================================
# gravity profiles
# ============================================================

_GRAVITY_DEPTH = None
_GRAVITY_G     = None

def load_gravity_profile():
    global _GRAVITY_DEPTH, _GRAVITY_G
    if _GRAVITY_DEPTH is not None:
        return
    khan = compute_khan_median()
    _GRAVITY_DEPTH = khan['gravity_z']
    _GRAVITY_G     = khan['gravity_g']
    print(f"  Gravity from Khan median: "
          f"surface {_GRAVITY_G[0]:.3f} m/s², "
          f"CMB (~1533 km) {np.interp(1533, _GRAVITY_DEPTH, _GRAVITY_G):.3f} m/s²")

def gravity_mars(depth_km):
    if _GRAVITY_DEPTH is None:
        load_gravity_profile()
    return np.interp(depth_km, _GRAVITY_DEPTH, _GRAVITY_G)


# ============================================================
# composition
# ============================================================

YM_BASE = {
    'Si': 4.01931, 'Mg': 4.08235, 'Fe': 1.08599,
    'Ca': 0.27259, 'Al': 0.37376, 'Na': 0.10105, 'Cr': 0.06146,
}
MGFE_TOTAL = YM_BASE['Mg'] + YM_BASE['Fe']

# ============================================================
# ★ STEP 5: BML_thickness added to PRIOR / STEP / START
# ============================================================

START_PARAMS = {
    'T_lit':        1952.92,
    'P_lit':        6.0755,
    'Mg#':          YM_BASE['Mg'] / (YM_BASE['Mg'] + YM_BASE['Fe']),
    'T_bml':        2500.0,
    'Mg#_bml':      0.7,
    'BML_thickness': 150.0,   # km
}

FIXED_PARAMS = {
    'Si': YM_BASE['Si'], 'Ca': YM_BASE['Ca'], 'Al': YM_BASE['Al'],
    'Na': YM_BASE['Na'], 'Cr': YM_BASE['Cr'],
}

PRIOR = {
    'T_lit':        (1000.0, 2600.0),
    'P_lit':        (1.5,    9.0),
    'Mg#':          (0.50,   0.86),
    'T_bml':        (1800.0, 3500.0),
    'Mg#_bml':      (0.50,   0.80),
    'BML_thickness': (40.0,  400.0),  # km  ← new
}

STEP = {
    'T_lit':        60.0,
    'P_lit':        0.3,
    'Mg#':          0.015,
    'T_bml':        60.0,
    'Mg#_bml':      0.015,
    'BML_thickness': 20.0,   # km  ← new
}

SIGMA = {
    'S-P': 10.0, 'pP-P': 3.0, 'sP-P': 5.0, 'PP-P': 8.0, 'PPP-P': 12.0,
    'sS-S': 5.0, 'SS-S': 5.0, 'SSS-S': 8.0, 'ScS-S': 12.0,
    'SS-PP': 10.0, 'SKS-PP': 10.0,
    'PP-PbdiffPcP': 10.0,   # ← new (Samuel 2026 ±10 s)
}


def s(a, sa, b, sb):
    return np.sqrt(sa**2 + sb**2)

# ============================================================
# ★ STEP 6: PbdiffPcP added to KHAN_DATA for S1000a
# ============================================================

KHAN_DATA = {
    'S0167b': {
        'delta': 72.5, 'depth': 31.2,
        'PP-P':  (37.0,  3.0),
        'S-P':   (414.5, 2.0),
        'SSS-S': (468.0 - 414.5, s(3,0, 2,0)),
    },
    'S0173a': {
        'delta': 30.6, 'depth': 28.7,
        'pP-P':  (11.1,  3.0),
        'S-P':   (174.8, 2.0),
        'sS-S':  (184.8 - 174.8, s(3,0, 2,0)),
        'SS-S':  (197.9 - 174.8, s(2,0, 2,0)),
        'ScS-S': (515.0 - 174.8, s(5,0, 2,0)),
    },
    'S0183a': {
        'delta': 47.9, 'depth': 31.2,
        'PP-P':  (24.5, 4.0),
        'PPP-P': (43.0, 7.0),
    },
    'S0185a': {
        'delta': 63.1, 'depth': 34.2,
        'S-P':   (360.2, 2.0),
        'sS-S':  (379.5 - 360.2, s(4,0, 2,0)),
        'SSS-S': (412.5 - 360.2, s(6,0, 2,0)),
    },
    'S0235b': {
        'delta': 29.6, 'depth': 27.4,
        'PP-P':  (17.4,  2.0),
        'PPP-P': (31.1,  7.0),
        'S-P':   (166.0, 3.0),
        'sS-S':  (178.7 - 166.0, s(3,0, 3,0)),
        'SS-S':  (193.1 - 166.0, s(3,0, 3,0)),
        'ScS-S': (512.0 - 166.0, s(8,0, 3,0)),
    },
    'S0325a': {
        'delta': 42.4, 'depth': 33.1,
        'pP-P':  (11.3,  3.0),
        'S-P':   (230.7, 3.0),
        'SS-S':  (260.7 - 230.7, s(4,0, 3,0)),
        'SSS-S': (281.0 - 230.7, s(6,0, 3,0)),
    },
    'S0407a': {
        'delta': 30.3, 'depth': 32.0,
        'PP-P':  (17.8,  5.0),
        'PPP-P': (33.0,  7.0),
        'S-P':   (172.2, 2.0),
        'sS-S':  (183.4 - 172.2, s(3,0, 2,0)),
        'SS-S':  (196.4 - 172.2, s(5,0, 2,0)),
    },
    'S0409d': {
        'delta': 29.8, 'depth': 31.2,
        'S-P':   (162.5, 3.0),
        'SS-S':  (184.9 - 162.5, s(4,0, 3,0)),
        'SSS-S': (207.7 - 162.5, s(6,0, 3,0)),
    },
    'S0484b': {
        'delta': 30.7, 'depth': 33.5,
        'PP-P':  (18.1,  5.0),
        'S-P':   (170.2, 1.0),
        'sS-S':  (184.0 - 170.2, s(3,0, 1,0)),
        'SS-S':  (196.8 - 170.2, s(3,0, 1,0)),
    },
    'S0784a': {
        'delta': 31.1, 'depth': 30.6,
        'PP-P':  (15.2,  4.0),
        'PPP-P': (29.5,  7.0),
        'S-P':   (173.0, 4.0),
        'sS-S':  (182.2 - 173.0, s(4,0, 4,0)),
        'SS-S':  (196.9 - 173.0, s(5,0, 4,0)),
        'SSS-S': (221.9 - 173.0, s(6,0, 4,0)),
    },
    'S0802a': {
        'delta': 31.9, 'depth': 31.2,
        'PP-P':  (17.3,  5.0),
        'S-P':   (176.3, 5.0),
        'SS-S':  (201.5 - 176.3, s(5,0, 5,0)),
        'SSS-S': (222.1 - 176.3, s(6,0, 5,0)),
    },
    'S0809a': {
        'delta': 31.3, 'depth': 31.2,
        'PP-P':  (15.5,  5.0),
        'PPP-P': (29.3,  7.0),
        'S-P':   (175.0, 1.0),
        'SS-S':  (197.5 - 175.0, s(3,0, 1,0)),
    },
    'S0820a': {
        'delta': 31.6, 'depth': 30.5,
        'PP-P':  (15.7,  5.0),
        'S-P':   (176.5, 2.0),
        'sS-S':  (186.2 - 176.5, s(3,0, 2,0)),
        'SS-S':  (201.4 - 176.5, s(5,0, 2,0)),
    },
    'S0864a': {
        'delta': 30.5, 'depth': 31.3,
        'PP-P':  (18.0,  5.0),
        'S-P':   (169.0, 3.0),
        'sS-S':  (181.2 - 169.0, s(4,0, 3,0)),
        'SS-S':  (194.0 - 169.0, s(3,0, 3,0)),
        'SSS-S': (216.1 - 169.0, s(6,0, 3,0)),
        'ScS-S': (505.0 - 169.0, s(8,0, 3,0)),
    },
    # ★ S1000a: add PP-PbdiffPcP constraint (Samuel 2023/2026)
    'S1000a': {
        'delta': 125.9, 'depth': 0.0,
        'SS-PP':         (749.0,   10.0),
        'SKS-PP':        (339.3,   10.0),
        'PP-PbdiffPcP':  (180.65,  10.0),  # ← new
    },
}

SAMUEL_DATA = {
    'S0154a': {'delta': 29.7,  'depth': 17.4, 'SS-S': 25.3, 'SSS-S': 35.0},
    'S0173a': {'delta': 30.9,  'depth': 28.4,
               'S-P': 178.8, 'sP-P': 9.43, 'PP-P': 19.9, 'PPP-P': 34.4,
               'sS-S': 13.2, 'SS-S': 24.4, 'SSS-S': 40.5, 'ScS-S': 345.2},
    'S0185a': {'delta': 54.8,  'depth': 17.4,
               'S-P': 327.28, 'pP-P': 4.0, 'PP-P': 22.47, 'PPP-P': 49.3,
               'sS-S': 10.0, 'SS-S': 30.9, 'SSS-S': 55.4, 'ScS-S': 152.3},
    'S0235b': {'delta': 30.5,  'depth': 26.1,
               'S-P': 171.4, 'PP-P': 18.6, 'PPP-P': 32.0,
               'sS-S': 9.2, 'SS-S': 23.2, 'SSS-S': 33.3, 'ScS-S': 343.9},
    'S0325a': {'delta': 42.0,  'depth': 33.8,
               'S-P': 229.3, 'pP-P': 9.8, 'PP-P': 21.1, 'PPP-P': 34.4,
               'sS-S': 13.8, 'SS-S': 26.1, 'SSS-S': 50.3, 'ScS-S': 220.4},
    'S0407a': {'delta': 29.1,  'depth': 31.3,
               'S-P': 170.7, 'pP-P': 6.77, 'PP-P': 23.38,
               'sS-S': 13.3, 'SS-S': 21.1, 'SSS-S': 33.1, 'ScS-S': 370.0},
    'S0409d': {'delta': 30.6,  'depth': 26.1,
               'S-P': 163.2, 'pP-P': 8.3, 'PP-P': 27.6, 'PPP-P': 36.94,
               'sS-S': 8.4, 'SS-S': 20.9, 'SSS-S': 39.8, 'ScS-S': 320.1},
    'S0474a': {'delta': 20.7,  'depth': 30.7,
               'S-P': 121.6, 'PP-P': 13.4, 'PPP-P': 24.8,
               'SS-S': 15.8, 'SSS-S': 32.4},
    'S0484b': {'delta': 31.3,  'depth': 24.9,
               'S-P': 173.1, 'pP-P': 5.5, 'PP-P': 19.73,
               'sS-S': 13.0, 'SS-S': 17.4, 'ScS-S': 322.3},
    'S0784a': {'delta': 30.2,  'depth': 16.8,
               'S-P': 179.3, 'pP-P': 6.5, 'PP-P': 13.7, 'PPP-P': 22.4,
               'sS-S': 7.2, 'SS-S': 19.6, 'SSS-S': 28.0},
    'S0802a': {'delta': 30.0,  'depth': 20.4,
               'S-P': 180.3, 'pP-P': 4.0, 'PP-P': 25.6, 'PPP-P': 33.9,
               'sS-S': 9.3, 'SS-S': 22.4, 'SSS-S': 36.5, 'ScS-S': 387.6},
    'S0809a': {'delta': 30.7,  'depth': 16.0,
               'S-P': 191.95, 'pP-P': 4.5, 'PP-P': 16.25, 'PPP-P': 29.65,
               'sS-S': 8.1, 'SS-S': 23.8, 'SSS-S': 39.3, 'ScS-S': 373.5},
    'S0820a': {'delta': 28.1,  'depth': 18.7,
               'S-P': 174.1, 'PP-P': 21.9, 'PPP-P': 32.1, 'sS-S': 8.5},
    'S0861a': {'delta': 54.5,  'depth': 15.5,
               'S-P': 319.3, 'PP-P': 19.6, 'PPP-P': 47.6, 'SS-S': 41.1},
    'S0864a': {'delta': 29.0,  'depth': 25.0,
               'S-P': 171.4, 'PP-P': 18.0, 'PPP-P': 27.9,
               'sS-S': 17.3, 'SS-S': 26.4},
    'S0916d': {'delta': 30.2,  'depth': 16.3,
               'S-P': 170.8, 'pP-P': 3.9, 'PP-P': 19.3, 'PPP-P': 36.1,
               'SS-S': 19.0, 'SSS-S': 42.9, 'ScS-S': 342.8},
    'S0918a': {'delta': 16.6,  'depth': 22.3,
               'S-P': 102.4, 'PP-P': 12.8, 'PPP-P': 22.5,
               'SS-S': 21.2, 'SSS-S': 35.0},
    'S0976a': {'delta': 144.0, 'depth': 30.0,
               'SS-PP': 854.4, 'SKS-PP': 303.9},
    'S1000a': {'delta': 125.9, 'depth': 0.0,
               'SS-PP': 749.0, 'SKS-PP': 339.3},
    'S1094b': {'delta': 58.5,  'depth': 0.0, 'S-P': 343.0},
    'S1222a': {'delta': 36.1,  'depth': 32.8,
               'S-P': 216.0, 'ScS-S': 258.0},
}


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
# HeFESTo control phases (unchanged)
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


def run_ngibbs(params):
    p     = composition_from_params(params)
    T_lit = params['T_lit']
    P_lit = params['P_lit']

    pressures_shallow = np.linspace(0.01, 3.0, 100)
    pressures_deep    = np.linspace(3.0, P_MAX_GPA, 101)[1:]
    pressures = np.concatenate([pressures_shallow, pressures_deep])
    n_pressures = len(pressures)
    P_lit_idx   = np.argmin(np.abs(pressures - P_lit))

    isentropic_IDX, isothermal_IDX = IDX_2D_Lithosphere(
        np.array([P_lit_idx]), n_pressures
    )
    T_headers = ['P(GPa)(System_main)', 'T(K)(System_main)',
                 'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
    S_headers = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)',
                 'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']
    O = compute_oxygen(p)
    comp_values = [p['Si'], p['Mg'], p['Fe'], p['Ca'],
                   p['Al'], p['Na'], p['Cr'], O]

    single_input = np.array([[P_lit, T_lit] + comp_values], dtype=np.float32)
    with torch.no_grad():
        out1 = HeFESToEmulatorCPU.ForwardMB(
            single_input, headers=T_headers, outputs=['component_moles']
        )

    PT_single = np.array([[P_lit, T_lit]], dtype=np.float64)
    burnman_s = HeFESToEmulatorCPU.get_property_burnman_vectorized_from_assemblage(
        torch.tensor(out1['component_moles'], dtype=torch.float64),
        torch.tensor(PT_single, dtype=torch.float64),
        property_names=['entropy_by_mass'],
    )
    S_lit = float(burnman_s['entropy_by_mass'][0]) / 1000.0
    print(f"    S_lit = {S_lit:.6f} J/g/K  @ P={P_lit:.2f} GPa, T={T_lit:.1f} K")

    properties = {k: np.zeros(n_pressures, dtype=np.float32)
                  for k in ['temperature', 'density', 'p_wave_velocity',
                             's_wave_velocity', 'isentropic_bulk_modulus_reuss']}

    n_isen = isentropic_IDX[1].shape[0]
    isen_input = np.zeros((n_isen, 2 + len(comp_values)), dtype=np.float32)
    isen_input[:, 0] = pressures[isentropic_IDX[1]]
    isen_input[:, 1] = S_lit
    isen_input[:, 2:] = comp_values

    with torch.no_grad():
        out2 = HeFESToEmulatorCPU.ForwardMB(
            isen_input, headers=S_headers,
            outputs=['component_moles', 'temperature']
        )

    T_isen = np.asarray(out2['temperature'].detach().cpu(), dtype=np.float64)
    properties['temperature'][isentropic_IDX[1]] = T_isen

    PT_isen = np.stack([isen_input[:, 0], T_isen], axis=1)
    burnman_isen = HeFESToEmulatorCPU.get_property_burnman_vectorized_from_assemblage(
        torch.tensor(out2['component_moles'], dtype=torch.float64),
        torch.tensor(PT_isen, dtype=torch.float64),
        property_names=['density', 'p_wave_velocity', 's_wave_velocity',
                        'isentropic_bulk_modulus_reuss'],
    )
    for k, v in burnman_isen.items():
        properties[k][isentropic_IDX[1]] = v

    T_lit_actual = float(properties['temperature'][P_lit_idx])
    T_cond = np.linspace(T_SURF, T_lit_actual, P_lit_idx + 1)[:-1]
    properties['temperature'][isothermal_IDX[1]] = T_cond

    n_iso = isothermal_IDX[1].shape[0]
    iso_input = np.zeros((n_iso, 2 + len(comp_values)), dtype=np.float32)
    iso_input[:, 0] = pressures[isothermal_IDX[1]]
    iso_input[:, 1] = T_cond
    iso_input[:, 2:] = comp_values

    with torch.no_grad():
        out3 = HeFESToEmulatorCPU.ForwardMB(
            iso_input, headers=T_headers, outputs=['component_moles']
        )

    PT_iso = np.stack([iso_input[:, 0], T_cond], axis=1)
    burnman_iso = HeFESToEmulatorCPU.get_property_burnman_vectorized_from_assemblage(
        torch.tensor(out3['component_moles'], dtype=torch.float64),
        torch.tensor(PT_iso, dtype=torch.float64),
        property_names=['density', 'p_wave_velocity', 's_wave_velocity',
                        'isentropic_bulk_modulus_reuss'],
    )
    for k, v in burnman_iso.items():
        properties[k][isothermal_IDX[1]] = v

    rho = properties['density']
    dP  = np.diff(pressures) * 1e9
    depth = np.zeros(n_pressures)
    for i in range(len(dP)):
        g_i        = gravity_mars(depth[i])
        rho_si     = ((rho[i] + rho[i+1]) / 2)
        depth[i+1] = depth[i] + dP[i] / (rho_si * g_i) / 1000.0

    return {
        'depth_km':  depth,
        'P_GPa':     pressures,
        'T_K':       properties['temperature'],
        'Vp':        properties['p_wave_velocity'] / 1000.0,
        'Vs':        properties['s_wave_velocity'] / 1000.0,
        'rho':       rho / 1000.0,
        'P_profile': pressures,
        'T_profile': properties['temperature'],
    }


# ============================================================
# ★ run_ngibbs_bml: now uses BML_thickness, applies Pierru correction
# ============================================================

def run_ngibbs_bml(params, T_bml, n_points=30):
    """
    BML from P_BML_TOP to P_BML_TOP + thickness.
    Returns solid velocities from HeFESTo, then applies Pierru melt correction.
    Also returns phi array and outer_core_depth for build_taup.
    """
    bml_thickness = params.get('BML_thickness', 150.0)   # km
    Mg_number_bml = params['Mg#_bml']

    p = composition_from_params({
        'Mg#':   Mg_number_bml,
        'T_lit': T_bml,
        'P_lit': P_BML_TOP,
    })
    O = compute_oxygen(p)
    comp_values = [p['Si'], p['Mg'], p['Fe'], p['Ca'],
                   p['Al'], p['Na'], p['Cr'], O]

    T_headers = ['P(GPa)(System_main)', 'T(K)(System_main)',
                 'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']

    # pressure range: P_BML_TOP to P_BML_BOTTOM (derived from thickness)
    # use a rough conversion: ~0.5 GPa per 50 km at CMB depth
    # more accurate: integrate later; for now use depth→pressure scaling
    # We store depth offset and compute absolute depth in forward()
    # Here we use pressure grid derived from thickness via gravity
    # Simple estimate: dP/dz = rho * g ≈ 4200 kg/m³ * 3.7 m/s² = 15540 Pa/m
    rho_bml_est = 4200.0   # kg/m³
    g_bml_est   = 3.7      # m/s²
    dP_per_km   = rho_bml_est * g_bml_est * 1000 / 1e9   # GPa/km
    P_BML_BOTTOM = P_BML_TOP + bml_thickness * dP_per_km

    pressures = np.linspace(P_BML_TOP, P_BML_BOTTOM, n_points)
    bml_input = np.zeros((n_points, 2 + len(comp_values)), dtype=np.float32)
    bml_input[:, 0] = pressures
    bml_input[:, 1] = T_bml
    bml_input[:, 2:] = comp_values

    with torch.no_grad():
        out = HeFESToEmulatorCPU.ForwardMB(
            bml_input, headers=T_headers, outputs=['component_moles']
        )

    burnman_out = HeFESToEmulatorCPU.get_property_burnman_vectorized_from_assemblage(
        torch.tensor(out['component_moles'], dtype=torch.float64),
        torch.tensor(bml_input[:, :2],      dtype=torch.float64),
        property_names=['density', 'p_wave_velocity', 's_wave_velocity'],
    )

    rho_solid = burnman_out['density']
    Vp_solid  = burnman_out['p_wave_velocity'] / 1000.0
    Vs_solid  = burnman_out['s_wave_velocity'] / 1000.0

    # ★ compute depth
    dP    = np.diff(pressures) * 1e9
    depth = np.zeros(n_points)
    for i in range(len(dP)):
        g_i        = gravity_mars(depth[i])
        rho_si     = ((rho_solid[i] + rho_solid[i+1]) / 2)
        depth[i+1] = depth[i] + dP[i] / (rho_si * g_i) / 1000.0

    # ★ STEP 2+3: compute φ, apply Pierru correction, find outer-core depth
    phi_arr    = np.zeros(n_points)
    Vp_corr    = np.zeros(n_points)
    Vs_corr    = np.zeros(n_points)
    outer_core_depth = depth[-1]   # default: bottom of BML

    for i in range(n_points):
        phi = compute_melt_fraction(T_bml, pressures[i], Mg_number_bml)
        phi_arr[i] = phi
        Vp_corr[i], Vs_corr[i] = apply_melt_correction_pierru(
            float(Vp_solid[i]), float(Vs_solid[i]), phi
        )
        # first point where φ >= 0.30 → outer-core marker
        if phi >= PHI_OUTER_CORE and outer_core_depth == depth[-1]:
            outer_core_depth = depth[i]

    print(f"    BML: thickness={bml_thickness:.0f} km  "
          f"P_bottom={P_BML_BOTTOM:.2f} GPa  "
          f"phi_top={phi_arr[0]:.3f}  phi_bot={phi_arr[-1]:.3f}  "
          f"outer_core_depth_offset={outer_core_depth:.1f} km")

    return {
        'depth_km':         depth,
        'P_GPa':            pressures,
        'Vp':               Vp_corr,
        'Vs':               Vs_corr,
        'rho':              rho_solid / 1000.0,
        'phi':              phi_arr,
        'outer_core_depth_offset': outer_core_depth,  # relative to BML top
    }


# ============================================================
# ★ STEP 4: build_taup — dynamic outer-core depth
# ============================================================

def build_taup(fort56_data, model_name, khan_cache, bml_data=None):
    os.makedirs(TAUP_WORK_DIR, exist_ok=True)
    model_name = model_name.replace(".npz", "")
    npz_path   = os.path.join(TAUP_WORK_DIR, f'{model_name}.npz')
    nd_path    = os.path.join(TAUP_WORK_DIR, f"{model_name}.nd")
    if os.path.exists(npz_path):
        return TauPyModel(model=npz_path)

    khan           = khan_cache
    lsl_top_depth  = khan['lsl_top_depth']
    true_cmb_depth = khan['true_cmb_depth']
    mantle_bottom  = lsl_top_depth

    hef_depth = fort56_data['depth_km']
    hef_Vp    = fort56_data['Vp']
    hef_Vs    = fort56_data['Vs']
    hef_rho   = fort56_data['rho']

    mantle_mask = (hef_depth >= 200.0) & (hef_depth <= mantle_bottom)
    man_depth   = hef_depth[mantle_mask]
    man_Vp      = hef_Vp[mantle_mask]
    man_Vs      = hef_Vs[mantle_mask]
    man_rho     = hef_rho[mantle_mask]
    if len(man_depth) == 0:
        raise ValueError("Mantle depth range insufficient")

    with open(nd_path, 'w') as f:
        # crust
        for d, vp, vs, r in zip(khan['crust_z'], khan['crust_vp'],
                                 khan['crust_vs'], khan['crust_rho']):
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
            bml_phi   = bml_data['phi']

            # ★ find outer_core_depth in absolute coordinates
            outer_core_abs = bml_data.get('outer_core_depth_abs', bml_depth[-1])

            bml_mask = (bml_depth >= mantle_bottom) & (bml_depth <= true_cmb_depth)

            # write partial melt region (Vs > 0, above outer_core_abs)
            partial_mask = bml_mask & (bml_depth < outer_core_abs)
            for d, vp, vs, r in zip(bml_depth[partial_mask],
                                     bml_vp[partial_mask],
                                     bml_vs[partial_mask],
                                     bml_rho[partial_mask]):
                f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

            # ★ discontinuity at outer_core_abs
            # find the index just at or above outer_core_abs
            idx_oc = np.searchsorted(bml_depth, outer_core_abs)
            idx_oc = min(idx_oc, len(bml_depth) - 1)

            # last solid line
            if idx_oc > 0:
                d_above = bml_depth[idx_oc - 1]
                f.write(f"{d_above:.3f}  "
                        f"{bml_vp[idx_oc-1]:.4f}  "
                        f"{bml_vs[idx_oc-1]:.4f}  "
                        f"{bml_rho[idx_oc-1]:.4f}\n")

            # outer-core marker + first liquid line
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
            core_z   = khan['core_z']
            core_vp  = khan['core_vp']
            core_rho = khan['core_rho']
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
            core_z   = khan['core_z']
            core_vp  = khan['core_vp']
            core_rho = khan['core_rho']
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
# Moment of Inertia (unchanged)
# ============================================================

def compute_mass_and_moi(fort56_data, khan_cache, bml_data=None):
    R                = MARS_RADIUS_M
    lsl_top_km       = khan_cache['lsl_top_depth']
    true_cmb_km      = khan_cache['true_cmb_depth']
    mantle_bottom_km = lsl_top_km

    crust_z      = khan_cache['crust_z']
    crust_rho    = khan_cache['crust_rho']
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

    core_z      = khan_cache['core_z']
    core_rho    = khan_cache['core_rho']
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
# Solidus penalty (updated to use Mg#-corrected solidus)
# ============================================================

def compute_solidus_penalty(fort56_data, params):
    penalty = 0.0

    P_prof = fort56_data.get('P_profile', None)
    T_prof = fort56_data.get('T_profile', None)

    if P_prof is not None and T_prof is not None:
        P_lit       = params['P_lit']
        mantle_mask = (P_prof >= P_lit) & (P_prof <= P_BML_TOP)
        P_m = P_prof[mantle_mask]
        T_m = T_prof[mantle_mask]

        if len(P_m) > 0:
            # use normal mantle Mg# for mantle solidus
            T_sol_arr     = solidus_duncan2018(P_m)
            excess        = T_m - T_sol_arr
            mantle_excess = float(np.sum(excess[excess > 0])) / 100.0
            if mantle_excess > 0:
                print(f"    Mantle solidus penalty = {mantle_excess:.4f}")
                penalty += mantle_excess

    if params is not None:
        T_bml         = params['T_bml']
        Mg_number_bml = params['Mg#_bml']
        bml_thickness = params.get('BML_thickness', 150.0)
        dP_per_km     = 4200.0 * 3.7 * 1000 / 1e9
        P_bml_bottom  = P_BML_TOP + bml_thickness * dP_per_km
        # BML must be above solidus (molten)
        T_sol_bml = solidus_bml(P_bml_bottom, Mg_number_bml)
        if T_bml < T_sol_bml:
            bml_pen = (T_sol_bml - T_bml) / 100.0
            print(f"    BML solidus penalty = {bml_pen:.4f} "
                  f"(T_bml={T_bml:.0f}K < T_sol={T_sol_bml:.0f}K)")
            penalty += bml_pen

    return penalty


# ============================================================
# ★ STEP 6: Misfit — add PP-PbdiffPcP
# ============================================================

def _diff(times, a, b):
    return times[a] - times[b] if a in times and b in times else None


def compute_misfit(taup_model, obs_dataset, fort56_data,
                   khan_cache, bml_data=None, params=None):

    tt_total = 0.0
    tt_n     = 0

    # ★ add Pdiff and PbdiffPcP to phase list for S1000a
    phases_standard   = ['P', 'S', 'pP', 'sP', 'PP', 'PPP',
                         'SS', 'SSS', 'sS', 'ScS', 'SKS']
    phases_with_pdiff = phases_standard + ['Pdiff']

    for event, obs in obs_dataset.items():
        delta = obs['delta']
        depth = obs.get('depth', 10.0)

        # use extended phase list only for S1000a (large distance, needs Pdiff)
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
            # ★ PP - PbdiffPcP: in TauPy, PbdiffPcP arrives BEFORE PP
            # observed: PP - PbdiffPcP = 180.65 s → PbdiffPcP arrives 180.65 s before PP
            # TauPy computes Pdiff which corresponds to PbdiffPcP in BML models
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
                      f"sigma={sigma:.2f}")
                continue
            tt_total += val
            tt_n     += 1

    tt_misfit = tt_total / tt_n if tt_n > 0 else 999.0
    if not np.isfinite(tt_misfit):
        tt_misfit = 999.0

    M_pred, moi_pred = compute_mass_and_moi(fort56_data, khan_cache,
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

def forward(params, run_dir, model_name, khan_cache,
            skip_bml_density_check=False):

    fort56_data = run_ngibbs(params)
    if fort56_data is None:
        return None, None, None, None

    # BML: use BML_thickness from params
    bml_raw = run_ngibbs_bml(params, T_bml=params['T_bml'])

    bml_data = None
    if bml_raw is not None:
        bml_top_km = khan_cache['lsl_top_depth']

        # convert relative depth to absolute
        bml_raw['depth_km'] = (bml_raw['depth_km']
                               - bml_raw['depth_km'][0] + bml_top_km)

        # ★ compute absolute outer_core_depth
        outer_core_offset = bml_raw['outer_core_depth_offset']
        # outer_core_offset is relative to BML top (depth[0] before shift)
        # after shift, outer_core_abs = bml_top_km + outer_core_offset
        bml_raw['outer_core_depth_abs'] = bml_top_km + outer_core_offset

        rho_mantle_bottom = float(fort56_data['rho'][-1])
        rho_bml_top       = float(bml_raw['rho'][0])
        rho_bml_bottom    = float(bml_raw['rho'][-1])

        core_z    = khan_cache['core_z']
        core_rho  = khan_cache['core_rho']
        true_cmb  = khan_cache['true_cmb_depth']
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
        print(f"    outer_core_depth = {bml_raw['outer_core_depth_abs']:.1f} km  "
              f"(phi=0.30 threshold)")

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
                                khan_cache, bml_data=bml_data)
    except Exception as e:
        print(f"    TauP failed: {e}")
        return None, None, None, None

    misfit, n_data, components = compute_misfit(
        taup_model, KHAN_DATA, fort56_data, khan_cache,
        bml_data=bml_data, params=params)

    if components is not None and bml_data is not None:
        components['upper_contrast'] = bml_data.get('upper_contrast')
        components['lower_contrast'] = bml_data.get('lower_contrast')

    return misfit, n_data, components, fort56_data


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
    khan_cache = compute_khan_median()

    rng     = np.random.default_rng(42 + chain_id)
    current = start_params.copy() if start_params else START_PARAMS.copy()

    # ensure BML_thickness exists in current (backward compatibility)
    if 'BML_thickness' not in current:
        current['BML_thickness'] = 150.0

    chain_file = os.path.join(chain_dir, "chain.json")
    chain      = []
    if os.path.exists(chain_file):
        with open(chain_file) as f:
            chain = json.load(f)
        if chain:
            current = chain[-1]['params']
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
                      f"Mg#_bml={trial['Mg#_bml']:.3f}  "
                      f"BML_thickness={trial['BML_thickness']:.0f}")

            current_misfit, _, current_components, _ = forward(
                trial, run_dir, model_name, khan_cache,
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

        proposed_misfit, n_data, components, fort56_data = forward(
            proposed, run_dir, model_name, khan_cache)

        if proposed_misfit is None or proposed_misfit >= 990.0:
            accepted        = False
            proposed_misfit = 999.0
            fort56_data     = None
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

            if fort56_data is not None:
                np.savez(
                    os.path.join(chain_dir, f"profile_s{step+1:05d}.npz"),
                    depth_km = fort56_data['depth_km'],
                    Vp       = fort56_data['Vp'],
                    Vs       = fort56_data['Vs'],
                    rho      = fort56_data['rho'],
                    T_K      = fort56_data['T_K'],
                    P_GPa    = fort56_data['P_GPa'],
                )

        if os.path.isdir(run_dir):
            for sub in list(os.listdir(run_dir)):
                sub_path = os.path.join(run_dir, sub)
                if sub in {'s3_final', 'bml'}:
                    if os.path.isdir(sub_path):
                        for fname in list(os.listdir(sub_path)):
                            if fname not in {'fort.56', 'ad.in', 'control'}:
                                fp = os.path.join(sub_path, fname)
                                try:
                                    if os.path.isfile(fp):
                                        os.remove(fp)
                                    elif os.path.isdir(fp):
                                        shutil.rmtree(fp, ignore_errors=True)
                                except Exception:
                                    pass
                else:
                    try:
                        if os.path.isfile(sub_path):
                            os.remove(sub_path)
                        elif os.path.isdir(sub_path):
                            shutil.rmtree(sub_path, ignore_errors=True)
                    except Exception:
                        pass

        elapsed     = (datetime.now() - t0).total_seconds()
        accept_rate = accept_count / (step - step_start + 1) * 100

        uc     = current_components.get('upper_contrast')
        uc_str = f"{uc:+.4f}" if uc is not None else "N/A"
        print(f"  Step {step+1:4d}: misfit={current_misfit:.4f}  "
              f"{'ACCEPT' if accepted else 'reject'}  "
              f"rate={accept_rate:.1f}%  "
              f"upper_contrast={uc_str}  ({elapsed:.0f}s)")

        chain.append({
            'step':             step + 1,
            'params':           current,
            'misfit':           current_misfit,
            'misfit_tt':        current_components.get('tt',      999.0),
            'misfit_mass':      current_components.get('mass',    999.0),
            'misfit_moi':       current_components.get('moi',     999.0),
            'misfit_solidus':   current_components.get('solidus', 0.0),
            'upper_contrast':   current_components.get('upper_contrast'),
            'lower_contrast':   current_components.get('lower_contrast'),
            'accepted':         bool(accepted),
            'accept_rate':      accept_rate,
        })

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
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--start',        type=str,  default=None)
    args = parser.parse_args()

    os.makedirs(MCMC_DIR, exist_ok=True)

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