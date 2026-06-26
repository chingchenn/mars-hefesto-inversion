#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
43_mcmc_hefesto_bml.py
MCMC inversion for Mars interior structure using HeFESTo + BML physics
"""

import os
import re
import io
import shutil
import subprocess
import numpy as np
import json
import argparse
import glob
import pandas as pd
from datetime import datetime
from config import *
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

# ── constants ─────────────────────────────────────────────────────────────────
MARS_RADIUS      = 3389.5
MARS_RADIUS_M    = MARS_RADIUS * 1000
T_SURF           = 220.0
GAMMA            = 1.1
MARS_MASS_OBS    = 6.4171e23
MARS_MASS_SIGMA  = MARS_MASS_OBS * 0.01
MOI_OBS          = 0.3634
MOI_SIGMA        = 0.0006
MCMC_TEMPERATURE = 1.0

TRUE_CMB_DEPTH   = 1743.3    # km  true CMB
CMB_VS_THRESHOLD = 0.1       # km/s

# ── MCMC parameters ──────────────────────────────────────────────────────────
YM_BASE    = {'Si': 4.01931, 'Mg': 4.08235, 'Fe': 1.08599,
              'Ca': 0.27259, 'Al': 0.37376, 'Na': 0.10105, 'Cr': 0.06146}
MGFE_TOTAL = YM_BASE['Mg'] + YM_BASE['Fe']

FIXED_PARAMS = {k: YM_BASE[k] for k in ('Si', 'Ca', 'Al', 'Na', 'Cr')}

START_PARAMS = {
    'T_lit':         1539.0,
    'P_lit':         3.69,
    'Mg#':           YM_BASE['Mg'] / MGFE_TOTAL,
    'T_bml':         2200.0,
    'Mg#_bml':       0.7,
    'BML_thickness': 168.7,
}

PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (1.5,    9.0),
    'Mg#':           (0.50,   0.86),
    'T_bml':         (1800.0, 3500.0),
    'Mg#_bml':       (0.50,   0.80),
    'BML_thickness': (50.0,   400.0),
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
    'SS-PP': 10.0, 'SKS-PP': 10.0, 'PP-PbdiffPcP': 10.0,
}

# ── Samuel 2023 paths ─────────────────────────────────────────────────────────
SAMUEL_DATA_DIR = ("/net/beno3/data1/jcchen2/Mars_Samuel_2023/"
                   "Nature_Samuel_s41586-023-06601-8/"
                   "METADATA_BML/DATA_FIG2/PANEL_B")

SAMUEL_RHO_PROFILE_PATH = ("/net/beno3/data1/jcchen2/Mars_Samuel_2023/"
                            "Nature_Samuel_s41586-023-06601-8/"
                            "METADATA_BML/DATA_FIG1/PANEL_J/rho_profile.dat")

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

# ── gravity and pressure profiles ─────────────────────────────────────────────
_GRAVITY_DEPTH  = None
_GRAVITY_G      = None
_PRESSURE_DEPTH = None
_PRESSURE_GPA   = None

def load_gravity_profile():
    global _GRAVITY_DEPTH, _GRAVITY_G, _PRESSURE_DEPTH, _PRESSURE_GPA
    if _GRAVITY_DEPTH is not None:
        return
    try:
        data  = np.loadtxt(SAMUEL_RHO_PROFILE_PATH)
        rho   = data[:, 0]
        r_km  = data[:, 1]
    except Exception as e:
        print(f"  no rho_profile.dat ({e}), using g=3.72 m/s²")
        _GRAVITY_DEPTH  = np.array([0.0, MARS_RADIUS])
        _GRAVITY_G      = np.array([3.72, 3.72])
        _PRESSURE_DEPTH = np.array([0.0, MARS_RADIUS])
        _PRESSURE_GPA   = np.array([0.0, 25.0])
        return

    idx   = np.argsort(r_km)
    rho   = rho[idx]
    r     = r_km[idx] * 1000   # m

    M_enc    = np.zeros(len(r))
    G_CONST  = 6.674e-11
    for i in range(1, len(r)):
        dr       = r[i] - r[i-1]
        M_enc[i] = M_enc[i-1] + 4*np.pi * ((rho[i]+rho[i-1])/2) * ((r[i]+r[i-1])/2)**2 * dr

    g         = np.zeros(len(r))
    g[1:]     = G_CONST * M_enc[1:] / r[1:]**2
    depth_km  = (r[-1] - r) / 1000.0
    sort_d    = np.argsort(depth_km)

    _GRAVITY_DEPTH = depth_km[sort_d]
    _GRAVITY_G     = g[sort_d]

    P_GPa     = np.zeros(len(_GRAVITY_DEPTH))
    rho_d     = rho[sort_d[::-1]][::-1]
    for i in range(1, len(_GRAVITY_DEPTH)):
        dz       = (_GRAVITY_DEPTH[i] - _GRAVITY_DEPTH[i-1]) * 1000.0
        P_GPa[i] = P_GPa[i-1] + ((rho_d[i]+rho_d[i-1])/2) * ((_GRAVITY_G[i]+_GRAVITY_G[i-1])/2) * dz / 1e9

    _PRESSURE_DEPTH = _GRAVITY_DEPTH.copy()
    _PRESSURE_GPA   = P_GPa
    print(f"  Gravity loaded: P at {TRUE_CMB_DEPTH:.0f} km = "
          f"{float(np.interp(TRUE_CMB_DEPTH, _PRESSURE_DEPTH, _PRESSURE_GPA)):.2f} GPa")


def gravity_mars(depth_km):
    if _GRAVITY_DEPTH is None:
        load_gravity_profile()
    return np.interp(depth_km, _GRAVITY_DEPTH, _GRAVITY_G)


def pressure_mars(depth_km):
    if _PRESSURE_DEPTH is None:
        load_gravity_profile()
    return np.interp(depth_km, _PRESSURE_DEPTH, _PRESSURE_GPA)

# ── Samuel 2023 median model ──────────────────────────────────────────────────
_SAMUEL_CACHE = None

def compute_samuel_median():
    global _SAMUEL_CACHE
    if _SAMUEL_CACHE is not None:
        return _SAMUEL_CACHE

    vp_files = sorted(glob.glob(os.path.join(SAMUEL_DATA_DIR, 'vp*.dat')))
    print(f"Loading {len(vp_files)} Samuel models...")

    crust_z = np.linspace(0, 100, 200)
    core_z  = np.linspace(1500, MARS_RADIUS, 200)

    crust_vp_all  = []; crust_vs_all = []; crust_rho_all = []
    core_vp_all   = []

    for vp_path in vp_files:
        m = re.search(r'vp(\d+)\.dat', os.path.basename(vp_path))
        if not m:
            continue
        vs_path = os.path.join(SAMUEL_DATA_DIR, f'vs{m.group(1)}.dat')
        if not os.path.exists(vs_path):
            continue
        try:
            vp_data = np.loadtxt(vp_path)
            vs_data = np.loadtxt(vs_path)
            vp_ms   = vp_data[:, 0].copy()
            vs_ms   = vs_data[:, 0].copy()
            vs_ms[-2:] = 0.0
            vp    = vp_ms / 1000.0
            vs    = vs_ms / 1000.0
            depth = MARS_RADIUS - vp_data[:, 1]
            idx   = np.argsort(depth)
            depth, vp, vs = depth[idx], vp[idx], vs[idx]
            rho   = 0.32 * vp + 0.77

            liquid_mask = vs < CMB_VS_THRESHOLD
            solid_mask  = ~liquid_mask
            if solid_mask.sum() < 5 or liquid_mask.sum() < 5:
                continue

            liq_depth = depth[liquid_mask]
            liq_vp    = vp[liquid_mask]
            dvp       = np.abs(np.diff(liq_vp))

            sd  = depth[solid_mask]
            if sd.max() > crust_z.max():
                crust_vp_all.append(np.interp(crust_z, sd, vp[solid_mask]))
                crust_vs_all.append(np.interp(crust_z, sd, vs[solid_mask]))
                crust_rho_all.append(np.interp(crust_z, sd, rho[solid_mask]))

            true_cmb    = liq_depth[np.argmax(dvp) + 1]
            core_liq    = liquid_mask & (depth >= true_cmb)
            cd, cvp     = depth[core_liq], vp[core_liq]
            if len(cd) > 0 and cd.max() >= core_z.max() * 0.9:
                core_vp_all.append(np.interp(core_z, cd, cvp))
        except Exception as e:
            print(f"  Warning: {e}")
            continue

    crust_vp_med  = np.nanmedian(crust_vp_all, axis=0)
    crust_vs_med  = np.nanmedian(crust_vs_all, axis=0)
    crust_rho_med = np.nanmedian(crust_rho_all, axis=0)
    core_vp_med   = np.nanmedian(core_vp_all, axis=0)

    try:
        rho_data  = np.loadtxt(SAMUEL_RHO_PROFILE_PATH)
        depth_rho = MARS_RADIUS - rho_data[:, 1]
        idx       = np.argsort(depth_rho)
        core_rho_med = np.interp(core_z, depth_rho[idx], rho_data[idx, 0] / 1000.0)
    except Exception as e:
        print(f"  rho_profile.dat failed ({e}), using Birch's law")
        core_rho_med = 0.32 * core_vp_med + 0.77

    print(f"  CMB={TRUE_CMB_DEPTH:.1f} km")

    _SAMUEL_CACHE = {
        'crust_z':        crust_z,
        'crust_vp':       crust_vp_med,
        'crust_vs':       crust_vs_med,
        'crust_rho':      crust_rho_med,
        'core_z':         core_z,
        'core_vp':        core_vp_med,
        'core_vs':        np.zeros(len(core_z)),
        'core_rho':       core_rho_med,
        'true_cmb_depth': TRUE_CMB_DEPTH,
    }
    return _SAMUEL_CACHE

# ── physics: solidus / liquidus / melt fraction ───────────────────────────────
MG_DUNCAN_REF  = 0.75
PHI_OUTER_CORE = 0.30

_PIERRU_PHI_NODES = np.array([0.000, 0.030, 0.050, 0.080, 0.130, 0.300, 1.000])
_PIERRU_DVP_NODES = np.array([0.000, 0.063, 0.081, 0.102, 0.124, 0.200, 0.200])
_PIERRU_DVS_NODES = np.array([0.000, 0.174, 0.215, 0.241, 0.271, 1.000, 1.000])


def solidus_duncan2018(P_GPa):
    P = float(P_GPa)
    if P <= 10.0:
        T_C = -4.877*P**2 + 120.2*P + 1088.0
    elif P <= 23.0:
        T_C = -1.323*(P-10.0)**2 + 38.18*(P-10.0) + 1802.0
    else:
        T_C = 77.75*(P-23.0) + 2075.0
    return T_C + 273.15


def solidus_bml(P_GPa, Mg_bml):
    Fe     = 100.0 * (1.0 - Mg_bml)
    Fe_ref = 100.0 * (1.0 - MG_DUNCAN_REF)
    return solidus_duncan2018(P_GPa) + (-6.0 * (Fe - Fe_ref))


def liquidus_bml(P_GPa, Mg_bml):
    Fe     = 100.0 * (1.0 - Mg_bml)
    Fe_ref = 100.0 * (1.0 - MG_DUNCAN_REF)
    liq    = 2160.6 + 64.7109*P_GPa - 3.97463*P_GPa**2 + 0.0957894*P_GPa**3
    return liq + (-6.0 * (Fe - Fe_ref))


def compute_melt_fraction(T_K, P_GPa, Mg_bml):
    T_sol = solidus_bml(P_GPa, Mg_bml)
    T_liq = liquidus_bml(P_GPa, Mg_bml)
    if T_liq <= T_sol:
        T_liq = T_sol + 200.0
    return float(np.clip((T_K - T_sol) / (T_liq - T_sol), 0.0, 1.0))


def apply_melt_correction_pierru(Vp, Vs, phi):
    if phi <= 0.0:
        return Vp, Vs
    dVp = float(np.interp(phi, _PIERRU_PHI_NODES, _PIERRU_DVP_NODES))
    dVs = float(np.interp(phi, _PIERRU_PHI_NODES, _PIERRU_DVS_NODES))
    return Vp * (1.0 - dVp), max(Vs * (1.0 - dVs), 0.0)

# ── composition ───────────────────────────────────────────────────────────────
def composition_from_params(params):
    mgnum = params['Mg#']
    Mg    = mgnum * MGFE_TOTAL
    Fe    = (1.0 - mgnum) * MGFE_TOTAL
    return {**FIXED_PARAMS,
            'Mg': Mg, 'Fe': Fe,
            'T_lit': params['T_lit'], 'P_lit': params['P_lit']}


def compute_oxygen(p):
    return (2.0*p['Si'] + p['Mg'] + p['Fe'] + p['Ca'] +
            1.5*p['Al'] + 0.5*p['Na'] + 1.5*p['Cr'])

# ── HeFESTo control file ──────────────────────────────────────────────────────
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
        f"Na      {p['Na']:.5f}     {p['Na']:.5f}    0",
        f"Cr      {p['Cr']:.5f}     {p['Cr']:.5f}    0",
        f" O      {O:.5f}     {O:.5f}    0",
        "1,1,1,1", PAR_DIR, "73", CONTROL_PHASES,
    ]

# ── HeFESTo runner ─────────────────────────────────────────────────────────────
def _cleanup(step_dir):
    keep = {'fort.56', 'ad.in', 'control'}
    if not os.path.isdir(step_dir):
        return
    for fname in os.listdir(step_dir):
        if fname not in keep:
            fpath = os.path.join(step_dir, fname)
            try:
                if os.path.isfile(fpath):   os.remove(fpath)
                elif os.path.isdir(fpath):  shutil.rmtree(fpath, ignore_errors=True)
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
    try:
        with open(os.path.join(run_dir, "hefesto.log"), 'w') as log:
            subprocess.run(["./main"], cwd=run_dir,
                           stdout=log, stderr=log, timeout=timeout)
    except Exception:
        return None
    fort56 = os.path.join(run_dir, "fort.56")
    if not os.path.exists(fort56) or os.path.getsize(fort56) == 0:
        return None
    return fort56


def read_fort56(fort56_path):
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
                if re.search(r'\d-\d', line):
                    continue
                clean_lines.append(line)

        if len(clean_lines) <= 2:
            return None

        df = pd.read_csv(io.StringIO(''.join(clean_lines)),
                         sep=r'\s+', skiprows=2, names=cols)
        if df.empty:
            return None
    except Exception:
        return None

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    key_cols = ['P(GPa)', 'T(K)', 'S(J/g/K)', 'rho(g/cm^3)', 'VS(km/s)', 'VP(km/s)']
    df = df.dropna(subset=[c for c in key_cols if c in df.columns])
    if len(df) < 1:
        return None

    P_GPa = df['P(GPa)'].values
    rho   = df['rho(g/cm^3)'].values
    depth = np.zeros(len(P_GPa))
    dP    = np.diff(P_GPa) * 1e9
    for i in range(len(dP)):
        dz         = dP[i] / (((rho[i]+rho[i+1])/2)*1000 * gravity_mars(depth[i])) / 1000
        depth[i+1] = depth[i] + dz

    return {
        'depth_km': depth,
        'P_GPa':    P_GPa,
        'T_K':      df['T(K)'].values,
        'S':        df['S(J/g/K)'].values,
        'Vp':       df['VP(km/s)'].values,
        'Vs':       df['VS(km/s)'].values,
        'rho':      rho,
    }


def run_hefesto(params, run_dir, P_bml_top=None):
    p   = composition_from_params(params)
    O   = compute_oxygen(p)
    T_lit, P_lit = p['T_lit'], p['P_lit']

    if P_bml_top is None:
        bml_top_km = TRUE_CMB_DEPTH - params.get('BML_thickness', 168.7)
        P_bml_top  = float(pressure_mars(bml_top_km))

    # step 1: S_lit
    dir1 = os.path.join(run_dir, "s1_npt")
    f1   = run_hefesto_single(dir1,
           make_control_lines(p, O,
           f"{P_lit:.4f},{P_lit:.4f},1,{T_lit:.2f},{T_lit:.2f},0,0,0,0"))
    if f1 is None:
        print("  Step 1 failed"); _cleanup(dir1); return None, None
    d1   = read_fort56(f1)
    if d1 is None:
        print("  Step 1 read failed"); _cleanup(dir1); return None, None
    S_lit = float(d1['S'][0])
    if not np.isfinite(S_lit) or S_lit <= 0 or S_lit > 10:
        print(f"  Step 1: invalid S_lit={S_lit}"); _cleanup(dir1); return None, None
    print(f"  S_lit={S_lit:.6f}  T={T_lit:.1f}K  P={P_lit:.2f}GPa")
    _cleanup(dir1)

    # step 2: NPT scan → adiabatic T(P)
    P_scan = np.concatenate([[P_lit],
                              np.array([9.0, 12.0, 15.0, 17.0])[np.array([9.0,12.0,15.0,17.0]) < P_bml_top],
                              [P_bml_top]])
    T_guess      = T_lit
    P_adiab_list = [P_lit]
    T_adiab_list = [T_lit]

    for i_p, P_target in enumerate(P_scan[1:]):
        d_lo = d_hi = None
        for dT in [20.0, 10.0, 5.0, 2.0]:
            T_lo, T_hi = T_guess - dT, T_guess + dT
            dir_lo = os.path.join(run_dir, f"s2_scan_{i_p}_lo")
            dir_hi = os.path.join(run_dir, f"s2_scan_{i_p}_hi")
            f_lo = run_hefesto_single(dir_lo, make_control_lines(p, O,
                   f"{P_target:.4f},{P_target:.4f},1,{T_lo:.2f},{T_lo:.2f},0,0,0,0"))
            f_hi = run_hefesto_single(dir_hi, make_control_lines(p, O,
                   f"{P_target:.4f},{P_target:.4f},1,{T_hi:.2f},{T_hi:.2f},0,0,0,0"))
            d_lo = read_fort56(f_lo) if f_lo else None
            d_hi = read_fort56(f_hi) if f_hi else None
            _cleanup(dir_lo); _cleanup(dir_hi)
            if d_lo is not None and d_hi is not None:
                break

        if d_lo is None or d_hi is None:
            if len(T_adiab_list) >= 2:
                dT_dP = ((T_adiab_list[-1] - T_adiab_list[-2]) /
                         (P_adiab_list[-1] - P_adiab_list[-2]))
                T_adiab_p = T_adiab_list[-1] + dT_dP * (P_target - P_adiab_list[-1])
            else:
                T_adiab_p = T_guess
        else:
            S_lo, S_hi = float(d_lo['S'][0]), float(d_hi['S'][0])
            if abs(S_hi - S_lo) < 1e-8:
                T_adiab_p = T_guess
            else:
                T_adiab_p = T_lo + (S_lit - S_lo) * (T_hi - T_lo) / (S_hi - S_lo)

        T_adiab_p = float(np.clip(T_adiab_p, 800, 4000))
        P_adiab_list.append(P_target)
        T_adiab_list.append(T_adiab_p)
        T_guess = T_adiab_p

    P_adiab = np.array(P_adiab_list)
    T_adiab = np.array(T_adiab_list)
    print(f"  Adiabat: T={T_adiab[0]:.1f}K@{P_adiab[0]:.2f}GPa → "
          f"T={T_adiab[-1]:.1f}K@{P_adiab[-1]:.2f}GPa")

    # build ad.in
    P_cond = np.linspace(1.04, P_lit, 100)
    T_cond = T_SURF + (T_lit - T_SURF) * (P_cond / P_lit)
    P_full = np.concatenate([P_cond, P_adiab])
    T_full = np.concatenate([T_cond, T_adiab])
    idx    = np.argsort(P_full)
    P_full, T_full = P_full[idx], T_full[idx]
    _, uniq = np.unique(P_full, return_index=True)
    P_full, T_full = P_full[uniq], T_full[uniq]
    ad_in  = "".join(f"{P:.6f} 0.000000 {T:.6f}\n" for P, T in zip(P_full, T_full))

    # step 3: full run
    dir3   = os.path.join(run_dir, "s3_final")
    f3     = run_hefesto_single(dir3,
             make_control_lines(p, O, f"0,{P_bml_top:.4f},50,0,0,0,-1,0,0"),
             ad_in_content=ad_in)
    _cleanup(dir3)
    if f3 is None:
        print("  Step 3 failed"); return None, None
    d3 = read_fort56(f3)
    if d3 is None:
        print("  Step 3 read failed"); return None, None

    d3['P_profile'] = P_full
    d3['T_profile'] = T_full
    print("  Step 3 OK")
    return f3, d3


def run_hefesto_bml(params, run_dir, T_bml, T_mantle_bottom, true_cmb_depth, n_points=20):
    bml_thickness = params['BML_thickness']
    Mg_bml        = params['Mg#_bml']
    bml_top_depth = true_cmb_depth - bml_thickness
    P_top         = max(float(pressure_mars(bml_top_depth)), 5.0)
    P_bottom      = float(pressure_mars(true_cmb_depth))

    p = composition_from_params({'Mg#': Mg_bml, 'T_lit': T_mantle_bottom, 'P_lit': P_top})
    O = compute_oxygen(p)

    P_range = np.linspace(P_top, P_bottom, n_points)
    T_range = np.linspace(T_mantle_bottom, T_bml, n_points)
    ad_in   = "".join(f"{P:.6f} 0.000000 {T:.6f}\n" for P, T in zip(P_range, T_range))

    dir_bml = os.path.join(run_dir, "bml")
    fort56  = run_hefesto_single(dir_bml,
              make_control_lines(p, O, f"{P_top:.4f},{P_bottom:.4f},{n_points},0,0,0,-1,0,0"),
              ad_in_content=ad_in)
    _cleanup(dir_bml)
    if fort56 is None:
        return None

    bml_raw = read_fort56(fort56)
    if bml_raw is None:
        return None

    # recompute depth from pressure (BML-local)
    rho   = bml_raw['rho']
    P_GPa = bml_raw['P_GPa']
    depth = np.zeros(len(P_GPa))
    dP    = np.diff(P_GPa) * 1e9
    for i in range(len(dP)):
        dz         = dP[i] / (((rho[i]+rho[i+1])/2)*1000 * gravity_mars(depth[i])) / 1000
        depth[i+1] = depth[i] + dz

    T_actual = bml_raw['T_K']
    phi_arr  = np.zeros(len(P_GPa))
    Vp_corr  = np.zeros(len(P_GPa))
    Vs_corr  = np.zeros(len(P_GPa))
    outer_core_offset = depth[-1]

    for i in range(len(P_GPa)):
        phi        = compute_melt_fraction(float(T_actual[i]), P_GPa[i], Mg_bml)
        phi_arr[i] = phi
        Vp_corr[i], Vs_corr[i] = apply_melt_correction_pierru(
            float(bml_raw['Vp'][i]), float(bml_raw['Vs'][i]), phi)
        if phi >= PHI_OUTER_CORE and outer_core_offset == depth[-1]:
            outer_core_offset = depth[i]

    print(f"  BML: thick={bml_thickness:.0f}km  "
          f"T={T_mantle_bottom:.0f}→{T_bml:.0f}K  "
          f"phi={phi_arr[0]:.3f}→{phi_arr[-1]:.3f}  "
          f"oc_offset={outer_core_offset:.1f}km")

    return {
        'depth_km':              depth,
        'P_GPa':                 P_GPa,
        'T_K':                   T_actual,
        'Vp':                    Vp_corr,
        'Vs':                    Vs_corr,
        'rho':                   rho,
        'phi':                   phi_arr,
        'outer_core_depth_offset': outer_core_offset,
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

        if bml_data is not None:
            bml_depth = bml_data['depth_km']
            bml_vp    = bml_data['Vp']
            bml_vs    = bml_data['Vs']
            bml_rho   = bml_data['rho']
            outer_core_abs = bml_data.get('outer_core_depth_abs', bml_depth[-1])
            bml_mask  = (bml_depth >= bml_top_depth) & (bml_depth <= true_cmb_depth)

            for d, vp, vs, r in zip(bml_depth[bml_mask & (bml_depth < outer_core_abs)],
                                     bml_vp[bml_mask & (bml_depth < outer_core_abs)],
                                     bml_vs[bml_mask & (bml_depth < outer_core_abs)],
                                     bml_rho[bml_mask & (bml_depth < outer_core_abs)]):
                f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

            idx_oc = min(np.searchsorted(bml_depth, outer_core_abs), len(bml_depth)-1)
            f.write(f"{outer_core_abs:.3f}  {bml_vp[idx_oc]:.4f}  {bml_vs[idx_oc]:.4f}  {bml_rho[idx_oc]:.4f}\n")
            f.write("outer-core\n")
            f.write(f"{outer_core_abs:.3f}  {bml_vp[idx_oc]:.4f}  0.0000  {bml_rho[idx_oc]:.4f}\n")

            for d, vp, r in zip(bml_depth[bml_mask & (bml_depth > outer_core_abs)],
                                  bml_vp[bml_mask & (bml_depth > outer_core_abs)],
                                  bml_rho[bml_mask & (bml_depth > outer_core_abs)]):
                f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")

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
    R              = MARS_RADIUS_M
    true_cmb_km    = samuel_cache['true_cmb_depth']
    bml_top_km     = float(bml_data['depth_km'][0])

    crust_z      = samuel_cache['crust_z']
    crust_rho    = samuel_cache['crust_rho']
    crust_mask   = crust_z <= 100.0
    crust_r      = R - crust_z[crust_mask] * 1000
    crust_rho_si = crust_rho[crust_mask] * 1000

    hef_depth    = fort56_data['depth_km']
    mantle_mask  = (hef_depth >= 100.0) & (hef_depth <= bml_top_km)
    man_r        = R - hef_depth[mantle_mask] * 1000
    man_rho_si   = fort56_data['rho'][mantle_mask] * 1000

    bml_depth  = bml_data['depth_km']
    bml_mask   = (bml_depth >= bml_top_km) & (bml_depth <= true_cmb_km)
    bml_r      = R - bml_depth[bml_mask] * 1000
    bml_rho_si = bml_data['rho'][bml_mask] * 1000

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
    T_sol_arr   = np.array([solidus_duncan2018(p) for p in P_m])
    excess      = T_m - T_sol_arr
    penalty     = float(np.sum(excess[excess > 0])) / 100.0
    if penalty > 0:
        print(f"  Solidus penalty = {penalty:.4f}")
    return penalty

# ── misfit ────────────────────────────────────────────────────────────────────
def compute_misfit(taup_model, obs_dataset, fort56_data, samuel_cache, bml_data, params=None):
    phases_std   = ['P', 'S', 'pP', 'sP', 'PP', 'PPP', 'SS', 'SSS', 'sS', 'ScS', 'SKS']
    phases_pdiff = phases_std + ['Pdiff']

    tt_total = 0.0
    tt_n     = 0

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

        for phase, obs_val in obs.items():
            if phase in ('delta', 'depth') or not isinstance(obs_val, tuple):
                continue
            p_val = pred.get(phase)
            if not p_val:
                continue
            obs_t, sigma = obs_val
            if sigma <= 0 or not np.isfinite(obs_t) or not np.isfinite(p_val):
                continue
            val = abs(obs_t - p_val) / sigma
            if np.isfinite(val):
                tt_total += val
                tt_n     += 1

    tt_misfit = tt_total / tt_n if tt_n > 0 else 999.0
    if not np.isfinite(tt_misfit):
        tt_misfit = 999.0

    M_pred, moi_pred = compute_mass_and_moi(fort56_data, samuel_cache, bml_data=bml_data)
    mass_misfit = abs(MARS_MASS_OBS - M_pred) / MARS_MASS_SIGMA
    moi_misfit  = abs(MOI_OBS - moi_pred)     / MOI_SIGMA
    if not np.isfinite(mass_misfit): mass_misfit = 999.0
    if not np.isfinite(moi_misfit):  moi_misfit  = 999.0

    solidus_penalty = compute_solidus_penalty(fort56_data, params) if params else 0.0

    total_n      = tt_n + 2
    total_misfit = (tt_total + mass_misfit + moi_misfit + solidus_penalty) / total_n
    if not np.isfinite(total_misfit):
        total_misfit = 999.0

    print(f"  TT={tt_misfit:.4f}(n={tt_n})  "
          f"mass={mass_misfit:.4f}  moi={moi_misfit:.4f}  "
          f"total={total_misfit:.4f}")

    return total_misfit, total_n, {
        'tt': tt_misfit, 'mass': mass_misfit,
        'moi': moi_misfit, 'solidus': solidus_penalty,
    }

# ── forward model ─────────────────────────────────────────────────────────────
def forward(params, run_dir, model_name, samuel_cache):
    true_cmb_depth = samuel_cache['true_cmb_depth']
    bml_top_km     = true_cmb_depth - params['BML_thickness']
    P_bml_top      = float(pressure_mars(bml_top_km))

    fort56, fort56_data = run_hefesto(params, run_dir, P_bml_top=P_bml_top)
    if fort56_data is None:
        return None, None, None, None, None

    T_profile = fort56_data.get('T_profile')
    P_profile = fort56_data.get('P_profile')
    T_mantle_bottom = (float(np.interp(P_bml_top, P_profile, T_profile))
                       if T_profile is not None else float(fort56_data['T_K'][-1]))
    print(f"  T_mantle_bottom={T_mantle_bottom:.1f}K  P_bml_top={P_bml_top:.2f}GPa")

    bml_raw = run_hefesto_bml(params, run_dir,
                              T_bml=params['T_bml'],
                              T_mantle_bottom=T_mantle_bottom,
                              true_cmb_depth=true_cmb_depth)
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
    bml_data = bml_raw

    try:
        taup_model = build_taup(fort56_data, model_name, samuel_cache, bml_data=bml_data)
    except Exception as e:
        print(f"  TauP failed: {e}"); return None, None, None, None, None

    misfit, n_data, components = compute_misfit(
        taup_model, SAMUEL_DATA, fort56_data, samuel_cache,
        bml_data=bml_data, params=params)

    if components is not None and bml_data is not None:
        components['upper_contrast'] = bml_data.get('upper_contrast')
        components['lower_contrast'] = bml_data.get('lower_contrast')

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

    load_gravity_profile()
    samuel_cache = compute_samuel_median()

    rng     = np.random.default_rng(42 + chain_id)
    current = (start_params or START_PARAMS).copy()

    chain_file = os.path.join(chain_dir, "chain.json")
    chain      = []
    if os.path.exists(chain_file):
        with open(chain_file) as f:
            chain = json.load(f)
        if chain:
            current = chain[-1]['params']
            print(f"Chain {chain_id}: resuming from step {len(chain)}")

    step_start = len(chain)
    accept_count = 0
    current_components = {'tt': 999.0, 'mass': 999.0, 'moi': 999.0,
                          'solidus': 0.0, 'upper_contrast': None, 'lower_contrast': None}

    if chain:
        current_misfit = chain[-1]['misfit']
        accept_count   = sum(1 for s in chain if s.get('accepted', False))
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
        for attempt in range(20):
            trial = current.copy() if attempt == 0 else {
                k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PRIOR.items()}
            if attempt > 0:
                print(f"  Retry {attempt}: T_lit={trial['T_lit']:.1f}  "
                      f"Mg#={trial['Mg#']:.3f}  T_bml={trial['T_bml']:.1f}")
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
            components      = {'tt': 999.0, 'mass': 999.0, 'moi': 999.0,
                               'solidus': 0.0, 'upper_contrast': None, 'lower_contrast': None}
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
                    })
                np.savez(os.path.join(chain_dir, f"profile_s{step+1:05d}.npz"), **npz_dict)

        elapsed     = (datetime.now() - t0).total_seconds()
        accept_rate = accept_count / (step - step_start + 1) * 100
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
            'misfit_mass':    current_components.get('mass',    999.0),
            'misfit_moi':     current_components.get('moi',     999.0),
            'misfit_solidus': current_components.get('solidus', 0.0),
            'upper_contrast': current_components.get('upper_contrast'),
            'lower_contrast': current_components.get('lower_contrast'),
            'accepted':       bool(accepted),
            'accept_rate':    accept_rate,
        })

        with open(chain_file, 'w') as f:
            json.dump(chain, f, indent=2)

    print(f"\nChain {chain_id} done  "
          f"accept_rate={accept_count/n_steps*100:.1f}%  "
          f"misfit={current_misfit:.4f}")

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