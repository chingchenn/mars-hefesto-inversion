"""
39_mcmc_noBML_nGibbs.py

noBML version of MCMC using nGibbs emulator instead of HeFESTo.
- No BML: mantle extends to CMB (~23 GPa)
- PRIOR: T_lit, P_lit, Mg# only (no T_bml, Mg#_bml)
- build_taup: mantle goes directly to true_cmb_depth with CMB discontinuity
"""

import os
import shutil
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

repo_root   = Path('/home/jcchen2/nGibbs/')
src_root    = repo_root / "src"
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
GAMMA       = 1.1
MARS_MASS_OBS   = 6.4171e23
MARS_MASS_SIGMA = MARS_MASS_OBS * 0.01
MOI_OBS         = 0.3634
MOI_SIGMA       = 0.0006
MARS_RADIUS_M   = MARS_RADIUS * 1000

P_BML_TOP      = 19.0
P_BML_BOTTOM   = 21.0
P_MAX_GPA_BML  = 19.0
P_MAX_GPA_NOBL = 23.0   # noBML: mantle to CMB
MCMC_TEMPERATURE = 1.0

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

START_PARAMS = {
    'T_lit':   1952.92,
    'P_lit':   6.0755,
    'Mg#':     YM_BASE['Mg'] / (YM_BASE['Mg'] + YM_BASE['Fe']),
}

FIXED_PARAMS = {
    'Si': YM_BASE['Si'], 'Ca': YM_BASE['Ca'], 'Al': YM_BASE['Al'],
    'Na': YM_BASE['Na'], 'Cr': YM_BASE['Cr'],
}

PRIOR = {
    'T_lit':   (1000.0, 2600.0),
    'P_lit':   (1.5,    9.0),
    'Mg#':     (0.50,   0.86),
}

STEP = {
    'T_lit':   60.0,
    'P_lit':   0.3,
    'Mg#':     0.015,
}

SIGMA = {
    'S-P': 10.0, 'pP-P': 3.0, 'sP-P': 5.0, 'PP-P': 8.0, 'PPP-P': 12.0,
    'sS-S': 5.0, 'SS-S': 5.0, 'SSS-S': 8.0, 'ScS-S': 12.0,
    'SS-PP': 10.0, 'SKS-PP': 10.0,
}

def s(a, sa, b, sb):
    return np.sqrt(sa**2 + sb**2)

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
# nGibbs emulator
# ============================================================

def run_ngibbs(params, use_bml=False):
    """
    nGibbs emulator replacing run_hefesto().
    use_bml=True  -> P_max = 19 GPa
    use_bml=False -> P_max = 23 GPa (mantle to CMB)
    """
    p     = composition_from_params(params)
    T_lit = params['T_lit']
    P_lit = params['P_lit']

    P_max = P_MAX_GPA_BML if use_bml else P_MAX_GPA_NOBL

    # 
    pressures_shallow = np.linspace(0.01, 3.0, 100)
    pressures_deep    = np.linspace(3.0, P_max, 101)[1:]
    pressures         = np.concatenate([pressures_shallow, pressures_deep])
    n_pressures       = len(pressures)
    P_lit_idx         = np.argmin(np.abs(pressures - P_lit))

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

    # Step 1: S_lit via Burnman
    single_input = np.array([[P_lit, T_lit] + comp_values], dtype=np.float32)
    with torch.no_grad():
        out1 = HeFESToEmulatorCPU.ForwardMB(
            single_input, headers=T_headers, outputs=['component_moles'])
    PT_single = np.array([[P_lit, T_lit]], dtype=np.float64)
    burnman_s = HeFESToEmulatorCPU.get_property_burnman_vectorized_from_assemblage(
        torch.tensor(out1['component_moles'], dtype=torch.float64),
        torch.tensor(PT_single, dtype=torch.float64),
        property_names=['entropy_by_mass'],
    )
    S_lit = float(burnman_s['entropy_by_mass'][0]) / 1000.0
    print(f"    S_lit = {S_lit:.6f} J/g/K  @ P={P_lit:.2f} GPa, T={T_lit:.1f} K")

    # Step 2: Isentropic segment (asthenosphere)
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
            outputs=['component_moles', 'temperature'])

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

    # Step 3: Conductive segment (lithosphere)
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
            iso_input, headers=T_headers, outputs=['component_moles'])

    PT_iso = np.stack([iso_input[:, 0], T_cond], axis=1)
    burnman_iso = HeFESToEmulatorCPU.get_property_burnman_vectorized_from_assemblage(
        torch.tensor(out3['component_moles'], dtype=torch.float64),
        torch.tensor(PT_iso, dtype=torch.float64),
        property_names=['density', 'p_wave_velocity', 's_wave_velocity',
                        'isentropic_bulk_modulus_reuss'],
    )
    for k, v in burnman_iso.items():
        properties[k][isothermal_IDX[1]] = v

    # depth integral
    rho   = properties['density']
    dP    = np.diff(pressures) * 1e9
    depth = np.zeros(n_pressures)
    for i in range(len(dP)):
        g_i        = gravity_mars(depth[i])
        rho_mid    = (rho[i] + rho[i+1]) / 2
        depth[i+1] = depth[i] + dP[i] / (rho_mid * g_i) / 1000.0

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


def run_ngibbs_bml(params, P_top, P_bottom, T_bml, n_points=20):
    """Isothermal BML using nGibbs."""
    p = composition_from_params(params)
    O = compute_oxygen(p)
    comp_values = [p['Si'], p['Mg'], p['Fe'], p['Ca'],
                   p['Al'], p['Na'], p['Cr'], O]

    T_headers = ['P(GPa)(System_main)', 'T(K)(System_main)',
                 'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr', 'O']

    pressures = np.linspace(P_top, P_bottom, n_points)
    bml_input = np.zeros((n_points, 2 + len(comp_values)), dtype=np.float32)
    bml_input[:, 0] = pressures
    bml_input[:, 1] = T_bml
    bml_input[:, 2:] = comp_values

    with torch.no_grad():
        out = HeFESToEmulatorCPU.ForwardMB(
            bml_input, headers=T_headers, outputs=['component_moles'])

    burnman_out = HeFESToEmulatorCPU.get_property_burnman_vectorized_from_assemblage(
        torch.tensor(out['component_moles'], dtype=torch.float64),
        torch.tensor(bml_input[:, :2], dtype=torch.float64),
        property_names=['density', 'p_wave_velocity'],
    )

    rho = burnman_out['density']
    dP  = np.diff(pressures) * 1e9
    depth = np.zeros(n_points)
    for i in range(len(dP)):
        g_i        = gravity_mars(depth[i])
        rho_mid    = (rho[i] + rho[i+1]) / 2
        depth[i+1] = depth[i] + dP[i] / (rho_mid * g_i) / 1000.0

    return {
        'depth_km': depth,
        'P_GPa':    pressures,
        'Vp':       burnman_out['p_wave_velocity'] / 1000.0,
        'Vs':       np.zeros(n_points),
        'rho':      burnman_out['density'] / 1000.0,
    }

# ============================================================
# TauP
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

    # noBML: mantle goes to true_cmb; BML: mantle goes to lsl_top
    mantle_bottom = lsl_top_depth if bml_data is not None else true_cmb_depth

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
        # Khan 0-200 km (crust + upper mantle)
        for d, vp, vs, r in zip(khan['crust_z'], khan['crust_vp'],
                                 khan['crust_vs'], khan['crust_rho']):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        f.write("mantle\n")
        for d, vp, vs, r in zip(man_depth, man_Vp, man_Vs, man_rho):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

        if bml_data is not None:
            bml_depth = bml_data['depth_km']
            bml_vp    = bml_data['Vp']
            bml_rho   = bml_data['rho']
            bml_mask  = ((bml_depth >= mantle_bottom) &
                         (bml_depth <= true_cmb_depth))

            bml_vp_top  = float(bml_vp[bml_mask][0])
            bml_rho_top = float(bml_rho[bml_mask][0])

            f.write(f"{man_depth[-1]:.3f}  "
                    f"{man_Vp[-1]:.4f}  {man_Vs[-1]:.4f}  {man_rho[-1]:.4f}\n")
            f.write(f"{man_depth[-1]:.3f}  "
                    f"{bml_vp_top:.4f}  0.0000  {bml_rho_top:.4f}\n")
            f.write("outer-core\n")
            bml_d = bml_depth[bml_mask]
            bml_v = bml_vp[bml_mask]
            bml_r = bml_rho[bml_mask]
            for d, vp, r in zip(bml_d[1:], bml_v[1:], bml_r[1:]):
                f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")

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
            # noBML: mantle直接到CMB，用雙行寫CMB不連續
            core_z   = khan['core_z']
            core_vp  = khan['core_vp']
            core_rho = khan['core_rho']
            mask     = core_z >= true_cmb_depth

            f.write(f"{man_depth[-1]:.3f}  "
                    f"{man_Vp[-1]:.4f}  {man_Vs[-1]:.4f}  {man_rho[-1]:.4f}\n")
            # CMB discon.
            f.write(f"{true_cmb_depth:.3f}  "
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

def compute_mass_and_moi(fort56_data, khan_cache, bml_data=None):
    R                = MARS_RADIUS_M
    lsl_top_km       = khan_cache['lsl_top_depth']
    true_cmb_km      = khan_cache['true_cmb_depth']
    mantle_bottom_km = lsl_top_km if bml_data is not None else true_cmb_km

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
# Solidus penalty
# ============================================================

def compute_solidus_penalty(fort56_data, params, use_bml=False):
    penalty = 0.0

    P_prof = fort56_data.get('P_profile', None)
    T_prof = fort56_data.get('T_profile', None)

    if P_prof is not None and T_prof is not None:
        P_lit        = params['P_lit']
        P_mantle_top = P_BML_TOP if use_bml else P_MAX_GPA_NOBL
        mantle_mask  = (P_prof >= P_lit) & (P_prof <= P_mantle_top)
        P_m = P_prof[mantle_mask]
        T_m = T_prof[mantle_mask]

        if len(P_m) > 0:
            excess        = T_m - solidus_duncan2018(P_m)
            mantle_excess = float(np.sum(excess[excess > 0])) / 100.0
            if mantle_excess > 0:
                print(f"    Mantle solidus penalty = {mantle_excess:.4f}")
                penalty += mantle_excess

    return penalty

# ============================================================
# Misfit
# ============================================================

def _diff(times, a, b):
    return times[a] - times[b] if a in times and b in times else None


def compute_misfit(taup_model, obs_dataset, fort56_data,
                   khan_cache, bml_data=None, params=None, use_bml=False):

    tt_total = 0.0
    tt_n     = 0
    phases   = ['P','S','pP','sP','PP','PPP','SS','SSS','sS','ScS','SKS']

    for event, obs in obs_dataset.items():
        delta = obs['delta']
        depth = obs.get('depth', 10.0)
        try:
            arrivals = taup_model.get_travel_times(
                source_depth_in_km=depth,
                distance_in_degree=delta,
                phase_list=phases)
        except Exception:
            continue
        times = {}
        for a in arrivals:
            if a.name not in times:
                times[a.name] = a.time

        pred = {
            'S-P':    _diff(times, 'S',   'P'),
            'pP-P':   _diff(times, 'pP',  'P'),
            'sP-P':   _diff(times, 'sP',  'P'),
            'PP-P':   _diff(times, 'PP',  'P'),
            'PPP-P':  _diff(times, 'PPP', 'P'),
            'sS-S':   _diff(times, 'sS',  'S'),
            'SS-S':   _diff(times, 'SS',  'S'),
            'SSS-S':  _diff(times, 'SSS', 'S'),
            'ScS-S':  _diff(times, 'ScS', 'S'),
            'SS-PP':  _diff(times, 'SS',  'PP'),
            'SKS-PP': _diff(times, 'SKS', 'PP'),
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

    solidus_penalty = compute_solidus_penalty(fort56_data, params,
                                              use_bml=use_bml)
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

    return total_misfit, total_n, {
        'tt':      tt_misfit,
        'mass':    mass_misfit,
        'moi':     moi_misfit,
        'solidus': solidus_penalty,
    }

# ============================================================
# forward model
# ============================================================

def forward(params, run_dir, model_name, khan_cache,
            use_bml=False, skip_bml_density_check=False):

    fort56_data = run_ngibbs(params, use_bml=use_bml)
    if fort56_data is None:
        return None, None, None, None

    bml_data = None
    if use_bml:
        bml_raw = run_ngibbs_bml(
            params   = {'T_lit': params['T_bml'],
                        'P_lit': P_BML_TOP,
                        'Mg#':   params['Mg#_bml']},
            P_top    = P_BML_TOP,
            P_bottom = P_BML_BOTTOM,
            T_bml    = params['T_bml'],
        )
        if bml_raw is not None:
            bml_top_km = khan_cache['lsl_top_depth']
            bml_raw['depth_km'] = (bml_raw['depth_km']
                                   - bml_raw['depth_km'][0] + bml_top_km)

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
        bml_data=bml_data, params=params, use_bml=use_bml)

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


def run_mcmc(chain_id, n_steps, start_params=None, prefix='chain', use_bml=False):
    chain_dir = os.path.join(MCMC_DIR, f"{prefix}_{chain_id:02d}")
    os.makedirs(chain_dir, exist_ok=True)

    load_gravity_profile()
    khan_cache = compute_khan_median()

    rng     = np.random.default_rng(42 + chain_id)
    current = start_params.copy() if start_params else START_PARAMS.copy()

    chain_file = os.path.join(chain_dir, "chain.json")
    chain      = []
    if os.path.exists(chain_file):
        with open(chain_file) as f:
            chain = json.load(f)
        if chain:
            current = chain[-1]['params']
            print(f"Chain {chain_id}: resuming from step {len(chain)}")

    step_start   = len(chain)
    accept_count = 0
    current_components = {
        'tt': 999.0, 'mass': 999.0, 'moi': 999.0, 'solidus': 0.0,
    }

    print(f"\nChain {chain_id} starting (use_bml={use_bml})")
    print(f"  Target steps: {n_steps}")
    print("=" * 60)

    if chain:
        current_misfit     = chain[-1]['misfit']
        accept_count       = sum(1 for s in chain if s.get('accepted', False))
        current_components = {
            'tt':      chain[-1].get('misfit_tt',      999.0),
            'mass':    chain[-1].get('misfit_mass',    999.0),
            'moi':     chain[-1].get('misfit_moi',     999.0),
            'solidus': chain[-1].get('misfit_solidus', 0.0),
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
                      f"Mg#={trial['Mg#']:.3f}")

            current_misfit, _, current_components, _ = forward(
                trial, run_dir, model_name, khan_cache,
                use_bml=use_bml, skip_bml_density_check=True)

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
                'tt': 999.0, 'mass': 999.0, 'moi': 999.0, 'solidus': 0.0,
            }

    print(f"  Initial misfit/datum = {current_misfit:.4f}")

    for step in range(step_start, step_start + n_steps):
        t0         = datetime.now()
        proposed   = propose(current, rng)
        run_dir    = os.path.join(chain_dir, f"step_{step+1:05d}")
        model_name = f"mcmc_{prefix}_c{chain_id:02d}_s{step+1:05d}"

        proposed_misfit, n_data, components, fort56_data = forward(
            proposed, run_dir, model_name, khan_cache, use_bml=use_bml)

        if proposed_misfit is None or proposed_misfit >= 990.0:
            accepted        = False
            proposed_misfit = 999.0
            fort56_data     = None
            components      = {
                'tt': 999.0, 'mass': 999.0, 'moi': 999.0, 'solidus': 0.0,
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

            # 存 profile
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

        elapsed     = (datetime.now() - t0).total_seconds()
        accept_rate = accept_count / (step - step_start + 1) * 100

        print(f"  Step {step+1:4d}: misfit={current_misfit:.4f}  "
              f"{'ACCEPT' if accepted else 'reject'}  "
              f"rate={accept_rate:.1f}%  ({elapsed:.0f}s)")

        chain.append({
            'step':           step + 1,
            'params':         current,
            'misfit':         current_misfit,
            'misfit_tt':      current_components.get('tt',      999.0),
            'misfit_mass':    current_components.get('mass',    999.0),
            'misfit_moi':     current_components.get('moi',     999.0),
            'misfit_solidus': current_components.get('solidus', 0.0),
            'accepted':       bool(accepted),
            'accept_rate':    accept_rate,
        })

        with open(chain_file, 'w') as f:
            json.dump(chain, f, indent=2)
        print(f"  [Saved {prefix}_{chain_id:02d}, total {len(chain)} steps]")

    print(f"\nChain {chain_id} complete!")
    print(f"  Total steps:       {step_start + n_steps}")
    print(f"  Final accept rate: {accept_count/n_steps*100:.1f}%")
    print(f"  Final misfit:      {current_misfit:.4f}")

# ============================================================
# entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain',   type=int,  default=0)
    parser.add_argument('--steps',   type=int,  default=100)
    parser.add_argument('--test',    action='store_true')
    parser.add_argument('--prefix',  type=str,  default='chain')
    parser.add_argument('--use_bml', action='store_true',
                        help='Include basal melt layer (BML) in forward model')
    args = parser.parse_args()

    os.makedirs(MCMC_DIR, exist_ok=True)

    if args.test:
        print("Test mode: running 1 step")
        run_mcmc(chain_id=0, n_steps=1, prefix=args.prefix,
                 use_bml=args.use_bml)
    else:
        run_mcmc(chain_id=args.chain, n_steps=args.steps,
                 prefix=args.prefix, use_bml=args.use_bml)