"""
13_mcmc_4p_v5.py
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
# 常數
# ============================================================

MARS_RADIUS = 3389.5
T_SURF      = 220.0
P_MAX_GPA   = 22.0
GAMMA       = 1.1    # Grüneisen parameter（火星地幔平均值）
MARS_MASS_OBS   = 6.4171e23
MARS_MASS_SIGMA = MARS_MASS_OBS * 0.01
MOI_OBS         = 0.3634
MOI_SIGMA       = 0.0006
MARS_RADIUS_M   = MARS_RADIUS * 1000  # 3389500 m

# ============================================================
# 重力剖面
# ============================================================

_GRAVITY_DEPTH = None
_GRAVITY_G     = None

def load_gravity_profile():
    global _GRAVITY_DEPTH, _GRAVITY_G
    if _GRAVITY_DEPTH is not None:
        return

    rho_path = SAMUEL_RHO_PROFILE_PATH
    try:
        data    = np.loadtxt(rho_path)
        rho_all = data[:, 0]
        r_all   = data[:, 1]
    except Exception as e:
        print(f"警告：無法讀取 rho_profile.dat ({e})，使用固定 g=3.45")
        _GRAVITY_DEPTH = np.array([0.0, 1600.0])
        _GRAVITY_G     = np.array([3.45, 3.45])
        return

    idx = np.argsort(r_all)
    rho = rho_all[idx]
    r   = r_all[idx] * 1000

    G_CONST = 6.674e-11
    M_enc   = np.zeros(len(r))
    for i in range(1, len(r)):
        dr      = r[i] - r[i-1]
        rho_mid = (rho[i] + rho[i-1]) / 2
        r_mid   = (r[i]   + r[i-1])   / 2
        M_enc[i] = M_enc[i-1] + 4 * np.pi * rho_mid * r_mid**2 * dr

    g = np.zeros(len(r))
    g[1:] = G_CONST * M_enc[1:] / r[1:]**2

    depth_km = (r[-1] - r) / 1000
    sort_idx       = np.argsort(depth_km)
    _GRAVITY_DEPTH = depth_km[sort_idx]
    _GRAVITY_G     = g[sort_idx]
    print(f"  重力剖面載入：地表 {_GRAVITY_G[0]:.3f} m/s²，"
          f"CMB (~1533 km) {np.interp(1533, _GRAVITY_DEPTH, _GRAVITY_G):.3f} m/s²")


def gravity_mars(depth_km):
    if _GRAVITY_DEPTH is None:
        load_gravity_profile()
    return np.interp(depth_km, _GRAVITY_DEPTH, _GRAVITY_G)

# ============================================================
# compsition 
# ============================================================

YM_BASE = {
    'Si': 4.01931, 'Mg': 4.08235, 'Fe': 1.08599,
    'Ca': 0.27259, 'Al': 0.37376, 'Na': 0.10105, 'Cr': 0.06146,
}
MGFE_TOTAL = YM_BASE['Mg'] + YM_BASE['Fe']

START_PARAMS = {
    'T_lit': 1952.92,
    'P_lit': 6.0755,
    'Mg#':   YM_BASE['Mg'] / (YM_BASE['Mg'] + YM_BASE['Fe']),
}

FIXED_PARAMS = {
    'Si': YM_BASE['Si'], 'Ca': YM_BASE['Ca'], 'Al': YM_BASE['Al'],
    'Na': YM_BASE['Na'], 'Cr': YM_BASE['Cr'],
}

PRIOR = {
    'T_lit': (1000.0, 2600.0),
    'P_lit': (1.5,    9.0),
    'Mg#':   (0.50,   0.86),
}

STEP = {
    'T_lit': 60.0,
    'P_lit': 0.3,
    'Mg#':   0.015,
}

SIGMA = {
    'S-P': 10.0, 'pP-P': 3.0, 'sP-P': 5.0, 'PP-P': 8.0, 'PPP-P': 12.0,
    'sS-S': 5.0, 'SS-S': 5.0, 'SSS-S': 8.0, 'ScS-S': 12.0,
    'SS-PP': 10.0, 'SKS-PP': 10.0,
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
# Khan 中位數
# ============================================================

_KHAN_CACHE = None

def compute_khan_median():
    global _KHAN_CACHE
    if _KHAN_CACHE is not None:
        return _KHAN_CACHE

    files = sorted(glob.glob(os.path.join(KHAN_MODEL_DIR, 'Model_*.txt')))
    print(f"讀取 {len(files)} 個 Khan models...")

    crust_z = np.linspace(0, 100, 200)
    core_z  = np.linspace(1500, MARS_RADIUS, 200)

    crust_vp_all = []; crust_vs_all = []; crust_rho_all = []
    core_vp_all  = []; core_rho_all = []
    cmb_depths   = []

    for fpath in files:
        try:
            data  = np.loadtxt(fpath, comments='#')
            depth = data[:, 0]; Vp = data[:, 1]
            Vs    = data[:, 2]; rho = data[:, 3]
            solid_mask = Vs > 0.01
            core_mask  = Vs < 0.01
            if solid_mask.sum() < 5 or core_mask.sum() < 5:
                continue
            cmb_depths.append(depth[core_mask][0])
            sd = depth[solid_mask]; svp = Vp[solid_mask]
            svs = Vs[solid_mask];   sr  = rho[solid_mask]
            if sd.max() > crust_z.max():
                crust_vp_all.append(np.interp(crust_z, sd, svp))
                crust_vs_all.append(np.interp(crust_z, sd, svs))
                crust_rho_all.append(np.interp(crust_z, sd, sr))
            cd = depth[core_mask]; cvp = Vp[core_mask]; cr = rho[core_mask]
            if cd.max() >= core_z.max() * 0.9:
                core_vp_all.append(np.interp(core_z, cd, cvp))
                core_rho_all.append(np.interp(core_z, cd, cr))
        except Exception:
            continue

    cmb_median = float(np.median(cmb_depths))
    print(f"  CMB 中位數深度：{cmb_median:.0f} km")
    _KHAN_CACHE = {
        'crust_z':   crust_z,
        'crust_vp':  np.nanmedian(crust_vp_all,  axis=0),
        'crust_vs':  np.nanmedian(crust_vs_all,  axis=0),
        'crust_rho': np.nanmedian(crust_rho_all, axis=0),
        'core_z':    core_z,
        'core_vp':   np.nanmedian(core_vp_all,  axis=0),
        'core_vs':   np.zeros(len(core_z)),
        'core_rho':  np.nanmedian(core_rho_all, axis=0),
        'cmb_depth': cmb_median,
    }
    return _KHAN_CACHE

# ============================================================
# 電荷平衡
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

def run_hefesto_single(run_dir, control_lines, ad_in_content=None, timeout=600):
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


def make_adiabatic_profile_from_fort56(fort56_path, P_lit, T_lit,
                                        P_max=P_MAX_GPA, n_points=50):
    """
    從 Step1 的 fort.56 讀出 K_S = rho * Vp²，
    用 dT/dP = γ·T / K_S 積分絕熱梯度。
    比 NPS ensemble 快很多，不需要額外跑 HeFESTo。
    """
    try:
        with open(fort56_path) as f:
            f.readline(); cols = f.readline().split()
        df = pd.read_csv(fort56_path, sep=r'\s+', skiprows=2, names=cols)
        if df.empty:
            return None, None

        P_col   = df.columns[0]   # P(GPa)
        rho_col = df.columns[3]   # rho(g/cm^3)
        Vp_col  = df.columns[6]   # VP(km/s)

        idx    = (df[P_col] - P_lit).abs().idxmin()
        rho_0  = float(df[rho_col].iloc[idx]) * 1000   # kg/m³
        Vp_0   = float(df[Vp_col].iloc[idx])  * 1000   # m/s
        K_S_0  = rho_0 * Vp_0**2 / 1e9                 # GPa

        print(f"    K_S at P_lit: {K_S_0:.1f} GPa  "
              f"→ dT/dP|adiab ≈ {GAMMA * T_lit / K_S_0:.1f} K/GPa")

    except Exception as e:
        print(f"    make_adiabatic_profile 失敗: {e}")
        return None, None

    # 積分
    P_adiab = np.linspace(P_lit, P_max, n_points)
    T_adiab = np.zeros(n_points)
    T_adiab[0] = T_lit
    for i in range(1, n_points):
        dP = P_adiab[i] - P_adiab[i-1]
        dTdP = GAMMA * T_adiab[i-1] / K_S_0   # 用 P_lit 處的 K_S 近似
        T_adiab[i] = T_adiab[i-1] + dTdP * dP

    return P_adiab, T_adiab


def run_hefesto(params, run_dir):
    """
    Step1 (NPT, P_lit, T_lit): 取得 K_S
    用 K_S 積分絕熱溫度剖面
    Step2 (NPT with ad.in): 用完整 T profile 跑最終計算
    """
    p = composition_from_params(params)
    O = compute_oxygen(p)
    T_lit = p['T_lit']
    P_lit = p['P_lit']

    # ── Step 1: NPT 在 P_lit, T_lit ──
    dir1  = os.path.join(run_dir, "s1_npt")
    line1 = f"{P_lit:.4f},{P_lit:.4f},1,{T_lit:.2f},{T_lit:.2f},0,0,0,0"
    fort56_1 = run_hefesto_single(dir1, make_control_lines(p, O, line1))
    if fort56_1 is None:
        print("    Step1 失敗")
        return None

    # ── 從 fort.56 積分絕熱溫度 ──
    P_adiab, T_adiab = make_adiabatic_profile_from_fort56(
        fort56_1, P_lit, T_lit)
    if P_adiab is None:
        return None

    # ── 合併：傳導段 + 絕熱段 ──
    P_cond = np.linspace(1.04, P_lit, 100)
    T_cond = T_SURF + (T_lit - T_SURF) * (P_cond / P_lit)

    P_full = np.concatenate([P_cond, P_adiab])
    T_full = np.concatenate([T_cond, T_adiab])
    sort_idx        = np.argsort(P_full)
    P_full, T_full  = P_full[sort_idx], T_full[sort_idx]
    _, uniq         = np.unique(P_full, return_index=True)
    P_full, T_full  = P_full[uniq], T_full[uniq]

    # ── Step 2 (原 Step3): 最終計算 ──
    dir2  = os.path.join(run_dir, "s2_final")
    line2 = f"0,{P_MAX_GPA:.0f},50,0,0,0,-1,0,0"
    ad_in = "".join(f"{P:.6f} 0.000000 {T:.6f}\n"
                    for P, T in zip(P_full, T_full))
    return run_hefesto_single(dir2, make_control_lines(p, O, line2),
                              ad_in_content=ad_in)

# ============================================================
# TauP
# ============================================================

def read_fort56(fort56_path):
    try:
        with open(fort56_path) as f:
            f.readline(); cols = f.readline().split()
        if not cols:
            return None
        df = pd.read_csv(fort56_path, sep=r'\s+', skiprows=2, names=cols)
        if df.empty:
            return None
    except Exception:
        return None

    P_GPa = df['P(GPa)'].values
    rho   = df['rho(g/cm^3)'].values
    Vs    = df['VS(km/s)'].values
    Vp    = df['VP(km/s)'].values

    dP    = np.diff(P_GPa) * 1e9
    rho_m = (rho[:-1] + rho[1:]) / 2
    depth = np.zeros(len(P_GPa))
    for i in range(len(dP)):
        g_i      = gravity_mars(depth[i])
        rho_si   = rho_m[i] * 1000
        dz_i     = dP[i] / (rho_si * g_i) / 1000
        depth[i+1] = depth[i] + dz_i

    return {'depth_km': depth, 'Vp': Vp, 'Vs': Vs, 'rho': rho}


def build_taup(fort56_data, model_name, khan_cache):
    os.makedirs(TAUP_WORK_DIR, exist_ok=True)
    model_name = model_name.replace(".npz", "")
    npz_path   = os.path.join(TAUP_WORK_DIR, f'{model_name}.npz')
    nd_path    = os.path.join(TAUP_WORK_DIR, f"{model_name}.nd")
    if os.path.exists(npz_path):
        return TauPyModel(model=npz_path)

    khan      = khan_cache 
    cmb_depth = khan['cmb_depth']
    hef_depth = fort56_data['depth_km']
    hef_Vp    = fort56_data['Vp']
    hef_Vs    = fort56_data['Vs']
    hef_rho   = fort56_data['rho']

    mantle_mask = (hef_depth >= 100.0) & (hef_depth <= cmb_depth)
    man_depth   = hef_depth[mantle_mask]
    man_Vp      = hef_Vp[mantle_mask]
    man_Vs      = hef_Vs[mantle_mask]
    man_rho     = hef_rho[mantle_mask]
    if len(man_depth) == 0:
        raise ValueError("地幔深度範圍不足")

    with open(nd_path, 'w') as f:
        for d, vp, vs, r in zip(khan['crust_z'], khan['crust_vp'],
                                 khan['crust_vs'], khan['crust_rho']):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        f.write("mantle\n")
        for d, vp, vs, r in zip(man_depth, man_Vp, man_Vs, man_rho):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        last_d   = man_depth[-1]
        last_vp  = man_Vp[-1]
        last_rho = man_rho[-1]
        f.write(f"{last_d:.3f}  {last_vp:.4f}  0.0000  {last_rho:.4f}\n")
        f.write("outer-core\n")
        core_z   = khan['core_z']
        core_vp  = khan['core_vp']
        core_rho = khan['core_rho']
        mask     = core_z >= last_d
        core_z   = core_z[mask]
        core_vp  = core_vp[mask]
        core_rho = core_rho[mask]
        f.write(f"{last_d:.3f}  {core_vp[0]:.4f}  0.0000  {core_rho[0]:.4f}\n")
        for d, vp, r in zip(core_z[1:], core_vp[1:], core_rho[1:]):
            f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")

    build_taup_model(nd_path, output_folder=TAUP_WORK_DIR)
    return TauPyModel(model=npz_path)
# ============================================================
# Moment of Interia
# ============================================================

def compute_mass_and_moi(fort56_data, khan_cache):
    """
    從 HeFESTo 地幔密度 + Khan 地殼/核密度
    積分計算 Total Mass 和 I/MR²
    """
    R = MARS_RADIUS_M
    cmb_depth_km = khan_cache['cmb_depth']

    # ── 地殼（Khan median, 0~100 km）──
    crust_z   = khan_cache['crust_z']       # km, 從地表往下
    crust_rho = khan_cache['crust_rho']     # g/cm³
    crust_mask = crust_z <= 100.0
    crust_r   = (R - crust_z[crust_mask] * 1000)   # m，從球心算
    crust_rho_si = crust_rho[crust_mask] * 1000     # kg/m³

    # ── 地幔（HeFESTo, 100 km ~ CMB）──
    hef_depth = fort56_data['depth_km']
    hef_rho   = fort56_data['rho']
    mantle_mask = (hef_depth >= 100.0) & (hef_depth <= cmb_depth_km)
    man_r     = (R - hef_depth[mantle_mask] * 1000)
    man_rho_si = hef_rho[mantle_mask] * 1000

    # ── 核（Khan median）──
    core_z   = khan_cache['core_z']         # km，從地表往下
    core_rho = khan_cache['core_rho']
    core_mask = core_z >= cmb_depth_km
    core_r   = (R - core_z[core_mask] * 1000)
    core_rho_si = core_rho[core_mask] * 1000

    # ── 合併，從球心（r小）到地表（r大）排序 ──
    all_r   = np.concatenate([core_r, man_r, crust_r])
    all_rho = np.concatenate([core_rho_si, man_rho_si, crust_rho_si])
    sort_idx = np.argsort(all_r)
    all_r   = all_r[sort_idx]
    all_rho = all_rho[sort_idx]

    # ── 積分 ──
    M   = 4 * np.pi * np.trapezoid(all_rho * all_r**2, all_r)
    I   = (8 * np.pi / 3) * np.trapezoid(all_rho * all_r**4, all_r)
    moi = I / (M * R**2)

    return M, moi

def compute_misfit(taup_model, obs_dataset, fort56_data, khan_cache):
    total = 0.0; n = 0
    phases = ['P','S','pP','sP','PP','PPP','SS','SSS','sS','ScS','SKS']
    for event, obs in obs_dataset.items():
        delta = obs['delta']; depth = obs.get('depth', 10.0)
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
        def diff(a, b):
            return times[a] - times[b] if a in times and b in times else None
        pred = {
            'S-P': diff('S','P'), 'pP-P': diff('pP','P'),
            'sP-P': diff('sP','P'), 'PP-P': diff('PP','P'),
            'PPP-P': diff('PPP','P'), 'sS-S': diff('sS','S'),
            'SS-S': diff('SS','S'), 'SSS-S': diff('SSS','S'),
            'ScS-S': diff('ScS','S'), 'SS-PP': diff('SS','PP'),
            'SKS-PP': diff('SKS','PP'),
        }
        for phase, obs_val in obs.items():
            if phase in ('delta','depth') or obs_val is None:
                continue
            if phase not in pred or pred[phase] is None:
                continue
            if phase not in SIGMA:
                continue
            total += abs(obs_val - pred[phase]) / SIGMA[phase]
            n     += 1
    M_pred, moi_pred = compute_mass_and_moi(fort56_data, khan_cache)

    mass_misfit = abs(MARS_MASS_OBS - M_pred) / MARS_MASS_SIGMA
    moi_misfit  = abs(MOI_OBS - moi_pred) / MOI_SIGMA

    total += mass_misfit
    total += moi_misfit
    n     += 2

    print(f"    Mass={M_pred:.4e} kg (obs={MARS_MASS_OBS:.4e}), "
          f"misfit={mass_misfit:.3f}")
    print(f"    MoI={moi_pred:.4f} (obs={MOI_OBS:.4f}), "
          f"misfit={moi_misfit:.3f}")

    return (total / n if n > 0 else 999.0), n
# ============================================================
# 正演
# ============================================================

def forward(params, run_dir, model_name, khan_cache):
    fort56 = run_hefesto(params, run_dir)
    if fort56 is None:
        return None, None
    fort56_data = read_fort56(fort56)
    if fort56_data is None:
        shutil.rmtree(run_dir, ignore_errors=True)
        return None, None
    try:
        taup_model = build_taup(fort56_data, model_name, khan_cache)
    except Exception as e:
        print(f"    TauP 失敗：{e}")
        shutil.rmtree(run_dir, ignore_errors=True)
        return None, None
    misfit, n_data = compute_misfit(taup_model, SAMUEL_DATA, fort56_data, khan_cache)
    shutil.rmtree(run_dir, ignore_errors=True)
    return misfit, n_data

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

    chain_file = os.path.join(chain_dir, "chain.json")
    chain      = []
    if os.path.exists(chain_file):
        with open(chain_file) as f:
            chain = json.load(f)
        if chain:
            current = chain[-1]['params']
            print(f"Chain {chain_id}：從第 {len(chain)} 步繼續")

    step_start   = len(chain)
    accept_count = 0

    print(f"\nChain {chain_id} 開始")
    print(f"  目標步數：{n_steps}")
    print("=" * 60)

    if chain:
        current_misfit = chain[-1]['misfit']
    else:
        run_dir    = os.path.join(chain_dir, "step_current")
        model_name = f"mcmc_c{chain_id:02d}_current"
        current_misfit, _ = forward(current, run_dir, model_name, khan_cache)
        if current_misfit is None:
            print("  起始點 HeFESTo 失敗！")
            return

    print(f"  起始 misfit/datum = {current_misfit:.4f}")

    for step in range(step_start, step_start + n_steps):
        t0 = datetime.now()
        proposed        = propose(current, rng)
        run_dir         = os.path.join(chain_dir, f"step_{step+1:05d}")
        model_name      = f"mcmc_c{chain_id:02d}_s{step+1:05d}"
        proposed_misfit, n_data = forward(proposed, run_dir, model_name, khan_cache)

        if proposed_misfit is None:
            accepted        = False
            proposed_misfit = 999.0
        else:
            delta_misfit = proposed_misfit - current_misfit
            if delta_misfit <= 0:
                accepted = True
            else:
                log_alpha = -delta_misfit * n_data
                accepted  = np.log(rng.uniform()) < log_alpha

        if accepted:
            current        = proposed
            current_misfit = proposed_misfit
            accept_count  += 1

        elapsed     = (datetime.now() - t0).total_seconds()
        accept_rate = accept_count / (step - step_start + 1) * 100

        print(f"  Step {step+1:4d}: misfit={current_misfit:.4f}  "
              f"{'ACCEPT' if accepted else 'reject'}  "
              f"rate={accept_rate:.1f}%  ({elapsed:.0f}s)")

        chain.append({
            'step':        step + 1,
            'params':      current,
            'misfit':      current_misfit,
            'accepted':    bool(accepted),
            'accept_rate': accept_rate,
        })

        if (step + 1) % 10 == 0:
            with open(chain_file, 'w') as f:
                json.dump(chain, f, indent=2)
            print(f"  [儲存 {prefix}_{chain_id:02d}，共 {len(chain)} 步]")

    with open(chain_file, 'w') as f:
        json.dump(chain, f, indent=2)

    print(f"\nChain {chain_id} 完成！")
    print(f"  總步數：{step_start + n_steps}")
    print(f"  最終 accept rate：{accept_count/n_steps*100:.1f}%")
    print(f"  最終 misfit：{current_misfit:.4f}")

# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain',      type=int,  default=0)
    parser.add_argument('--steps',      type=int,  default=100)
    parser.add_argument('--test',       action='store_true')
    parser.add_argument('--prefix',     type=str,  default='chain')
    parser.add_argument('--verify_moi', action='store_true',
                        help='測試 Mass 和 MoI 計算是否正確')
    args = parser.parse_args()

    os.makedirs(MCMC_DIR, exist_ok=True)

    if args.verify_moi:
        print("=" * 55)
        print("驗證 Mass 和 MoI 計算")
        print("=" * 55)
        load_gravity_profile()
        khan = compute_khan_median()

        # 用 START_PARAMS 跑一次 HeFESTo
        print(f"\n起始參數：{START_PARAMS}")
        test_dir = os.path.join(MCMC_DIR, "verify_moi_test")
        fort56 = run_hefesto(START_PARAMS, test_dir)

        if fort56 is None:
            print("HeFESTo 失敗，無法驗證")
        else:
            fort56_data = read_fort56(fort56)
            if fort56_data is None:
                print("fort.56 讀取失敗")
            else:
                M, moi = compute_mass_and_moi(fort56_data, khan)
                print(f"\n結果：")
                print(f"  Mass = {M:.4e} kg")
                print(f"  Mass obs = {MARS_MASS_OBS:.4e} kg")
                print(f"  Mass 差異 = {abs(M - MARS_MASS_OBS)/MARS_MASS_OBS*100:.2f}%")
                print(f"")
                print(f"  MoI  = {moi:.5f}")
                print(f"  MoI obs = {MOI_OBS:.5f}")
                print(f"  MoI 差異 = {abs(moi - MOI_OBS):.5f}")
                print(f"")
                if abs(moi - MOI_OBS) < 0.01:
                    print("  ✓ MoI 在合理範圍內")
                else:
                    print("  ✗ MoI 偏差過大，請檢查密度積分")
                if abs(M - MARS_MASS_OBS) / MARS_MASS_OBS < 0.05:
                    print("  ✓ Mass 在合理範圍內")
                else:
                    print("  ✗ Mass 偏差過大，請檢查密度積分")

    elif args.test:
        print("測試模式：跑 1 步")
        run_mcmc(chain_id=0, n_steps=1, prefix=args.prefix)

    else:
        run_mcmc(chain_id=args.chain, n_steps=args.steps, prefix=args.prefix)
