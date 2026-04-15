"""
MCMC 反演火星地幔溫度和組成
=====================================
算法：Metropolis-Hastings
數據：Samuel 2023 走時數據
正演：HeFESTo + TauP

使用方式：
    python mcmc.py --test              # 測試 1 步
    python mcmc.py --chain 0 --steps 100  # 跑 chain 0，100 步
    python mcmc.py --chain 1 --steps 100  # 跑 chain 1，100 步（同時開不同 terminal）
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
# 路徑設定
# ============================================================

G_MARS        = 3.35
MARS_RADIUS   = 3389.5
T_SURF        = 220.0
P_MAX_GPA     = 22.0

# ============================================================
# 起始點（model_001281，misfit/datum = 0.749）
# ============================================================

START_PARAMS = {
    'T_lit': 1952.92,
    'P_lit': 6.0755,
    'dTdP':  8.8266,
    'Si':    4.0002,
    'Mg':    4.3764,
    'Fe':    1.0730,
    'Ca':    0.5881,
    'Al':    0.3798,
}

FIXED_PARAMS = {
    'Na': 0.2056,
    'Cr': 0.0035,
}

# ============================================================
# Prior 範圍
# ============================================================

PRIOR = {
    'T_lit': (1000.0, 2600.0),
    'P_lit': (1.5,    9.0),
    'dTdP':  (5.0,    20.0),
    'Si':    (3.0,    5.5),
    'Mg':    (2.5,    5.5),
    'Fe':    (0.4,    2.5),
    'Ca':    (0.1,    0.6),
    'Al':    (0.1,    0.8),
}

# ============================================================
# 初始步長（prior 範圍的 10%）
# ============================================================

STEP = {
    'T_lit': 100.0,
    'P_lit': 0.75,
    'dTdP':  1.5,
    'Si':    0.25,
    'Mg':    0.30,
    'Fe':    0.21,
    'Ca':    0.05,
    'Al':    0.07,
}

# ============================================================
# 觀測數據誤差
# ============================================================

SIGMA = {
    'S-P':    10.0,
    'pP-P':    3.0,
    'sP-P':    5.0,
    'PP-P':    8.0,
    'PPP-P':  12.0,
    'sS-S':    5.0,
    'SS-S':    5.0,
    'SSS-S':   8.0,
    'ScS-S':  12.0,
    'SS-PP':  10.0,
    'SKS-PP': 10.0,
}

# Samuel 2023 觀測數據
SAMUEL_DATA = {
    'S0154a': {'delta': 29.9, 'depth': 22.0,
               'SS-S': 25.3, 'SSS-S': 35.0},
    'S0173a': {'delta': 30.9, 'depth': 20.9,
               'S-P': 178.8, 'sP-P': 9.43, 'PP-P': 19.9, 'PPP-P': 34.4,
               'sS-S': 13.2, 'SS-S': 24.4, 'SSS-S': 40.5, 'ScS-S': 345.2},
    'S0235b': {'delta': 30.1, 'depth': 21.2,
               'S-P': 171.4, 'PP-P': 18.6, 'PPP-P': 32.0,
               'sS-S': 9.2, 'SS-S': 23.2, 'SSS-S': 33.3, 'ScS-S': 343.9},
    'S0325a': {'delta': 41.9, 'depth': 30.9,
               'S-P': 229.3, 'pP-P': 9.8, 'PP-P': 21.1, 'PPP-P': 34.4,
               'sS-S': 13.8, 'SS-S': 26.1, 'SSS-S': 50.3, 'ScS-S': 220.4},
    'S0407a': {'delta': 28.7, 'depth': 21.1,
               'S-P': 170.7, 'pP-P': 6.77, 'PP-P': 23.38,
               'sS-S': 13.3, 'SS-S': 21.1, 'SSS-S': 33.1, 'ScS-S': 370.0},
    'S0409d': {'delta': 29.7, 'depth': 26.2,
               'S-P': 163.2, 'pP-P': 8.3, 'PP-P': 27.6, 'PPP-P': 36.94,
               'sS-S': 8.4, 'SS-S': 20.9, 'SSS-S': 39.8, 'ScS-S': 320.1},
    'S0474a': {'delta': 20.5, 'depth': 24.1,
               'S-P': 121.6, 'PP-P': 13.4, 'PPP-P': 24.8,
               'SS-S': 15.8, 'SSS-S': 32.4},
    'S0484b': {'delta': 28.7, 'depth': 21.1,
               'S-P': 173.1, 'pP-P': 5.5, 'PP-P': 19.73,
               'sS-S': 13.0, 'SS-S': 17.4, 'ScS-S': 322.3},
    'S0784a': {'delta': 30.0, 'depth': 12.9,
               'S-P': 179.3, 'pP-P': 6.5, 'PP-P': 13.7, 'PPP-P': 22.4,
               'sS-S': 7.2, 'SS-S': 19.6, 'SSS-S': 28.0},
    'S0802a': {'delta': 28.9, 'depth': 15.8,
               'S-P': 180.3, 'pP-P': 4.0, 'PP-P': 25.6, 'PPP-P': 33.9,
               'sS-S': 9.3, 'SS-S': 22.4, 'SSS-S': 36.5, 'ScS-S': 387.6},
    'S0809a': {'delta': 30.6, 'depth': 12.1,
               'S-P': 191.95, 'pP-P': 4.5, 'PP-P': 16.25, 'PPP-P': 29.65,
               'sS-S': 8.1, 'SS-S': 23.8, 'SSS-S': 39.3, 'ScS-S': 373.5},
    'S0820a': {'delta': 30.1, 'depth': 15.4,
               'S-P': 174.1, 'PP-P': 21.9, 'PPP-P': 32.1, 'sS-S': 8.5},
    'S0861a': {'delta': 55.3, 'depth': 34.3,
               'S-P': 319.3, 'PP-P': 19.6, 'PPP-P': 47.6, 'SS-S': 41.1},
    'S0864a': {'delta': 29.9, 'depth': 23.9,
               'S-P': 171.4, 'PP-P': 18.0, 'PPP-P': 27.9,
               'sS-S': 17.3, 'SS-S': 26.4},
    'S0916d': {'delta': 30.2, 'depth': 11.1,
               'S-P': 170.8, 'pP-P': 3.9, 'PP-P': 19.3, 'PPP-P': 36.1,
               'SS-S': 19.0, 'SSS-S': 42.9, 'ScS-S': 342.8},
    'S0918a': {'delta': 16.9, 'depth': 16.9,
               'S-P': 102.4, 'PP-P': 12.8, 'PPP-P': 22.5,
               'SS-S': 21.2, 'SSS-S': 35.0},
    'S0976a': {'delta': 143.6, 'depth': 13.5,
               'SS-PP': 854.4, 'SKS-PP': 303.9},
    'S1000a': {'delta': 125.9, 'depth': 0.0,
               'SS-PP': 749.0, 'SKS-PP': 339.3},
    'S1094b': {'delta': 58.5,  'depth': 0.0,
               'S-P': 343.0},
    'S1222a': {'delta': 37.3,  'depth': 14.9,
               'S-P': 216.0, 'ScS-S': 258.0},
}

# ============================================================
# Khan 中位數（全域快取）
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

def compute_oxygen(p):
    return (2.0 * p['Si'] + p['Mg'] + p['Fe'] + p['Ca'] +
            1.5 * p['Al'] + 0.5 * p.get('Na', FIXED_PARAMS['Na']) +
            1.5 * p.get('Cr', FIXED_PARAMS['Cr']))

# ============================================================
# HeFESTo 正演
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

def make_pressure_array():
    return np.concatenate([
        np.linspace(1.04,  6.82,  100),
        np.linspace(6.88,  13.97, 100),
        np.linspace(14.03, 22.93, 115),
    ])

def make_temperature_profile(P_array, T_lit, P_lit, dTdP):
    T_start = 800.0  # 地幔頂部溫度（約對應 Moho 溫度）
    return np.where(
        P_array <= P_lit,
        T_start + (T_lit - T_start) * (P_array / P_lit),
        T_lit + dTdP * (P_array - P_lit)
    )


def run_hefesto(params, run_dir):
    """跑 HeFESTo，回傳 fort.56 路徑或 None"""
    os.makedirs(run_dir, exist_ok=True)

    p = {**params, **FIXED_PARAMS}
    O = compute_oxygen(p)

    # 寫 control
    lines = [
        f"0,{P_MAX_GPA:.0f},50,0,0,0,-1,0,0",
        "8,2,4,2", "oxides",
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
    with open(os.path.join(run_dir, "control"), 'w') as f:
        f.write("\n".join(lines))

    # 寫 ad.in
    P_array = make_pressure_array()
    T_array = make_temperature_profile(P_array, p['T_lit'], p['P_lit'], p['dTdP'])
    with open(os.path.join(run_dir, "ad.in"), 'w') as f:
        for P, T in zip(P_array, T_array):
            f.write(f"{P:.6f} 0.000000 {T:.6f}\n")

    # cp main
    main_dst = os.path.join(run_dir, "main")
    if not os.path.exists(main_dst):
        shutil.copy2(HEFESTO_MAIN, main_dst)
        os.chmod(main_dst, 0o755)

    # 執行
    log_path = os.path.join(run_dir, "hefesto.log")
    try:
        with open(log_path, 'w') as log:
            result = subprocess.run(
                ["./main"], cwd=run_dir,
                stdout=log, stderr=log, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    fort56 = os.path.join(run_dir, "fort.56")
    if not os.path.exists(fort56) or os.path.getsize(fort56) == 0:
        return None
    return fort56

# ============================================================
# TauP 正演
# ============================================================

def read_fort56(fort56_path):
    try:
        with open(fort56_path) as f:
            f.readline()
            cols = f.readline().split()
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
    dz    = dP / (rho_m * 1000 * G_MARS) / 1000
    depth = np.zeros(len(P_GPa))
    depth[1:] = np.cumsum(dz)

    return {'depth_km': depth, 'Vp': Vp, 'Vs': Vs, 'rho': rho}

def build_taup(fort56_data, model_name):
    os.makedirs(TAUP_WORK_DIR, exist_ok=True)

    # 🔥 防止 .npz.npz
    model_name = model_name.replace(".npz", "")

    npz_path = os.path.join(TAUP_WORK_DIR, f'{model_name}.npz')
    nd_path  = os.path.join(TAUP_WORK_DIR, f"{model_name}.nd")

    if os.path.exists(npz_path):
        return TauPyModel(model=npz_path)

    khan      = compute_khan_median()
    cmb_depth = khan['cmb_depth']

    hef_depth = fort56_data['depth_km']
    hef_Vp    = fort56_data['Vp']
    hef_Vs    = fort56_data['Vs']
    hef_rho   = fort56_data['rho']

    mantle_mask = (hef_depth >= 100.0) & (hef_depth <= cmb_depth)
    man_depth = hef_depth[mantle_mask]
    man_Vp    = hef_Vp[mantle_mask]
    man_Vs    = hef_Vs[mantle_mask]
    man_rho   = hef_rho[mantle_mask]

    if len(man_depth) == 0:
        raise ValueError("地幔深度範圍不足")

    with open(nd_path, 'w') as f:
        for d, vp, vs, r in zip(khan['crust_z'], khan['crust_vp'],
                                  khan['crust_vs'], khan['crust_rho']):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

        f.write("mantle\n")

        for d, vp, vs, r in zip(man_depth, man_Vp, man_Vs, man_rho):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

        last_d = man_depth[-1]
        last_vp = man_Vp[-1]
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

def compute_misfit(taup_model, obs_dataset):
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
            'S-P':    diff('S','P'),   'pP-P':  diff('pP','P'),
            'sP-P':   diff('sP','P'),  'PP-P':  diff('PP','P'),
            'PPP-P':  diff('PPP','P'), 'sS-S':  diff('sS','S'),
            'SS-S':   diff('SS','S'),  'SSS-S': diff('SSS','S'),
            'ScS-S':  diff('ScS','S'), 'SS-PP': diff('SS','PP'),
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
            n += 1

    return total / n if n > 0 else 999.0, n

# ============================================================
# 完整正演（HeFESTo + TauP + misfit）
# ============================================================

def forward(params, run_dir, model_name):
    """給定參數，回傳 misfit/datum"""
    fort56 = run_hefesto(params, run_dir)
    if fort56 is None:
        return None, None

    fort56_data = read_fort56(fort56)
    if fort56_data is None:
        return None, None

    try:
        taup_model = build_taup(fort56_data, model_name)
    except Exception as e:
        print(f"    TauP 失敗：{e}")
        return None, None

    misfit, n_data = compute_misfit(taup_model, SAMUEL_DATA)
    return misfit, n_data

# ============================================================
# MCMC Metropolis-Hastings
# ============================================================

def propose(current, rng):
    """提議新參數（Gaussian proposal，在 prior 範圍內）"""
    proposed = {}
    for key in PRIOR:
        lo, hi = PRIOR[key]
        while True:
            val = current[key] + rng.normal(0, STEP[key])
            if lo <= val <= hi:
                proposed[key] = val
                break
    return proposed

def run_mcmc(chain_id, n_steps, start_params=None):
    """
    執行單條 MCMC chain

    chain_id : int，chain 編號（0, 1, 2, ...）
    n_steps  : int，總步數
    """
    chain_dir = os.path.join(MCMC_DIR, f"chain_{chain_id:02d}")
    os.makedirs(chain_dir, exist_ok=True)

    # 讀取 Khan 中位數（一次）
    compute_khan_median()

    # 初始化
    rng = np.random.default_rng(42 + chain_id)

    # 起始點：用 start_params 或 START_PARAMS
    current = start_params.copy() if start_params else START_PARAMS.copy()

    # 如果有之前的 chain 記錄，從最後一步繼續
    chain_file = os.path.join(chain_dir, "chain.json")
    chain = []
    if os.path.exists(chain_file):
        with open(chain_file) as f:
            chain = json.load(f)
        if chain:
            current = chain[-1]['params']
            print(f"Chain {chain_id}：從第 {len(chain)} 步繼續")

    step_start = len(chain)
    accept_count = 0

    print(f"\nChain {chain_id} 開始")
    print(f"  起始 misfit：計算中...")
    print(f"  目標步數：{n_steps}")
    print("=" * 60)

    # 計算起始點的 misfit
    if chain:
        current_misfit = chain[-1]['misfit']
    else:
        run_dir    = os.path.join(chain_dir, "step_current")
        model_name = f"mcmc_c{chain_id:02d}_current"
        current_misfit, _ = forward(current, run_dir, model_name)
        if current_misfit is None:
            print("  起始點 HeFESTo 失敗！請換一個起始點")
            return

    print(f"  起始 misfit/datum = {current_misfit:.4f}")

    # MCMC 主迴圈
    for step in range(step_start, step_start + n_steps):
        t0 = datetime.now()

        # 提議新參數
        proposed = propose(current, rng)

        # 計算新 misfit
        run_dir    = os.path.join(chain_dir, f"step_{step+1:05d}")
        model_name = f"mcmc_c{chain_id:02d}_s{step+1:05d}"
        proposed_misfit, n_data = forward(proposed, run_dir, model_name)

        if proposed_misfit is None:
            # HeFESTo 失敗 → reject
            accepted = False
            proposed_misfit = 999.0
        else:
            # Metropolis acceptance criterion
            # misfit 是 L1 norm，likelihood ∝ exp(-misfit)
            delta_misfit = proposed_misfit - current_misfit

            if delta_misfit <= 0:
                accepted = True
            else:
                log_alpha = -delta_misfit * n_data  # L1 norm likelihood
                accepted  = np.log(rng.uniform()) < log_alpha

        if accepted:
            current        = proposed
            current_misfit = proposed_misfit
            accept_count  += 1

        elapsed = (datetime.now() - t0).total_seconds()
        accept_rate = accept_count / (step - step_start + 1) * 100

        print(f"  Step {step+1:4d}: misfit={current_misfit:.4f}  "
              f"{'ACCEPT' if accepted else 'reject'}  "
              f"accept_rate={accept_rate:.1f}%  "
              f"({elapsed:.0f}s)")

        # 記錄
        record = {
            'step':         step + 1,
            'params':       current,
            'misfit':       current_misfit,
            'accepted':     bool(accepted),
            'accept_rate':  accept_rate,
        }
        chain.append(record)

        # 每 10 步存一次
        if (step + 1) % 10 == 0:
            with open(chain_file, 'w') as f:
                json.dump(chain, f, indent=2)
            print(f"  [儲存 chain_{chain_id:02d}，共 {len(chain)} 步]")

    # 最終儲存
    with open(chain_file, 'w') as f:
        json.dump(chain, f, indent=2)

    total_steps = step_start + n_steps
    print(f"\nChain {chain_id} 完成！")
    print(f"  總步數：{total_steps}")
    print(f"  最終 accept rate：{accept_count/(n_steps)*100:.1f}%")
    print(f"  最終 misfit：{current_misfit:.4f}")

# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=int, default=0,
                        help='Chain 編號（0, 1, 2, ...）')
    parser.add_argument('--steps', type=int, default=100,
                        help='步數（預設 100）')
    parser.add_argument('--test', action='store_true',
                        help='只跑 1 步測試')
    args = parser.parse_args()

    os.makedirs(MCMC_DIR, exist_ok=True)

    if args.test:
        print("測試模式：跑 1 步")
        run_mcmc(chain_id=0, n_steps=1)
    else:
        run_mcmc(chain_id=args.chain, n_steps=args.steps)
