#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:33:17 2026

@author: chingchen
"""
"""
TauP 走時計算腳本（完整版）
=============================
速度模型拼接：
    地殼 + 核心  <- Khan 2023 1000 個 model 的中位數
    地幔         <- HeFESTo fort.56 輸出

深度計算：
    用 HeFESTo 輸出的 rho 積分，火星 g = 3.35 m/s²

使用方式：
    python compute_traveltime.py --test
    python compute_traveltime.py --run_dir runs/model_000001 --dataset samuel
    python compute_traveltime.py --all --dataset samuel
"""

import os
import glob
import numpy as np
import pandas as pd
import json
import argparse

from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

# ============================================================
# 路徑設定
# ============================================================

RUNS_DIR       = "/Users/chingchen/Desktop/HeFESTo/runs"
KHAN_MODEL_DIR = "/Users/chingchen/Desktop/Lunar/Mars_Khan_2023/LSL_Models"
TAUP_DATA_DIR  = "/opt/anaconda3/lib/python3.9/site-packages/obspy/taup/data"

# 火星常數
G_MARS      = 3.35    # m/s²
MARS_RADIUS = 3389.5  # km

# ============================================================
# 觀測數據誤差（秒）
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

# ============================================================
# Samuel 2023 觀測數據
# ============================================================

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
# Drilleau 2024 觀測數據
# ============================================================

DRILLEAU_DATA = {
    'S0154a': {'delta': 29.9, 'depth': 10.0,
               'S-P': 174.4, 'SS-S': 25.3, 'SSS-S': 35.0},
    'S0173a': {'delta': 30.2, 'depth': 10.0,
               'S-P': 178.8, 'sP-P': 9.43, 'PP-P': 19.9, 'PPP-P': 34.4,
               'sS-S': 13.2, 'SS-S': 24.4, 'SSS-S': 40.5, 'ScS-S': 345.2},
    'S0185a': {'delta': 54.7, 'depth': 10.0,
               'S-P': 327.28, 'pP-P': 4.0, 'PP-P': 22.47, 'PPP-P': 49.3,
               'sS-S': 10.0, 'SS-S': 30.9, 'SSS-S': 55.4, 'ScS-S': 152.3},
    'S0235b': {'delta': 29.2, 'depth': 10.0,
               'S-P': 171.4, 'PP-P': 18.6, 'PPP-P': 32.0,
               'sS-S': 9.2, 'SS-S': 23.2, 'SSS-S': 33.3, 'ScS-S': 343.9},
    'S0325a': {'delta': 41.5, 'depth': 10.0,
               'S-P': 229.3, 'pP-P': 9.8, 'PP-P': 21.1, 'PPP-P': 34.4,
               'sS-S': 13.8, 'SS-S': 26.1, 'SSS-S': 50.3, 'ScS-S': 220.4},
    'S0407a': {'delta': 27.9, 'depth': 10.0,
               'S-P': 170.7, 'pP-P': 6.77, 'PP-P': 23.38,
               'sS-S': 13.3, 'SS-S': 21.1, 'SSS-S': 33.1, 'ScS-S': 370.0},
    'S0409d': {'delta': 30.3, 'depth': 10.0,
               'S-P': 163.2, 'pP-P': 8.3, 'PP-P': 27.6, 'PPP-P': 36.94,
               'sS-S': 8.4, 'SS-S': 20.9, 'SSS-S': 39.8, 'ScS-S': 320.1},
    'S0474a': {'delta': 19.7, 'depth': 10.0,
               'S-P': 121.6, 'PP-P': 13.4, 'PPP-P': 24.8,
               'SS-S': 15.8, 'SSS-S': 32.4},
    'S0484b': {'delta': 30.9, 'depth': 10.0,
               'S-P': 173.1, 'pP-P': 5.5, 'PP-P': 19.73,
               'sS-S': 13.0, 'SS-S': 17.4, 'ScS-S': 322.3},
    'S0784a': {'delta': 30.6, 'depth': 10.0,
               'S-P': 179.3, 'pP-P': 6.5, 'PP-P': 13.7, 'PPP-P': 22.4,
               'sS-S': 7.2, 'SS-S': 19.6, 'SSS-S': 28.0},
    'S0802a': {'delta': 27.9, 'depth': 10.0,
               'S-P': 180.3, 'pP-P': 4.0, 'PP-P': 25.6, 'PPP-P': 33.9,
               'sS-S': 9.3, 'SS-S': 22.4, 'SSS-S': 36.5, 'ScS-S': 387.6},
    'S0809a': {'delta': 29.4, 'depth': 10.0,
               'S-P': 191.95, 'pP-P': 4.5, 'PP-P': 16.25, 'PPP-P': 29.65,
               'sS-S': 8.1, 'SS-S': 23.8, 'SSS-S': 39.3, 'ScS-S': 373.5},
    'S0820a': {'delta': 29.4, 'depth': 10.0,
               'S-P': 174.1, 'PP-P': 21.9, 'PPP-P': 32.1, 'sS-S': 8.5},
    'S0861a': {'delta': 55.1, 'depth': 10.0,
               'S-P': 319.3, 'PP-P': 19.6, 'PPP-P': 47.6, 'SS-S': 41.1},
    'S0864a': {'delta': 29.5, 'depth': 10.0,
               'S-P': 171.4, 'PP-P': 18.0, 'PPP-P': 27.9,
               'sS-S': 17.3, 'SS-S': 26.4},
    'S0916d': {'delta': 29.4, 'depth': 10.0,
               'S-P': 170.8, 'pP-P': 3.9, 'PP-P': 19.3, 'PPP-P': 36.1,
               'SS-S': 19.0, 'SSS-S': 42.9, 'ScS-S': 342.8},
    'S0918a': {'delta': 16.8, 'depth': 10.0,
               'S-P': 102.4, 'PP-P': 12.8, 'PPP-P': 22.5,
               'SS-S': 21.2, 'SSS-S': 35.0},
    'S0976a': {'delta': 143.6, 'depth': 10.0,
               'SS-PP': 854.4, 'SKS-PP': 303.9},
    'S1000a': {'delta': 126.09, 'depth': 0.0,
               'SS-PP': 749.0, 'SKS-PP': 339.3},
    'S1012d': {'delta': 37.6, 'depth': 10.0,
               'S-P': 221.7, 'pP-P': 6.7, 'PP-P': 18.0, 'PPP-P': 27.7,
               'sS-S': 11.0, 'SS-S': 24.5, 'SSS-S': 40.65},
    'S1022a': {'delta': 29.4, 'depth': 10.0,
               'S-P': 177.0, 'pP-P': 9.0, 'PP-P': 19.3, 'PPP-P': 27.5,
               'sS-S': 11.0, 'SS-S': 25.0, 'SSS-S': 55.0},
    'S1039b': {'delta': 30.1, 'depth': 10.0,
               'S-P': 176.1, 'pP-P': 7.8, 'PP-P': 12.3, 'PPP-P': 28.7,
               'SS-S': 14.9},
    'S1048d': {'delta': 30.8, 'depth': 10.0,
               'S-P': 179.19, 'pP-P': 7.49, 'PP-P': 18.05, 'SS-S': 24.3},
    'S1094b': {'delta': 58.5, 'depth': 0.0,
               'S-P': 343.0},
    'S1102a': {'delta': 75.7, 'depth': 10.0,
               'S-P': 443.7},
    'S1133c': {'delta': 29.5, 'depth': 10.0,
               'S-P': 175.3, 'pP-P': 5.8, 'PP-P': 16.7, 'PPP-P': 28.46,
               'sS-S': 9.1, 'SS-S': 17.4, 'SSS-S': 35.4},
    'S1153a': {'delta': 82.2, 'depth': 10.0,
               'S-P': 467.95},
    'S1157a': {'delta': 36.9, 'depth': 10.0,
               'S-P': 215.8, 'PP-P': 18.7, 'sS-S': 14.6, 'SSS-S': 39.6},
    'S1197a': {'delta': 32.2, 'depth': 10.0,
               'S-P': 191.2, 'pP-P': 8.4, 'PP-P': 15.4, 'sS-S': 14.7},
    'S1222a': {'delta': 37.6, 'depth': 10.0,
               'S-P': 216.0, 'PP-P': 28.0, 'PPP-P': 37.8, 'ScS-S': 258.0},
}


# ============================================================
# 1. 讀取 Khan 1000 models，計算地殼和核心的中位數
# ============================================================

_KHAN_MEDIAN_CACHE = None

def compute_khan_median():
    global _KHAN_MEDIAN_CACHE
    if _KHAN_MEDIAN_CACHE is not None:
        return _KHAN_MEDIAN_CACHE

    files = sorted(glob.glob(os.path.join(KHAN_MODEL_DIR, 'Model_*.txt')))
    print(f"讀取 {len(files)} 個 Khan models...")

    crust_z = np.linspace(0, 100, 200)
    core_z  = np.linspace(1500, MARS_RADIUS, 200)

    crust_vp_all  = []
    crust_vs_all  = []
    crust_rho_all = []
    core_vp_all   = []
    core_rho_all  = []
    cmb_depths    = []

    for fpath in files:
        try:
            data  = np.loadtxt(fpath, comments='#')
            depth = data[:, 0]
            Vp    = data[:, 1]
            Vs    = data[:, 2]
            rho   = data[:, 3]

            solid_mask = Vs > 0.01
            core_mask  = Vs < 0.01

            if solid_mask.sum() < 5 or core_mask.sum() < 5:
                continue

            cmb_depths.append(depth[core_mask][0])

            solid_depth = depth[solid_mask]
            solid_Vp    = Vp[solid_mask]
            solid_Vs    = Vs[solid_mask]
            solid_rho   = rho[solid_mask]

            if solid_depth.max() > crust_z.max():
                crust_vp_all.append(np.interp(crust_z, solid_depth, solid_Vp))
                crust_vs_all.append(np.interp(crust_z, solid_depth, solid_Vs))
                crust_rho_all.append(np.interp(crust_z, solid_depth, solid_rho))

            core_depth = depth[core_mask]
            core_Vp    = Vp[core_mask]
            core_rho   = rho[core_mask]

            if core_depth.max() >= core_z.max() * 0.9:
                core_vp_all.append(np.interp(core_z, core_depth, core_Vp))
                core_rho_all.append(np.interp(core_z, core_depth, core_rho))

        except Exception:
            continue

    crust_vp_med  = np.nanmedian(np.array(crust_vp_all),  axis=0)
    crust_vs_med  = np.nanmedian(np.array(crust_vs_all),  axis=0)
    crust_rho_med = np.nanmedian(np.array(crust_rho_all), axis=0)
    core_vp_med   = np.nanmedian(np.array(core_vp_all),   axis=0)
    core_rho_med  = np.nanmedian(np.array(core_rho_all),  axis=0)
    cmb_median    = float(np.median(cmb_depths))

    print(f"  地殼深度範圍：0 – {crust_z.max():.0f} km")
    print(f"  CMB 中位數深度：{cmb_median:.0f} km")
    print(f"  核心 Vp 範圍：{core_vp_med.min():.2f} – {core_vp_med.max():.2f} km/s")

    _KHAN_MEDIAN_CACHE = {
        'crust_z':   crust_z,
        'crust_vp':  crust_vp_med,
        'crust_vs':  crust_vs_med,
        'crust_rho': crust_rho_med,
        'core_z':    core_z,
        'core_vp':   core_vp_med,
        'core_vs':   np.zeros(len(core_z)),
        'core_rho':  core_rho_med,
        'cmb_depth': cmb_median,
    }
    return _KHAN_MEDIAN_CACHE


# ============================================================
# 2. 讀取 fort.56，計算火星深度
# ============================================================

def read_fort56(fort56_path):
    """讀取 fort.56，用 rho 積分算火星真實深度"""

    # 空檔案檢查
    if os.path.getsize(fort56_path) == 0:
        return None

    try:
        with open(fort56_path) as f:
            f.readline()
            cols = f.readline().split()

        if not cols:  # header 是空的
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
    T     = df['T(K)'].values

    dP      = np.diff(P_GPa) * 1e9
    rho_mid = (rho[:-1] + rho[1:]) / 2
    dz      = dP / (rho_mid * 1000 * G_MARS) / 1000
    depth   = np.zeros(len(P_GPa))
    depth[1:] = np.cumsum(dz)

    return {
        'P_GPa':    P_GPa,
        'depth_km': depth,
        'Vp':       Vp,
        'Vs':       Vs,
        'rho':      rho,
        'T':        T,
    }


# ============================================================
# 3. 建立 TauP 速度模型（有快取）
# ============================================================

def build_taup_model_combined(fort56_data, model_name):
    """
    地殼（Khan 中位數）+ HeFESTo 地幔 + 核心（Khan 中位數）
    如果 .npz 已存在，直接讀取不重新建立
    """
    # 快取檢查：如果 .npz 已存在，直接讀取
    npz_path = os.path.join(TAUP_DATA_DIR, f'{model_name}.npz')
    if os.path.exists(npz_path):
        return TauPyModel(model=model_name)

    khan      = compute_khan_median()
    cmb_depth = khan['cmb_depth']

    hef_depth = fort56_data['depth_km']
    hef_Vp    = fort56_data['Vp']
    hef_Vs    = fort56_data['Vs']
    hef_rho   = fort56_data['rho']

    crust_max_depth = khan['crust_z'].max()

    mantle_mask = (hef_depth >= crust_max_depth) & (hef_depth <= cmb_depth)
    man_depth   = hef_depth[mantle_mask]
    man_Vp      = hef_Vp[mantle_mask]
    man_Vs      = hef_Vs[mantle_mask]
    man_rho     = hef_rho[mantle_mask]

    if len(man_depth) == 0:
        raise ValueError("HeFESTo 地幔深度範圍不足")

    nd_path = f"{model_name}.nd"
    with open(nd_path, 'w') as f:
        # 地殼
        for d, vp, vs, r in zip(
                khan['crust_z'], khan['crust_vp'],
                khan['crust_vs'], khan['crust_rho']):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        f.write("mantle\n")

        # HeFESTo 地幔
        for d, vp, vs, r in zip(man_depth, man_Vp, man_Vs, man_rho):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")

        # CMB 不連續面
        last_d   = man_depth[-1]
        last_vp  = man_Vp[-1]
        last_rho = man_rho[-1]
        f.write(f"{last_d:.3f}  {last_vp:.4f}  0.0000  {last_rho:.4f}\n")
        f.write("outer-core\n")

        # 核心（從 CMB 深度開始）
        core_z   = khan['core_z']
        core_vp  = khan['core_vp']
        core_rho = khan['core_rho']
        core_mask = core_z >= last_d
        core_z    = core_z[core_mask]
        core_vp   = core_vp[core_mask]
        core_rho  = core_rho[core_mask]

        f.write(f"{last_d:.3f}  {core_vp[0]:.4f}  0.0000  {core_rho[0]:.4f}\n")
        for d, vp, r in zip(core_z[1:], core_vp[1:], core_rho[1:]):
            f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")

    build_taup_model(nd_path)
    return TauPyModel(model=model_name)


# ============================================================
# 4. 計算差分走時
# ============================================================

def get_differential_times(model, delta_deg, depth_km):
    phases = ['P', 'S', 'pP', 'sP', 'PP', 'PPP', 'SS', 'SSS',
              'sS', 'ScS', 'SKS']
    try:
        arrivals = model.get_travel_times(
            source_depth_in_km=depth_km,
            distance_in_degree=delta_deg,
            phase_list=phases
        )
    except Exception:
        return {}

    times = {}
    for arr in arrivals:
        if arr.name not in times:
            times[arr.name] = arr.time

    def diff(a, b):
        if a in times and b in times:
            return times[a] - times[b]
        return None

    return {
        'S-P':    diff('S',   'P'),
        'pP-P':   diff('pP',  'P'),
        'sP-P':   diff('sP',  'P'),
        'PP-P':   diff('PP',  'P'),
        'PPP-P':  diff('PPP', 'P'),
        'sS-S':   diff('sS',  'S'),
        'SS-S':   diff('SS',  'S'),
        'SSS-S':  diff('SSS', 'S'),
        'ScS-S':  diff('ScS', 'S'),
        'SS-PP':  diff('SS',  'PP'),
        'SKS-PP': diff('SKS', 'PP'),
    }


# ============================================================
# 5. 計算 Misfit（L1 norm）
# ============================================================

def compute_misfit(model, obs_dataset):
    total_misfit  = 0.0
    n_data        = 0
    event_misfits = {}

    for event_name, obs in obs_dataset.items():
        delta = obs['delta']
        depth = obs.get('depth', 10.0)

        pred = get_differential_times(model, delta, depth)
        if not pred:
            continue

        event_misfit = 0.0
        event_n      = 0

        for phase, obs_val in obs.items():
            if phase in ('delta', 'depth') or obs_val is None:
                continue
            if phase not in pred or pred[phase] is None:
                continue
            if phase not in SIGMA:
                continue

            residual = abs(obs_val - pred[phase]) / SIGMA[phase]
            event_misfit += residual
            event_n      += 1

        if event_n > 0:
            total_misfit += event_misfit
            n_data       += event_n
            event_misfits[event_name] = {
                'misfit':   event_misfit,
                'n_phases': event_n,
            }

    return total_misfit, n_data, event_misfits


# ============================================================
# 6. 處理單一 model
# ============================================================

def process_model(run_dir, dataset='samuel'):
    obs_data = SAMUEL_DATA if dataset == 'samuel' else DRILLEAU_DATA

    fort56_path = os.path.join(run_dir, "fort.56")

    # 檔案不存在或是空的，跳過
    if not os.path.exists(fort56_path):
        print(f"  找不到 fort.56，跳過")
        return None
    if os.path.getsize(fort56_path) == 0:
        print(f"  fort.56 是空的（HeFESTo 失敗），跳過")
        return None

    params = {}
    params_path = os.path.join(run_dir, "params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)

    model_name = os.path.basename(run_dir)

    fort56_data = read_fort56(fort56_path)
    if fort56_data is None:
        print(f"  fort.56 讀取失敗，跳過")
        return None

    try:
        taup_model = build_taup_model_combined(fort56_data, model_name)
    except Exception as e:
        print(f"  TauP 模型建立失敗：{e}")
        return None

    misfit, n_data, event_misfits = compute_misfit(taup_model, obs_data)

    result = {
        'run_dir':          run_dir,
        'params':           params,
        'misfit':           misfit,
        'n_data':           n_data,
        'misfit_per_datum': misfit / n_data if n_data > 0 else None,
        'event_misfits':    event_misfits,
        'dataset':          dataset,
    }

    with open(os.path.join(run_dir, f"misfit_{dataset}.json"), 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# 7. 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',    action='store_true',
                        help='用 model_000001 測試')
    parser.add_argument('--run_dir', type=str,
                        help='單一 model 資料夾')
    parser.add_argument('--all',     action='store_true',
                        help='處理所有 model')
    parser.add_argument('--dataset', type=str, default='samuel',
                        choices=['samuel', 'drilleau'])
    args = parser.parse_args()

    if args.test:
        run_dir = os.path.join(RUNS_DIR, "model_000001")
        print(f"測試：{run_dir}\n")
        result = process_model(run_dir, args.dataset)
        if result:
            print(f"\n結果：")
            print(f"  Misfit          = {result['misfit']:.2f}")
            print(f"  觀測數          = {result['n_data']}")
            print(f"  每觀測平均      = {result['misfit_per_datum']:.2f}")
            print(f"\n各事件 misfit：")
            for event, m in sorted(result['event_misfits'].items()):
                print(f"  {event}: {m['misfit']:.2f} ({m['n_phases']} 個震相)")

    elif args.run_dir:
        result = process_model(args.run_dir, args.dataset)
        if result:
            print(f"Misfit = {result['misfit']:.2f} ({result['n_data']} 個觀測)")

    elif args.all:
        model_dirs = sorted([
            os.path.join(RUNS_DIR, d)
            for d in os.listdir(RUNS_DIR)
            if d.startswith("model_")
        ])
        print(f"找到 {len(model_dirs)} 個 model")

        all_results = []
        for i, run_dir in enumerate(model_dirs):
            model_name = os.path.basename(run_dir)

            # 跳過空的 fort.56
            fort56_path = os.path.join(run_dir, "fort.56")
            if not os.path.exists(fort56_path) or \
               os.path.getsize(fort56_path) == 0:
                continue

            # 已有結果就跳過
            result_path = os.path.join(run_dir, f"misfit_{args.dataset}.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    result = json.load(f)
                all_results.append(result)
                print(f"  [{model_name}] 已有結果，跳過  misfit={result['misfit']:.2f}")
                continue

            print(f"  處理 {model_name} ({i+1}/{len(model_dirs)})...")
            result = process_model(run_dir, args.dataset)
            if result:
                all_results.append(result)
                print(f"    misfit = {result['misfit']:.2f}")

        # 儲存所有結果
        summary_path = os.path.join(RUNS_DIR, f"all_misfits_{args.dataset}.json")
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # 印出 Top 10
        all_results.sort(key=lambda x: x.get('misfit_per_datum') or 999)
        print(f"\n總共處理 {len(all_results)} 個 model")
        print(f"結果儲存至 {summary_path}")
        print(f"\nTop 100 最低 misfit：")
        for r in all_results[:100]:
            mid = r['params'].get('model_id', '?')
            print(f"  model_{mid:06d}: {r['misfit_per_datum']:.3f} "
                  f"(T_lit={r['params'].get('T_lit',0):.0f}K "
                  f"P_lit={r['params'].get('P_lit',0):.2f}GPa)")