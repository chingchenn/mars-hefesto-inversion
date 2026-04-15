"""
計算火星質量和轉動慣量（MoI），加入 misfit 計算
=====================================================
觀測約束：
    平均密度  rho_bar = 3.9350 ± 0.0012 g/cm³
    MoI       I/MR²  = 0.3639 ± 0.0001
    
計算方式：
    地殼 + 地幔：從 HeFESTo fort.56 的 rho(z) 積分
    核心：從 Khan 1000 models 的中位數核心密度積分

使用方式：
    python compute_moi.py --test
    python compute_moi.py --all
    python compute_moi.py --update   # 更新已有的 misfit json，加入 MoI
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import argparse

# ============================================================
# 路徑設定
# ============================================================

RUNS_DIR       = "/Users/chingchen/Desktop/HeFESTo/runs"
KHAN_MODEL_DIR = "/Users/chingchen/Desktop/Lunar/Mars_Khan_2023/LSL_Models"

# 火星常數
G_MARS      = 3.35     # m/s²
MARS_RADIUS = 3389.5   # km
MARS_RADIUS_M = MARS_RADIUS * 1e3  # m

# ============================================================
# 火星觀測約束
# ============================================================

# 平均密度（Rivoldini et al. 2011）
RHO_BAR_OBS   = 3.9350   # g/cm³
RHO_BAR_SIGMA = 0.0012   # g/cm³

# 轉動慣量（Konopliv et al. 2020）
MOI_OBS   = 0.3639    # I/MR²
MOI_SIGMA = 0.0001

# 火星質量（用於轉換）
MARS_MASS = 6.417e23   # kg

# ============================================================
# Khan 核心中位數（全域快取）
# ============================================================

_KHAN_CORE_CACHE = None

def compute_khan_core_median():
    """
    讀取 Khan 1000 models，計算核心密度剖面中位數
    Khan model 格式：col 0: depth(km), col 2: Vs, col 3: rho(g/cm³)
    核心 = Vs < 0.01 的部分
    """
    global _KHAN_CORE_CACHE
    if _KHAN_CORE_CACHE is not None:
        return _KHAN_CORE_CACHE

    files = sorted(glob.glob(os.path.join(KHAN_MODEL_DIR, 'Model_*.txt')))
    print(f"讀取 {len(files)} 個 Khan models 計算核心中位數...")

    # 共同深度格網（核心部分）
    core_z = np.linspace(1500, MARS_RADIUS, 300)

    core_rho_all = []
    cmb_depths   = []

    for fpath in files:
        try:
            data  = np.loadtxt(fpath, comments='#')
            depth = data[:, 0]
            Vs    = data[:, 2]
            rho   = data[:, 3]

            core_mask = Vs < 0.01
            if core_mask.sum() < 5:
                continue

            cmb_depth  = depth[core_mask][0]
            core_depth = depth[core_mask]
            core_rho   = rho[core_mask]

            cmb_depths.append(cmb_depth)

            if core_depth.max() >= core_z.max() * 0.9:
                core_rho_all.append(
                    np.interp(core_z, core_depth, core_rho))

        except Exception:
            continue

    core_rho_med = np.nanmedian(np.array(core_rho_all), axis=0)
    cmb_median   = float(np.median(cmb_depths))

    print(f"  CMB 中位數深度：{cmb_median:.0f} km")
    print(f"  核心密度範圍：{core_rho_med.min():.2f} – {core_rho_med.max():.2f} g/cm³")

    _KHAN_CORE_CACHE = {
        'core_z':   core_z,
        'core_rho': core_rho_med,
        'cmb_depth': cmb_median,
    }
    return _KHAN_CORE_CACHE


# ============================================================
# 計算 MoI 和質量
# ============================================================

def compute_mass_and_moi(fort56_path):
    """
    從 fort.56 讀取地幔密度，加上 Khan 核心中位數，
    計算火星的質量和轉動慣量

    積分公式：
        M   = 4π ∫ ρ(r) r² dr
        I   = (8π/3) ∫ ρ(r) r⁴ dr
        MoI = I / (M * R²)
    """
    # 讀取 fort.56
    if not os.path.exists(fort56_path):
        return None
    if os.path.getsize(fort56_path) == 0:
        return None

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

    P_GPa     = df['P(GPa)'].values
    rho_mantle = df['rho(g/cm^3)'].values

    # 用火星 g 積分算深度
    dP      = np.diff(P_GPa) * 1e9
    rho_mid = (rho_mantle[:-1] + rho_mantle[1:]) / 2
    dz      = dP / (rho_mid * 1000 * G_MARS) / 1000  # km
    depth   = np.zeros(len(P_GPa))
    depth[1:] = np.cumsum(dz)

    # 轉換成半徑（從地心出發）
    # r = R_total - depth（地殼頂部 ≈ MARS_RADIUS）
    r_mantle = MARS_RADIUS - depth  # km

    # Khan 核心
    khan = compute_khan_core_median()
    cmb_depth = khan['cmb_depth']
    core_z    = khan['core_z']
    core_rho  = khan['core_rho']
    r_core    = MARS_RADIUS - core_z  # km，從地心出發

    # ── 合併地幔 + 核心 ──
    # 只取地幔中深度 < CMB 的部分
    mantle_mask = depth <= cmb_depth
    r_m   = r_mantle[mantle_mask][::-1]   # 從地心往外
    rho_m = rho_mantle[mantle_mask][::-1]

    # 核心：從地心（r=0）到 CMB
    r_c   = r_core[::-1]    # 從地心往外
    rho_c = core_rho[::-1]

    # 拼接：核心 + 地幔
    r_all   = np.concatenate([r_c, r_m]) * 1e3     # 轉成 m
    rho_all = np.concatenate([rho_c, rho_m]) * 1e3  # 轉成 kg/m³

    # 數值積分（梯形法）
    r2   = r_all ** 2
    r4   = r_all ** 4

    # M = 4π ∫ ρ r² dr
    M = 4 * np.pi * np.trapz(rho_all * r2, r_all)

    # I = (8π/3) ∫ ρ r⁴ dr
    I = (8 * np.pi / 3) * np.trapz(rho_all * r4, r_all)

    # 平均密度
    V       = (4/3) * np.pi * (MARS_RADIUS_M ** 3)
    rho_bar = M / V / 1000  # g/cm³

    # MoI
    moi = I / (M * MARS_RADIUS_M ** 2)

    return {
        'mass_kg':  M,
        'rho_bar':  rho_bar,
        'moi':      moi,
    }


# ============================================================
# 計算 MoI + 質量的 misfit
# ============================================================

def compute_geophysical_misfit(fort56_path):
    """
    計算地球物理約束的 misfit（L1 norm）：
        S_geo = |rho_bar_pred - rho_bar_obs| / sigma_rho
              + |MoI_pred - MoI_obs| / sigma_MoI
    """
    result = compute_mass_and_moi(fort56_path)
    if result is None:
        return None

    rho_bar = result['rho_bar']
    moi     = result['moi']

    # L1 norm
    misfit_rho = abs(rho_bar - RHO_BAR_OBS) / RHO_BAR_SIGMA
    misfit_moi = abs(moi     - MOI_OBS)     / MOI_SIGMA

    return {
        'rho_bar_pred':  rho_bar,
        'moi_pred':      moi,
        'misfit_rho':    misfit_rho,
        'misfit_moi':    misfit_moi,
        'misfit_geo':    misfit_rho + misfit_moi,
    }


# ============================================================
# 更新已有的 misfit json，加入 MoI 和質量
# ============================================================

def update_all_misfits(dataset='samuel'):
    """
    讀取已有的 all_misfits_samuel.json，
    對每個 model 加入 MoI 和質量的 misfit，
    輸出 all_misfits_combined.json
    """
    summary_path = os.path.join(RUNS_DIR, f"all_misfits_{dataset}.json")
    if not os.path.exists(summary_path):
        print(f"找不到 {summary_path}，請先跑 compute_traveltime.py --all")
        return

    with open(summary_path) as f:
        all_results = json.load(f)

    print(f"讀取 {len(all_results)} 個 model 的走時 misfit...")

    # 計算核心中位數（一次）
    compute_khan_core_median()

    updated = []
    for result in all_results:
        run_dir     = result['run_dir']
        fort56_path = os.path.join(run_dir, "fort.56")

        geo = compute_geophysical_misfit(fort56_path)
        if geo is None:
            continue

        # 合併 misfit
        tt_misfit  = result.get('misfit', 0)
        n_data     = result.get('n_data', 1)
        geo_misfit = geo['misfit_geo']

        # 總 misfit = 走時 misfit + MoI misfit（各自 normalized）
        total_misfit = tt_misfit / n_data + geo_misfit

        result['rho_bar_pred']    = geo['rho_bar_pred']
        result['moi_pred']        = geo['moi_pred']
        result['misfit_rho']      = geo['misfit_rho']
        result['misfit_moi']      = geo['misfit_moi']
        result['misfit_geo']      = geo_misfit
        result['misfit_tt_norm']  = tt_misfit / n_data
        result['misfit_total']    = total_misfit

        updated.append(result)

        model_name = os.path.basename(run_dir)
        print(f"  {model_name}: "
              f"tt={tt_misfit/n_data:.3f}  "
              f"rho={geo['rho_bar_pred']:.4f}g/cm³(Δ={geo['misfit_rho']:.1f}σ)  "
              f"MoI={geo['moi_pred']:.4f}(Δ={geo['misfit_moi']:.1f}σ)  "
              f"total={total_misfit:.3f}")

    # 排序
    updated.sort(key=lambda x: x.get('misfit_total', 999))

    # 儲存
    out_path = os.path.join(RUNS_DIR, "all_misfits_combined.json")
    with open(out_path, 'w') as f:
        json.dump(updated, f, indent=2)

    print(f"\n總共處理 {len(updated)} 個 model")
    print(f"結果儲存至 {out_path}")
    print(f"\nTop 10 最低總 misfit（走時 + MoI + 質量）：")
    for r in updated[:10]:
        mid = r['params'].get('model_id', '?')
        print(f"  model_{mid:06d}: total={r['misfit_total']:.3f}  "
              f"tt={r['misfit_tt_norm']:.3f}  "
              f"rho={r['rho_bar_pred']:.4f}  "
              f"MoI={r['moi_pred']:.4f}")

    return updated


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',   action='store_true',
                        help='用 model_000001 測試')
    parser.add_argument('--all',    action='store_true',
                        help='計算所有 model 的 MoI 和質量')
    parser.add_argument('--update', action='store_true',
                        help='更新已有 misfit json，加入 MoI 約束')
    parser.add_argument('--dataset', type=str, default='samuel',
                        choices=['samuel', 'drilleau'])
    args = parser.parse_args()

    if args.test:
        run_dir     = os.path.join(RUNS_DIR, "model_000001")
        fort56_path = os.path.join(run_dir, "fort.56")
        print(f"測試：{run_dir}\n")

        compute_khan_core_median()

        geo = compute_geophysical_misfit(fort56_path)
        if geo:
            print(f"\n地球物理約束結果：")
            print(f"  預測平均密度：{geo['rho_bar_pred']:.4f} g/cm³")
            print(f"  觀測平均密度：{RHO_BAR_OBS:.4f} ± {RHO_BAR_SIGMA:.4f} g/cm³")
            print(f"  密度 misfit：{geo['misfit_rho']:.2f} σ")
            print(f"")
            print(f"  預測 MoI：{geo['moi_pred']:.5f}")
            print(f"  觀測 MoI：{MOI_OBS:.5f} ± {MOI_SIGMA:.5f}")
            print(f"  MoI misfit：{geo['misfit_moi']:.2f} σ")
            print(f"")
            print(f"  地球物理總 misfit：{geo['misfit_geo']:.2f}")

    elif args.update or args.all:
        update_all_misfits(args.dataset)
