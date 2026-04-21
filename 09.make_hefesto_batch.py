#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:52:33 2026

@author: chingchen
"""

"""
HeFESTo 批次執行腳本（最終版）
================================
參數空間：
    T_lit  : 岩石圈底部溫度 (K)
    P_lit  : 岩石圈底部壓力 (GPa)
    dTdP   : 絕熱梯度 (K/GPa)
    組成   : Mg, Fe, Ca, Al, Na, Cr（Si 由電荷平衡導出）

資料結構：
    runs/
    ├── model_000001/
    │   ├── main     ← cp 自 HEFESTO_MAIN
    │   ├── control  ← 每個 model 不同
    │   ├── ad.in    ← 每個 model 不同
    │   └── fort.56  ← HeFESTo 輸出

使用方式：
    python run_hefesto_batch.py --test        # 先測試 1 個 model
    python run_hefesto_batch.py --n 1000      # 跑 1000 個 model
    python run_hefesto_batch.py --start 500   # 從第 500 個續跑
"""

import os
import shutil
from config import *
import subprocess
import numpy as np
import json
from datetime import datetime
from multiprocessing import Pool

# ============================================================
# Path
# ============================================================

HEFESTO_MAIN = HEFESTO_MAIN
PAR_DIR      = PAR_DIR
RUNS_DIR     = RUNS_DIR

# ============================================================
# 先驗範圍
# ============================================================

T_SURF = 220.0  # K，火星地表溫度，固定

PRIOR = {
    'T_lit': (1000.0, 2000.0),   # K
    'P_lit': (1.5,    9.0),      # GPa，對應深度約 100-500 km
    'dTdP':  (5.0,    20.0),     # K/GPa，涵蓋傳導到對流
    'Mg':    (2.5,    5.5),      # molar amount
    'Fe':    (0.4,    2.5),
    'Ca':    (0.1,    0.6),
    'Al':    (0.1,    0.8),
    'Na':    (0.0,    0.3),
    'Cr':    (0.0,    0.15),
}

P_MAX_GPA = 22.0  # 火星 CMB 壓力

# ============================================================
# 相位列表（從您的 control 複製，固定不變）
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

# ============================================================
# 1. 抽樣參數
# ============================================================

def sample_parameters(n_samples, seed=42):
    """從均勻先驗抽樣 n_samples 組參數"""
    rng = np.random.default_rng(seed)
    params_list = []

    for i in range(n_samples):
        p = {'model_id': i + 1}  # 從 1 開始，對應 model_000001

        for key, (lo, hi) in PRIOR.items():
            p[key] = float(rng.uniform(lo, hi))

        # Si 由電荷平衡計算（不是獨立參數）
        # Si 參考您的範本：Si ~ 4.0，這裡用簡單比例估算
        # 保持總 cation 數合理
        p['Si'] = float(rng.uniform(3.0, 5.5))

        # O 由電荷平衡決定（假設 Fe = FeO，全部 Fe²⁺）
        p['O'] = compute_oxygen(p)

        params_list.append(p)

    return params_list


def compute_oxygen(p):
    """電荷平衡計算 O"""
    return (2.0 * p['Si'] +
            1.0 * p['Mg'] +
            1.0 * p['Fe'] +
            1.0 * p['Ca'] +
            1.5 * p['Al'] +
            0.5 * p['Na'] +
            1.5 * p['Cr'])

# ============================================================
# 2. 產生 ad.in（溫度剖面）
# ============================================================

def make_pressure_array(n_points=315):
    """
    產生壓力點陣列，與您原始 ad.in 的分布一致：
    從地表（~0.27 GPa）到 CMB（~22.9 GPa）
    """
    P = np.concatenate([
        np.linspace(0.27,  6.82,  100),
        np.linspace(6.88,  13.97, 100),
        np.linspace(14.03, 22.93, 115),
    ])
    return P


def make_temperature_profile(P_array, T_lit, P_lit, dTdP):
    """
    在壓力空間建立溫度剖面

    傳導段（P <= P_lit）：線性，從 T_surf 到 T_lit
    絕熱段（P >  P_lit）：T_lit + dTdP * (P - P_lit)

    完全不依賴熱演化模型，假設最少
    """
    T = np.where(
        P_array <= P_lit,
        T_SURF + (T_lit - T_SURF) * (P_array / P_lit),  # 傳導段
        T_lit  + dTdP * (P_array - P_lit)                # 絕熱段
    )
    return T


def write_ad_in(P_array, T_array, run_dir):
    """寫入 ad.in，格式：P(GPa)  0.000000  T(K)"""
    path = os.path.join(run_dir, "ad.in")
    with open(path, 'w') as f:
        for P, T in zip(P_array, T_array):
            f.write(f"{P:.6f} 0.000000 {T:.6f}\n")

# ============================================================
# 3. 產生 control 檔
# ============================================================

def write_control(p, run_dir):
    """
    寫入 control 檔
    格式與您的原始 control 完全一致
    iensem = -1：從 ad.in 讀溫度剖面
    """
    O = compute_oxygen(p)

    lines = [
        f"0,{P_MAX_GPA:.0f},50,0,0,0,-1,0,0",
        "8,2,4,2",
        "oxides",
        f"Si      {p['Si']:.5f}     {p['Si']:.5f}    0",
        f"Mg      {p['Mg']:.5f}     {p['Mg']:.5f}    0",
        f"Fe      {p['Fe']:.5f}     {p['Fe']:.5f}    0",
        f"Ca      {p['Ca']:.5f}     {p['Ca']:.5f}    0",
        f"Al      {p['Al']:.5f}     {p['Al']:.5f}    0",
        f"Na      {p['Na']:.5f}     {p['Na']:.5f}    0",
        f"Cr      {p['Cr']:.5f}     {p['Cr']:.5f}    0",
        f" O      {O:.5f}     {O:.5f}    0",
        "1,1,1,1",
        PAR_DIR,
        "73",
        CONTROL_PHASES,
    ]

    path = os.path.join(run_dir, "control")
    with open(path, 'w') as f:
        f.write("\n".join(lines))

# ============================================================
# 4. 執行單一 model
# ============================================================

def run_single_model(job):
    """
    建立資料夾，寫入 control 和 ad.in，cp main，執行 HeFESTo
    """
    params, run_dir = job
    status_file = os.path.join(run_dir, "status.txt")
    if os.path.exists(status_file):
        with open(status_file) as f:
            if f.read().strip() == "success":
                return params['model_id'], True, "skipped"
            
    os.makedirs(run_dir, exist_ok=True)

    # 儲存參數記錄
    with open(os.path.join(run_dir, "params.json"), 'w') as f:
        json.dump(params, f, indent=2)

    # 產生壓力點和溫度剖面
    P_array = make_pressure_array()
    T_array = make_temperature_profile(
        P_array,
        params['T_lit'],
        params['P_lit'],
        params['dTdP'],
    )

    # 寫入 ad.in 和 control
    write_ad_in(P_array, T_array, run_dir)
    write_control(params, run_dir)

    # cp main 執行檔進資料夾
    main_dst = os.path.join(run_dir, "main")
    if not os.path.exists(main_dst):
        shutil.copy2(HEFESTO_MAIN, main_dst)
        os.chmod(main_dst, 0o755)  # 確保可執行

    # 執行 HeFESTo（在 run_dir 裡執行，它讀 ./control 和 ./ad.in）
    log_path = os.path.join(run_dir, "hefesto.log")
    try:
        with open(log_path, 'w') as log:
            result = subprocess.run(
                ["./main"],
                cwd=run_dir,
                stdout=log,
                stderr=log,
                timeout=600,   # 10 分鐘 timeout
            )
        success = (result.returncode == 0)

    except subprocess.TimeoutExpired:
        success = False
        with open(log_path, 'a') as f:
            f.write("\nERROR: TIMEOUT after 600s\n")

    except FileNotFoundError:
        success = False
        with open(log_path, 'w') as f:
            f.write(f"ERROR: 找不到執行檔 {HEFESTO_MAIN}\n")

    # 記錄狀態
    with open(os.path.join(run_dir, "status.txt"), 'w') as f:
        f.write("success" if success else "failed")

    return params['model_id'], success, "ran"

# ============================================================
# 5. 批次執行
# ============================================================

def run_batch(n_samples=1000, seed=42, start_from=0, n_cpu=8):
    """批次平行執行所有 model"""
    os.makedirs(RUNS_DIR, exist_ok=True)

    if not os.path.exists(HEFESTO_MAIN):
        print(f"ERROR: 找不到 HeFESTo 執行檔：{HEFESTO_MAIN}")
        return

    params_list = sample_parameters(n_samples, seed=seed)

    print("=" * 65)
    print(f"HeFESTo 批次執行")
    print(f"執行檔  ：{HEFESTO_MAIN}")
    print(f"輸出目錄：{RUNS_DIR}")
    print(f"總 model 數：{n_samples}   種子：{seed}   起始：{start_from}")
    print(f"CPU 核心：{n_cpu}")

    # 建立 job list
    jobs = []
    for i, params in enumerate(params_list):
        if i < start_from:
            continue
        model_name = f"model_{params['model_id']:06d}"
        run_dir    = os.path.join(RUNS_DIR, model_name)
        jobs.append((params, run_dir))

    print(f"待執行：{len(jobs)} 個 model")
    print(f"預計時間：~{len(jobs) * 3 / n_cpu / 60:.1f} 小時")
    print("=" * 65)

    success_count = 0
    fail_count    = 0
    skip_count    = 0
    t0 = datetime.now()

    with Pool(processes=n_cpu) as pool:
        for model_id, success, status in pool.imap_unordered(
                run_single_model, jobs):
            if status == "skipped":
                skip_count += 1
                print(f"  [model_{model_id:06d}] 已完成，跳過")
            elif success:
                success_count += 1
                elapsed = (datetime.now() - t0).total_seconds()
                done = success_count + fail_count
                eta = elapsed / done * (len(jobs) - done - skip_count) if done > 0 else 0
                print(f"  [model_{model_id:06d}] OK  "
                      f"({success_count}/{len(jobs)-skip_count}  "
                      f"ETA {int(eta//60)}m{int(eta%60)}s)")
            else:
                fail_count += 1
                print(f"  [model_{model_id:06d}] FAIL <- 查看 hefesto.log")

    print("=" * 65)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"完成！耗時 {int(elapsed//3600)}h {int(elapsed%3600//60)}m")
    print(f"成功：{success_count}  失敗：{fail_count}  跳過：{skip_count}")

    summary = {
        'timestamp':    datetime.now().isoformat(),
        'n_samples':    n_samples,
        'seed':         seed,
        'n_cpu':        n_cpu,
        'success':      success_count,
        'failed':       fail_count,
        'skipped':      skip_count,
        'prior':        PRIOR,
    }
    with open(os.path.join(RUNS_DIR, "batch_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
# ============================================================
# 6. 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HeFESTo 批次執行")
    parser.add_argument('--n',     type=int, default=1000,
                        help='model 總數（預設 1000）')
    parser.add_argument('--seed',  type=int, default=42,
                        help='隨機種子（預設 42）')
    parser.add_argument('--start', type=int, default=0,
                        help='從第幾個 model 開始，用於斷點續跑')
    parser.add_argument('--cpu',   type=int, default=8,
                        help='CPU 核心數（預設 8）')
    parser.add_argument('--test',  action='store_true',
                        help='只跑 1 個 model 測試（單核）')
    args = parser.parse_args()

    if args.test:
        print("測試模式：只跑 model_000001（單核）")
        run_batch(n_samples=1, seed=args.seed, start_from=0, n_cpu=1)
        print("\n請檢查：")
        print(f"  {RUNS_DIR}/model_000001/control      <- 組成設定")
        print(f"  {RUNS_DIR}/model_000001/ad.in        <- 溫度剖面")
        print(f"  {RUNS_DIR}/model_000001/fort.56      <- HeFESTo 輸出")
        print(f"  {RUNS_DIR}/model_000001/hefesto.log  <- 執行記錄")
    else:
        run_batch(
            n_samples=args.n,
            seed=args.seed,
            start_from=args.start,
            n_cpu=args.cpu,
        )
