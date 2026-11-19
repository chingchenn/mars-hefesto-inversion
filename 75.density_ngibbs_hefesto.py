#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# nGibbs vs HeFESTo:地函密度的逐點對照
#
# 為什麼:
#   77 腳本顯示模型地函在同深度比 Samuel 密 +0.86%,但依組成
#   (a38B Mg# 中位 0.754 vs Samuel 0.75,幾乎相同)與溫度(你更熱,
#   密度應更低 0.3-0.5%)推算,應該【低】約 0.2-0.4%。
#   落差約 1.1-1.3%,對應行星質量 0.8-0.9%,佔 a38B 質量偏移 +1.95% 的四成多。
#   若來源是 nGibbs emulator,則 a39B 把 Mg# 從 0.754 推到 0.87
#   是在吸收 emulator 偏差,不是組成訊號。
#
# 做法:
#   只比【單點】 rho(P, T, X)。不需要 S_lit,不需要等熵搜尋,
#   所以 HeFESTo 只要一次呼叫(~30-60 s),不是整條剖面的 20-30 min。
#   P-T 直接取自 profile npz,保證落在模型走過的地溫線上。
#   BML 與核不碰(BML 佔質量 4.75% 且液相用 Thomas EoS,不是 nGibbs)。
#
# 為什麼不 import 50:
#   50.mcmc_main.py 開頭是 `from config import *`,舊模組名(現為 mars_config),
#   import 不動。這裡只需要 control 檔組裝與單點執行,直接內含。
#
# 質量影響:
#   地函約佔行星質量 70%  ->  dM/M = 0.70 * (drho/rho)_mantle

import os, sys, json, glob, re, shutil, subprocess
import importlib.util
import numpy as np
import pandas as pd
from mars_config import RUN_BASE
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

# ── config ────────────────────────────────────────────────────────────────────
INV_DIR   = HERE
SCRIPT_70 = os.path.join(INV_DIR, '70.mcmc_core_nGibbs.py')
SCRIPT_50 = os.path.join(INV_DIR, '50.mcmc_main.py')      # 只讀文字,不 import

RUNS      = RUN_BASE
CHAIN_DIR = os.path.join(RUNS, 'mcmc', 'chain_a38B_00')
WORK_DIR  = os.path.join(RUNS, 'hef_vs_ng')

N_POINTS  = 6
P_MIN     = 2.0          # GPa
MANTLE_LAYER = 1
HEF_TIMEOUT  = 1600      # s

sys.path.insert(0, INV_DIR)
from mars_config import PAR_DIR, HEFESTO_MAIN

print('PAR_DIR      =', PAR_DIR, os.path.isdir(PAR_DIR))
print('HEFESTO_MAIN =', HEFESTO_MAIN, os.path.exists(HEFESTO_MAIN))

# ── 載入 70(nGibbs) ──────────────────────────────────────────────────────────
def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


print('\nloading 70 (nGibbs) ...')
NG = load_mod(SCRIPT_70, 'mod70')

# ── 從 50 的原始碼取出相位清單(只讀文字) ─────────────────────────────────────
_src = open(SCRIPT_50).read()
_m = re.search(r'CONTROL_PHASES\s*=\s*"""\\?\n(.*?)"""', _src, re.S)
if _m is None:
    raise SystemExit('抓不到 CONTROL_PHASES,直接把 50 裡那段相位清單貼進本檔')
CONTROL_PHASES = _m.group(1)
print(f'CONTROL_PHASES: {len(CONTROL_PHASES.splitlines())} lines')


# ── HeFESTo 單點 ──────────────────────────────────────────────────────────────
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

f56 = os.path.join(run_dir, "fort.56")
    if os.path.exists(f56):
        os.remove(f56)
def run_hefesto_single(run_dir, control_lines):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "control"), 'w') as f:
        f.write("\n".join(control_lines))
    main_dst = os.path.join(run_dir, "main")
    if not os.path.exists(main_dst):
        shutil.copy2(HEFESTO_MAIN, main_dst)
        os.chmod(main_dst, 0o755)
    with open(os.path.join(run_dir, "hefesto.log"), 'w') as log:
        subprocess.run(["./main"], cwd=run_dir, stdout=log, stderr=log,
                       timeout=HEF_TIMEOUT)
    f56 = os.path.join(run_dir, "fort.56")
    if not os.path.exists(f56) or os.path.getsize(f56) == 0:
        return None
    return f56


def read_fort56_rho(path):
    # fort.56 第一行是標題,第二行是欄名
    with open(path) as f:
        f.readline()
        cols = f.readline().split()
    df = pd.read_csv(path, sep=r'\s+', skiprows=2, names=cols)
    rho_col = [c for c in cols if c.lower().startswith('rho')]
    if not rho_col:
        print('  fort.56 欄名:', cols)
        return None
    v = pd.to_numeric(df[rho_col[0]], errors='coerce').dropna()
    return float(v.iloc[0]) if len(v) else None


# ── 取 profile 與 params ──────────────────────────────────────────────────────
npz_files = sorted(glob.glob(os.path.join(CHAIN_DIR, 'profile_s*.npz')))
if not npz_files:
    raise SystemExit(f'no profile npz in {CHAIN_DIR}')
npz_path = npz_files[len(npz_files) // 2]
step = int(os.path.basename(npz_path).replace('profile_s', '').replace('.npz', ''))
npz = np.load(npz_path)
print(f'\nprofile: {os.path.basename(npz_path)}  (step {step})')

params = None
with open(os.path.join(CHAIN_DIR, 'chain.jsonl')) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('step') == step:
            params = r['params']
            break
if params is None:
    raise SystemExit(f'step {step} not found in chain.jsonl')
print('params:', {k: round(v, 4) for k, v in params.items()})

p  = NG.composition_from_params(params)
O  = NG.compute_oxygen(p)
cv = NG._comp_values(p, O)

# ── 挑地函點 ──────────────────────────────────────────────────────────────────
z   = np.asarray(npz['z_km'], float)
lay = np.asarray(npz['layer'], int)
P   = np.asarray(npz['P_GPa'], float)
T   = np.asarray(npz['T_K'], float)
rho = np.asarray(npz['rho'], float)        # g/cm3

m = (lay == MANTLE_LAYER) & (P >= P_MIN)
if m.sum() < N_POINTS:
    raise SystemExit(f'mantle nodes with P >= {P_MIN}: only {m.sum()}')

idx = np.linspace(0, m.sum() - 1, N_POINTS).astype(int)
Pp, Tp, Rp, Zp = P[m][idx], T[m][idx], rho[m][idx], z[m][idx]

# ── nGibbs ────────────────────────────────────────────────────────────────────
print('\nnGibbs ...')
rho_ng = np.zeros(N_POINTS)
for i in range(N_POINTS):
    out = NG._ng_forward([Pp[i]], [Tp[i]], cv,
                         NG.NG_T_HEADERS, ['component_moles'])
    pr = NG._ng_props(out['component_moles'], [Pp[i]], [Tp[i]], ['rho'])
    rho_ng[i] = float(pr['rho'][0])
    print(f'  {i}: P={Pp[i]:6.2f} T={Tp[i]:7.1f}  rho={rho_ng[i]:.4f}  '
          f'(npz {Rp[i]:.4f})')

# ── HeFESTo ───────────────────────────────────────────────────────────────────
print('\nHeFESTo ...')
os.makedirs(WORK_DIR, exist_ok=True)
rho_hf = np.full(N_POINTS, np.nan)
for i in range(N_POINTS):
    d = os.path.join(WORK_DIR, f'pt_{i:02d}')
    line1 = f"{Pp[i]:.4f},{Pp[i]:.4f},1,{Tp[i]:.2f},{Tp[i]:.2f},0,0,0,0"
    try:
        f56 = run_hefesto_single(d, make_control_lines(p, O, line1))
    except subprocess.TimeoutExpired:
        print(f'  {i}: timeout'); continue
    if f56 is None:
        print(f'  {i}: no fort.56  -> see {d}/hefesto.log'); continue
    v = read_fort56_rho(f56)
    if v is None:
        print(f'  {i}: rho column not found'); continue
    rho_hf[i] = v
    print(f'  {i}: P={Pp[i]:6.2f} T={Tp[i]:7.1f}  rho={v:.4f}')

# ── 結果 ──────────────────────────────────────────────────────────────────────
print('\n' + '=' * 92)
print('nGibbs vs HeFESTo   mantle density   (same composition, same P-T)')
print('=' * 92)
print(f"{'z (km)':>9}{'P (GPa)':>9}{'T (K)':>8}"
      f"{'npz':>9}{'nGibbs':>9}{'HeFESTo':>9}{'ng-hf':>9}{'ng-hf %':>9}")
for i in range(N_POINTS):
    d = rho_ng[i] - rho_hf[i]
    pc = 100 * d / rho_hf[i] if np.isfinite(rho_hf[i]) else np.nan
    print(f'{Zp[i]:9.1f}{Pp[i]:9.2f}{Tp[i]:8.1f}'
          f'{Rp[i]:9.4f}{rho_ng[i]:9.4f}{rho_hf[i]:9.4f}{d:+9.4f}{pc:+9.2f}')

print(f'\n  npz vs nGibbs self-check: max |diff| = '
      f'{np.max(np.abs(Rp - rho_ng)):.5f} g/cm3   (應接近 0)')

ok = np.isfinite(rho_hf)
if ok.sum() == 0:
    raise SystemExit('HeFESTo 全部失敗,先看 hefesto.log')

pc_all = 100 * (rho_ng[ok] - rho_hf[ok]) / rho_hf[ok]
mean_pc = float(np.mean(pc_all))
print(f'\n  mean = {mean_pc:+.3f} %   '
      f'range = [{pc_all.min():+.3f}, {pc_all.max():+.3f}] %   n = {ok.sum()}')

print('\n' + '=' * 92)
print('mass impact   (mantle ~70 % of planet mass)')
print('=' * 92)
print(f'  dM/M from this bias  = 0.70 x {mean_pc:+.3f} % = {0.70*mean_pc:+.3f} %')
print(f'  observed a38B offset = +1.95 %')
if abs(0.70 * mean_pc) > 0.8:
    print('\n  -> emulator 偏差足以解釋大部分質量偏移。')
    print('     a39B 把 Mg# 推到 0.87 是在吸收 emulator 誤差,不是組成訊號。')
    print('     停 a39B,先找 Ben 修校準。')
elif abs(mean_pc) < 0.3:
    print('\n  -> emulator 沒有明顯偏差。Mg# 撞牆是真實的模型不足訊號,')
    print('     a39B 的結論成立。')
else:
    print('\n  -> 偏差中等,解釋部分而非全部。兩者都要在論文裡報告。')


import warnings
for i in range(6):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = NG._ng_forward([Pp[i]], [Tp[i]], cv, NG.NG_T_HEADERS, ['component_moles'])
        pr  = NG._ng_props(out['component_moles'], [Pp[i]], [Tp[i]], ['rho'])
    print(f'point {i}  P={Pp[i]:.2f} T={Tp[i]:.0f}  n_warn={len(w)}')