#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 92.khan_member_forward.py
#
# 89 用的 khan_median 落在 ensemble 的 99.5 百分位(91 的結果),不是代表性成員,
# 而且逐點中位把 Moho 抹成 0-50 km 的連續梯度。89 對 khan 的走時評估因此無效。
#
# 這裡從 1000 個成員裡按 M_col 分位挑 5 個真實成員,各跑一次 R=1600 forward,
# 看兩件事:
#   (1) 89 裡 khan 在 S0918a (delta=16.8) / S0474a (delta=19.7) 的
#       3 個 nopred + 3 筆 4000 秒級分支跳是否消失
#       -> 消失 = khan_median 的中位化人造物,可換
#       -> 不消失 = Khan 系地殼共有的低速層,換過去要連帶接受
#   (2) mass / moi 落在哪
#
# 挑選準則是 M_col 分位,與 likelihood 正交 —— 不是挑 mass/moi 最好的那個。

import os
import sys
import glob
import importlib.util
import numpy as np
from scipy.interpolate import interp1d
from mars_config import KHAN_MODEL_DIR
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

MCMC_PY = f'{HERE}/82.mcmc_core_nGibbs_wideMoI.py'
LSL_DIR = KHAN_MODEL_DIR
R_TEST  = 1600.0        # 89 的 khan 在此同時 mass 0.06 / moi 0.54
QUANT   = [25, 40, 50, 60, 75]

sys.path.insert(0, os.path.dirname(MCMC_PY))
spec = importlib.util.spec_from_file_location('M', MCMC_PY)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
M.CF.configure(route='B', verbose=False)

Z_GRID = np.linspace(0.0, M.SC_Z_FIXED, 200)    # 與 _samuel_cache 同網格

def member_cache(path):
    # Khan 的 .txt 是分段常數(每段起點給值)。用 previous 而非 linear:
    # 線性插值會把 Moho 抹成過渡帶,那正是 khan_median 的問題。
    d = np.loadtxt(path)
    z, vp, vs, rho = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    f = lambda y: interp1d(z, y, kind='previous', bounds_error=False,
                           fill_value=(y[0], y[-1]))(Z_GRID)
    return {'crust_z': Z_GRID, 'crust_vp': f(vp),
            'crust_vs': f(vs), 'crust_rho': f(rho)}

def column_M(cache):
    # 球殼:  dM = 4pi/3 rho (r2^3 - r1^3)
    r = (M.MARS_RADIUS - cache['crust_z'])*1e3
    o = np.argsort(r); ra, ro = r[o], cache['crust_rho'][o]*1e3
    rm = 0.5*(ro[1:] + ro[:-1])
    return float(np.sum(4*np.pi/3*rm*(ra[1:]**3 - ra[:-1]**3)))

# ── 依 M_col 排序,取分位 ──────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(LSL_DIR, 'Model_*.txt')))
Mcol  = np.array([column_M(member_cache(f)) for f in files])
order = np.argsort(Mcol)
pick  = [order[int(q/100*(len(order)-1))] for q in QUANT]

M_sam = column_M(M._samuel_cache)
print(f'Samuel M_col = {100*M_sam/M.MARS_MASS_OBS:.3f} % Mars')
print(f'{"quantile":>10}{"model":>14}{"M_col %":>10}{"rho@0":>8}'
      f'{"rho@50":>8}{"Moho jump":>11}')
for q, i in zip(QUANT, pick):
    c = member_cache(files[i])
    j = int(np.argmax(np.abs(np.diff(c['crust_rho']))))
    print(f'{q:>9}%{os.path.basename(files[i]):>14}'
          f'{100*Mcol[i]/M.MARS_MASS_OBS:>10.3f}'
          f'{c["crust_rho"][0]:>8.3f}{np.interp(50.,Z_GRID,c["crust_rho"]):>8.3f}'
          f'{np.max(np.abs(np.diff(c["crust_rho"]))):>8.3f}@{Z_GRID[j]:.0f}km')

# ── forward ───────────────────────────────────────────────────────────────
run_dir = os.path.join(M.MCMC_DIR, 'khan_member')
os.makedirs(run_dir, exist_ok=True)

WATCH = ('S0918a', 'S0474a')     # 89 裡 khan 出問題的兩個最短距離事件

print()
print('=' * 84)
print(f'forward at R_cmb = {R_TEST:.0f}')
print('=' * 84)
print(f'{"model":>14}{"mass_sig":>10}{"moi_sig":>9}{"TT":>9}'
      f'{"miss":>6}{"cap":>5}{"res/present":>12}  watch')

rows = []
for tag, cache in ([('samuel', M._samuel_cache)]
                   + [(os.path.basename(files[i])[:-4], member_cache(files[i]))
                      for i in pick]):
    p = {**M.START_PARAMS, 'R_cmb': R_TEST}
    out = M.forward(p, run_dir, f'km_{tag}_R{R_TEST:.0f}', cache)
    c = out[2]
    if c is None:
        print(f'{tag:>14}   reject'); continue
    n_miss, n_cap = c['tt_n_miss'], c['tt_n_capped']
    n_pres = c['tt_n_ph']
    res = (c['misfit_tt'] - M.MISS_PENALTY*n_miss)/n_pres
    bad = [s for s in c['tt_miss_list']
           if s.split(':')[0] in WATCH
           and (s.endswith('nopred') or ':capped@' in s)]
    print(f'{tag:>14}{c["mass_sigma"]:>10.2f}{c["moi_sigma"]:>9.2f}'
          f'{c["misfit_tt"]:>9.1f}{n_miss:>6}{n_cap:>5}{res:>12.3f}  {len(bad)}')
    for s in bad:
        print(f'{"":>14}    {s}')
    rows.append((tag, c['mass_sigma'], c['moi_sigma'], res, len(bad)))

print()
print('watch = S0918a/S0474a 的 nopred + capped 筆數。89 的 khan_median 是 6。')
print('若成員普遍 <= 1 -> 中位化人造物,可換;若普遍 ~6 -> Khan 系共有,換要接受。')