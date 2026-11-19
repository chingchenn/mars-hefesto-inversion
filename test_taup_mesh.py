#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 88.calibrate_step.py
#
# 量單一參數擾動 +STEP / -STEP 造成的 |dS|(total misfit 變化),把步長校到
# 每步約 TARGET_DS 個 misfit 單位。
#
# a39B 的教訓:STEP 從 a24 一路繼承,因為 sigma 一直沒變;a39B 首次改 sigma
# 之後似然面陡一個數量級,同樣的 STEP 單步跨 11-14 個單位 -> 必然被拒。
# a44 同時換了地殼(Samuel -> Khan Model_820)與 sigma(2.63e-3 -> 3.05e-4,
# 6.4e-4 -> 6e-5),兩者都改變似然面曲率,必須重校。
#
# a44 test 的實測:單步 dS = 11.0,其中 TT +5.9 / mass +2.6 / moi +2.5。
# TT 竟是主要來源,這是先前只算地測方向時沒預期到的 -> 逐項拆開看。
#
# 17 次 forward(1 baseline + 8 參數 x 2 方向),約 13 分鐘。

import os
import sys
import importlib.util
import numpy as np

MCMC_PY   = '/home/jcchen2/mars-hefesto-inversion/82.mcmc_core_nGibbs_wideMoI.py'
ROUTE     = 'B'
TARGET_DS = 1.0       # 目標單步 misfit 變化。exp(-1) = 0.37,配上八維
R_START   = 1620.0
TAG       = 'calibA44b'   # 設定變了就改,避免 build_taup 載到舊 .npz

sys.path.insert(0, os.path.dirname(MCMC_PY))
spec = importlib.util.spec_from_file_location('M', MCMC_PY)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
M.CF.configure(route=ROUTE, verbose=False)

CACHE = M._khan_cache
P0 = {**M.START_PARAMS, 'R_cmb': R_START}

run_dir = os.path.join(M.MCMC_DIR, 'calib_step')
os.makedirs(run_dir, exist_ok=True)

def run(params, tag):
    out = M.forward(params, run_dir, tag, CACHE)
    c = out[2]
    if c is None or out[0] is None:
        return None
    n_miss = int(c['tt_n_miss'])
    return {'S': float(out[0]),
            # 罰則階梯不是似然的曲率:miss 每變動一筆就跳 MISS_PENALTY,
            # 而步長要校的是曲率。扣掉之後才是可外插的部分。
            'S_adj': float(out[0]) - M.MISS_PENALTY*n_miss,
            'tt': float(c['misfit_tt']),
            'mass': float(c['mass_sigma']),
            'moi': float(c['moi_sigma']),
            'sol': float(c['misfit_solidus']),
            'h_sol': float(c['h_solid_km']),
            'h_liq': float(c['h_liquid_km']),
            'miss': n_miss,
            'branch': int(c['tt_n_branch'])}

def state(r):
    return 'solid' if r['h_liq'] < 1 else ('molten' if r['h_sol'] < 1 else 'layered')

# ── baseline ──────────────────────────────────────────────────────────────
b = run(P0, f'{TAG}_base')
if b is None:
    raise SystemExit('baseline forward failed')

print()
print('=' * 92)
print(f"baseline   S={b['S']:.3f}   tt={b['tt']:.2f}  mass={b['mass']:.2f}  "
      f"moi={b['moi']:.2f}  sol={b['sol']:.2f}   "
      f"h_sol={b['h_sol']:.0f}/h_liq={b['h_liq']:.0f} [{state(b)}]   "
      f"miss={b['miss']} branch={b['branch']}")
print('=' * 92)

# ── 逐參數 +/- STEP ───────────────────────────────────────────────────────
print()
print(f"{'param':<16}{'STEP':>9}{'dS(+)':>9}{'dS(-)':>9}{'|dS|':>8}"
      f"{'dTT':>8}{'dmass':>8}{'dmoi':>8}{'dmiss':>8}{'suggest':>10}{'curv':>8}  state")

rows = []
for k in M.PRIOR:
    d = {}
    for sgn, lab in ((+1, '+'), (-1, '-')):
        p = {**P0, k: P0[k] + sgn*M.STEP[k]}
        lo, hi = M.PRIOR[k]
        if not (lo <= p[k] <= hi):
            d[lab] = None
            continue
        d[lab] = run(p, f'{TAG}_{k}_{lab}')

    ok = [v for v in d.values() if v is not None]
    if not ok:
        print(f'{k:<16}   both directions failed')
        continue

    dS = {lab: (v['S_adj'] - b['S_adj']) if v is not None else np.nan
          for lab, v in d.items()}
    # 幾何平均:曲率為負時(校準點不在極小值附近)算術平均會被梯度污染
    a = np.array([abs(x) for x in dS.values() if np.isfinite(x)])
    dS_mean = float(np.exp(np.mean(np.log(a[a > 1e-6])))) if (a > 1e-6).any() else np.nan

    # 各項貢獻取兩方向的最大絕對變化,看誰主導
    dtt = max(abs(v['tt'] - b['tt']) for v in ok)
    dm  = max(abs(v['mass'] - b['mass']) for v in ok)
    dmo = max(abs(v['moi'] - b['moi']) for v in ok)

    sug  = M.STEP[k]*TARGET_DS/dS_mean if np.isfinite(dS_mean) and dS_mean > 0 else np.nan
    curv = 0.5*(dS.get('+', np.nan) + dS.get('-', np.nan))
    sts  = '/'.join(sorted({state(v) for v in ok}))
    dmiss = '/'.join(str(v['miss'] - b['miss']) if v is not None else '-'
                     for v in (d['+'], d['-']))

    print(f'{k:<16}{M.STEP[k]:>9.4f}{dS["+"]:>+9.2f}{dS["-"]:>+9.2f}'
          f'{dS_mean:>8.2f}{dtt:>8.2f}{dm:>8.2f}{dmo:>8.2f}{dmiss:>8}'
          f'{sug:>10.4f}{curv:>+8.2f}  {sts}')
    rows.append((k, M.STEP[k], dS_mean, sug, curv, sts))

# ── 建議的 STEP dict ──────────────────────────────────────────────────────
print()
print('=' * 92)
print(f'STEP = {{                      # 88 校準, target dS = {TARGET_DS}')
for k, cur, dsm, sug, curv, sts in rows:
    note = ''
    if curv < 0:
        note = '  # 曲率為負,校準點不在極小值附近,|dS| 被梯度污染'
    elif state(b) not in sts or len(sts.split('/')) > 1:
        note = '  # 擾動翻了三態,|dS| 含相變不連續'
    print(f"'{k}': {sug:.4g},{note}")
print('}')

print()
print('曲率 = (dS+ + dS-)/2。正值表示校準點在局部極小附近,|dS| 主要反映曲率;')
print('負值表示位在斜坡上,|dS| 主要反映梯度,建議步長會被高估。')
print('三態欄若與 baseline 不同,該參數的 |dS| 含相變造成的不連續,不是曲率。')