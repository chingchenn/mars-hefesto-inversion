#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:53:30 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 101: SKS 存在窗口的震央距掃描
#
# 問題:solid 模型 48/48 掉 S1000a 的 SKS,molten 只掉 10/48。這是
# 「固態 BML 關閉了 SKS 的射線參數窗口」還是「87 度本來就在窗口邊緣」?
#
# 做法:重建 .nd,對 DELTA_GRID 逐點問 TauP 有沒有 SKS,畫出存在窗口。
# 標兩條線:
#   87.0   —— 目前 OBS_DATASET 給 S1000a 的距離
#   125.9  —— Irving 2023 由軌道影像定位的 S1000a 距離(震源深度固定為 0)
# 兩者差 39 度。若窗口涵蓋 126 而不涵蓋 87,那 SKS 的缺失來自距離設定,
# 與 BML 相態無關。
#
# Irving 2023 的觀測值(Table 1,方法 A/B/C 平均):
#   S1000a  SKS-PP = 339.3 s  (SD 4.7)
#   S0976a  SKS-PP = 303.9 s  (SD 7.1)
# 目前 OBS_DATASET 寫的是 SKS-PP = 87.0 +- 10.0,與 delta 同值 —— 先確認。
# 本腳本會把每個模型在兩個距離的 SKS-PP 預測值印出來,直接對照。

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from mars_config import MCMC_DIR, FIG_DIR

# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = MCMC_DIR
PREFIXES  = ['chain_a49B', 'chain_a51B']     # solid 樣本多的兩個 run
WORK_DIR  = '/tmp/taup_sks_scan'

BURNIN_FRAC = 0.4
DROP_LAST   = 1
MIN_STEPS   = 400
H_TOL       = 1.0
H_MIN_LAYER = 1.0
N_PER_STATE = 6              # 每個 state 掃幾個模型

DELTA_GRID  = np.arange(60.0, 140.01, 1.0)
SRC_DEPTH   = 0.0            # Irving 由軌道影像把 S1000a 震源深度固定為 0
DELTA_OBS   = 87.0           # 目前 OBS_DATASET 的值
DELTA_IRV   = 125.9          # Irving 2023 軌道影像定位
SKS_PP_IRV  = (339.3, 4.7)   # Irving Table 1,S1000a,方法 A/B/C 平均與 SD

fontsize = 14
TICKSIZE = 14
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
STATES      = ['solid', 'layered', 'molten']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}

# ══════════════════════════════════════════════════════════════════════════
def load_chain_dir(dd):
    p = os.path.join(dd, 'chain.jsonl')
    if not os.path.exists(p):
        return None
    recs = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return recs or None

def classify_state(h_sol, h_liq, tol=H_TOL):
    st  = np.full(np.shape(h_sol), 'fail', dtype='<U8')
    ok  = np.isfinite(h_sol) & np.isfinite(h_liq)
    s_y = ok & (h_sol >= tol)
    l_y = ok & (h_liq >= tol)
    st[ s_y &  l_y] = 'layered'
    st[~s_y &  l_y] = 'molten'
    st[ s_y & ~l_y] = 'solid'
    return st

def write_nd(z_npz, nd_path):
    z   = np.asarray(z_npz['z_km'], float)
    Vp  = np.asarray(z_npz['Vp'], float)
    Vs  = np.asarray(z_npz['Vs'], float)
    rho = np.asarray(z_npz['rho'], float)
    lay = np.asarray(z_npz['layer'], int)
    z_cmb = float(np.asarray(z_npz['z_cmb']).ravel()[0])
    m_cru, m_man = (lay == 0), (lay == 1)
    m_bml, m_cor = (lay == 2), (lay == 3)
    if not (m_man.any() and m_cor.any()):
        return False
    with open(nd_path, 'w') as f:
        for d, vp, vs, r in zip(z[m_cru], Vp[m_cru], Vs[m_cru], rho[m_cru]):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        f.write("mantle\n")
        for d, vp, vs, r in zip(z[m_man], Vp[m_man], Vs[m_man], rho[m_man]):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        d_b, vp_b, vs_b, rho_b = z[m_bml], Vp[m_bml], Vs[m_bml], rho[m_bml]
        zero = np.where(vs_b <= 1e-6)[0]
        i0   = int(zero[0]) if len(zero) else len(d_b)
        if i0 < len(d_b) and (z_cmb - d_b[i0]) < H_MIN_LAYER:
            d_b, vp_b, vs_b, rho_b = d_b[:i0], vp_b[:i0], vs_b[:i0], rho_b[:i0]
            i0 = len(d_b)
        if i0 > 0 and (d_b[i0-1] - d_b[0]) < H_MIN_LAYER:
            d_b, vp_b, vs_b, rho_b = d_b[i0:], vp_b[i0:], vs_b[i0:], rho_b[i0:]
            i0 = 0
        for d, vp, vs, r in zip(d_b[:i0], vp_b[:i0], vs_b[:i0], rho_b[:i0]):
            f.write(f"{d:.3f}  {vp:.4f}  {vs:.4f}  {r:.4f}\n")
        if i0 < len(d_b):
            if i0 > 0:
                f.write(f"{d_b[i0]:.3f}  {vp_b[i0-1]:.4f}  {vs_b[i0-1]:.4f}  "
                        f"{rho_b[i0-1]:.4f}\n")
            f.write("outer-core\n")
            for d, vp, r in zip(d_b[i0:], vp_b[i0:], rho_b[i0:]):
                f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")
        else:
            if len(d_b) and z_cmb - d_b[-1] > 1e-6:
                f.write(f"{z_cmb:.3f}  {vp_b[-1]:.4f}  {vs_b[-1]:.4f}  {rho_b[-1]:.4f}\n")
            f.write("outer-core\n")
        d_c, vp_c, rho_c = z[m_cor], Vp[m_cor], rho[m_cor]
        f.write(f"{z_cmb:.3f}  {vp_c[0]:.4f}  0.0000  {rho_c[0]:.4f}\n")
        for d, vp, r in zip(d_c[1:], vp_c[1:], rho_c[1:]):
            f.write(f"{d:.3f}  {vp:.4f}  0.0000  {r:.4f}\n")
    zz = np.array([float(l.split()[0]) for l in open(nd_path)
                   if l.strip() and not l[0].isalpha()])
    _, cnt = np.unique(zz, return_counts=True)
    return not (cnt > 2).any()

def first_time(model, delta, phase):
    try:
        arr = model.get_travel_times(source_depth_in_km=SRC_DEPTH,
                                     distance_in_degree=float(delta),
                                     phase_list=[phase])
    except Exception:
        return np.nan
    return arr[0].time if len(arr) else np.nan

# ══════════════════════════════════════════════════════════════════════════
picks = {s: [] for s in STATES}
for pf in PREFIXES:
    for dd in sorted(glob.glob(os.path.join(CHAIN_DIR, f'{pf}_*'))):
        ch = load_chain_dir(dd)
        if ch is None:
            continue
        if DROP_LAST > 0 and len(ch) > DROP_LAST:
            ch = ch[:-DROP_LAST]
        if len(ch) < MIN_STEPS:
            continue
        recs = ch[int(len(ch)*BURNIN_FRAC):]
        hs = np.array([np.nan if s.get('h_solid_km') is None else float(s['h_solid_km'])
                       for s in recs])
        hl = np.array([np.nan if s.get('h_liquid_km') is None else float(s['h_liquid_km'])
                       for s in recs])
        st = classify_state(hs, hl)
        for i, s in enumerate(recs):
            if st[i] not in STATES or not s.get('accepted'):
                continue
            p = os.path.join(dd, f"profile_s{int(s['step']):05d}.npz")
            if os.path.exists(p):
                picks[st[i]].append((os.path.basename(dd), int(s['step']), p, s))

os.makedirs(WORK_DIR, exist_ok=True)
RES = {s: [] for s in STATES}
print('=' * 96)
print(f'SKS 存在窗口掃描   source depth = {SRC_DEPTH:.1f} km')
print('=' * 96)
for st in STATES:
    if not picks[st]:
        print(f'{st}: 找不到留有 profile npz 的樣本'); continue
    idx = np.unique(np.linspace(0, len(picks[st])-1,
                                min(N_PER_STATE, len(picks[st]))).astype(int))
    for i in idx:
        name, step, path, rec = picks[st][i]
        nd = os.path.join(WORK_DIR, f'{name}_s{step:05d}.nd')
        try:
            z = np.load(path)
            if 'layer' not in z.files or not write_nd(z, nd):
                continue
            build_taup_model(nd, output_folder=WORK_DIR, verbose=False)
            model = TauPyModel(model=nd.replace('.nd', '.npz'))
        except Exception as e:
            print(f'  {name} s{step}: 建模失敗 {e}'); continue

        t_sks = np.array([first_time(model, d, 'SKS') for d in DELTA_GRID])
        t_pp  = np.array([first_time(model, d, 'PP')  for d in DELTA_GRID])
        ok = np.isfinite(t_sks)
        lo = DELTA_GRID[ok].min() if ok.any() else np.nan
        hi = DELTA_GRID[ok].max() if ok.any() else np.nan
        d_obs = np.nan if not ok.any() else (lo - DELTA_OBS if DELTA_OBS < lo else
                                             (DELTA_OBS - hi if DELTA_OBS > hi else 0.0))
        sksppo = first_time(model, DELTA_OBS, 'SKS') - first_time(model, DELTA_OBS, 'PP')
        skspi  = first_time(model, DELTA_IRV, 'SKS') - first_time(model, DELTA_IRV, 'PP')
        RES[st].append(dict(name=name, step=step, t_sks=t_sks, t_pp=t_pp,
                            lo=lo, hi=hi, d_obs=d_obs,
                            sks_pp_obs=sksppo, sks_pp_irv=skspi,
                            h_liq=float(rec.get('h_liquid_km') or 0.0),
                            R_cmb=float(rec['params'].get('R_cmb', np.nan))))
        print(f'  {st:<8s}{name:<16s} s{step:<7d} SKS window '
              f'{lo:6.1f} - {hi:6.1f} deg   '
              f'87 deg {"inside" if d_obs == 0 else f"outside by {d_obs:.1f}"}   '
              f'SKS-PP@126 = {skspi:7.1f} s')

print('\n' + '=' * 96)
print('彙總:SKS 窗口下界與兩個距離的 SKS-PP 預測')
print('=' * 96)
print(f"{'state':<10s}{'n':>4s}{'window lo':>12s}{'window hi':>12s}"
      f"{'87 inside':>12s}{'SKS-PP@87':>12s}{'SKS-PP@125.9':>14s}")
for st in STATES:
    if not RES[st]:
        continue
    lo = np.array([r['lo'] for r in RES[st]])
    hi = np.array([r['hi'] for r in RES[st]])
    ins = np.mean([r['d_obs'] == 0 for r in RES[st]])
    a = np.array([r['sks_pp_obs'] for r in RES[st]])
    b = np.array([r['sks_pp_irv'] for r in RES[st]])
    print(f'{st:<10s}{len(RES[st]):4d}{np.nanmedian(lo):12.1f}{np.nanmedian(hi):12.1f}'
          f'{100*ins:11.0f}%{np.nanmedian(a):12.1f}{np.nanmedian(b):14.1f}')
print(f"\n  Irving 2023 觀測 SKS-PP (S1000a) = {SKS_PP_IRV[0]:.1f} +- {SKS_PP_IRV[1]:.1f} s "
      f"at delta = {DELTA_IRV:.1f} deg")

# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
ax = axes[0]
for st in STATES:
    for k, r in enumerate(RES[st]):
        ax.plot(DELTA_GRID, np.where(np.isfinite(r['t_sks']), 1.0, np.nan)
                * (1.0 + 0.06*k + 0.4*STATES.index(st)),
                lw=3.5, color=STATE_COLOR[st], solid_capstyle='butt',
                label=st if k == 0 else None)
ax.axvline(DELTA_OBS, color='k', ls='--', lw=1.8)
ax.axvline(DELTA_IRV, color='#C0392B', ls='-', lw=1.8)
ax.text(DELTA_OBS, ax.get_ylim()[1], ' OBS 87', fontsize=10, va='top')
ax.text(DELTA_IRV, ax.get_ylim()[1], ' Irving 125.9', fontsize=10,
        va='top', color='#C0392B')
ax.set_xlabel('epicentral distance (deg)', fontsize=fontsize)
ax.set_ylabel('SKS exists (one bar per model)', fontsize=fontsize)
ax.set_yticklabels([])
ax.tick_params(labelsize=TICKSIZE)
ax.legend(fontsize=11, loc='lower right')

ax = axes[1]
for st in STATES:
    for k, r in enumerate(RES[st]):
        ax.plot(DELTA_GRID, r['t_sks'] - r['t_pp'], lw=1.4,
                color=STATE_COLOR[st], alpha=0.8,
                label=st if k == 0 else None)
ax.errorbar([DELTA_IRV], [SKS_PP_IRV[0]], yerr=[SKS_PP_IRV[1]], fmt='o',
            color='k', ms=7, capsize=4, label='Irving 2023 obs')
ax.axvline(DELTA_OBS, color='k', ls='--', lw=1.5)
ax.set_xlabel('epicentral distance (deg)', fontsize=fontsize)
ax.set_ylabel('SKS - PP (s)', fontsize=fontsize)
ax.tick_params(labelsize=TICKSIZE)
ax.legend(fontsize=11)
fig.suptitle('SKS existence window and differential travel time', fontsize=fontsize + 2)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '101_sks_window.png'), dpi=150)
plt.show()