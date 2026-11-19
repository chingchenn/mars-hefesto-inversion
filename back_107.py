#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 12:46:59 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 107: 固態 BML 與「無 BML」在地震上分不分得開
#
# 問題:BML thickness -> 0 等同沒有 BML。固態 BML 厚度有限但若
#      Mg#_bulk_bml 貼近 Mg#(地函),BML 頂就沒有速度對比,
#      走時看不見它,兩種情形實質上不可分。
#
# 三段:
#   A. 固態樣本裡 Δ = Mg#_bulk_bml - Mg# 的分佈,按 BML 厚度分箱。
#      同時看 Mg#_bulk_bml 的後驗 sd 對 prior sd 的比值:
#      h 小 -> 該參數對觀測沒有影響 -> 後驗 = 先驗 -> 比值 ~ 1(不可辨識)。
#   B. 質量加權的 bulk silicate mantle Mg#。h -> 0 時平滑趨近 Mg#(地函),
#      沒有參數化退化,是該報的量。質量用 profile npz 的 rho 積分球殼。
#   C. 直接從 profile npz 量 BML 頂界面兩側的 Vs/Vp/rho 跳躍,
#      不用從 Mg# 推導。跳躍再小,也要跟走時的 σ 比才知道看不看得見。
#
# 注意:後驗裡沒有 BML thickness = 0 的樣本(state counts 的 fail = 0),
#      所以「無 BML」只能用最薄的那一箱當代理,不是真正的對照組。

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = MCMC_DIR
PREFIX    = 'chain_a52B'
EXCLUDE   = ['chain_a52B_04']

BURNIN_FRAC = 0.30
DROP_LAST   = 1
MIN_STEPS   = 400
H_TOL       = 1.0

# BML 厚度分箱(km)
BINS      = [0, 20, 50, 100, 250, 400]

# Mg#_bulk_bml 的 prior,用來算後驗/先驗寬度比(要與 run 的 PRIOR 一致)
MGB_LO, MGB_HI = 0.20, 0.80

N_PROFILE = 200      # 每箱抓幾個 profile npz 量界面跳躍

fontsize = 14
TICKSIZE = 14
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

# ══════════════════════════════════════════════════════════════════════════
# 讀鏈
# ══════════════════════════════════════════════════════════════════════════
recs_all, dirs_all = [], []
for dd in sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '_*'))):
    if os.path.basename(dd) in EXCLUDE:
        continue
    p = os.path.join(dd, 'chain.jsonl')
    if not os.path.exists(p):
        continue
    recs = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    if DROP_LAST > 0 and len(recs) > DROP_LAST:
        recs = recs[:-DROP_LAST]
    if len(recs) < MIN_STEPS:
        continue
    b = int(len(recs)*BURNIN_FRAC)
    for s in recs[b:]:
        recs_all.append(s)
        dirs_all.append(dd)

if not recs_all:
    raise SystemExit('no records')

hs   = np.array([np.nan if s.get('h_solid_km')  is None else float(s['h_solid_km'])
                 for s in recs_all])
hl   = np.array([np.nan if s.get('h_liquid_km') is None else float(s['h_liquid_km'])
                 for s in recs_all])
mgm  = np.array([float(s['params']['Mg#'])           for s in recs_all])
mgb  = np.array([float(s['params']['Mg#_bulk_bml'])  for s in recs_all])
hbml = np.array([float(s['params']['BML_thickness']) for s in recs_all])
rcmb = np.array([float(s['params']['R_cmb'])         for s in recs_all])
mis  = np.array([float(s['misfit'])                  for s in recs_all])
step = np.array([int(s['step'])                      for s in recs_all])

okh    = np.isfinite(hs) & np.isfinite(hl)
solid  = okh & (hs >= H_TOL) & (hl <  H_TOL)
molten = okh & (hs <  H_TOL) & (hl >= H_TOL)
layer  = okh & (hs >= H_TOL) & (hl >= H_TOL)

print('=' * 100)
print('%s   n = %d   solid %d  layered %d  molten %d  (h 無效 %d)'
      % (PREFIX, len(recs_all), solid.sum(), layer.sum(), molten.sum(),
         int((~okh).sum())))
print('BML thickness 最小值 = %.2f km  (若 >> 0,後驗裡沒有真正的「無 BML」樣本)'
      % np.nanmin(hbml))
print('=' * 100)

# ══════════════════════════════════════════════════════════════════════════
# A. Δ = Mg#_bulk_bml - Mg#,按厚度分箱
# ══════════════════════════════════════════════════════════════════════════
prior_sd = (MGB_HI - MGB_LO)/np.sqrt(12.0)

print('')
print('A. 固態樣本:Mg# 對比與可辨識性')
print('   post_sd/prior_sd ~ 1  ->  該箱的 Mg#_bulk_bml 不受資料約束')
print('-' * 100)
print('%-16s%8s%12s%12s%12s%12s%12s' %
      ('h_bml bin (km)', 'n', 'Mg#_mant', 'Mg#_bml', 'delta_med', 'sd_ratio',
       'misfit_med'))
for i in range(len(BINS) - 1):
    m = solid & (hbml >= BINS[i]) & (hbml < BINS[i+1])
    if m.sum() < 10:
        print('%-16s%8d   (樣本不足)' % ('%d-%d' % (BINS[i], BINS[i+1]), m.sum()))
        continue
    d = mgb[m] - mgm[m]
    print('%-16s%8d%12.3f%12.3f%12.3f%12.2f%12.2f' %
          ('%d-%d' % (BINS[i], BINS[i+1]), int(m.sum()),
           np.median(mgm[m]), np.median(mgb[m]), np.median(d),
           np.std(mgb[m], ddof=1)/prior_sd, np.median(mis[m])))

print('')
print('全體固態樣本 delta = Mg#_bulk_bml - Mg#:')
d_all = mgb[solid] - mgm[solid]
print('  5/25/50/75/95 %% = %.3f / %.3f / %.3f / %.3f / %.3f'
      % tuple(np.percentile(d_all, [5, 25, 50, 75, 95])))
print('  |delta| < 0.02 的比例 = %.1f%%   (幾乎無對比 -> 界面看不見)'
      % (100*np.mean(np.abs(d_all) < 0.02)))
print('  delta < 0 的比例 = %.1f%%' % (100*np.mean(d_all < 0)))

# ══════════════════════════════════════════════════════════════════════════
# B. bulk silicate mantle Mg#(質量加權)
# ══════════════════════════════════════════════════════════════════════════
# 質量用球殼近似:M ∝ ∫ rho r^2 dr。這裡沒有逐樣本密度,
# 先用等密度近似 M ∝ (R_out^3 - R_in^3),再於 C 段用 profile 校正。
R_bml_top = rcmb + hbml
R_SURF    = 3389.5
V_bml     = R_bml_top**3 - rcmb**3
V_man     = R_SURF**3 - R_bml_top**3
f_bml     = V_bml/(V_bml + V_man)
mg_bsm    = (V_man*mgm + V_bml*mgb)/(V_man + V_bml)

print('')
print('B. bulk silicate Mg#(等密度球殼近似,C 段會用 profile 校正)')
print('-' * 100)
print('%-16s%8s%12s%12s%12s%12s' %
      ('h_bml bin (km)', 'n', 'V_bml frac', 'Mg#_mant', 'Mg#_BSM', 'BSM sd'))
for i in range(len(BINS) - 1):
    m = solid & (hbml >= BINS[i]) & (hbml < BINS[i+1])
    if m.sum() < 10:
        continue
    print('%-16s%8d%12.4f%12.3f%12.3f%12.4f' %
          ('%d-%d' % (BINS[i], BINS[i+1]), int(m.sum()),
           np.median(f_bml[m]), np.median(mgm[m]), np.median(mg_bsm[m]),
           np.std(mg_bsm[m], ddof=1)))

for lab, m in [('solid', solid), ('molten', molten), ('layered', layer)]:
    if m.sum() < 10:
        continue
    q = np.percentile(mg_bsm[m], [5, 50, 95])
    print('  %-8s Mg#_BSM 5/50/95 = %.4f / %.4f / %.4f   sd = %.4f'
          % (lab, q[0], q[1], q[2], np.std(mg_bsm[m], ddof=1)))
print('  對比 Mg#(地函) 全體 sd = %.4f' % np.std(mgm, ddof=1))

# ══════════════════════════════════════════════════════════════════════════
# C. 直接從 profile 量 BML 頂的速度跳躍
# ══════════════════════════════════════════════════════════════════════════
print('')
print('C. BML 頂界面的實際跳躍(從 profile npz 直讀,不由 Mg# 推導)')
print('-' * 100)

idx_solid = np.flatnonzero(solid)
jump = []
for i in idx_solid:
    p = os.path.join(dirs_all[i], 'profile_s%05d.npz' % step[i])
    if not os.path.exists(p):
        continue
    try:
        z = np.load(p)
        if 'layer' not in z.files:
            continue
        lay = np.asarray(z['layer'], int)
        zz  = np.asarray(z['z_km'], float)
        vp  = np.asarray(z['Vp'], float)
        vs  = np.asarray(z['Vs'], float)
        rho = np.asarray(z['rho'], float)
    except Exception:
        continue
    m_man, m_bml = (lay == 1), (lay == 2)
    if not (m_man.any() and m_bml.any()):
        continue
    # 地函最深一點 vs BML 最淺一點
    j_man = np.flatnonzero(m_man)[np.argmax(zz[m_man])]
    j_bml = np.flatnonzero(m_bml)[np.argmin(zz[m_bml])]
    jump.append((hbml[i], mgb[i] - mgm[i],
                 100.0*(vs[j_bml] - vs[j_man])/max(vs[j_man], 1e-9),
                 100.0*(vp[j_bml] - vp[j_man])/max(vp[j_man], 1e-9),
                 100.0*(rho[j_bml] - rho[j_man])/max(rho[j_man], 1e-9)))
    if len(jump) >= N_PROFILE*(len(BINS) - 1):
        break

if not jump:
    print('  找不到留有 profile npz 的固態樣本 -> C 段跳過')
else:
    J = np.array(jump)
    print('  量到 %d 個固態 profile' % len(J))
    print('%-16s%8s%12s%12s%12s%12s' %
          ('h_bml bin (km)', 'n', 'delta_Mg#', 'dVs %', 'dVp %', 'drho %'))
    for i in range(len(BINS) - 1):
        m = (J[:, 0] >= BINS[i]) & (J[:, 0] < BINS[i+1])
        if m.sum() < 5:
            continue
        print('%-16s%8d%12.3f%12.2f%12.2f%12.2f' %
              ('%d-%d' % (BINS[i], BINS[i+1]), int(m.sum()),
               np.median(J[m, 1]), np.median(J[m, 2]),
               np.median(J[m, 3]), np.median(J[m, 4]))) 
    print('')
    print('  |dVs| < 1%% 的比例 = %.1f%%' % (100*np.mean(np.abs(J[:, 2]) < 1.0)))
    print('  |dVs| < 2%% 的比例 = %.1f%%' % (100*np.mean(np.abs(J[:, 2]) < 2.0)))
    print('  dVs 5/50/95 %% = %.2f / %.2f / %.2f'
          % tuple(np.percentile(J[:, 2], [5, 50, 95])))

# ══════════════════════════════════════════════════════════════════════════
# 圖 1:delta 對 BML 厚度
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(hbml[molten], (mgb - mgm)[molten], s=6, color='#CD5C5C',
           alpha=0.35, edgecolors='none', label='molten')
ax.scatter(hbml[solid], (mgb - mgm)[solid], s=8, color='#4E6E8E',
           alpha=0.6, edgecolors='none', label='solid')
ax.axhline(0.0, color='k', lw=1.2, ls='--')
ax.set_xlabel('BML thickness (km)', fontsize=fontsize)
ax.set_ylabel('Mg#$_{bulk,bml}$ - Mg#$_{mantle}$', fontsize=fontsize)
ax.tick_params(labelsize=TICKSIZE)
ax.legend(loc='lower right', fontsize=fontsize-2)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(os.path.join(FIG_DIR, '107_delta_Mg_vs_thickness.png'), dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 圖 2:Mg#_BSM 與 Mg#(地函) 的後驗寬度對比
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(mgm[solid],    bins=50, density=True, histtype='step', lw=2.0,
        color='#4E6E8E', label='Mg# mantle (solid)')
ax.hist(mg_bsm[solid], bins=50, density=True, histtype='step', lw=2.0,
        color='#7B9E87', label='Mg# BSM (solid)')
ax.hist(mgb[solid],    bins=50, density=True, histtype='step', lw=2.0,
        color='#C0392B', label='Mg# bulk_bml (solid)')
ax.set_xlabel('Mg#', fontsize=fontsize)
ax.set_ylabel('density', fontsize=fontsize)
ax.tick_params(labelsize=TICKSIZE)
ax.legend(loc='upper left', fontsize=fontsize-2)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(os.path.join(FIG_DIR, '107_Mg_BSM_vs_mantle.png'), dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 圖 3:速度跳躍
# ══════════════════════════════════════════════════════════════════════════
if jump:
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(J[:, 0], J[:, 2], c=J[:, 1], s=14, cmap='coolwarm',
                    edgecolors='none')
    ax.axhline(0.0, color='k', lw=1.2, ls='--')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Mg#$_{bml}$ - Mg#$_{mantle}$', fontsize=fontsize)
    cb.ax.tick_params(labelsize=TICKSIZE)
    ax.set_xlabel('BML thickness (km)', fontsize=fontsize)
    ax.set_ylabel('Vs jump at BML top (%)', fontsize=fontsize)
    ax.tick_params(labelsize=TICKSIZE)
    fig.tight_layout()
    if SAVE_FIG:
        fig.savefig(os.path.join(FIG_DIR, '107_Vs_jump_at_BML_top.png'), dpi=150)
    plt.show()