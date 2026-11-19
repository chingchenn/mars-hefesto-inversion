#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:19:12 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 104: 質量與慣量矩的後驗預測分佈
#
# 這張圖顯示的是 posterior predictive distribution,不是「通過與否」——
# 在本研究的 likelihood 裡質量與 MoI 是 L1 軟罰項,沒有門檻。
#
# sigma 的身分(寫進圖說):
#   MARS_MASS_SIGMA = 2.7e-3 = sqrt(0.031^2 + 0.032^2 + 0.259^2) %
#       0.031  平均密度 3935 +- 1.2 kg/m3 (Rivoldini 2011)
#       0.032  nGibbs vs HeFESTo, 2-19 GPa, 同組成同 P-T
#       0.259  固定 0-100 km 地殼的前向敏感度(Samuel 2023 換成 Kim 2023)
#   MOI_SIGMA = 6.4e-4 = Konopliv 2020 的約 10 倍
#       同一個固定地殼的理由,且 I ~ r^4 對淺部的加權大於 M ~ r^2
#       (記錄 §2.1:換殼時 MoI 的效應是質量的 2.8 倍)
#
# 三態分開報:若三態的 M/MoI 分佈重疊,兩階段過濾是中性的;
# 若分開,過濾會直接改變三態比例,那時 sigma 的選擇就會影響結論。

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = MCMC_DIR
PREFIXES  = ['chain_a52B']
EXCLUDE   = []

BURNIN_FRAC = 0.4
DROP_LAST   = 1
MIN_STEPS   = 400
H_TOL       = 1.0

MARS_MASS_OBS = 6.4171e23
MOI_OBS       = 0.3634            # Konopliv 2020
MASS_SIG_OBS  = 3.05e-4           # 相對值,Rivoldini 2011
MOI_SIG_OBS   = 6.0e-5            # Konopliv 2020

MASS_SIG_RUN  = 2.7e-3            # 相對值,本研究使用
MOI_SIG_RUN   = 6.4e-4            # 本研究使用

# 對照(uu 先前以同一積分器算得;設 None 則不畫)
KHAN_DM   = (0.028, 0.034, 0.040)     # % , 16/50/84
KHAN_MOI  = (0.36340, 0.36341, 0.36342)
SAMUEL_DM  = 0.598                    # %
SAMUEL_MOI = 0.36529

K_LEVELS = [1, 2, 3, 5, 10]

fontsize = 14
TICKSIZE = 14
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
STATES      = ['solid', 'layered', 'molten']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}

# ══════════════════════════════════════════════════════════════════════════
def load_chain_dir(dd):
    p = os.path.join(dd, 'chain.jsonl')
    if not os.path.exists(p):
        p = os.path.join(dd, 'chain.json')
        if not os.path.exists(p):
            return None
        try:
            return json.loads(open(p).read())
        except json.JSONDecodeError:
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

def load_run(prefix):
    recs, cid = [], []
    for c, dd in enumerate(sorted(glob.glob(os.path.join(CHAIN_DIR, f'{prefix}_*')))):
        if os.path.basename(dd) in EXCLUDE:
            continue
        ch = load_chain_dir(dd)
        if ch is None:
            continue
        if DROP_LAST > 0 and len(ch) > DROP_LAST:
            ch = ch[:-DROP_LAST]
        if len(ch) < MIN_STEPS:
            continue
        b = int(len(ch)*BURNIN_FRAC)
        recs.extend(ch[b:]); cid.extend([c]*(len(ch)-b))
    if not recs:
        return None

    def g(k):
        return np.array([np.nan if s.get(k) is None else float(s[k]) for s in recs])

    d = dict(M=g('M_pred'), moi=g('moi_pred'),
             hs=g('h_solid_km'), hl=g('h_liquid_km'),
             mass_sigma=g('mass_sigma'), moi_sigma=g('moi_sigma'),
             chain_id=np.array(cid), n_chain=len(set(cid)))
    d['state'] = classify_state(d['hs'], d['hl'])
    d['dM']    = 100.0*(d['M'] - MARS_MASS_OBS)/MARS_MASS_OBS      # %
    d['dmoi']  = d['moi'] - MOI_OBS
    # 以觀測 sigma 為單位
    d['zM']    = np.abs(d['dM']/100.0)/MASS_SIG_OBS
    d['zmoi']  = np.abs(d['dmoi'])/MOI_SIG_OBS
    return d

RUNS = {}
for pf in PREFIXES:
    r = load_run(pf) if glob.glob(os.path.join(CHAIN_DIR, f'{pf}_*')) else None
    if r is None:
        print(f'{pf}: 無可用鏈 -> 跳過')
    else:
        RUNS[pf] = r
if not RUNS:
    raise SystemExit('no runs loaded')

# ══════════════════════════════════════════════════════════════════════════
print('=' * 104)
print('後驗預測的質量與慣量矩   (dM 為 %, MoI 為絕對值)')
print(f'  觀測: M sigma = {MASS_SIG_OBS:.3e} (相對), MoI = {MOI_OBS} +- {MOI_SIG_OBS:.1e}')
print(f'  本研究: M sigma = {MASS_SIG_RUN:.3e} ({MASS_SIG_RUN/MASS_SIG_OBS:.1f}x), '
      f'MoI sigma = {MOI_SIG_RUN:.1e} ({MOI_SIG_RUN/MOI_SIG_OBS:.1f}x)')
print('=' * 104)
print(f"{'run':<14s}{'N':>8s}{'chains':>7s}"
      f"{'dM 5/50/95 (%)':>28s}{'MoI 5/50/95':>30s}")
for pf, d in RUNS.items():
    a = np.nanpercentile(d['dM'], [5, 50, 95])
    b = np.nanpercentile(d['moi'], [5, 50, 95])
    print(f'{pf:<14s}{len(d["dM"]):8d}{d["n_chain"]:7d}'
          f'{a[0]:9.3f}{a[1]:9.3f}{a[2]:9.3f}'
          f'{b[0]:10.5f}{b[1]:10.5f}{b[2]:10.5f}')

print('\n' + '=' * 104)
print('按三態:dM 與 MoI 的中位數與 5/95 —— 若三態重疊,兩階段過濾是中性的')
print('=' * 104)
print(f"{'run':<14s}{'state':<9s}{'n':>7s}"
      f"{'dM 5/50/95 (%)':>28s}{'MoI 5/50/95':>30s}")
for pf, d in RUNS.items():
    for st in STATES:
        m = (d['state'] == st)
        if m.sum() < 3:
            continue
        a = np.nanpercentile(d['dM'][m], [5, 50, 95])
        b = np.nanpercentile(d['moi'][m], [5, 50, 95])
        print(f'{pf:<14s}{st:<9s}{int(m.sum()):7d}'
              f'{a[0]:9.3f}{a[1]:9.3f}{a[2]:9.3f}'
              f'{b[0]:10.5f}{b[1]:10.5f}{b[2]:10.5f}')
    print()

print('=' * 104)
print('落在觀測 sigma 的 k 倍以內的樣本比例')
print('=' * 104)
hdr = ''.join(f'{f"k={k}":>10s}' for k in K_LEVELS)
print(f"{'run':<14s}{'quantity':<12s}{hdr}")
for pf, d in RUNS.items():
    for nm, z in [('mass', d['zM']), ('MoI', d['zmoi']),
                  ('both', np.maximum(d['zM'], d['zmoi']))]:
        row = ''.join(f'{100*np.nanmean(z < k):9.1f}%' for k in K_LEVELS)
        print(f'{pf:<14s}{nm:<12s}{row}')
    print()

# ══════════════════════════════════════════════════════════════════════════
# 圖:dM 與 MoI 直方圖,按三態
# ══════════════════════════════════════════════════════════════════════════
n = len(RUNS)
fig, axes = plt.subplots(2, n, figsize=(4.9*n, 8.4), squeeze=False)
for k, (pf, d) in enumerate(RUNS.items()):
    for row, (key, xlab) in enumerate([('dM', r'$\Delta M/M_{obs}$ (%)'),
                                       ('moi', r'$I/MR^2$')]):
        ax = axes[row, k]
        v = d[key]
        fin = np.isfinite(v)
        bins = np.histogram_bin_edges(v[fin], bins=50)
        for st in STATES:
            m = fin & (d['state'] == st)
            if m.sum():
                ax.hist(v[m], bins=bins, histtype='step', lw=1.8,
                        color=STATE_COLOR[st], label=st)
        if key == 'dM':
            ax.axvline(0.0, color='k', ls='--', lw=1.6)
            ax.axvspan(-100*MASS_SIG_RUN, 100*MASS_SIG_RUN,
                       color='0.85', zorder=0)
            if SAMUEL_DM is not None:
                ax.axvline(SAMUEL_DM, color='#C0392B', lw=1.8, ls='-')
            if KHAN_DM is not None:
                ax.axvspan(KHAN_DM[0], KHAN_DM[2], color='#4198B9', alpha=0.35,
                           zorder=1)
        else:
            ax.axvline(MOI_OBS, color='k', ls='--', lw=1.6)
            ax.axvspan(MOI_OBS - MOI_SIG_RUN, MOI_OBS + MOI_SIG_RUN,
                       color='0.85', zorder=0)
            if SAMUEL_MOI is not None:
                ax.axvline(SAMUEL_MOI, color='#C0392B', lw=1.8, ls='-')
            if KHAN_MOI is not None:
                ax.axvspan(KHAN_MOI[0], KHAN_MOI[2], color='#4198B9', alpha=0.35,
                           zorder=1)
        ax.set_xlabel(xlab, fontsize=fontsize)
        ax.set_ylabel('count', fontsize=fontsize)
        ax.tick_params(labelsize=TICKSIZE)
        if row == 0:
            ax.set_title(pf.replace('chain_', ''), fontsize=fontsize)
        if row == 0 and k == 0:
            ax.legend(fontsize=10)
fig.suptitle('posterior predictive M and MoI   '
             'grey band = sigma used here, dashed = observed, '
             'blue = Khan 2023, red = Samuel 2023', fontsize=fontsize)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '104_mass_moi_predictive.png'), dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 圖:dM vs MoI 散點,按三態。看兩者是否沿補償方向相關
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, n, figsize=(4.7*n, 4.4), squeeze=False)
for k, (pf, d) in enumerate(RUNS.items()):
    ax = axes[0, k]
    for st in STATES:
        m = (d['state'] == st) & np.isfinite(d['dM']) & np.isfinite(d['moi'])
        if m.sum():
            ax.scatter(d['dM'][m], d['moi'][m], s=4, alpha=0.35,
                       color=STATE_COLOR[st], linewidths=0, label=st)
    ax.axvline(0.0, color='k', ls='--', lw=1.3)
    ax.axhline(MOI_OBS, color='k', ls='--', lw=1.3)
    ax.axvspan(-100*MASS_SIG_RUN, 100*MASS_SIG_RUN, color='0.9', zorder=0)
    ax.axhspan(MOI_OBS - MOI_SIG_RUN, MOI_OBS + MOI_SIG_RUN,
               color='0.9', zorder=0)
    ax.set_xlabel(r'$\Delta M/M_{obs}$ (%)', fontsize=fontsize)
    ax.set_ylabel(r'$I/MR^2$', fontsize=fontsize)
    ax.set_title(pf.replace('chain_', ''), fontsize=fontsize)
    ax.tick_params(labelsize=TICKSIZE)
    if k == 0:
        ax.legend(fontsize=10, markerscale=3)
fig.suptitle('mass vs MoI, coloured by BML state', fontsize=fontsize + 1)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '104_mass_vs_moi.png'), dpi=150)
plt.show()