#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:58:49 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 98: 內核尺寸 x BML 三態 —— dynamo 論證的後驗預測檢驗
#
# 論證:
#   內核成核 -> 排出輕元素 + 釋放潛熱 -> 組成對流 -> dynamo。
#   火星表面磁場約 4.1 Ga 消失,且核只會持續冷卻,內核不會停止成長。
#   因此「現在存在大內核」蘊含「現在存在 dynamo」,與觀測矛盾。
#
# 這個約束沒有進 likelihood(70 的 GAP_ON = False),所以是乾淨的
# posterior predictive check —— 模型不知道有這回事,自己吐出來的解被
# 一個獨立的觀測事實排除。
#
# R_ic 的身分(必須寫進論文):
#   find_R_IC 是靜態溫度計 —— 絕熱線與 T_peritectic(Sakai 2026b Eq.2)的
#   最淺交點。不含潛熱、結晶分異、固核密度回饋。液相線的定義域是
#   P >= 21 GPa(程式用 P_valid = 27),而火星核頂只有 19-22 GPa,所以
#   低 T_core 的解在核頂附近是外插。R_ic 是指標,不是預測。
#
# 另外:build_taup 從 CMB 到地心 Vs 全部寫 0,沒有 inner-core 界面。
# 也就是說 R_ic > 0 的樣本,其 seismic forward model 仍是全液態核。
# 這是模型的內部不一致,SKS 走的是不含內核的路徑 —— 要寫進 limitations,
# 但也正因如此,R_ic 完全沒有回饋進 misfit,檢驗不循環。

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR

# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = MCMC_DIR
PREFIXES  = ['chain_a48B', 'chain_a49B', 'chain_a50B', 'chain_a51B', 'chain_a52B']
EXCLUDE   = []

BURNIN_FRAC = 0.4
DROP_LAST   = 1
MIN_STEPS   = 400
H_TOL       = 1.0        # km,固/液次層厚度判準

RIC_MAX_KM  = 750.0      # Irving 2023 SKS 最大穿透深度(主結果門檻)
RIC_ZERO    = 1.0        # km,視為「無內核」的門檻(敏感度用)
WS_EOS_MAX  = 0.36485    # core_fes.wS_to_chi 的硬 clip = 純 FeS 端元
                         # 超過此值 EoS 未定義,樣本會 silent 塌到 chi=1
DROP_ABOVE_EOS = True    # 主結果先切掉 EoS 定義域外的樣本

fontsize  = 14
TICKSIZE  = 14
SAVE      = False

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

    d = {p: np.array([float(s['params'].get(p, np.nan)) for s in recs])
         for p in ['T_core', 'w_S', 'R_cmb', 'BML_thickness', 'Mg#_bulk_bml']}
    for k in ['R_ic_km', 'gap_min', 'h_solid_km', 'h_liquid_km', 'misfit',
              'Vp_core_cmb', 'P_cmb', 'rho_core_mean', 'tt_n_miss']:
        d[k] = g(k)
    d['state']    = classify_state(d['h_solid_km'], d['h_liquid_km'])
    d['chain_id'] = np.array(cid)
    d['n_chain']  = len(set(cid))

    # S1000a 是唯一帶 SKS 的事件;solid 分支若系統性 miss,代表資料已在
    # 透過 MISS_PENALTY 懲罰它們,那是另一條機制,要跟 R_ic 分開講
    d['sks_miss'] = np.array([
        float((s.get('tt_miss_by_event') or {}).get('S1000a', 0)) for s in recs])
    return d

def frac(st, mask=None):
    m = (st != 'fail') if mask is None else ((st != 'fail') & mask)
    n = m.sum()
    return ({k: np.nan for k in STATES}, 0) if n == 0 else \
           ({k: (st[m] == k).sum()/n for k in STATES}, int(n))

# ══════════════════════════════════════════════════════════════════════════
RUNS = {}
for pf in PREFIXES:
    d = load_run(pf) if glob.glob(os.path.join(CHAIN_DIR, f'{pf}_*')) else None
    if d is None:
        print(f'{pf}: 無可用鏈 -> 跳過')
    else:
        RUNS[pf] = d
if not RUNS:
    raise SystemExit('no runs loaded')

print('=' * 96)
print('載入')
print('=' * 96)
for pf, d in RUNS.items():
    n_eos = int(np.sum(d['w_S'] > WS_EOS_MAX))
    print(f'{pf:<14s} N={len(d["misfit"]):7d}  chains={d["n_chain"]:3d}  '
          f'w_S > {WS_EOS_MAX:.3f}(EoS 定義域外): {n_eos:5d} '
          f'({100*n_eos/len(d["misfit"]):.2f}%)')

if DROP_ABOVE_EOS:
    print(f'\n已切除 w_S > {WS_EOS_MAX:.5f} 的樣本(core_fes.wS_to_chi 在此硬 clip,')
    print('該區間 EoS 平坦,likelihood 無梯度,樣本全部等價於純 FeS 端元)')
    for pf, d in RUNS.items():
        keep = d['w_S'] <= WS_EOS_MAX
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape:
                d[k] = v[keep]

# ══════════════════════════════════════════════════════════════════════════
# 交叉列表:三態 x 內核尺寸
# ══════════════════════════════════════════════════════════════════════════
BINS = [(0.0, RIC_ZERO, 'R_ic = 0'),
        (RIC_ZERO, RIC_MAX_KM, f'0 < R_ic < {RIC_MAX_KM:.0f}'),
        (RIC_MAX_KM, np.inf, f'R_ic > {RIC_MAX_KM:.0f}')]

print('\n' + '=' * 96)
print('交叉列表:BML 三態 x 內核尺寸   (每格為佔該 run 全體的 %)')
print('=' * 96)
for pf, d in RUNS.items():
    st, ric = d['state'], d['R_ic_km']
    ok = (st != 'fail') & np.isfinite(ric)
    N  = int(ok.sum())
    print(f'\n{pf}   (N = {N})')
    print(f"{'':<10s}" + ''.join(f'{lab:>22s}' for _, _, lab in BINS) + f"{'row %':>10s}")
    for s in STATES:
        row, tot = '', 0.0
        for lo, hi, _ in BINS:
            m = ok & (st == s) & (ric >= lo) & (ric < hi)
            p = 100*m.sum()/N
            tot += p
            row += f'{p:22.2f}'
        print(f'{s:<10s}{row}{tot:10.2f}')
    col = ''
    for lo, hi, _ in BINS:
        m = ok & (ric >= lo) & (ric < hi)
        col += f'{100*m.sum()/N:22.2f}'
    print(f"{'col %':<10s}{col}{100.0:10.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 過濾前後的三態比例
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 96)
print('套用 dynamo 約束後的三態比例')
print('=' * 96)
print(f"{'run':<14s}{'filter':<22s}{'N':>8s}" +
      ''.join(f'{s:>11s}' for s in STATES))
for pf, d in RUNS.items():
    st, ric = d['state'], d['R_ic_km']
    fin = np.isfinite(ric)
    for lab, m in [('none', np.ones(len(st), bool)),
                   (f'R_ic < {RIC_MAX_KM:.0f} km', fin & (ric < RIC_MAX_KM)),
                   (f'R_ic < {RIC_ZERO:.0f} km', fin & (ric < RIC_ZERO)),
                   ('gap_min > 0', np.isfinite(d['gap_min']) & (d['gap_min'] > 0))]:
        f, n = frac(st, m)
        print(f'{pf:<14s}{lab:<22s}{n:8d}' +
              ''.join(f'{100*f[s]:10.1f}%' for s in STATES))
    print()

# ══════════════════════════════════════════════════════════════════════════
# solid 分支的 SKS miss rate:資料是否已在懲罰它們
# ══════════════════════════════════════════════════════════════════════════
print('=' * 96)
print('S1000a(唯一帶 SKS 的事件)的 miss rate 與總 miss 數,按三態')
print('=' * 96)
print(f"{'run':<14s}{'state':<10s}{'n':>8s}{'SKS miss':>11s}{'tt_n_miss':>12s}"
      f"{'Vp_cmb':>10s}{'misfit':>10s}")
for pf, d in RUNS.items():
    for s in STATES:
        m = (d['state'] == s)
        if not m.any():
            continue
        print(f'{pf:<14s}{s:<10s}{int(m.sum()):8d}'
              f'{np.nanmean(d["sks_miss"][m]):11.3f}'
              f'{np.nanmean(d["tt_n_miss"][m]):12.2f}'
              f'{np.nanmedian(d["Vp_core_cmb"][m]):10.3f}'
              f'{np.nanmedian(d["misfit"][m]):10.3f}')
    print()

# ══════════════════════════════════════════════════════════════════════════
# solid 分支的 R_ic 分位數
# ══════════════════════════════════════════════════════════════════════════
print('=' * 96)
print('R_ic 分位數(只取 R_ic > 0 的樣本),按三態')
print('=' * 96)
print(f"{'run':<14s}{'state':<10s}{'n(R_ic>0)':>11s}{'P(R_ic>0)':>11s}"
      f"{'5%':>9s}{'50%':>9s}{'95%':>9s}")
for pf, d in RUNS.items():
    for s in STATES:
        m = (d['state'] == s) & np.isfinite(d['R_ic_km'])
        if not m.any():
            continue
        pos = m & (d['R_ic_km'] > RIC_ZERO)
        if pos.sum() < 3:
            print(f'{pf:<14s}{s:<10s}{int(pos.sum()):11d}'
                  f'{100*pos.sum()/m.sum():10.2f}%' + ' ' * 27)
            continue
        q = np.percentile(d['R_ic_km'][pos], [5, 50, 95])
        print(f'{pf:<14s}{s:<10s}{int(pos.sum()):11d}'
              f'{100*pos.sum()/m.sum():10.2f}%'
              f'{q[0]:9.0f}{q[1]:9.0f}{q[2]:9.0f}')
    print()

# ══════════════════════════════════════════════════════════════════════════
# 圖 A:R_ic vs T_core / w_S,按三態
# ══════════════════════════════════════════════════════════════════════════
n = len(RUNS)
fig, axes = plt.subplots(2, n, figsize=(4.8*n, 9.0), squeeze=False)
for k, (pf, d) in enumerate(RUNS.items()):
    for row, (xk, xlab) in enumerate([('T_core', 'T$_{core}$ (K)'),
                                      ('w_S', 'w$_S$')]):
        ax = axes[row, k]
        for s in STATES:
            m = (d['state'] == s) & np.isfinite(d['R_ic_km'])
            if m.any():
                ax.scatter(d[xk][m], d['R_ic_km'][m], s=5, alpha=0.45,
                           color=STATE_COLOR[s], label=s, linewidths=0)
        ax.axhline(RIC_MAX_KM, color='k', ls='--', lw=1.5)
        ax.text(0.02, RIC_MAX_KM + 25, f'Irving <{RIC_MAX_KM:.0f} km',
                transform=ax.get_yaxis_transform(), fontsize=9, va='bottom')
        ax.set_xlabel(xlab, fontsize=fontsize)
        ax.set_ylabel('R$_{ic}$ (km)', fontsize=fontsize)
        ax.tick_params(labelsize=TICKSIZE)
        if row == 0:
            ax.set_title(pf.replace('chain_', ''), fontsize=fontsize)
        if row == 0 and k == 0:
            ax.legend(fontsize=10, loc='upper right', markerscale=2.5)
fig.suptitle('inner-core size vs core temperature and light-element content',
             fontsize=fontsize + 2)
fig.tight_layout()
if SAVE: fig.savefig('98_ric_vs_Tcore_wS.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 圖 B:三態比例對 R_ic 門檻的響應
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, n, figsize=(4.6*n, 4.4), squeeze=False)
thr = np.concatenate([[RIC_ZERO], np.linspace(50.0, 1500.0, 40)])
for k, (pf, d) in enumerate(RUNS.items()):
    ax = axes[0, k]
    curves = {s: [] for s in STATES}
    nkeep = []
    for t in thr:
        m = np.isfinite(d['R_ic_km']) & (d['R_ic_km'] < t)
        f, nn = frac(d['state'], m)
        for s in STATES:
            curves[s].append(100*f[s])
        nkeep.append(nn)
    for s in STATES:
        ax.plot(thr, curves[s], color=STATE_COLOR[s], lw=2.0, label=s)
    ax.axvline(RIC_MAX_KM, color='k', ls='--', lw=1.5)
    ax.set_xlabel('R$_{ic}$ threshold (km)', fontsize=fontsize)
    ax.set_ylabel('posterior fraction (%)', fontsize=fontsize)
    ax.set_title(pf.replace('chain_', ''), fontsize=fontsize)
    ax.tick_params(labelsize=TICKSIZE)
    if k == 0:
        ax.legend(fontsize=10, loc='center right')
    axt = ax.twinx()
    axt.plot(thr, 100*np.array(nkeep)/max(nkeep[-1], 1), color='0.55',
             lw=1.2, ls=':')
    axt.set_ylabel('samples retained (%)', fontsize=fontsize - 2, color='0.45')
    axt.tick_params(labelsize=TICKSIZE, colors='0.45')
fig.suptitle('dashed = Irving 2023 SKS penetration limit   '
             'dotted grey = fraction of samples retained', fontsize=fontsize)
fig.tight_layout()
if SAVE: fig.savefig('98_state_vs_ric_threshold.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 圖 C:gap_min 分佈,按三態。gap <= 0 表示絕熱線碰到液相線 -> 有內核
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, n, figsize=(4.6*n, 4.2), squeeze=False)
for k, (pf, d) in enumerate(RUNS.items()):
    ax = axes[0, k]
    g = d['gap_min']
    fin = np.isfinite(g)
    bins = np.histogram_bin_edges(g[fin], bins=45)
    for s in STATES:
        m = fin & (d['state'] == s)
        if m.any():
            ax.hist(g[m], bins=bins, histtype='step', lw=1.8,
                    color=STATE_COLOR[s], label=s)
    ax.axvline(0.0, color='k', ls='--', lw=1.5)
    ax.set_xlabel('gap$_{min}$ = min(T$_{ad}$ - T$_{liq}$)  (K)', fontsize=fontsize)
    ax.set_ylabel('count', fontsize=fontsize)
    ax.set_title(pf.replace('chain_', ''), fontsize=fontsize)
    ax.tick_params(labelsize=TICKSIZE)
    if k == 0:
        ax.legend(fontsize=10)
fig.suptitle('gap$_{min}$ <= 0: adiabat reaches the liquidus, inner core nucleates',
             fontsize=fontsize)
fig.tight_layout()
if SAVE: fig.savefig('98_gapmin_by_state.png', dpi=150)
plt.show()