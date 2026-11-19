#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:19:42 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 102: 島化是什麼造成的?
#
# 兩個假說對「凍結的鏈」給出相反的預測:
#   H1 缺失相位  -> tt_n_miss 高、misfit_tt 主導、geodetic 項不特別高
#                   MISS_PENALTY 在似然面上造出位置隨參數移動的牆
#   H2 geodetic 太緊 -> mass_sigma 或 moi_sigma 高、misfit_tt 正常
#
# 把每個 run 的鏈依末段 acceptance 分成 frozen / alive 兩組,逐項比較。
# 對照組:a48B-a51B 用同一套 Khan 地殼卻沒有島化,所以地殼本身不是充分條件。

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR

# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = MCMC_DIR
# 島化的 run 放前面,沒島化的對照組放後面
PREFIXES  = ['chain_a44B', 'chain_a45B', 'chain_a46B', 'chain_a48B']

TAIL_FRAC   = 0.3        # 用末段多少比例判斷凍結
FROZEN_ACC  = 0.02       # 末段 acceptance 低於此值 -> frozen
MIN_STEPS   = 300
DROP_LAST   = 1

KEYS = ['misfit', 'misfit_tt', 'misfit_solidus', 'mass_sigma', 'moi_sigma',
        'tt_n_miss', 'tt_n_ph', 'tt_n_capped', 'tt_n_branch', 'tt_n_ev']

fontsize = 14
TICKSIZE = 14
SAVE     = False
C_FROZEN, C_ALIVE = '#CD5C5C', '#4E6E8E'

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

def col(recs, k):
    return np.array([np.nan if s.get(k) is None else float(s[k]) for s in recs])

def load_run(prefix):
    out = []
    for dd in sorted(glob.glob(os.path.join(CHAIN_DIR, f'{prefix}_*'))):
        ch = load_chain_dir(dd)
        if ch is None:
            continue
        if DROP_LAST > 0 and len(ch) > DROP_LAST:
            ch = ch[:-DROP_LAST]
        if len(ch) < MIN_STEPS:
            continue
        acc = np.array([1.0 if s.get('accepted') else 0.0 for s in ch])
        t0  = int(len(ch)*(1.0 - TAIL_FRAC))
        d = dict(name=os.path.basename(dd), n=len(ch),
                 acc_all=acc.mean(), acc_tail=acc[t0:].mean(),
                 recs_tail=ch[t0:])
        for k in KEYS:
            d[k] = np.nanmedian(col(ch[t0:], k))
        out.append(d)
    return out or None

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
print('=' * 108)
print(f'逐鏈狀態  (末段 {TAIL_FRAC:.0%},acceptance < {FROZEN_ACC:.0%} 判為 frozen)')
print('=' * 108)
for pf, chains in RUNS.items():
    nf = sum(c['acc_tail'] < FROZEN_ACC for c in chains)
    print(f'\n{pf}   {len(chains)} chains,frozen {nf},alive {len(chains)-nf}')
    print(f"  {'chain':<18s}{'n':>7s}{'acc_all':>9s}{'acc_tail':>10s}"
          f"{'misfit':>9s}{'tt':>8s}{'mass_s':>9s}{'moi_s':>8s}"
          f"{'n_miss':>8s}{'n_ph':>7s}")
    for c in sorted(chains, key=lambda x: x['acc_tail']):
        tag = ' FROZEN' if c['acc_tail'] < FROZEN_ACC else ''
        print(f"  {c['name']:<18s}{c['n']:7d}{100*c['acc_all']:8.1f}%"
              f"{100*c['acc_tail']:9.1f}%{c['misfit']:9.2f}{c['misfit_tt']:8.2f}"
              f"{c['mass_sigma']:9.2f}{c['moi_sigma']:8.2f}"
              f"{c['tt_n_miss']:8.1f}{c['tt_n_ph']:7.1f}{tag}")

print('\n' + '=' * 108)
print('frozen vs alive 的中位數對照 —— 哪一項在凍結的鏈裡特別高?')
print('=' * 108)
print(f"{'run':<14s}{'group':<9s}{'n':>4s}" +
      ''.join(f'{k.replace("misfit_",""):>14s}' for k in KEYS))
for pf, chains in RUNS.items():
    for lab, sel in [('frozen', lambda c: c['acc_tail'] < FROZEN_ACC),
                     ('alive',  lambda c: c['acc_tail'] >= FROZEN_ACC)]:
        g = [c for c in chains if sel(c)]
        if not g:
            continue
        print(f'{pf:<14s}{lab:<9s}{len(g):4d}' +
              ''.join(f'{np.nanmedian([c[k] for c in g]):14.2f}' for k in KEYS))
    print()

print('=' * 108)
print('判讀')
print('=' * 108)
print('  frozen 的 tt_n_miss 明顯高於 alive,而 mass_sigma/moi_sigma 相當')
print('    -> H1:MISS_PENALTY 造出的移動牆。解法是處理缺失相位,不是放鬆 geodetic。')
print('  frozen 的 mass_sigma 或 moi_sigma 明顯高於 alive,而 tt_n_miss 相當')
print('    -> H2:geodetic 過緊。解法是放鬆 sigma 或降低權重。')
print('  兩者都高 -> 交互作用:geodetic 把鏈推到 forward model 容易失敗的角落。')

# ══════════════════════════════════════════════════════════════════════════
# 圖:acceptance vs 各項,frozen/alive 分色
# ══════════════════════════════════════════════════════════════════════════
SHOW = ['tt_n_miss', 'mass_sigma', 'moi_sigma', 'misfit_tt']
fig, axes = plt.subplots(len(RUNS), len(SHOW),
                         figsize=(3.9*len(SHOW), 3.4*len(RUNS)), squeeze=False)
for i, (pf, chains) in enumerate(RUNS.items()):
    for j, k in enumerate(SHOW):
        ax = axes[i, j]
        for c in chains:
            fr = c['acc_tail'] < FROZEN_ACC
            ax.scatter(c[k], 100*c['acc_tail'], s=55, linewidths=0,
                       color=C_FROZEN if fr else C_ALIVE, alpha=0.85)
        ax.axhline(100*FROZEN_ACC, color='k', ls='--', lw=1.2)
        ax.set_xlabel(k, fontsize=fontsize)
        if j == 0:
            ax.set_ylabel(f"{pf.replace('chain_','')}\ntail acceptance (%)",
                          fontsize=fontsize)
        ax.tick_params(labelsize=TICKSIZE)
fig.suptitle('red = frozen chains; a horizontal spread in one column identifies the cause',
             fontsize=fontsize + 1)
fig.tight_layout()
if SAVE: fig.savefig('102_islanding_cause.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 逐事件缺失:凍結的鏈卡在哪些事件
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 108)
print('凍結鏈的逐事件平均缺失相位數(只列非零者)')
print('=' * 108)
for pf, chains in RUNS.items():
    fro = [c for c in chains if c['acc_tail'] < FROZEN_ACC]
    ali = [c for c in chains if c['acc_tail'] >= FROZEN_ACC]
    if not fro:
        print(f'\n{pf}: 無凍結鏈'); continue
    evs = set()
    for c in chains:
        for s in c['recs_tail']:
            evs |= set((s.get('tt_miss_by_event') or {}).keys())
    if not evs:
        print(f'\n{pf}: chain.jsonl 無 tt_miss_by_event 欄位'); continue
    print(f'\n{pf}')
    print(f"  {'event':<12s}{'frozen':>10s}{'alive':>10s}{'diff':>10s}")
    rows = []
    for ev in sorted(evs):
        def m(g):
            v = [float((s.get('tt_miss_by_event') or {}).get(ev, 0))
                 for c in g for s in c['recs_tail']]
            return np.mean(v) if v else np.nan
        a, b = m(fro), m(ali) if ali else np.nan
        if np.nansum([a, b]) > 0:
            rows.append((ev, a, b))
    for ev, a, b in sorted(rows, key=lambda r: -(r[1] - (r[2] if r[2] == r[2] else 0))):
        print(f'  {ev:<12s}{a:10.3f}{b:10.3f}{a-b:10.3f}')