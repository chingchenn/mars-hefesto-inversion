#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:35:02 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 100: 後驗的模態結構
#   A. layered 是走廊還是第三個盆地?
#        走廊  -> 停留片段短、前後接不同狀態、只出現在有轉換的鏈
#        真盆地 -> 有長片段、前後接同一狀態、純 molten 鏈也產生 layered
#   B. 模態內收斂:鏈依 solid 比例分組,各組單獨算參數 R-hat 與分位數。
#      跨模態的 R-hat 沒有意義(2.3-5.7),但模態內部若收斂,
#      就能報「固態解對應 T_core = X ± Y」這種有內容的數字。
#   C. 轉換事件的位置:唯一發生過模態切換的那幾條鏈,轉換前後的參數。
#      那是屏障最薄的地方,是量化屏障位置的唯一資訊來源。
#
# 注意:這裡的所有比例都不是後驗機率。鏈之間沒有混合(見 99A),
# 合併平均只是在數有幾條鏈落在哪個盆地。

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = MCMC_DIR
PREFIXES  = ['chain_a52B']
EXCLUDE   = ['chain_a52B_04']

BURNIN_FRAC = 0.3
DROP_LAST   = 1
MIN_STEPS   = 400
H_TOL       = 1.0

MODE_HI     = 0.5        # 鏈的 solid 比例 > 此值 -> solid 模態
MODE_LO     = 0.05       # < 此值 -> molten 模態;之間 = mixed(發生過轉換)
RUN_MIN_N   = 50         # 進 R-hat 的最短鏈長
LONG_RUN    = 20         # 停留片段長於此步數視為「長片段」

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml',
          'BML_thickness', 'w_S', 'R_cmb']
STATES      = ['solid', 'layered', 'molten']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}

fontsize = 14
TICKSIZE = 14
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

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

def rhat_ess(mat):
    m, n = mat.shape
    if m < 2 or n < 10 or not np.all(np.isfinite(mat)):
        return np.nan, np.nan
    W = mat.var(axis=1, ddof=1).mean()
    B = n*mat.mean(axis=1).var(ddof=1)
    if W <= 0:
        return np.nan, np.nan
    vp = (n-1)/n*W + B/n
    def acov(c_):
        c_ = c_ - c_.mean()
        f = np.fft.rfft(c_, 2*n)
        return np.fft.irfft(f*np.conj(f))[:n]/n
    gamma = np.mean([acov(mat[j]) for j in range(m)], axis=0)
    rho = 1.0 - (W - gamma)/vp
    s_, t_ = 0.0, 1
    while t_ + 1 < n:
        pair = rho[t_] + rho[t_+1]
        if pair < 0:
            break
        s_ += pair; t_ += 2
    return np.sqrt(vp/W), m*n/(1.0 + 2.0*s_)

def runs_of(state, target):
    # 回傳 [(起點, 長度, 前一個狀態, 後一個狀態), ...]
    out, i, n = [], 0, len(state)
    while i < n:
        if state[i] != target:
            i += 1; continue
        j = i
        while j < n and state[j] == target:
            j += 1
        out.append((i, j-i,
                    state[i-1] if i > 0 else None,
                    state[j] if j < n else None))
        i = j
    return out

def load_run(prefix):
    chains = []
    for dd in sorted(glob.glob(os.path.join(CHAIN_DIR, f'{prefix}_*'))):
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
        recs = ch[b:]
        hs = np.array([np.nan if s.get('h_solid_km') is None else float(s['h_solid_km'])
                       for s in recs])
        hl = np.array([np.nan if s.get('h_liquid_km') is None else float(s['h_liquid_km'])
                       for s in recs])
        st = classify_state(hs, hl)
        d = {p: np.array([float(s['params'].get(p, np.nan)) for s in recs])
             for p in PARAMS}
        d['misfit'] = np.array([np.nan if s.get('misfit') is None else float(s['misfit'])
                                for s in recs])
        d['h_liquid_km'] = hl
        chains.append(dict(name=os.path.basename(dd), state=st, n=len(recs), **d))
    return chains or None

RUNS = {}
for pf in PREFIXES:
    c = load_run(pf) if glob.glob(os.path.join(CHAIN_DIR, f'{pf}_*')) else None
    if c is None:
        print(f'{pf}: 無可用鏈 -> 跳過')
    else:
        RUNS[pf] = c
if not RUNS:
    raise SystemExit('no runs loaded')

# ══════════════════════════════════════════════════════════════════════════
# A. layered 是走廊還是盆地
# ══════════════════════════════════════════════════════════════════════════
print('=' * 100)
print('A1. layered 停留片段的長度分佈')
print('    走廊的特徵:片段短(中位數 1-5 步)、幾乎沒有長片段')
print('=' * 100)
print(f"{'run':<14s}{'n_segments':>12s}{'median':>9s}{'mean':>9s}{'max':>7s}"
      f"{'len=1':>9s}{f'len>{LONG_RUN}':>10s}{'in long %':>11s}")
SEGS = {}
for pf, chains in RUNS.items():
    segs = [s for c in chains for s in runs_of(c['state'], 'layered')]
    SEGS[pf] = segs
    if not segs:
        print(f'{pf:<14s}  無 layered 片段'); continue
    L = np.array([s[1] for s in segs])
    print(f'{pf:<14s}{len(L):12d}{np.median(L):9.1f}{L.mean():9.2f}{L.max():7d}'
          f'{100*np.mean(L == 1):8.1f}%{100*np.mean(L > LONG_RUN):9.1f}%'
          f'{100*L[L > LONG_RUN].sum()/L.sum():10.1f}%')

print('\n' + '=' * 100)
print('A2. layered 片段的前後鄰居')
print('    走廊:前後不同(solid<->molten 穿越);盆地:前後相同(進去又原路出來)')
print('=' * 100)
print(f"{'run':<14s}{'cross':>9s}{'bounce_m':>10s}{'bounce_s':>10s}{'edge':>7s}"
      f"{'cross %':>10s}")
for pf, segs in SEGS.items():
    if not segs:
        continue
    cross = bm = bs = edge = 0
    for _, _, a, b in segs:
        if a is None or b is None:
            edge += 1
        elif a != b:
            cross += 1
        elif a == 'molten':
            bm += 1
        else:
            bs += 1
    tot = max(cross + bm + bs, 1)
    print(f'{pf:<14s}{cross:9d}{bm:10d}{bs:10d}{edge:7d}{100*cross/tot:9.1f}%')
print('\n  bounce_m = 從 molten 進入又回到 molten;bounce_s 同理但為 solid')

print('\n' + '=' * 100)
print('A3. 純 molten 鏈(solid 比例 < %.0f%%)是否也產生 layered' % (100*MODE_LO))
print('    若是,layered 至少有一部分是 molten 旁的真實鄰域,不只是穿越軌跡')
print('=' * 100)
print(f"{'run':<14s}{'pure-molten chains':>20s}{'their layered %':>17s}"
      f"{'mixed chains':>14s}{'their layered %':>17s}")
for pf, chains in RUNS.items():
    fs = np.array([(c['state'] == 'solid').mean() for c in chains])
    pure = [c for c, f in zip(chains, fs) if f < MODE_LO]
    mixed = [c for c, f in zip(chains, fs) if MODE_LO <= f <= MODE_HI]
    def lay(cs):
        if not cs:
            return np.nan
        tot = sum(c['n'] for c in cs)
        return 100*sum((c['state'] == 'layered').sum() for c in cs)/max(tot, 1)
    print(f'{pf:<14s}{len(pure):20d}{lay(pure):16.2f}%'
          f'{len(mixed):14d}{lay(mixed):16.2f}%')

fig, axes = plt.subplots(1, len(RUNS), figsize=(4.4*len(RUNS), 4.0), squeeze=False)
for k, (pf, segs) in enumerate(SEGS.items()):
    ax = axes[0, k]
    if segs:
        L = np.array([s[1] for s in segs])
        ax.hist(L, bins=np.arange(0.5, min(L.max(), 120) + 1.5),
                color=STATE_COLOR['layered'], edgecolor='none')
        ax.set_yscale('log')
        ax.axvline(LONG_RUN, color='k', ls='--', lw=1.5)
    ax.set_xlabel('layered segment length (steps)', fontsize=fontsize)
    ax.set_ylabel('count', fontsize=fontsize)
    ax.set_title(pf.replace('chain_', ''), fontsize=fontsize)
    ax.tick_params(labelsize=TICKSIZE)
fig.suptitle('short segments dominate => layered is a corridor, not a basin',
             fontsize=fontsize)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '100_layered_segment_lengths.png'), dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# B. 模態內收斂
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('B. 模態內收斂:鏈依 solid 比例分組,各組單獨算 R-hat')
print('=' * 100)
MODES = {}
for pf, chains in RUNS.items():
    fs = np.array([(c['state'] == 'solid').mean() for c in chains])
    grp = {'solid':  [c for c, f in zip(chains, fs) if f > MODE_HI],
           'molten': [c for c, f in zip(chains, fs) if f < MODE_LO],
           'mixed':  [c for c, f in zip(chains, fs) if MODE_LO <= f <= MODE_HI]}
    MODES[pf] = grp
    print(f'\n{pf}   solid-mode {len(grp["solid"])} chains, '
          f'molten-mode {len(grp["molten"])}, mixed {len(grp["mixed"])}')
    for mode in ['solid', 'molten']:
        cs = [c for c in grp[mode] if c['n'] >= RUN_MIN_N]
        if len(cs) < 2:
            print(f'  {mode}-mode: 少於 2 條鏈,無法算 R-hat')
            continue
        n2 = min(c['n'] for c in cs)
        print(f'  {mode}-mode  ({len(cs)} chains, aligned n={n2})')
        print(f"    {'param':<15s}{'R-hat':>8s}{'ESS':>8s}"
              f"{'5%':>10s}{'50%':>10s}{'95%':>10s}")
        for p in PARAMS + ['misfit']:
            mat = np.array([c[p][-n2:] for c in cs])
            if not np.all(np.isfinite(mat)):
                continue
            r, e = rhat_ess(mat)
            q = np.percentile(mat.ravel(), [5, 50, 95])
            flag = '' if (r == r and r < 1.1) else '  *'
            print(f'    {p:<15s}{r:8.3f}{e:8.0f}'
                  f'{q[0]:10.3f}{q[1]:10.3f}{q[2]:10.3f}{flag}')

fig, axes = plt.subplots(2, 4, figsize=(18, 8.4))
for ax, p in zip(axes.ravel(), PARAMS):
    for pf, grp in MODES.items():
        for mode, c_ in [('solid', STATE_COLOR['solid']),
                         ('molten', STATE_COLOR['molten'])]:
            cs = grp[mode]
            if not cs:
                continue
            v = np.concatenate([c[p] for c in cs])
            v = v[np.isfinite(v)]
            if v.size:
                ax.hist(v, bins=40, density=True, histtype='step',
                        lw=1.6, color=c_, alpha=0.75)
    ax.set_xlabel(p, fontsize=fontsize)
    ax.set_ylabel('density', fontsize=fontsize)
    ax.set_yticklabels([])
    ax.tick_params(labelsize=TICKSIZE)
fig.suptitle('parameter distributions within each mode, all runs overlaid '
             '(blue = solid mode, red = molten mode)', fontsize=fontsize + 1)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '100_within_mode_params.png'), dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# C. 轉換事件:屏障在哪
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('C. solid <-> molten 轉換事件的參數位置')
print('   只有 mixed 鏈會貢獻。這是屏障最薄處的唯一直接證據。')
print('=' * 100)
ALLT = []
for pf, chains in RUNS.items():
    ev = []
    for c in chains:
        st = c['state']
        ch = np.flatnonzero(st[1:] != st[:-1]) + 1
        for i in ch:
            a, b = st[i-1], st[i]
            if {a, b} == {'solid', 'molten'} or \
               (a == 'layered' and b in ('solid', 'molten')) or \
               (b == 'layered' and a in ('solid', 'molten')):
                ev.append({p: c[p][i] for p in PARAMS} | {'from': a, 'to': b})
    if not ev:
        print(f'\n{pf}: 無轉換事件'); continue
    ALLT.extend(ev)
    print(f'\n{pf}   {len(ev)} 次轉換,轉換點的參數 5/50/95%')
    print(f"    {'param':<15s}{'5%':>10s}{'50%':>10s}{'95%':>10s}"
          f"{'  (all samples 50%)':>22s}")
    allv = {p: np.concatenate([c[p] for c in chains]) for p in PARAMS}
    for p in PARAMS:
        v = np.array([e[p] for e in ev])
        v = v[np.isfinite(v)]
        if v.size < 3:
            continue
        q = np.percentile(v, [5, 50, 95])
        med_all = np.nanmedian(allv[p])
        print(f'    {p:<15s}{q[0]:10.3f}{q[1]:10.3f}{q[2]:10.3f}{med_all:22.3f}')

if ALLT:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.0))
    for ax, p in zip(axes.ravel(), PARAMS):
        v = np.array([e[p] for e in ALLT])
        v = v[np.isfinite(v)]
        allv = np.concatenate([c[p] for chains in RUNS.values() for c in chains])
        allv = allv[np.isfinite(allv)]
        if allv.size:
            ax.hist(allv, bins=40, density=True, color='0.82',
                    edgecolor='none', label='all samples')
        if v.size:
            ax.hist(v, bins=25, density=True, histtype='step', lw=2.2,
                    color='#C0392B', label='at transition')
        ax.set_xlabel(p, fontsize=fontsize)
        ax.set_ylabel('density', fontsize=fontsize)
        ax.set_yticklabels([])
        ax.tick_params(labelsize=TICKSIZE)
    axes[0, 0].legend(fontsize=10)
    fig.suptitle('where the sampler crosses between modes '
                 '(narrow red peak = the barrier coordinate)',
                 fontsize=fontsize + 1)
    fig.tight_layout()
    if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '100_transition_locations.png'), dpi=150)
    plt.show()