#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:58:16 2026

@author: chingchen
"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

CHAIN_DIR = MCMC_DIR
OUT_DIR   = FIG_DIR
SAVE_FIG  = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
fontsize  = 14

# 組成 -> [(標籤, prefix)],舊在前、新在後;prefix=None 表示沒有舊 run
RUNS = {
    'TA':   [('a53B', 'chain_a53B'), ('a60', 'chain_a60_TA')],
    'EH45': [('a54B', 'chain_a54B'), ('a60', 'chain_a60_EH45')],
    'YM20': [('a52B',  'chain_a52B'),  ('a60', 'chain_a60_YM20')],   # 填入舊 YM20 prefix
}
EXCLUDE   = ['chain_a54B_00', 'chain_a54B_09', 'chain_a60_EH45_00']
LEAVE_OUT = []      # 例:['chain_a60_EH45_01'],測試單一 chain 的影響

BURNIN_FRAC, DROP_LAST, MIN_STEPS = 0.4, 1, 400
T_SOL_CAL = 2200.0  # a60 相圖校準點 (17.7 GPa) 的 T_sol

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness',
          'w_S', 'R_cmb']
# w_S 用 0.40 當畫圖範圍,涵蓋舊版上界 (0.40) 與 a60 (0.365)
PRIOR = {'T_lit': (1000, 2600), 'P_lit': (1.4, 9.0), 'Mg#': (0.50, 0.90),
         'T_core': (1550, 3600), 'Mg#_bulk_bml': (0.20, 0.80),
         'BML_thickness': (0, 400), 'w_S': (0.05, 0.40), 'R_cmb': (1400, 1900)}
EXTRA   = ['P_cmb', 'misfit_tt', 'moi_sigma', 'mass_sigma', 'h_solid_km',
           'h_liquid_km', 'T_solidus_bml', 'T_liquidus_bml']
DERIVED = ['R_bml_top', 'R_scs', 'P_cmb', 'misfit_tt', 'moi_sigma']
UNITS = {'T_lit': 'K', 'P_lit': 'GPa', 'T_core': 'K', 'BML_thickness': 'km',
         'R_cmb': 'km', 'R_bml_top': 'km', 'R_scs': 'km', 'P_cmb': 'GPa'}
C_OLD, C_NEW = '0.45', '#CD5C5C'


# ── 讀檔 ──────────────────────────────────────────────────────────────────
def load_jsonl(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break                     # 只有最後一行可能寫到一半
    return recs


def _num(s, k):
    v = s.get(k, s.get('params', {}).get(k))
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_run(prefix):
    if prefix is None:
        return None
    samples, n_chain = [], 0
    for d in sorted(glob.glob(os.path.join(CHAIN_DIR, prefix + '_*'))):
        name = os.path.basename(d)
        if name in EXCLUDE or name in LEAVE_OUT:
            continue
        path = os.path.join(d, 'chain.jsonl')
        if not os.path.exists(path):
            continue
        ch = load_jsonl(path)
        ch = ch[:-DROP_LAST] if DROP_LAST else ch
        if len(ch) < MIN_STEPS:
            continue
        samples.extend(ch[int(len(ch) * BURNIN_FRAC):])
        n_chain += 1
    if not samples:
        print(f'  {prefix}: no usable chains')
        return None
    post = {k: np.array([_num(s, k) for s in samples]) for k in PARAMS + EXTRA}
    post['R_bml_top'] = post['R_cmb'] + post['BML_thickness']
    post['R_scs']     = post['R_cmb'] + post['h_liquid_km']   # ScS 反射面
    post['dT_sol']    = post['T_core'] - post['T_solidus_bml']
    post['branch']    = np.array([str(s.get('phase_branch', '')) for s in samples])
    hs, hl = post['h_solid_km'], post['h_liquid_km']
    st = np.full(len(hs), 'fail', dtype='<U8')
    st[(hs >= 1) & (hl >= 1)] = 'layered'
    st[(hs < 1) & (hl >= 1)]  = 'molten'
    st[(hs >= 1) & (hl < 1)]  = 'solid'
    post['state'] = st
    print(f'  {prefix}: {n_chain} chains, {len(samples)} samples')
    return post


# ── 數字表 ────────────────────────────────────────────────────────────────
def q(v):
    v = v[np.isfinite(v)]
    return np.percentile(v, [5, 50, 95]) if v.size else np.full(3, np.nan)


def table(comp, old, new, lo, ln):
    print('\n' + '=' * 92)
    print(f'{comp}: {lo} -> {ln}   median [5, 95]   shift = dmedian / sd({lo})')
    print('=' * 92)
    for k in PARAMS + DERIVED:
        a, b = q(old[k]), q(new[k])
        sd = np.nanstd(old[k])
        sh = (b[1] - a[1]) / sd if sd > 0 else np.nan
        flag = '  <--' if abs(sh) > 0.5 else ''
        print(f'  {k:<14s}{a[1]:10.4g} [{a[0]:8.4g},{a[2]:8.4g}]   '
              f'{b[1]:10.4g} [{b[0]:8.4g},{b[2]:8.4g}]   {sh:+6.2f}{flag}')
    # state 比例只看方向,不報告
    probes = [('P(T_core < 2200 K)', lambda p: p['T_core'] < T_SOL_CAL),
              ('P(w_S > 0.20)',      lambda p: p['w_S'] > 0.20),
              ('P(P_lit < 2.4 GPa)', lambda p: p['P_lit'] < 2.4),
              ('P(Mg# < 0.74)',      lambda p: p['Mg#'] < 0.74),
              ('P(P_cmb > 23 GPa)',  lambda p: p['P_cmb'] > 23.0),
              ('P(state = solid)',   lambda p: p['state'] == 'solid')]
    print()
    for name, f in probes:
        print(f'  {name:<22s}{np.mean(f(old)):8.1%}  ->{np.mean(f(new)):8.1%}')
        print('\n  within-state medians (old -> new)')
    for st_ in ['molten', 'solid']:
        mo, mn = old['state'] == st_, new['state'] == st_
        if mo.sum() < 100 or mn.sum() < 100:
            continue
        print(f'  [{st_}]  n = {mo.sum()} -> {mn.sum()}')
        for k in ['R_cmb', 'w_S', 'R_scs', 'T_core', 'P_cmb']:
            a, b = np.nanmedian(old[k][mo]), np.nanmedian(new[k][mn])
            print(f'    {k:<10s}{a:10.4g} -> {b:10.4g}   ({b - a:+.4g})')


# ── 圖 1:參數疊圖 ────────────────────────────────────────────────────────
def overlay(comp, old, new, lo, ln):
    keys = PARAMS + DERIVED
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.6 * nrow))
    axes = axes.ravel()
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    for ax, k in zip(axes, keys):
        vals = [p[k][np.isfinite(p[k])] for p in (old, new) if p is not None]
        both = np.concatenate(vals)
        if both.size == 0:
            continue
        rng = PRIOR.get(k, tuple(np.percentile(both, [0.5, 99.5])))
        for p, lab, c in ((old, lo, C_OLD), (new, ln, C_NEW)):
            if p is None:
                continue
            v = p[k][np.isfinite(p[k])]
            if v.size == 0:
                continue
            ax.hist(v, bins=40, range=rng, density=True, histtype='step',
                    lw=2, color=c, label=lab)
            ax.axvline(np.median(v), color=c, ls='--', lw=1.5)
        if k == 'T_core':
            ax.axvline(T_SOL_CAL, color='k', ls=':', lw=1.5)
        u = UNITS.get(k, '')
        ax.set_xlabel(f'{k} [{u}]' if u else k, fontsize=fontsize)
        ax.tick_params(labelsize=14)
    axes[0].legend(fontsize=12)
    fig.suptitle(f'{comp}: {lo} (grey) vs {ln} (red)', fontsize=fontsize + 2)
    fig.tight_layout()
    if SAVE_FIG:
        fig.savefig(os.path.join(OUT_DIR, f'137_{comp}_overlay.png'), dpi=150)
    plt.show()


# ── 圖 2:T_core - T_solidus_bml(只有 a60 有) ────────────────────────────
def solidus_fig(news):
    fig, axes = plt.subplots(1, len(news), figsize=(5.5 * len(news), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (comp, p) in zip(axes, news.items()):
        d = p['dT_sol']
        if not np.isfinite(d).any():
            ax.set_title(f'{comp}: T_solidus_bml not found', fontsize=fontsize)
            continue
        rng = tuple(np.nanpercentile(d, [0.5, 99.5]))
        for br in sorted(set(p['branch'])):
            m = (p['branch'] == br) & np.isfinite(d)
            if m.sum():
                ax.hist(d[m], bins=50, range=rng, histtype='stepfilled',
                        alpha=0.5, label=f'{br or "?"} (n={m.sum()})')
        ax.axvline(0, color='k', lw=1.5)
        ax.set_xlabel('T_core - T_solidus,BML (K)', fontsize=fontsize)
        ax.set_title(comp, fontsize=fontsize)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=10)
    fig.tight_layout()
    if SAVE_FIG:
        fig.savefig(os.path.join(OUT_DIR, '137_dT_solidus.png'), dpi=150)
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
news = {}
for comp, ((lo, po), (ln, pn)) in RUNS.items():
    print(f'\n[{comp}]')
    old, new = load_run(po), load_run(pn)
    if new is None:
        continue
    news[comp] = new
    if old is not None:
        table(comp, old, new, lo, ln)
    overlay(comp, old, new, lo, ln)
if news:
    solidus_fig(news)
    
    