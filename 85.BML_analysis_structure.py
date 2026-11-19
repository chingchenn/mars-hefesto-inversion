#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:50:55 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 把 solid / layered / molten 三種 BML 狀態的 Vp, Vs, rho, T 剖面分色畫出來
#
# 三態分類(1 km 門檻):
#   h_liquid < 1   -> solid
#   h_solid  < 1   -> molten
#   兩者都 >= 1    -> layered
# 注意 chain.jsonl 的 thermal_state 存的是熱傳機制,不是這個三態,要自己重建

import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

MCMC_BASE_DIR = MCMC_DIR
CHAIN_PREFIX  = 'chain_a52B'

H_THRESH      = 1.0     # km
N_PER_STATE   = 40      # 每個 state 畫幾條
BURNIN_FRAC   = 0.3
SAVE_FIG      = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

COLOR = {'solid': '#2C3E50', 'layered': '#E67E22', 'molten': '#C0392B'}


# ── 第一遍:掃過所有鏈,只記下 (misfit, npz 路徑, 狀態) ────────────────────────
cand = {'solid': [], 'layered': [], 'molten': []}

for chain_dir in sorted(glob.glob(f'{MCMC_BASE_DIR}/{CHAIN_PREFIX}_*')):
    recs = []
    with open(os.path.join(chain_dir, 'chain.jsonl')) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break          # 只有最後一行可能寫到一半

    if len(recs) < 400:
        continue
    burn = BURNIN_FRAC * max(r['step'] for r in recs)

    for r in recs:
        if not r.get('accepted') or r['step'] < burn:
            continue
        npz_path = f"{chain_dir}/profile_s{r['step']:05d}.npz"
        if not os.path.exists(npz_path):
            continue

        npz = np.load(npz_path)
        h_solid  = float(npz['h_solid_km'])
        h_liquid = float(npz['h_liquid_km'])

        if h_liquid < H_THRESH:
            state = 'solid'
        elif h_solid < H_THRESH:
            state = 'molten'
        else:
            state = 'layered'

        cand[state].append((r['misfit'], npz_path, r['params'],
                            os.path.basename(chain_dir), r['step']))

for state in ['solid', 'layered', 'molten']:
    print(f"{state:8s}: {len(cand[state]):5d} 個 post-burn-in accepted 步")


# ── 第二遍:每個 state 依 misfit 由小到大取前 N,把剖面陣列讀進來 ─────────────
prof = {'solid': [], 'layered': [], 'molten': []}

for state in ['solid', 'layered', 'molten']:
    cand[state].sort()
    for misfit, npz_path, params, chain, step in cand[state][:N_PER_STATE]:
        npz = np.load(npz_path)
        prof[state].append({
            'z':      npz['z_km'],        # a36+ 是單一全行星陣列,不用切段再拼
            'Vp':     npz['Vp'],
            'Vs':     npz['Vs'],
            'rho':    npz['rho'],
            'T':      npz['T_K'],
            'z_top':  float(npz['z_bml_top']),
            'z_cmb':  float(npz['z_cmb']),
            'h_s':    float(npz['h_solid_km']),
            'h_l':    float(npz['h_liquid_km']),
            'misfit': misfit,
            'params': params,
            'chain':  chain,
            'step':   step,
        })
    print(f"{state:8s}: 載入 {len(prof[state])} 條剖面")

states = [s for s in ['solid', 'layered', 'molten'] if prof[s]]

# BML zoom 的深度視窗:所有剖面 z_bml_top 最小值到 z_cmb 最大值,上下各留 80 km
tops = [p['z_top'] for s in states for p in prof[s]]
bots = [p['z_cmb'] for s in states for p in prof[s]]
ZOOM = (max(bots) + 80, min(tops) - 80)
print(f"BML 視窗: {ZOOM[1]:.0f} - {ZOOM[0]:.0f} km")


# ── Figure 1: BML 區間放大 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 9))
fig.patch.set_facecolor('#FAFAFA')

for ax, key, xlab in zip(axes,
                         ['Vp', 'Vs', 'rho', 'T'],
                         ['Vp (km/s)', 'Vs (km/s)', 'ρ (g/cm³)', 'T (K)']):
    for state in states:
        c = COLOR[state]
        for p in prof[state][1:]:
            ax.plot(p[key], p['z'], color=c, lw=0.8, alpha=0.18)
        b = prof[state][0]
        ax.plot(b[key], b['z'], color=c, lw=2.5,
                label=f"{state} best (misfit={b['misfit']:.2f}, n={len(prof[state])})")
        ax.axhline(b['z_top'], color=c, ls='--', lw=1.0, alpha=0.5)
        ax.axhline(b['z_cmb'], color=c, ls=':',  lw=1.0, alpha=0.5)
    ax.set_ylim(*ZOOM)
    ax.set_xlabel(xlab, fontsize=13)
    ax.grid(True, ls='--', alpha=0.3, lw=0.8)
    ax.tick_params(labelsize=14)
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)

axes[0].set_ylabel('Depth (km)', fontsize=13)
axes[0].set_xlim(6.5, 9.5)
axes[1].set_xlim(0.0, 5.0)
axes[2].set_xlim(3.6, 6.0)
axes[0].legend(fontsize=9, loc='lower right', framealpha=0.85)

fig.suptitle(f'{CHAIN_PREFIX}: BML zoom, 依三態分色 (每態 top {N_PER_STATE} by misfit)',
             fontsize=11, y=0.98)
plt.tight_layout()
if SAVE_FIG:
    fig.savefig(os.path.join(FIG_DIR, f'69b_{CHAIN_PREFIX}_bml_zoom_by_state.png'), dpi=300,
                bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()


# ── Figure 2: 全深度 ──────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 9))
fig2.patch.set_facecolor('#FAFAFA')

for ax, key, xlab in zip(axes2,
                         ['Vp', 'Vs', 'rho', 'T'],
                         ['Vp (km/s)', 'Vs (km/s)', 'ρ (g/cm³)', 'T (K)']):
    for state in states:
        c = COLOR[state]
        for p in prof[state][1:]:
            ax.plot(p[key], p['z'], color=c, lw=0.8, alpha=0.18)
        b = prof[state][0]
        ax.plot(b[key], b['z'], color=c, lw=2.5, label=f"{state} best")
    ax.set_ylim(3300, 0)
    ax.set_xlabel(xlab, fontsize=13)
    ax.grid(True, ls='--', alpha=0.3, lw=0.8)
    ax.tick_params(labelsize=14)
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)

axes2[0].set_ylabel('Depth (km)', fontsize=13)
axes2[3].set_xlim(0, 3800)
axes2[0].legend(fontsize=9, loc='lower right', framealpha=0.85)

fig2.suptitle(f'{CHAIN_PREFIX}: 全深度,依三態分色', fontsize=11, y=0.98)
plt.tight_layout()
if SAVE_FIG:
    fig2.savefig(os.path.join(FIG_DIR, f'69b_{CHAIN_PREFIX}_fulldepth_by_state.png'), dpi=300,
                 bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.show()


# ── 數值摘要 ──────────────────────────────────────────────────────────────────
for state in states:
    b = prof[state][0]
    print(f"\n{state}  {b['chain']} step {b['step']}  misfit={b['misfit']:.4f}")
    print(f"  h_solid={b['h_s']:.1f} km  h_liquid={b['h_l']:.1f} km  "
          f"z_bml_top={b['z_top']:.1f}  z_cmb={b['z_cmb']:.1f}")
    print(f"  {b['params']}")

    for key in ['Vp', 'Vs', 'rho', 'T']:
        # BML 段 = z_bml_top 到 z_cmb 之間
        v = [np.mean(p[key][(p['z'] >= p['z_top']) & (p['z'] <= p['z_cmb'])])
             for p in prof[state]]
        print(f"  <{key:3s}>_BML  中位數={np.median(v):8.3f}  "
              f"16-84% = {np.percentile(v, 16):8.3f} - {np.percentile(v, 84):8.3f}")