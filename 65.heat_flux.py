#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
105.core_adiabat_flux.py

BML 相態 -> CMB 熱通量 -> 核是否超絕熱 -> 熱驅動 dynamo 開關

沿用 68.heat_flux.py 的傳導公式,但比較對象換成核側的絕熱熱通量:

    q_cmb = k_bml * (T_interface - T_mantle_bottom) / h_solid     傳導蓋層
    q_ad  = k_core * alpha * g * T_core / Cp                      核側絕熱值

    q_cmb < q_ad  ->  核次絕熱,熱對流關閉,無熱驅動 dynamo
    q_cmb > q_ad  ->  核超絕熱,可能有熱驅動 dynamo

g 由存檔量算,不需要額外假設:
    M_core = 4/3 pi R_cmb^3 rho_core_mean
    g_cmb  = G M_core / R_cmb^2 = 4/3 pi G rho_core_mean R_cmb

限制(必須寫進圖說):
  * 全熔態沒有傳導蓋層,h_solid = 0,q_cmb 在靜態模型裡算不出來。
    液態 BML 以對流輸熱,效率遠高於傳導,所以全熔態的 q_cmb 應該高於
    固態/分層態 —— 但高多少要解對流,本腳本不做。
  * 組成驅動的浮力來自內核成長排出輕元素,正比於 dR_ic/dt,
    那是時間演化問題,靜態剖面給不出來。本腳本只談熱驅動。
  * q_ad 正比於 k_core * alpha / Cp,所以換一組值可以直接等比例縮放。
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR

# ── 設定 ──────────────────────────────────────────────────────────────────────
CHAIN_DIR = MCMC_DIR
OUT_DIR   = '/Users/chingchen/Desktop/HeFESTo/mcmc/figures'
PREFIX    = 'chain_a52B'

BURNIN_FRAC = 0.4
MIN_STEPS   = 400
SAVE        = False

K_BML   = 4.0        # W/m/K   固態 BML 熱導率(與 70b 的 K_SOLID 一致)
H_MIN   = 1.0        # km      三態分類門檻

# 核側絕熱熱通量的三個常數。q_ad 正比於 K_CORE * ALPHA / CP。
K_CORE_list = [20.0, 30.0, 40.0]   # W/m/K   液態 Fe-S 的熱導率,文獻範圍很寬
ALPHA_CORE  = 1.0e-4               # 1/K     液態 Fe-S 的熱膨脹係數
CP_CORE     = 800.0                # J/kg/K  液態 Fe-S 的定壓比熱

G_GRAV = 6.674e-11

bwith, fontsize = 1.5, 12
COLORS = ['#CD5C5C', '#35838D', '#414F67', '#7B9E87', '#C47F3E']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)


def load_chain_json(path):
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    if not txt.strip():
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        cut = txt.rfind('},')
        if cut < 0:
            return None
        try:
            return json.loads(txt[:cut + 1] + ']')
        except json.JSONDecodeError:
            return None


def load_chain_jsonl(path):
    if not os.path.exists(path):
        return None
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return recs if recs else None


def load_chain_dir(chain_dir):
    jsonl_path = os.path.join(chain_dir, 'chain.jsonl')
    if os.path.exists(jsonl_path):
        return load_chain_jsonl(jsonl_path)
    return load_chain_json(os.path.join(chain_dir, 'chain.json'))


records = []
n_kept = 0
for dd in sorted(glob.glob(os.path.join(CHAIN_DIR, f'{PREFIX}*'))):
    rec = load_chain_dir(dd)
    name = os.path.basename(dd)
    if rec is None:
        print(f'skip {name}  (no readable chain file)')
        continue
    if len(rec) < MIN_STEPS:
        print(f'skip {name}  ({len(rec)} steps < MIN_STEPS={MIN_STEPS})')
        continue
    records += rec[int(len(rec) * BURNIN_FRAC):]
    n_kept += 1

print(f'{n_kept} chains, {len(records)} post-burn-in samples')
if not records:
    raise SystemExit(f"no samples for prefix '{PREFIX}'")


def col(key):
    out = []
    for r in records:
        v = r.get(key)
        if v is None:
            v = r.get('params', {}).get(key)
        out.append(np.nan if v is None else float(v))
    return np.array(out)


h_solid  = col('h_solid_km')
h_liquid = col('h_liquid_km')
T_int    = col('T_interface')
T_mantle = col('T_mantle_bottom')
T_core   = col('T_core')
R_cmb    = col('R_cmb')
rho_core = col('rho_core_mean')      # g/cm3
R_ic     = col('R_ic_km')

# ── 三態 ──────────────────────────────────────────────────────────────────────
state = np.full(len(records), 'fail', dtype='<U8')
state[(h_solid >= H_MIN) & (h_liquid >= H_MIN)] = 'layered'
state[(h_solid <  H_MIN) & (h_liquid >= H_MIN)] = 'molten'
state[(h_solid >= H_MIN) & (h_liquid <  H_MIN)] = 'solid'
print('state counts:', {s: int((state == s).sum())
                        for s in ['solid', 'layered', 'molten', 'fail']})

# ── CMB 重力加速度:g = 4/3 pi G rho_core_mean R_cmb ──────────────────────────
g_cmb = 4.0/3.0 * np.pi * G_GRAV * (rho_core * 1e3) * (R_cmb * 1e3)

# ── 傳導蓋層的熱通量(只有 solid 與 layered 有蓋層)──────────────────────────
dT    = T_int - T_mantle
h_m   = h_solid * 1e3
q_cmb = np.where(h_solid >= H_MIN, K_BML * dT / h_m * 1e3, np.nan)   # mW/m2
q_cmb = np.where(dT > 0, q_cmb, np.nan)     # dT <= 0 的樣本不列入

n_neg = int(np.sum((h_solid >= H_MIN) & (dT <= 0)))
if n_neg:
    print(f'  警告: {n_neg} 個有蓋層的樣本 dT <= 0(界面比地函底部冷),已排除')

print(f'\ng_cmb  5/50/95 = ' +
      '  '.join(f'{v:.3f}' for v in np.nanpercentile(g_cmb, [5, 50, 95])) + ' m/s2')
print(f'T_core 5/50/95 = ' +
      '  '.join(f'{v:.0f}' for v in np.nanpercentile(T_core, [5, 50, 95])) + ' K')

# ── 主表:各 k_core 下的 q_ad 與超絕熱比例 ────────────────────────────────────
print('\n' + '=' * 96)
print(f'q_ad = k_core * alpha * g * T_core / Cp    '
      f'(alpha = {ALPHA_CORE:.1e} /K, Cp = {CP_CORE:.0f} J/kg/K)')
print(f'q_cmb = {K_BML:.1f} * (T_int - T_mantle_bottom) / h_solid   [傳導蓋層]')
print('=' * 96)
print(f"{'k_core':>8s}{'state':>10s}{'n':>8s}"
      f"{'q_ad 50':>10s}{'q_cmb 5/50/95':>26s}{'q_cmb/q_ad 50':>15s}"
      f"{'超絕熱 %':>12s}")

Q_AD = {}
for K_CORE in K_CORE_list:
    q_ad = K_CORE * ALPHA_CORE * g_cmb * T_core / CP_CORE * 1e3      # mW/m2
    Q_AD[K_CORE] = q_ad
    for st in ['solid', 'layered', 'molten']:
        m = (state == st)
        if m.sum() == 0:
            continue
        qa = q_ad[m]
        qc = q_cmb[m]
        ok = np.isfinite(qc)
        if ok.sum() == 0:
            print(f'{K_CORE:8.0f}{st:>10s}{int(m.sum()):8d}'
                  f'{np.nanmedian(qa):10.1f}'
                  f'{"  無傳導蓋層,靜態算不出":>26s}'
                  f'{"--":>15s}{"--":>12s}')
            continue
        p = np.percentile(qc[ok], [5, 50, 95])
        ratio = qc[ok] / qa[ok]
        print(f'{K_CORE:8.0f}{st:>10s}{int(ok.sum()):8d}'
              f'{np.nanmedian(qa):10.1f}'
              f'{p[0]:8.1f}{p[1]:9.1f}{p[2]:9.1f}'
              f'{np.median(ratio):15.3f}{100*np.mean(ratio > 1.0):11.1f}%')
    print()

# ── 內核與熱狀態的關係 ────────────────────────────────────────────────────────
print('=' * 96)
print('內核與傳導蓋層')
print('=' * 96)
print(f"{'state':<10s}{'n':>8s}{'P(R_ic>0)':>12s}"
      f"{'R_ic>0 的 q_cmb 50':>20s}{'R_ic=0 的 q_cmb 50':>20s}")
for st in ['solid', 'layered', 'molten']:
    m = (state == st)
    if m.sum() == 0:
        continue
    has = m & np.isfinite(R_ic) & (R_ic > 1.0)
    non = m & np.isfinite(R_ic) & (R_ic <= 1.0)
    q1 = np.nanmedian(q_cmb[has]) if has.sum() else np.nan
    q0 = np.nanmedian(q_cmb[non]) if non.sum() else np.nan
    print(f'{st:<10s}{int(m.sum()):8d}{100*has.sum()/max(m.sum(),1):11.1f}%'
          f'{q1:20.1f}{q0:20.1f}')

# ══════════════════════════════════════════════════════════════════════════════
# 圖 1:q_cmb 與 q_ad 的分佈
# ══════════════════════════════════════════════════════════════════════════════
BINS = np.logspace(np.log10(0.5), np.log10(500), 60)

fig, ax = plt.subplots(figsize=(8.0, 5.2))
for st in ['solid', 'layered']:
    m = (state == st) & np.isfinite(q_cmb)
    if m.sum():
        ax.hist(q_cmb[m], bins=BINS, color=STATE_COLOR[st], alpha=0.55,
                edgecolor='none', label=f'q$_{{cmb}}$  {st}  n={m.sum()}')
for mm, K_CORE in enumerate(K_CORE_list):
    qa = Q_AD[K_CORE]
    ax.axvline(np.nanmedian(qa), color=COLORS[mm], lw=2.0, ls='--',
               label=f'q$_{{ad}}$  k$_{{core}}$={K_CORE:.0f}  '
                     f'{np.nanmedian(qa):.0f} mW/m$^2$')
ax.set_xscale('log')
ax.set_xlabel(r'heat flux (mW/m$^2$)', fontsize=fontsize)
ax.set_ylabel('count', fontsize=fontsize)
ax.set_title(f'{PREFIX}   conductive CMB flux vs core adiabat', fontsize=fontsize)
ax.legend(fontsize=9, loc='upper right')
ax.tick_params(labelsize=14)
ax.grid(alpha=0.3)
for a in ['top', 'bottom', 'left', 'right']:
    ax.spines[a].set_linewidth(bwith)
plt.tight_layout()
if SAVE:
    plt.savefig(os.path.join(OUT_DIR, f'{PREFIX}_105_flux_vs_adiabat.png'), dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 圖 2:q_cmb / q_ad 的分佈,1 是熱對流的開關
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(K_CORE_list), figsize=(5.0*len(K_CORE_list), 4.4))
RBINS = np.logspace(-2, 1.5, 50)
for mm, K_CORE in enumerate(K_CORE_list):
    ax = axes[mm]
    qa = Q_AD[K_CORE]
    for st in ['solid', 'layered']:
        m = (state == st) & np.isfinite(q_cmb)
        if m.sum():
            ax.hist(q_cmb[m]/qa[m], bins=RBINS, color=STATE_COLOR[st],
                    alpha=0.55, edgecolor='none', label=st)
    ax.axvline(1.0, color='k', lw=2.0, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel(r'q$_{cmb}$ / q$_{ad}$', fontsize=fontsize)
    ax.set_ylabel('count' if mm == 0 else '', fontsize=fontsize)
    ax.set_title(f'k$_{{core}}$ = {K_CORE:.0f} W/m/K', fontsize=fontsize)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    if mm == 0:
        ax.legend(fontsize=10)
    for a in ['top', 'bottom', 'left', 'right']:
        ax.spines[a].set_linewidth(bwith)
fig.suptitle('left of the dashed line: core is subadiabatic, '
             'thermal convection shuts down', fontsize=fontsize + 1)
plt.tight_layout()
if SAVE:
    plt.savefig(os.path.join(OUT_DIR, f'{PREFIX}_105_flux_ratio.png'), dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 圖 3:q_cmb vs T_core,按三態。全熔態畫在 q_cmb = nan,只標 q_ad
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for st in ['solid', 'layered']:
    m = (state == st) & np.isfinite(q_cmb)
    if m.sum():
        ax.scatter(T_core[m], q_cmb[m], s=6, alpha=0.4, linewidths=0,
                   color=STATE_COLOR[st], label=st)
for mm, K_CORE in enumerate(K_CORE_list):
    o = np.argsort(T_core)
    ax.plot(T_core[o], Q_AD[K_CORE][o], color=COLORS[mm], lw=1.8, ls='--',
            label=f'q$_{{ad}}$  k$_{{core}}$={K_CORE:.0f}')
ax.set_yscale('log')
ax.set_xlabel(r'T$_{core}$ (K)', fontsize=fontsize)
ax.set_ylabel(r'heat flux (mW/m$^2$)', fontsize=fontsize)
ax.set_title(f'{PREFIX}   molten state has no conductive lid and is not shown',
             fontsize=fontsize)
ax.legend(fontsize=9)
ax.tick_params(labelsize=14)
ax.grid(alpha=0.3)
for a in ['top', 'bottom', 'left', 'right']:
    ax.spines[a].set_linewidth(bwith)
plt.tight_layout()
if SAVE:
    plt.savefig(os.path.join(OUT_DIR, f'{PREFIX}_105_flux_vs_Tcore.png'), dpi=150)
plt.show()