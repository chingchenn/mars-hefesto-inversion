#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:14:45 2026

@author: chingchen
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:14:45 2026
@author: chingchen

MCMC 鏈的三張診斷圖:trace / misfit history / acceptance rate。
讀檔支援 chain.jsonl(a34 起)與 chain.json(舊版,含 salvage)。
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

# ── 設定 ─────────────────────────────────────────────────────────────────────
CHAIN_DIR = '/Users/chingchen/Desktop/HeFESTo/mcmc/mcmc_chain'
OUT_DIR   = '/Users/chingchen/Desktop/HeFESTo/mcmc/figures'
PREFIX    = 'chain_a41B'

plt.rcParams['font.family'] = 'DejaVu Sans'

DROP_LAST = 1
MIN_STEPS = 5      # 設 400 會剔掉 chain_13(396 步);trace 圖希望看到全部鏈
SAVE      = False

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness']
LABELS = ['T_lit (K)', 'P_lit (GPa)', 'Mg# (mantle)', 'T_core (K)',
          'Mg#_bulk_bml', 'BML thickness (km)']
PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core',
          'Mg#_bulk_bml', 'BML_thickness', 'R_cmb', 'w_S']
LABELS = ['T_lit (K)', 'P_lit (GPa)', 'Mg# (mantle)', 'T_core (K)',
          'Mg#_bulk_bml', 'BML thickness (km)', 'R_cmb (km)', 'w_S (wt%)']
COLORS = ['#CD5C5C', '#35838D', '#849DAB', '#414F67', '#97795D', '#7B9E87',
          '#9B6B8A', '#4E6E8E', '#C47F3E', '#5C7A5C', '#8B6F6F', '#4A7C7C',
          '#7A6B9B', '#6B8E7A']
bwith, fontsize = 1.5, 12

MISFIT_KEYS = [
    ('misfit',         'Total misfit'),
    ('misfit_tt',      'TT misfit'),
    ('misfit_solidus', 'Solidus penalty'),
    ('mass_sigma',     'Mass (sigma)'),
    ('moi_sigma',      'MoI (sigma)'),
]

os.makedirs(OUT_DIR, exist_ok=True)
TAG = f'{PREFIX}_' if PREFIX else ''


# ── chain 載入: chain.jsonl(逐行) + chain.json(salvage) ─────────────────────
def load_chain_json(path):
    if not os.path.exists(path):
        return None, 'missing'
    with open(path) as fh:
        text = fh.read()
    if not text.strip():
        return None, 'empty'
    try:
        return json.loads(text), 'ok'
    except json.JSONDecodeError:
        pass
    dec, i, recs = json.JSONDecoder(), text.find('[') + 1, []
    while True:
        while i < len(text) and text[i] in ' \t\r\n,':
            i += 1
        if i >= len(text) or text[i] == ']':
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        recs.append(obj)
    return (recs, 'salvaged') if recs else (None, 'unparseable')


def load_chain_jsonl(path):
    """逐行讀;只有最後一行可能寫到一半,壞掉就丟棄該行"""
    if not os.path.exists(path):
        return None, 'missing'
    recs, n_bad = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
                break
    if not recs:
        return None, 'empty'
    return recs, ('ok' if n_bad == 0 else 'truncated_last_line')


def load_chain_dir(chain_dir):
    jsonl_path = os.path.join(chain_dir, 'chain.jsonl')
    if os.path.exists(jsonl_path):
        return load_chain_jsonl(jsonl_path)
    return load_chain_json(os.path.join(chain_dir, 'chain.json'))


chain_dirs = sorted(glob.glob(os.path.join(CHAIN_DIR, f'{PREFIX}*')))
all_chains, chain_names, load_report = [], [], []

for dd in chain_dirs:
    name = os.path.basename(dd)
    ch, status = load_chain_dir(dd)
    n_raw = 0 if ch is None else len(ch)
    if ch is not None and DROP_LAST > 0:
        ch = ch[:-DROP_LAST] if len(ch) > DROP_LAST else []
    if ch is None or len(ch) < MIN_STEPS:
        load_report.append((name, status, n_raw, 'DROPPED'))
        continue
    all_chains.append(ch)
    chain_names.append(name)
    load_report.append((name, status, n_raw, f'kept {len(ch)}'))

print(f"{'chain':<20s} {'status':<18s} {'n_raw':>7s}  action")
for name, status, n_raw, action in load_report:
    print(f"{name:<20s} {status:<18s} {n_raw:>7d}  {action}")
print(f"\nusable chains: {len(all_chains)}")
if not all_chains:
    raise SystemExit(
        f"no usable chains for prefix '{PREFIX}'.\n"
        f"  檢查: (1) {CHAIN_DIR} 下是否有 chain.jsonl\n"
        f"        (2) MIN_STEPS={MIN_STEPS} 是否高於實際鏈長")


def series(ch, key):
    """取單一欄位,None -> NaN"""
    return np.array([np.nan if s.get(key) is None else float(s[key]) for s in ch])


# ── 圖1: trace(完整鏈,不套 burn-in) ────────────────────────────────────────
#fig, axes = plt.subplots(3, 2, figsize=(16, 8))
fig, axes = plt.subplots(4, 2, figsize=(16, 11))
axes = axes.flatten()
fig.suptitle(f'{PREFIX} Trace plots', fontsize=fontsize + 5)
for i, (param, label) in enumerate(zip(PARAMS, LABELS)):
    ax = axes[i]
    for ci, ch in enumerate(all_chains):
        ax.plot([s['step'] for s in ch], [s['params'][param] for s in ch],
                lw=1, color=COLORS[ci % len(COLORS)])
    ax.set_ylabel(label, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=14); ax.grid(0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(bwith)
axes[-1].set_xlabel('Step', fontsize=fontsize)
axes[-2].set_xlabel('Step', fontsize=fontsize)
plt.tight_layout()
if SAVE: plt.savefig(os.path.join(OUT_DIR, TAG + '01_trace.png'), dpi=150)
plt.show()


# ── 圖2: misfit history ─────────────────────────────────────────────────────
# y 範圍用「全部鏈合併」的分位數。原版用迴圈洩漏的 v,只反映最後一條鏈,
# 且欄位為 None 時 np.percentile 回傳 NaN -> set_ylim 靜默失效。
fig, axes = plt.subplots(5, 1, figsize=(12, 13))
fig.suptitle(f'{PREFIX} Misfit history', fontsize=13)
for ax, (key, label) in zip(axes, MISFIT_KEYS):
    pooled = []
    for ci, ch in enumerate(all_chains):
        v = series(ch, key)
        pooled.append(v)
        ax.plot([s['step'] for s in ch], v, lw=1, color=COLORS[ci % len(COLORS)])
    pooled = np.concatenate(pooled)
    if np.isfinite(pooled).any():
        ymin, ymax = np.nanpercentile(pooled, [2, 98])
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.text(0.5, 0.5, f'{key}: no finite values', transform=ax.transAxes,
                ha='center', va='center', fontsize=fontsize, color='gray')
    ax.set_ylabel(label, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=14); ax.grid(0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(bwith)
axes[-1].set_xlabel('Step', fontsize=fontsize)
plt.tight_layout()
if SAVE: plt.savefig(os.path.join(OUT_DIR, TAG + '02_misfit_history.png'), dpi=150)
plt.show()


# ── 圖3: acceptance rate ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for ci, ch in enumerate(all_chains):
    ax.plot([s['step'] for s in ch], series(ch, 'accept_rate'),
            lw=0.8, color=COLORS[ci % len(COLORS)], label=chain_names[ci])
ax.axhline(30, color='gray', ls=':', lw=1)
ax.axhline(50, color='gray', ls=':', lw=1)
ax.set_xlabel('Step', fontsize=fontsize)
ax.set_ylabel('Acceptance rate (%)', fontsize=fontsize)
ax.legend(fontsize=7, ncol=4)
ax.tick_params(labelsize=14)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(bwith)
plt.tight_layout()
if SAVE: plt.savefig(os.path.join(OUT_DIR, TAG + '03_accept_rate.png'), dpi=150)
plt.show()