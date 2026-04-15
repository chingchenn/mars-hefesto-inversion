#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:53:40 2026

@author: chingchen
"""

"""
分析 HeFESTo 批次執行結果
畫出成功和失敗 model 的參數分布 histogram
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = "/Users/chingchen/Desktop/HeFESTo/runs"

# ============================================================
# 讀取所有 model 的參數和狀態
# ============================================================

success_params = []
failed_params  = []

for status_file in sorted(glob.glob(os.path.join(RUNS_DIR, "model_*/status.txt"))):
    run_dir    = os.path.dirname(status_file)
    params_file = os.path.join(run_dir, "params.json")

    if not os.path.exists(params_file):
        continue

    with open(status_file) as f:
        status = f.read().strip()
    with open(params_file) as f:
        params = json.load(f)

    if status == "success":
        success_params.append(params)
    elif status == "failed":
        failed_params.append(params)

print(f"成功：{len(success_params)} 個")
print(f"失敗：{len(failed_params)} 個")
print(f"失敗率：{len(failed_params)/(len(success_params)+len(failed_params))*100:.1f}%")

# ============================================================
# 整理成 numpy array
# ============================================================

def extract(params_list, key):
    return np.array([p[key] for p in params_list])

keys = ['T_lit', 'P_lit', 'dTdP', 'Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr']
labels = {
    'T_lit': 'T_lit (K)',
    'P_lit': 'P_lit (GPa)',
    'dTdP':  'dT/dP (K/GPa)',
    'Si':    'Si (molar)',
    'Mg':    'Mg (molar)',
    'Fe':    'Fe (molar)',
    'Ca':    'Ca (molar)',
    'Al':    'Al (molar)',
    'Na':    'Na (molar)',
    'Cr':    'Cr (molar)',
}

# ============================================================
# 畫圖
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for ax, key in zip(axes, keys):
    s_vals = extract(success_params, key) if success_params else np.array([])
    f_vals = extract(failed_params,  key) if failed_params  else np.array([])

    # 決定 bin 範圍
    all_vals = np.concatenate([s_vals, f_vals])
    lo, hi   = all_vals.min(), all_vals.max()
    bins     = np.linspace(lo, hi, 25)

    ax.hist(s_vals, bins=bins, alpha=0.6, color='steelblue',
            label=f'Success ({len(s_vals)})', density=True)
    ax.hist(f_vals, bins=bins, alpha=0.6, color='tomato',
            label=f'Failed ({len(f_vals)})',  density=True)

    ax.set_xlabel(labels[key], fontsize=11)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 印出基本統計
    if len(s_vals) > 0 and len(f_vals) > 0:
        print(f"\n{key}:")
        print(f"  成功：mean={s_vals.mean():.2f}  std={s_vals.std():.2f}  "
              f"range=[{s_vals.min():.2f}, {s_vals.max():.2f}]")
        print(f"  失敗：mean={f_vals.mean():.2f}  std={f_vals.std():.2f}  "
              f"range=[{f_vals.min():.2f}, {f_vals.max():.2f}]")

fig.suptitle(f'HeFESTo 參數分布：成功 vs 失敗\n'
             f'(成功 {len(success_params)} / 失敗 {len(failed_params)} / '
             f'失敗率 {len(failed_params)/(len(success_params)+len(failed_params))*100:.1f}%)',
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RUNS_DIR, "param_histogram.png"),
            dpi=150, bbox_inches='tight')
plt.show()
print(f"\n圖片已儲存至 {RUNS_DIR}/param_histogram.png")