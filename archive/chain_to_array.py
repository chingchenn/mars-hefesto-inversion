#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把所有 chain 讀成 numpy array。跑完直接在 namespace 裡用。"""

import json, glob, os
import numpy as np
from mars_config import MCMC_DIR

CHAIN_DIR = MCMC_DIR
PREFIX    = 'chain_a36A'      # a36A / a36B 分開讀,不要寫 chain_a36
BURNIN    = 0.3
MIN_STEPS = 300

# ── 讀檔 ──────────────────────────────────────────────────────────────────────
samples, chain_id, chain_name = [], [], []
for d in sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '*'))):
    recs = []
    for line in open(os.path.join(d, 'chain.jsonl')):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break                      # 最後一行可能寫到一半
    recs = recs[:-1]                   # 丟掉最後一步
    if len(recs) < MIN_STEPS:
        continue
    recs = recs[int(len(recs) * BURNIN):]
    samples += recs
    chain_id += [len(set(chain_name))] * len(recs)
    chain_name += [os.path.basename(d)] * len(recs)
    print(f'{os.path.basename(d):<20s} {len(recs)}')

# ── 轉成 array ────────────────────────────────────────────────────────────────
# params 展平到頂層,dict 欄位(tt_miss_by_event)跳過
D = {}
for k in sorted({k for s in samples for k in s['params']}):
    D[k] = np.array([s['params'].get(k, np.nan) for s in samples], float)

for k in sorted({k for s in samples for k in s if k != 'params'}):
    v0 = next(s[k] for s in samples if s.get(k) is not None)
    if isinstance(v0, dict):
        continue
    if isinstance(v0, str):
        D[k] = np.array([s.get(k, '') for s in samples])
    else:
        D[k] = np.array([np.nan if s.get(k) is None else float(s[k])
                         for s in samples])

D['chain']      = np.array(chain_id)
D['chain_name'] = np.array(chain_name)

# ── 常用推導量 ────────────────────────────────────────────────────────────────
D['cmb_depth_km']     = 3389.5 - D['R_cmb']
D['bml_top_depth_km'] = D['cmb_depth_km'] - D['BML_thickness']

hs, hl = D['h_solid_km'], D['h_liquid_km']
D['state'] = np.where(D['BML_thickness'] < 2.0, 'no_bml',
             np.where((hs >= 1) & (hl >= 1), 'layered',
             np.where(hl >= 1, 'molten', 'solid')))

# 每個欄位也丟成獨立變數,方便直接打 T_core / R_cmb
globals().update(D)

print(f'\n{len(samples)} samples, {len(D)} fields')
print(sorted(D))
