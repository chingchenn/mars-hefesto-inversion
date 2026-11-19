#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 90: pick ONE representative posterior model for the method figure (panels 91-95 read it)
#
# Selection: a60 YM20, all records after burn-in (rejected steps repeat the state, so each
# unique accepted state is weighted by how many records sit on it), layered BML only,
# then the unique state closest to the layered posterior median in standardized units:
#     d^2 = sum_k ((x_k - median_k) / IQR_k)^2,   k = 8 parameters + h_liquid
# Set OVERRIDE to a (chain name, step) pair to use a model of your own choice instead.

import os
import glob
import json
import numpy as np
from mars_config import MCMC_DIR

# ── config ────────────────────────────────────────────────────────────────────
CHAIN_ROOT = MCMC_DIR
PREFIX     = 'chain_a60_YM20_'
EXCLUDE    = []                 # e.g. 'chain_a60_YM20_05'
BURN_FRAC  = 0.40
H_MIN      = 1.0                # km, three-state threshold (same as chainio.py)
STATE      = 'layered'
OVERRIDE   = None               # e.g. ('chain_a60_YM20_03', 9858)
OUT_JSON   = '90_rep_model.json'
PAR = ['T_lit', 'P_lit', 'Mg#', 'Mg#_bulk_bml', 'BML_thickness', 'T_core', 'w_S', 'R_cmb']
KEEP = ['h_solid_km', 'h_liquid_km', 'Mg_solid', 'Mg_liquid', 'T_interface', 'P_interface',
        'P_cmb', 'P_bml_top', 'T_mantle_bottom', 'T_solidus_bml', 'T_liquidus_bml', 'z_lit_km',
        'phase_branch', 'thermal_state', 'misfit', 'misfit_tt', 'mass_sigma', 'moi_sigma']

# ── read chains: each post-burn-in record sits on the last accepted state ─────
weights, rec_of = {}, {}
for path in sorted(glob.glob(os.path.join(CHAIN_ROOT, PREFIX + '*', 'chain.jsonl'))):
    chain_dir = os.path.dirname(path)
    if os.path.basename(chain_dir) in EXCLUDE:
        continue
    lines = [json.loads(l) for l in open(path) if l.strip().endswith('}')]   # skip a half-written line
    n_burn = int(BURN_FRAC * len(lines))
    last_acc = None
    for i, rec in enumerate(lines):
        if rec['accepted']:
            last_acc = rec['step']
            rec_of[(chain_dir, last_acc)] = rec
        if i >= n_burn and last_acc is not None:
            key = (chain_dir, last_acc)
            weights[key] = weights.get(key, 0) + 1
    print(f"{os.path.basename(chain_dir)}: {len(lines)} records")

keys = sorted(weights)
hs = np.array([rec_of[k]['h_solid_km'] for k in keys])
hl = np.array([rec_of[k]['h_liquid_km'] for k in keys])
state = np.full(len(keys), 'fail', dtype='<U8')
state[(hs >= H_MIN) & (hl >= H_MIN)] = 'layered'
state[(hs < H_MIN) & (hl >= H_MIN)] = 'molten'
state[(hs >= H_MIN) & (hl < H_MIN)] = 'solid'
w = np.array([weights[k] for k in keys], float)
for s in ['solid', 'layered', 'molten', 'fail']:
    print(f"  {s:8s}: {np.sum(state == s):6d} unique states, {w[state == s].sum() / w.sum() * 100:5.1f}% of records")

# ── choose the model ──────────────────────────────────────────────────────────
if OVERRIDE is None:
    m = state == STATE
    cand = [keys[j] for j in np.flatnonzero(m)]
    X = np.array([[rec_of[k]['params'][p] for p in PAR] + [rec_of[k]['h_liquid_km']] for k in cand])
    Xr = np.repeat(X, w[m].astype(int), axis=0)          # weighted percentiles
    med = np.median(Xr, axis=0)
    iqr = np.percentile(Xr, 75, axis=0) - np.percentile(Xr, 25, axis=0)
    d2 = np.sum(((X - med) / iqr) ** 2, axis=1)
    order = np.argsort(d2)
    pick = None
    for j in order:                                       # first one with a saved profile
        k = cand[j]
        if os.path.exists(os.path.join(k[0], f"profile_s{k[1]:05d}.npz")):
            pick = k
            break
    print(f"\n{STATE} posterior median:")
    for p, v in zip(PAR + ['h_liquid_km'], med):
        print(f"  {p:14s} {v:10.4g}")
else:
    pick = (os.path.join(CHAIN_ROOT, OVERRIDE[0]), OVERRIDE[1])

rec = rec_of[pick]
out = dict(chain_dir=pick[0], chain=os.path.basename(pick[0]), step=int(pick[1]),
           profile=os.path.join(pick[0], f"profile_s{pick[1]:05d}.npz"),
           params=rec['params'], **{k: rec.get(k) for k in KEEP})
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=1)

print(f"\npicked {out['chain']} step {out['step']}  (weight {weights.get(pick, 0)} records)")
for p in PAR:
    print(f"  {p:14s} {rec['params'][p]:10.4g}")
for k in KEEP:
    print(f"  {k:14s} {rec.get(k)}")
print(f"saved {OUT_JSON}")
