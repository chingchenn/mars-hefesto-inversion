#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 155: first look at a70 (Fe-S-C core) and a71 (Fe-S-H core), stage 1 (diagonal STEP, 1500 steps),
#      both with SKS-PP of S1000a in the likelihood. Compared with a64 (Fe-S, no SKS).
#
#   python 155.check_a70_a71.py
#
# 1. bookkeeping : steps, accept rate per chain
# 2. SKS         : fraction of steps where S1000a has a missing phase (= no SKS at 126.09 deg),
#                  per chain, first vs second half of the kept window. Stage 2 makes sense only
#                  if most chains have moved to SKS-producing models.
# 3. w_C / w_H   : posterior 5/50/95, fraction near the prior walls; for H also P(w_H < 0.15 wt%)
#                  (Samuel 2023 upper bound)
# 4. correlations of w_X with w_S, R_cmb, T_core, BML_thickness (the ridge stage 2 has to follow)
# 5. core        : Vp_cmb, rho_core_mean, w_S vs a64; misfit_tt
# 6. BML states  : none / solid / layered / molten, pooled and per chain (stage 1: not citable)
import os, json
import numpy as np
from mars_config import MCMC_DIR

RUNS   = {'a70': ('chain_a70_YM20', 'w_C', (0.0, 0.02)),
          'a71': ('chain_a71_YM20', 'w_H', (0.0, 0.01))}
REF    = 'chain_a64_YM20'
CHAINS = range(16)
BURN_FRAC = 0.4          # same as 63v2 / 81
WALL   = 0.02            # fraction of the prior width counted as "at the wall"
H_MAX_SAMUEL = 0.0015    # Samuel 2023: absolute upper bound for H in Mars's core


def load(prefix, i):
    recs = []
    fp = os.path.join(MCMC_DIR, f'{prefix}_{i:02d}', 'chain.jsonl')
    if not os.path.exists(fp):
        return recs
    for line in open(fp):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return recs


def state(r):
    # same as 153: none if h < 1 km, solid if h_liquid < 1, molten if h_solid < 1, else layered
    if r['params']['BML_thickness'] < 1.0:
        return 'n'
    hs, hl = r.get('h_solid_km'), r.get('h_liquid_km')
    if hs is None or hl is None:
        return 'f'
    if hl < 1:
        return 's'
    if hs < 1:
        return 'm'
    return 'l'


def no_sks(r):
    return (r.get('tt_miss_by_event') or {}).get('S1000a', 0) > 0


q = lambda x: np.percentile(x, [5, 50, 95])

# reference a64, same burn-in fraction
ref = []
for i in CHAINS:
    R = load(REF, i)
    ref += R[int(len(R)*BURN_FRAC):]
print(f'{REF}: {len(ref)} records after burn-in {BURN_FRAC}')

for run, (prefix, wk, (lo, hi)) in RUNS.items():
    print(f'\n{"=" * 80}\n{run}  ({prefix}, {wk} prior {lo}-{hi})\n{"=" * 80}')
    chains = {i: load(prefix, i) for i in CHAINS}
    chains = {i: R for i, R in chains.items() if R}
    if not chains:
        print('  no chains found in', MCMC_DIR)
        continue

    # ── 1. bookkeeping ──────────────────────────────────────────────────────
    print('\n1./2. per chain: steps, accept rate, no-SKS fraction (1st / 2nd half of kept window),')
    print('      last w_X, last state')
    print(f"{'chain':>5} {'steps':>6} {'acc%':>6} {'noSKS 1st':>10} {'2nd':>6} {wk:>8} {'state':>6}")
    kept = []
    for i, R in chains.items():
        X = np.array([[r['params'][k] for k in r['params']] for r in R])
        acc = 100*np.mean(np.any(np.diff(X, axis=0) != 0, axis=1))
        K = R[int(len(R)*BURN_FRAC):]
        h = len(K)//2
        f1 = np.mean([no_sks(r) for r in K[:h]]) if h else np.nan
        f2 = np.mean([no_sks(r) for r in K[h:]]) if h else np.nan
        print(f'{i:5d} {len(R):6d} {acc:6.1f} {f1:10.2f} {f2:6.2f} {R[-1]["params"][wk]:8.4f} {state(R[-1]):>6}')
        kept += K
    print(f'  pooled no-SKS fraction (kept window): {np.mean([no_sks(r) for r in kept]):.3f}'
          f'   a64 (SKS not in likelihood): {np.mean([no_sks(r) for r in ref]):.3f}')

    # ── 3. light element ────────────────────────────────────────────────────
    w = np.array([r['params'][wk] for r in kept])
    a = q(w)
    d = WALL*(hi - lo)
    print(f'\n3. {wk} [wt fraction]: 5/50/95 = {a[0]:.4f} {a[1]:.4f} {a[2]:.4f}'
          f'   near low wall {np.mean(w < lo + d):.3f}  near high wall {np.mean(w > hi - d):.3f}'
          f'   (uniform: {WALL:.3f} each)')
    if wk == 'w_H':
        print(f'   P(w_H < {H_MAX_SAMUEL}) = {np.mean(w < H_MAX_SAMUEL):.3f}'
              f'   (prior {H_MAX_SAMUEL/(hi - lo):.3f})')

    # ── 4. correlations ─────────────────────────────────────────────────────
    print(f'\n4. correlation of {wk} with (kept window, pooled):')
    for k in ('w_S', 'R_cmb', 'T_core', 'BML_thickness', 'Mg#_bulk_bml'):
        x = np.array([r['params'][k] for r in kept])
        print(f'   {k:>14}  {np.corrcoef(w, x)[0, 1]:+.3f}')
    ws = np.array([r['params']['w_S'] for r in kept])
    print(f'   corr(w_S, R_cmb) = {np.corrcoef(ws, [r["params"]["R_cmb"] for r in kept])[0, 1]:+.3f}'
          f'   (a64: {np.corrcoef([r["params"]["w_S"] for r in ref], [r["params"]["R_cmb"] for r in ref])[0, 1]:+.3f})')

    # ── 5. core and misfit ──────────────────────────────────────────────────
    print(f'\n5. 5/50/95 {"":>10}{run:>24}{"a64":>26}')
    for key, lab in (('Vp_core_cmb', 'Vp_cmb'), ('rho_core_mean', 'rho_core'), ('misfit_tt', 'misfit_tt'),
                     ('misfit', 'misfit')):
        a = q([r[key] for r in kept]); b = q([r[key] for r in ref])
        print(f'   {lab:>10}   {a[0]:7.3f} {a[1]:7.3f} {a[2]:7.3f}     {b[0]:7.3f} {b[1]:7.3f} {b[2]:7.3f}')
    for k in ('w_S', 'R_cmb', 'T_core'):
        a = q([r['params'][k] for r in kept]); b = q([r['params'][k] for r in ref])
        print(f'   {k:>10}   {a[0]:7.4g} {a[1]:7.4g} {a[2]:7.4g}     {b[0]:7.4g} {b[1]:7.4g} {b[2]:7.4g}')
    print('   misfit_tt here includes SKS-PP (or its MISS_PENALTY); a64 does not')

    # ── 6. states ───────────────────────────────────────────────────────────
    st = np.array([state(r) for r in kept]); st0 = np.array([state(r) for r in ref])
    print('\n6. BML states (stage 1, not citable)   none / solid / layered / molten')
    print(f'   {run}: ' + '  '.join(f'{np.mean(st == s):.3f}' for s in 'nslm'))
    print(f'   a64: ' + '  '.join(f'{np.mean(st0 == s):.3f}' for s in 'nslm'))
    for s, name in (('s', 'solid'), ('m', 'molten')):
        sel = st == s
        if sel.any():
            print(f'   {name:>7}: n={sel.sum():6d}  median {wk} {np.median(w[sel]):.4f}  '
                  f'Vp_cmb {np.median([r["Vp_core_cmb"] for r, t in zip(kept, sel) if t]):.3f}  '
                  f'no-SKS {np.mean([no_sks(r) for r, t in zip(kept, sel) if t]):.2f}')
