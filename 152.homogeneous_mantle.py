#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
152: how much posterior mass sits on a "homogeneous mantle"

    python 152.homogeneous_mantle.py --prefix chain_a60_YM20

Homogeneous means either
  H1  no BML:            BML_thickness -> 0
  H2  chemically same:   solid BML with Mg#_bulk_bml ~ Mg# (mantle).
      The solid BML is computed by nGibbs with the mantle NCFMAS composition and only
      Mg# replaced (composition_from_params), so equal Mg# = identical composition;
      only the temperature profile differs.

H1: Savage-Dickey BF01(eps) = P(h < eps | d) / (eps/400). 151 showed the forward
    model is smooth in h from 1 km up and h < 1 km is rejected by build_taup, so
    the limit is estimated from small-h samples.
H2: posterior fraction only. Its prior fraction needs a prior-only run (state
    depends on the phase diagram), so no BF yet.
a60 mixes poorly in the deep parameters: per-chain spread is printed; nothing here
is citable.
"""
import os, glob, json, argparse
import numpy as np
from mars_config import MCMC_DIR

ap = argparse.ArgumentParser()
ap.add_argument('--prefix',    default='chain_a60_YM20')
ap.add_argument('--burn_frac', type=float, default=0.4)
ap.add_argument('--hmax',      type=float, default=400.0)
ap.add_argument('--bml_mg_max', type=float, default=0.80)   # Mg#_bulk_bml prior upper bound in a60
a = ap.parse_args()


def load_jsonl(fp):
    recs = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return recs


def state(hs, hl):
    if hs is None or hl is None:
        return 'fail'
    if hl < 1:
        return 'solid'
    if hs < 1:
        return 'molten'
    return 'layered'


C = []          # one dict of arrays per chain
for d in sorted(glob.glob(os.path.join(MCMC_DIR, f'{a.prefix}_[0-9][0-9]'))):
    recs = load_jsonl(os.path.join(d, 'chain.jsonl'))
    if len(recs) < 300:
        continue
    kept = recs[int(len(recs) * a.burn_frac):]
    C.append(dict(
        h=np.array([r['params']['BML_thickness'] for r in kept]),
        mg=np.array([r['params']['Mg#'] for r in kept]),
        mgb=np.array([r['params']['Mg#_bulk_bml'] for r in kept]),
        st=np.array([state(r.get('h_solid_km'), r.get('h_liquid_km')) for r in kept]),
        uc=np.array([r.get('upper_contrast') if r.get('upper_contrast') is not None
                     else np.nan for r in kept], dtype=float)))
A = {k: np.concatenate([c[k] for c in C]) for k in C[0]}
N = len(A['h'])
print(f'{a.prefix}: {len(C)} chains, {N} samples, min h = {A["h"].min():.2f} km')

# ── where is the mantle Mg# relative to the BML prior wall ──────────────────
q = np.percentile(A['mg'], [5, 50, 95])
print(f'\nmantle Mg# 5/50/95: {q.round(3)}   Mg#_bulk_bml prior upper bound: {a.bml_mg_max}')
print(f'P(mantle Mg# > {a.bml_mg_max}) = {np.mean(A["mg"] > a.bml_mg_max):.3f}'
      '   (these samples cannot have Mg#_bml = Mg#_mantle under the a60 prior)')

# ── H1: no BML ───────────────────────────────────────────────────────────────
print(f"\nH1  no BML (Savage-Dickey)\n{'eps(km)':>8} {'n':>7} {'post':>8} {'prior':>7} "
      f"{'BF01':>6} {'chain min / median / max':>26} {'chains>0':>9}")
for eps in (5, 10, 20, 40, 80):
    prior = eps / a.hmax
    per = np.array([np.mean(c['h'] < eps) / prior for c in C])
    print(f'{eps:8.0f} {int(np.sum(A["h"] < eps)):7d} {np.mean(A["h"] < eps):8.4f} '
          f'{prior:7.4f} {np.mean(A["h"] < eps)/prior:6.2f} '
          f'{per.min():9.2f} / {np.median(per):5.2f} / {per.max():6.2f} '
          f'{int(np.sum(per > 0)):5d}/{len(C)}')
small = A['h'] < 20
print('state of h < 20 km samples: ' + '  '.join(
    f'{s} {np.mean(A["st"][small] == s):.2f}' for s in ('solid', 'layered', 'molten'))
    + f'   (all samples: ' + '  '.join(
    f'{s} {np.mean(A["st"] == s):.2f}' for s in ('solid', 'layered', 'molten')) + ')')

# ── H2: solid BML with the mantle composition ───────────────────────────────
dmg = A['mgb'] - A['mg']
solid = A['st'] == 'solid'
print(f"\nH2  solid & |Mg#_bml - Mg#| < delta   (posterior fraction only)")
print(f"{'delta':>6} {'post':>8} {'chain min / median / max':>26} "
      f"{'upper_contrast 5/50/95 (g/cm3)':>32}")
for dl in (0.02, 0.05, 0.10):
    m = solid & (np.abs(dmg) < dl)
    per = np.array([np.mean((c['st'] == 'solid') & (np.abs(c['mgb'] - c['mg']) < dl)) for c in C])
    uc = np.nanpercentile(A['uc'][m], [5, 50, 95]).round(3) if m.any() else '-'
    print(f'{dl:6.2f} {np.mean(m):8.4f} {per.min():9.4f} / {np.median(per):6.4f} / '
          f'{per.max():6.4f} {str(uc):>32}')
print(f'solid samples: Mg#_bml - Mg# 5/50/95 = '
      f'{np.percentile(dmg[solid], [5, 50, 95]).round(3) if solid.any() else "-"}')

# ── either ───────────────────────────────────────────────────────────────────
hom = (A['h'] < 20) | (solid & (np.abs(dmg) < 0.05))
print(f'\nhomogeneous-like (h < 20 km  or  solid & |dMg#| < 0.05): {np.mean(hom):.4f}')