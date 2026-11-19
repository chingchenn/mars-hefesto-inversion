#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
151: forward tests at the BML_thickness -> 0 limit and at w_S -> 0

    python 151.thin_bml_test.py --prefix chain_a60_YM20

Takes one molten and one solid posterior state (last record of the a60 chains)
and runs the forward model with h = 0 ... 20 km in two geometries:
    fixR   : R_cmb fixed (core size kept, BML top moves)
    fixTop : BML top fixed (R_cmb = R_top - h, the ScS reflector stays put)
plus w_S = 0 ... 0.05. Prints the misfit or the rejection reason.

Question: is the no-BML limit (h -> 0) reachable by the sampler at all?
"""
import os, sys, glob, json, argparse, io, contextlib, importlib.util
import numpy as np
from mars_config import MCMC_DIR

ap = argparse.ArgumentParser()
ap.add_argument('--prefix', default='chain_a60_YM20')
ap.add_argument('--main',   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 '26fall_main_mcmc.py'))
a = ap.parse_args()
comp = a.prefix.split('_')[-1]

# the main script reads --comp from sys.argv at import time
sys.argv = [a.main, '--comp', comp]
spec = importlib.util.spec_from_file_location('M', a.main)
M = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(M)
    M.CF.configure(route='B', verbose=False)
print(f'main: {a.main}   comp: {comp}')


def last_record(fp):
    last = None
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    break
    return last


def state(hs, hl):
    if hs is None or hl is None:
        return 'unknown'
    if hl < 1:
        return 'solid'
    if hs < 1:
        return 'molten'
    return 'layered'


# ── pick one molten and one solid posterior state ────────────────────────────
picked = {}
for d in sorted(glob.glob(os.path.join(MCMC_DIR, f'{a.prefix}_[0-9][0-9]'))):
    r = last_record(os.path.join(d, 'chain.jsonl'))
    if r is None:
        continue
    s = state(r.get('h_solid_km'), r.get('h_liquid_km'))
    if s in ('molten', 'solid') and s not in picked:
        picked[s] = (os.path.basename(d), r)
for s, (name, r) in picked.items():
    p = r['params']
    print(f'{s:7s} from {name}: R_cmb {p["R_cmb"]:.1f}  h {p["BML_thickness"]:.1f}  '
          f'w_S {p["w_S"]:.3f}  T_core {p["T_core"]:.0f}  misfit {r["misfit"]:.3f}')
if 'solid' not in picked:
    print('no chain ends in the solid state; only the molten case is tested')

KEYS = ('rejected', 'reject', 'failed', 'FAILED', 'NGFAIL', 'error', 'Error')

def run(p, tag):
    name = f't151_{tag}_{os.getpid()}'.replace('.', 'p')   # unique: build_taup caches by name
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mis, _, c, prof, _ = M.forward(p, '.', name, M._samuel_cache)
    except Exception as e:                                  # anything forward() does not catch
        return f'EXCEPTION {type(e).__name__}: {e}'
    if mis is None or mis >= 990:
        why = [l.strip() for l in buf.getvalue().splitlines() if any(k in l for k in KEYS)]
        return 'REJECT   ' + (why[-1][:110] if why else '(no message)')
    return (f'misfit {mis:7.3f}  tt {c["misfit_tt"]:6.3f}  miss {c["tt_n_miss"]}  '
            f'h_s {c["h_solid_km"]:6.2f}  h_l {c["h_liquid_km"]:6.2f}  '
            f'R_ScS {p["R_cmb"] + c["h_liquid_km"]:7.1f}  moi {c["moi_sigma"]:.2f}')


H = (0.0, 0.1, 0.5, 0.99, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0)
for s, (name, r) in picked.items():
    base = {k: float(v) for k, v in r['params'].items()}
    R_top = base['R_cmb'] + base['BML_thickness']
    print(f'\n==== {s} ({name}) ====')
    for mode in ('fixR', 'fixTop'):
        for h in H:
            p = dict(base, BML_thickness=h)
            if mode == 'fixTop':
                p['R_cmb'] = R_top - h
            print(f'  {mode:6s} h={h:5.2f}  {run(p, f"{s}_{mode}_h{h:g}")}', flush=True)
    for ws in (0.0, 0.01, 0.03, 0.05):
        print(f'  w_S   ={ws:5.2f}  {run(dict(base, w_S=ws), f"{s}_ws{ws:g}")}', flush=True)

# ── core EoS alone at w_S = 0 ─────────────────────────────────────────────────
pr = M.CF.core_properties(np.array([20.0, 30.0, 40.0]), np.array([2200.0, 2400.0, 2600.0]), 0.0)
print('\ncore_fes at w_S = 0:')
for k in ('rho', 'Vp', 'dTdP'):
    print(f'  {k:5s}', np.asarray(pr[k]))