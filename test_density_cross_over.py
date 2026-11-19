#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 96: J = d(M/sigma_M, I/sigma_I)/dp  (2 x 8), central difference at one reference
#     point on the mass/MoI shell. Output feeds propose() in 82 (a47):
#         p' = p + N eps + G eta
#     N = null space of J (8x6, directions that leave M and I unchanged)
#     G = row  space of J (8x2, directions that cross the walls)
# Coordinates are scaled by 1% of prior width, NOT by STEP: STEP['R_cmb']=1.0 is
# itself an artefact of the isotropic-proposal era and would drag the old problem
# into the new basis.
# Cost: 17 forwards + up to 2 verification forwards ~ 19 x 58 s.

import sys, os
import glob
import json
import numpy as np
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'm82', '/home/jcchen2/mars-hefesto-inversion/82.mcmc_core_nGibbs_wideMoI.py')
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

# ── config ────────────────────────────────────────────────────────────────────
ROUTE       = 'B'
PREFIX      = 'chain_a46B'          # where to pick the reference point
REF_FILE    = None                  # json with 8 params; None -> best a46 chain state
MAX_GEO     = 10.0                  # reference: mass_sigma + moi_sigma below this
RUN_DIR     = 'jac96'
S_PAR_TEST  = 3.0                   # verification step along tangent directions (scaled units)

# finite-difference half steps; inside the linear range (94: +-20 km in R_cmb),
# well above the self-consistent-field noise (SC_TOL_P = 0.01 GPa)
H = {'T_lit': 10.0, 'P_lit': 0.10, 'Mg#': 0.005, 'T_core': 50.0,
     'Mg#_bulk_bml': 0.02, 'BML_thickness': 10.0, 'w_S': 0.005, 'R_cmb': 5.0}

sys.path.insert(0, '/home/jcchen2/mars-hefesto-inversion')
m.CF.configure(route=ROUTE, verbose=False)
KEYS  = list(m.PRIOR)
WIDTH = np.array([m.PRIOR[k][1] - m.PRIOR[k][0] for k in KEYS])
SCALE = WIDTH/100.0
# T_lit 16K  P_lit 0.076  Mg# 0.004  T_core 20.5K  Mg#bml 0.006  D 4km  w_S 0.0035  R_cmb 5km

# ── reference point ───────────────────────────────────────────────────────────
# three requirements, all violated by the naive "min mass+moi" pick (which sat at
# R_cmb=1895 / T_core=3555 / D=13, on three prior walls and grav-unstable):
#   on the shell        mass + moi < MAX_GEO
#   gravitationally OK  grav_sigma ~ 0 (Kono penalty off the linearisation point)
#   away from walls     finite differences and tangent tests must not leave the prior
def ok_ref(rec, why=False):
    p = rec['params']
    g = rec.get('grav_sigma')
    if g is None or g > 0.1:
        return 'grav' if why else False
    if (rec.get('mass_sigma') or 9e9) + (rec.get('moi_sigma') or 9e9) > MAX_GEO:
        return 'geo' if why else False
    for k in KEYS:
        lo, hi = m.PRIOR[k]
        if p[k] - H[k] < lo or p[k] + H[k] > hi:      # 只要求 FD 不出界
            return k if why else False
    return 'OK' if why else True

def best_chain_state():
    cands = []
    for f in sorted(glob.glob(os.path.join(m.MCMC_DIR, f'{PREFIX}_*', 'chain.jsonl'))):
        last = None
        for line in open(f):
            line = line.strip()
            if line:
                try: last = json.loads(line)
                except json.JSONDecodeError: break
        if last is None:
            continue
        geo_ = (last.get('mass_sigma') or 9e9) + (last.get('moi_sigma') or 9e9)
        tag = ok_ref(last, why=True)
        print(f'{f.split("/")[-2]:16s} step={last["step"]:5d} '
              f'mass={last.get("mass_sigma"):7.2f} moi={last.get("moi_sigma"):7.2f} '
              f'grav={last.get("grav_sigma"):5.2f} tt={last.get("misfit_tt"):7.1f} '
              f'R={last["params"]["R_cmb"]:6.1f} D={last["params"]["BML_thickness"]:6.1f} '
               f'{tag}')
        if tag == 'OK':
            cands.append((geo_, last))
    if not cands:
        raise SystemExit('no usable reference point: every chain is on a wall, '
                         'grav-unstable, or off the shell. Let a46 run further.')
    return min(cands, key=lambda t: t[0])[1]

if REF_FILE is not None:
    REF = json.load(open(REF_FILE))
else:
    rec = best_chain_state()
    REF = rec['params']
    print(f'\nreference: step {rec["step"]}  mass={rec["mass_sigma"]:.2f}  '
          f'moi={rec["moi_sigma"]:.2f}  h_sol={rec["h_solid_km"]:.1f} km')
print('REF =', json.dumps(REF, indent=1))

def in_prior(p):
    return all(m.PRIOR[k][0] <= p[k] <= m.PRIOR[k][1] for k in KEYS)

for j, k in enumerate(KEYS):
    for sgn in (+1, -1):
        if not (m.PRIOR[k][0] <= REF[k] + sgn*H[k] <= m.PRIOR[k][1]):
            raise SystemExit(f'REF +- H leaves prior on {k}: '
                             f'{REF[k]:.4g} {sgn:+d}*{H[k]:.4g}')

# ── forward wrapper: returns (M/sigma_M, I/sigma_I) as signed offsets ─────────
run_dir = os.path.join(m.MCMC_DIR, RUN_DIR)
os.makedirs(run_dir, exist_ok=True)

def geo(p, name):
    c = m.forward(p, run_dir, f'jac96_{name}', m._khan_cache)[2]
    if c is None:
        raise RuntimeError(f'forward failed at {name}: {p}')
    return np.array([(c['M_pred']   - m.MARS_MASS_OBS)/m.MARS_MASS_SIGMA,
                     (c['moi_pred'] - m.MOI_OBS)      /m.MOI_SIGMA]), c

# ── Jacobian ──────────────────────────────────────────────────────────────────
# J_ij = [f_i(p + h_j e_j) - f_i(p - h_j e_j)] / (2 h_j)
g0, c0 = geo(REF, 'ref')
print(f'\nref: mass={g0[0]:+.2f}sig  moi={g0[1]:+.2f}sig  h_sol={c0["h_solid_km"]:.1f}  '
      f'state={c0["thermal_state"]}  melt={c0["melting"]}  '
      f'miss={c0["tt_n_miss"]}  grav={c0["grav_sigma"]:.2f}')

J = np.zeros((2, 8))
state_flip = []
for j, k in enumerate(KEYS):
    gp = {}
    for sgn in (+1, -1):
        p = dict(REF); p[k] += sgn*H[k]
        gp[sgn], c = geo(p, f'{k}_{"p" if sgn > 0 else "m"}')
        if (c['h_solid_km'] >= 1.0) != (c0['h_solid_km'] >= 1.0):
            state_flip.append((k, sgn))
    J[:, j] = (gp[+1] - gp[-1])/(2*H[k])
    print(f'{k:14s}  dM/dp = {J[0,j]:+12.4e} sig/unit   dI/dp = {J[1,j]:+12.4e} sig/unit')

if state_flip:
    print('\nWARNING: three-state flip in finite differences (J unreliable there):', state_flip)

# ── cross-check against 94 (94 was taken at R_cmb=1620, w_S=0.170; the ridge ──
# steepens away from there, so agreement is order-of-magnitude, not exact) ─────
iR, iW = KEYS.index('R_cmb'), KEYS.index('w_S')
print(f'\n94 check:  dmass/dR_cmb = {J[0,iR]:+.3f} (94 @1620: +0.524)   '
      f'dmass/dw_S = {J[0,iW]:+.1f} (94: -854)   '
      f'mass ridge = {-J[0,iW]/J[0,iR]:+.0f} km/unit (94: 1630)   '
      f'MoI ridge = {-J[1,iW]/J[1,iR]:+.0f} km/unit')

# ── SVD in prior-scaled coordinates ───────────────────────────────────────────
# x = p / SCALE  ->  J_s = J * SCALE = U diag(s) V^T
# rows 0-1 of V^T span the wall directions, rows 2-7 the shell tangent space
Js = J*SCALE
U, s, Vt = np.linalg.svd(Js)
N = Vt[2:].T          # 8 x 6
G = Vt[:2].T          # 8 x 2

print(f'\nsingular values (sigma per 1% prior width): {s[0]:.2f}  {s[1]:.2f}')
print('wall directions:')
for i in range(2):
    print(f'  g{i}: ' + '  '.join(f'{k}={Vt[i,j]:+.2f}' for j, k in enumerate(KEYS)))
print('tangent directions:')
for i in range(6):
    print(f'  n{i}: ' + '  '.join(f'{k}={Vt[2+i,j]:+.2f}' for j, k in enumerate(KEYS)))

np.savez('96_jacobian_a46.npz', J=J, SCALE=SCALE, N=N, G=G, s=s,
         REF=np.array([REF[k] for k in KEYS]), KEYS=np.array(KEYS),
         H=np.array([H[k] for k in KEYS]))

# ── verification: walk S_PAR_TEST scaled units along tangent directions ───────
# expected: |dM|, |dI| << 1 sigma; larger means the linear range < S_PAR_TEST
rng = np.random.default_rng(96)
for t in range(2):
    for _try in range(20):
        eps = rng.normal(0, S_PAR_TEST, 6)
        dx  = N @ eps
        p   = dict(zip(KEYS, np.array([REF[k] for k in KEYS]) + dx*SCALE))
        if in_prior(p):
            break
    else:
        print(f'\ntangent test {t}: 20 draws all left the prior -> REF too close to '
              f'a wall or S_PAR_TEST too large; skipped')
        continue
    g, c = geo(p, f'tan{t}')
    print(f'\ntangent test {t}: |dx| = {np.linalg.norm(dx):.2f} units  '
          f'dM = {g[0]-g0[0]:+.3f}sig  dI = {g[1]-g0[1]:+.3f}sig  '
          f'dTT = {c["misfit_tt"]-c0["misfit_tt"]:+.2f}  h_sol={c["h_solid_km"]:.1f}')
    print('  dp: ' + '  '.join(f'{k}={dx[j]*SCALE[j]:+.4g}' for j, k in enumerate(KEYS)))

print('\nsaved 96_jacobian_a46.npz')