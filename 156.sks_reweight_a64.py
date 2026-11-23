#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 156: is the thinner BML in a70/a71 caused by SKS EXISTING at 126.09 deg, or by the light element?
#      Answered without a new MCMC: take the a64 posterior (Fe-S core, SKS not in the likelihood),
#      compute SKS at S1000a for ~2000 posterior samples, then
#        A. keep only the samples that HAVE SKS            -> effect of SKS existence alone
#        B. reweight by the SKS-PP term a70 uses            -> approximate "Fe-S + SKS" posterior
#      If A already moves molten BML thickness from ~240 to ~110-140 km (a70/a71), the thinning is
#      the SKS-existence constraint, not the light element.
#
#   beno3 (TauP + the main script; split over workers, resumable, run again after a TauP crash):
#       for k in 0 1 2 3 4 5 6 7; do nohup python 156.sks_reweight_a64.py run $k 8 > log_156_$k.txt 2>&1 & done
#   anywhere (after rsync, reads only the result files):
#       python 156.sks_reweight_a64.py
#
# Posterior samples: random chain RECORDS after burn-in (records are chain states, so this keeps
# the dwell-time weighting). A record's model is the profile_sNNNNN.npz of the last accepted step
# at or before it (profiles are written only at accepted steps).
#
# Weight B reproduces the a70 likelihood change for event S1000a. In a64 the event misfit is the
# SS-PP term x; with --sks it is (x + y)/2, y = min(|SKS-PP res|/sigma, RES_CAP) or MISS_PENALTY if
# there is no SKS. So  w = exp(-[(x + y)/2 - x]) = exp(-(y - x)/2).
import os, sys, json, glob
import numpy as np
from mars_config import MCMC_DIR

PREFIX     = 'chain_a64_YM20'
MAIN       = '26fall_main_mcmc_a63.py'      # build_taup and the Samuel crust come from here
CHAINS     = range(16)
BURN_FRAC  = 0.4
N_SAMP     = 2000
SEED       = 156
DELTA      = 126.09                         # S1000a, orbital imaging (fixed)
DEPTH      = 0.0
SS_PP_OBS  = (749.0, 10.8)                  # same as SAMUEL_DATA['S1000a'] in the main script
SKS_PP_OBS = (339.3, 10.8)                  # same as --sks in 26fall_main_mcmc_a70.py
RES_CAP, MISS_PENALTY = 10.0, 12.0
OUT_DIR    = os.path.join(MCMC_DIR, '156_sks_' + PREFIX.replace('chain_', ''))
STATES     = ['none', 'solid', 'layered', 'molten']

# ── posterior sample list (same on every machine: fixed seed) ───────────────
recs = []                                   # (chain, step, record)
for i in CHAINS:
    fp = os.path.join(MCMC_DIR, f'{PREFIX}_{i:02d}', 'chain.jsonl')
    R = []
    for line in open(fp):
        line = line.strip()
        if not line:
            continue
        try:
            R.append(json.loads(line))
        except json.JSONDecodeError:
            break
    for r in R[int(len(R)*BURN_FRAC):]:
        recs.append((i, r['step'], r))
pick = np.random.default_rng(SEED).choice(len(recs), N_SAMP, replace=False)
pick.sort()
print(f'{PREFIX}: {len(recs)} records after burn-in, {N_SAMP} picked')

# ══════════════════════════════════════════════════════════════════════════
# run mode (beno3): TauP for this worker's share of the samples
# ══════════════════════════════════════════════════════════════════════════
if len(sys.argv) > 1 and sys.argv[1] == 'run':
    worker, n_worker = int(sys.argv[2]), int(sys.argv[3])
    import importlib.util
    spec = importlib.util.spec_from_file_location('mainmc', os.path.join(os.path.dirname(os.path.abspath(__file__)), MAIN))
    M = importlib.util.module_from_spec(spec)
    sys.argv = [MAIN, '--comp', 'YM20']      # the main script parses --comp at import
    spec.loader.exec_module(M)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'worker{worker:02d}.jsonl')
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)['idx'])
            except Exception:
                pass
    mine = [j for n, j in enumerate(pick) if n % n_worker == worker and int(j) not in done]
    print(f'worker {worker}/{n_worker}: {len(mine)} to do ({len(done)} already done)', flush=True)
    prof_steps = {}
    for i in CHAINS:
        fs = glob.glob(os.path.join(MCMC_DIR, f'{PREFIX}_{i:02d}', 'profile_s*.npz'))
        prof_steps[i] = np.sort([int(os.path.basename(f)[9:14]) for f in fs])
    for j in mine:
        i, step, r = recs[int(j)]
        ps = prof_steps[i][prof_steps[i] <= step]
        if len(ps) == 0:
            continue
        z = np.load(os.path.join(MCMC_DIR, f'{PREFIX}_{i:02d}', f'profile_s{ps[-1]:05d}.npz'))
        lay = z['layer']
        prof = dict(z_km=z['z_km'], Vp=z['Vp'], Vs=z['Vs'], rho=z['rho'],
                    m_crust=lay == 0, m_mantle=lay == 1, m_bml=lay == 2, m_core=lay == 3,
                    z_cmb=float(z['z_cmb'][0]), bml=dict(thermal_state=str(z['thermal_state'][0])))
        res = dict(idx=int(j), chain=i, step=step, prof_step=int(ps[-1]),
                   BML_thickness=r['params']['BML_thickness'], R_cmb=r['params']['R_cmb'],
                   T_core=r['params']['T_core'], w_S=r['params']['w_S'],
                   Mgbml=r['params']['Mg#_bulk_bml'], h_solid=r.get('h_solid_km'),
                   h_liquid=r.get('h_liquid_km'), Vp_core_cmb=r.get('Vp_core_cmb'))
        try:
            tm = M.build_taup(prof, f'p156_w{worker:02d}', M._samuel_cache)
            arr = tm.get_travel_times(source_depth_in_km=DEPTH, distance_in_degree=DELTA,
                                      phase_list=['PP', 'SS', 'SKS'])
            t = {}
            for a in arr:
                if a.name not in t:
                    t[a.name] = a.time
            res['PP'], res['SS'], res['SKS'] = t.get('PP'), t.get('SS'), t.get('SKS')
            res['ok'] = True
        except Exception as e:
            res['ok'] = False
            res['err'] = str(e)[:200]
        with open(out, 'a') as f:
            f.write(json.dumps(res) + '\n')
        print(f"  idx {j}  chain {i} step {step}  SKS={res.get('SKS')}", flush=True)
    print('worker done', flush=True)
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════
# analysis mode: read the worker files
# ══════════════════════════════════════════════════════════════════════════
rows = []
for fp in sorted(glob.glob(os.path.join(OUT_DIR, 'worker*.jsonl'))):
    for line in open(fp):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
rows = [r for r in rows if r.get('ok') and r.get('PP') is not None and r.get('SS') is not None]
print(f'{len(rows)} samples with PP and SS at {DELTA} deg (of {N_SAMP})')
if not rows:
    sys.exit('no results yet')


def state(r):
    if r['BML_thickness'] < 1.0:
        return 'none'
    if r['h_liquid'] is None or r['h_solid'] is None:
        return 'fail'
    if r['h_liquid'] < 1:
        return 'solid'
    if r['h_solid'] < 1:
        return 'molten'
    return 'layered'


st   = np.array([state(r) for r in rows])
sks  = np.array([r['SKS'] is not None for r in rows])
H    = np.array([r['BML_thickness'] for r in rows])
Rc   = np.array([r['R_cmb'] for r in rows])
hl   = np.array([r['h_liquid'] or 0.0 for r in rows])
Tc   = np.array([r['T_core'] for r in rows])
wS   = np.array([r['w_S'] for r in rows])
x    = np.array([min(abs(r['SS'] - r['PP'] - SS_PP_OBS[0])/SS_PP_OBS[1], RES_CAP) for r in rows])
res  = np.array([(r['SKS'] - r['PP'] - SKS_PP_OBS[0]) if r['SKS'] is not None else np.nan for r in rows])
y    = np.where(sks, np.minimum(np.abs(res)/SKS_PP_OBS[1], RES_CAP), MISS_PENALTY)
w    = np.exp(-(y - x)/2.0)
w    = w/w.sum()
ess  = 1.0/np.sum(w**2)


def wq(v, wt, p=(5, 50, 95)):
    o = np.argsort(v); c = np.cumsum(wt[o]); c = c/c[-1]
    return [float(np.interp(pp/100.0, c, v[o])) for pp in p]


print(f'\n1. SKS exists at {DELTA} deg: {sks.mean():.3f} of the a64 posterior')
for s in STATES:
    m = st == s
    if m.any():
        print(f'   {s:>8}: n={m.sum():5d}  SKS {sks[m].mean():.2f}')
print(f'   SKS-PP residual (model - obs, s), samples with SKS: 5/50/95 = '
      f'{np.nanpercentile(res, 5):+.1f} {np.nanpercentile(res, 50):+.1f} {np.nanpercentile(res, 95):+.1f}'
      f'   (sigma {SKS_PP_OBS[1]})')

print(f'\n2. molten / all: median [5, 95]   all a64  |  A: SKS exists  |  B: reweighted (ESS {ess:.0f})')
for name, v in (('BML_thickness', H), ('h_liquid', hl), ('R_cmb', Rc), ('R_cmb+h_liq', Rc + hl),
                ('T_core', Tc), ('w_S', wS)):
    for lab, m in (('molten', st == 'molten'), ('all', np.ones(len(st), bool))):
        a = np.percentile(v[m], [5, 50, 95])
        b = np.percentile(v[m & sks], [5, 50, 95]) if (m & sks).sum() > 10 else [np.nan]*3
        c = wq(v[m], w[m]) if w[m].sum() > 0 else [np.nan]*3
        print(f'   {name:>13} {lab:>6}  {a[1]:8.4g} [{a[0]:.4g}, {a[2]:.4g}]  |  '
              f'{b[1]:8.4g} [{b[0]:.4g}, {b[2]:.4g}]  |  {c[1]:8.4g} [{c[0]:.4g}, {c[2]:.4g}]')

print('\n3. state fractions   all a64 | A: SKS exists | B: reweighted')
for s in STATES:
    print(f'   {s:>8}  {np.mean(st == s):.3f} | {np.mean(st[sks] == s):.3f} | {w[st == s].sum():.3f}')
print('\n   compare with a70 / a71 stage 1 (155, 113): molten BML_thickness median 139 / 108 km,'
      ' R_cmb 1619 / 1651 km')
