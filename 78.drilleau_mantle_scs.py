#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
78: Where must the ScS reflector sit for a given mantle?
    Drilleau 2026 mantles vs this study, in our forward model and with our data.

Question
  Drilleau 2026 put the ScS reflector at ~1816-1838 km; a64 puts it at ~1770 km.
  Is the difference set by the mantle (Mg#, geotherm), or by their method
  (SLB21, inverted epicentral distances, different likelihood)?

Test
  For each mantle case: no BML (BML_thickness = 0, state 'none', so R_ScS = R_CMB),
  scan R_CMB, compute the ScS-S residuals of our data set with our fixed event
  locations. The R that minimizes the ScS-S L1 misfit is the reflector this mantle
  requires. If Drilleau's mantles give ~1820 km here, the difference is the mantle;
  if they give ~1770 km, the difference is their method.

Drilleau 2026 cases (Table 3, mean values)
  our geotherm: linear from T_SURF to T_lit at P_lit, then isentrope
  their equivalent: T_lit = Tm (uppermost mantle temperature),
                    z_lit = Dlu (lithosphere + upper TBL thickness), P_lit = P(Dlu)
  Table 3 has BML columns for EH45 and TA only; YM appears only as non-BML.

  case             comp  Tm    Dlu   Mg#    their reflector
  D_TA_BML         TA    1756  334   0.773  1656 + 160 = 1816
  D_EH45_BML       EH45  1778  363   0.755  1659 + 162 = 1821
  D_YM_F2_noBML    YM20  1916  644   0.780  1838
  D_YM_F1_noBML    YM20  1945  852   0.768  1835

Decomposition cases (YM20, swap one thing at a time against a64)
  ours                   a64 posterior medians
  ours_T + D_TA_Mg#      our geotherm, Mg# 0.773
  D_TA_T + ours_Mg#      Drilleau TA BML geotherm, our Mg#
  D_TA_BML_YM20          Drilleau TA BML geotherm and Mg#, our YM20 composition

Not tested here: SLB21 vs our nGibbs (cannot swap), their crust (67 km inverted;
ours is the fixed Samuel 0-100 km column), their deep TBL above the BML.

Run on beno3 (nGibbs).  python 78.drilleau_mantle_scs.py
"""

import os, sys, json, glob
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mars_config as cfg

# ══════════════════════════════════════════════════════════════════════════
# settings
# ══════════════════════════════════════════════════════════════════════════
FORWARD_FILE = '26fall_main_mcmc_a63.py'   # same forward model as a64
CHAIN_DIR    = cfg.MCMC_DIR
PREFIX       = 'chain_a64_YM20'
BURNIN_FRAC  = 0.4
DROP_LAST    = 1
MIN_STEPS    = 100

R_GRID   = np.arange(1680.0, 1901.0, 10.0)   # km, R_CMB = R_ScS (no BML)
P_TP     = 0.01                              # GPa, isentrope foot used for Tp
Z_VS_AVG = (600.0, 1500.0)                   # km depth, mantle Vs average window
FS       = 14
SAVE     = True
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))   # beno3 mars_config has no FIG_DIR

# a64 medians are read from the chains; fill these to skip the chain read
OURS_OVERRIDE = None    # e.g. dict(T_lit=1600., P_lit=3.7, **{'Mg#': 0.80}, T_core=2400., w_S=0.15)

# (label, comp, T_lit source, z_lit (km) or None for our P_lit, Mg# source, their reflector)
#   'ours' means take the a64 median
CASES = [
    ('ours',               'YM20', 'ours', None,  'ours', np.nan),
    ('ours_T + D_TA_Mg#',  'YM20', 'ours', None,  0.773,  np.nan),
    ('D_TA_T + ours_Mg#',  'YM20', 1756.0, 334.0, 'ours', np.nan),
    ('D_TA_BML_YM20',      'YM20', 1756.0, 334.0, 0.773,  1816.0),
    ('D_TA_BML',           'TA',   1756.0, 334.0, 0.773,  1816.0),
    ('D_EH45_BML',         'EH45', 1778.0, 363.0, 0.755,  1821.0),
    ('D_YM_F2_noBML',      'YM20', 1916.0, 644.0, 0.780,  1838.0),
    ('D_YM_F1_noBML',      'YM20', 1945.0, 852.0, 0.768,  1835.0),
]
CASE_COLOR = ['k', '#555555', '#999999', '#E08E2B', '#C0392B', '#8E44AD', '#2E86C1', '#5DADE2']
CASE_LS    = ['-', '--', ':', '-', '-', '-', '-.', '-.']

plt.rcParams['font.family'] = 'DejaVu Sans'

# ══════════════════════════════════════════════════════════════════════════
# a64 posterior medians
# ══════════════════════════════════════════════════════════════════════════
if OURS_OVERRIDE is None:
    samples = []
    for d in sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '_*'))):
        path = os.path.join(d, 'chain.jsonl')
        if not os.path.exists(path):
            continue
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
        recs = recs[:-DROP_LAST or None]
        if len(recs) < MIN_STEPS:
            continue
        samples += recs[int(len(recs) * BURNIN_FRAC):]
    print(f'{PREFIX}: {len(samples)} post burn-in samples')
    OURS = {}
    for k in ('T_lit', 'P_lit', 'Mg#', 'T_core', 'w_S', 'Mg#_bulk_bml'):
        v = np.array([s.get('params', {}).get(k, s.get(k, np.nan)) for s in samples], float)
        OURS[k] = float(np.nanmedian(v))
else:
    OURS = dict(OURS_OVERRIDE)
    OURS.setdefault('Mg#_bulk_bml', 0.6)
print('a64 medians: ' + '  '.join(f'{k}={v:.4g}' for k, v in OURS.items()))

# ══════════════════════════════════════════════════════════════════════════
# forward modules, one per bulk composition (composition is fixed at import)
# ══════════════════════════════════════════════════════════════════════════
sys.argv = [sys.argv[0]]                 # the forward file parses --comp from argv
MODS = {}
for comp in sorted(set(c[1] for c in CASES)):
    os.environ['MARS_COMP'] = comp
    spec = importlib.util.spec_from_file_location(f'fwd_{comp}', FORWARD_FILE)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    MODS[comp] = mod
    print(f'loaded {FORWARD_FILE} as {comp}')

M0 = MODS['YM20']
# ScS-S observations: (event, delta, depth, t_obs, sigma)
SCS = [(ev, o['delta'], o.get('depth', 10.0), o['ScS-S'][0], o['ScS-S'][1])
       for ev, o in M0.SAMUEL_DATA.items() if 'ScS-S' in o]
print(f'{len(SCS)} ScS-S observations')

# ══════════════════════════════════════════════════════════════════════════
# scan
# ══════════════════════════════════════════════════════════════════════════
res = {}
for ic, (lab, comp, Tsrc, zlit, mgsrc, R_their) in enumerate(CASES):
    M = MODS[comp]
    T_lit = OURS['T_lit'] if Tsrc == 'ours' else Tsrc
    P_lit = OURS['P_lit'] if zlit is None else float(M.pressure_mars(zlit))
    Mg    = OURS['Mg#'] if mgsrc == 'ours' else mgsrc
    print(f'\n=== {lab}  comp={comp}  T_lit={T_lit:.0f}  P_lit={P_lit:.2f}  Mg#={Mg:.3f}')

    L1   = np.full(len(R_GRID), np.nan)     # sum |res| / sigma over ScS-S
    bias = np.full(len(R_GRID), np.nan)     # mean (obs - pred) / sigma
    tt   = np.full(len(R_GRID), np.nan)     # full misfit_tt (all phases)
    keep = None
    for ir, R in enumerate(R_GRID):
        params = {'T_lit': T_lit, 'P_lit': P_lit, 'Mg#': Mg,
                  'T_core': OURS['T_core'], 'Mg#_bulk_bml': OURS['Mg#_bulk_bml'],
                  'BML_thickness': 0.0, 'w_S': OURS['w_S'], 'R_cmb': float(R)}
        try:
            prof = M.sc_build_profile(params, M._samuel_cache)
            if prof is None:
                continue
            taup = M.build_taup(prof, f'78_{ic}_{int(R)}', M._samuel_cache)
        except Exception as e:
            print(f'  R={R:.0f}: forward failed ({e})')
            continue

        r_sig = []
        for ev, dl, dp, t_obs, sig in SCS:
            arr = taup.get_travel_times(source_depth_in_km=dp, distance_in_degree=dl,
                                        phase_list=['S', 'ScS'])
            t = {}
            for a in arr:
                if a.name not in t:
                    t[a.name] = a.time
            if 'S' in t and 'ScS' in t:
                r_sig.append((t_obs - (t['ScS'] - t['S'])) / sig)
            else:
                r_sig.append(np.nan)
        r_sig = np.array(r_sig)
        L1[ir]   = np.nansum(np.abs(r_sig))
        bias[ir] = np.nanmean(r_sig)
        _, _, comp_out = M.compute_misfit(taup, M.SAMUEL_DATA, prof)
        tt[ir] = comp_out['misfit_tt']
        print(f'  R={R:.0f}  ScS L1={L1[ir]:.2f}  bias={bias[ir]:+.2f} sig  '
              f'misfit_tt={tt[ir]:.2f}  n_ScS={np.sum(np.isfinite(r_sig))}')
        if keep is None or abs(R - 1770.0) < 5.0:
            keep = prof                      # profile shown in panels (b) and (c)

    # Tp: temperature at P_TP on the same isentrope
    p  = M.composition_from_params({'Mg#': Mg, 'T_lit': T_lit, 'P_lit': P_lit})
    cv = M._comp_values(p, M.compute_oxygen(p))
    o  = M._ng_forward([P_TP], [keep['S_lit']], cv, M.NG_S_HEADERS, ['temperature'])
    Tp = float(np.asarray(o['temperature'].detach().cpu()).ravel()[0])

    zz, mm = keep['z_km'], keep['m_mantle']
    win = mm & (zz >= Z_VS_AVG[0]) & (zz <= Z_VS_AVG[1])
    Vs_avg = float(np.mean(keep['Vs'][win]))

    i_best = int(np.nanargmin(L1))
    i0 = np.where(np.isfinite(bias[:-1]) & np.isfinite(bias[1:])
                  & (np.sign(bias[:-1]) != np.sign(bias[1:])))[0]
    R_zero = (R_GRID[i0[0]] - bias[i0[0]] * (R_GRID[i0[0] + 1] - R_GRID[i0[0]])
              / (bias[i0[0] + 1] - bias[i0[0]])) if len(i0) else np.nan
    res[lab] = dict(L1=L1, bias=bias, tt=tt, R_best=R_GRID[i_best], R_zero=R_zero,
                    Tp=Tp, Vs_avg=Vs_avg, T_lit=T_lit, P_lit=P_lit, Mg=Mg, comp=comp,
                    R_their=R_their, z=zz[mm], Vs=keep['Vs'][mm], T=keep['T_K'][mm])

# ══════════════════════════════════════════════════════════════════════════
# summary (terminal)
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 104)
print(f'{"case":<20s} {"comp":<5s} {"T_lit":>6s} {"P_lit":>6s} {"Mg#":>6s} {"Tp":>6s} '
      f'{"Vs600-1500":>10s} {"R_best":>7s} {"R_bias0":>8s} {"R_their":>8s} {"tt@best":>8s}')
for lab, r in res.items():
    ib = int(np.argmin(np.abs(R_GRID - r['R_best'])))
    print(f'{lab:<20s} {r["comp"]:<5s} {r["T_lit"]:6.0f} {r["P_lit"]:6.2f} {r["Mg"]:6.3f} '
          f'{r["Tp"]:6.0f} {r["Vs_avg"]:10.3f} {r["R_best"]:7.0f} {r["R_zero"]:8.1f} '
          f'{r["R_their"]:8.0f} {r["tt"][ib]:8.2f}')
print('R_best  = argmin of the ScS-S L1 misfit on the grid')
print('R_bias0 = R where the mean ScS-S residual crosses zero (linear interpolation)')

if SAVE:
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, '78_drilleau_mantle_scs.npz'),
             R_GRID=R_GRID, labels=np.array(list(res)),
             L1=np.array([r['L1'] for r in res.values()]),
             bias=np.array([r['bias'] for r in res.values()]),
             tt=np.array([r['tt'] for r in res.values()]),
             R_best=np.array([r['R_best'] for r in res.values()]),
             R_zero=np.array([r['R_zero'] for r in res.values()]),
             Tp=np.array([r['Tp'] for r in res.values()]),
             Vs_avg=np.array([r['Vs_avg'] for r in res.values()]))

# ══════════════════════════════════════════════════════════════════════════
# figure: (a) ScS-S mean residual vs R, (b) mantle Vs, (c) mantle T
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
for ic, (lab, r) in enumerate(res.items()):
    c, ls = CASE_COLOR[ic], CASE_LS[ic]
    ax[0].plot(R_GRID, r['bias'], color=c, ls=ls, lw=2, label=lab)
    if np.isfinite(r['R_their']):
        ax[0].axvline(r['R_their'], color=c, ls=ls, lw=1, alpha=0.6)
    ax[1].plot(r['Vs'], r['z'], color=c, ls=ls, lw=2)
    ax[2].plot(r['T'], r['z'], color=c, ls=ls, lw=2)
ax[0].axhline(0.0, color='k', lw=0.8)
ax[0].set_xlabel('ScS reflector radius (km)', fontsize=FS)
ax[0].set_ylabel('mean ScS-S residual (obs - pred) / sigma', fontsize=FS)
ax[1].set_xlabel('Vs (km/s)', fontsize=FS)
ax[2].set_xlabel('Temperature (K)', fontsize=FS)
for a in ax[1:]:
    a.set_ylabel('Depth (km)', fontsize=FS)
    a.invert_yaxis()
for a in ax:
    a.tick_params(labelsize=14)
ax[0].legend(fontsize=10, loc='lower right')
plt.tight_layout()
if SAVE:
    fig.savefig(os.path.join(OUT_DIR, '78_drilleau_mantle_scs.png'), dpi=200)
    print(f'saved {OUT_DIR}/78_drilleau_mantle_scs.png')
