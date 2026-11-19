#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 132  bulk Mg/Si - Fe/Si discrimination diagram
#
# Standalone. Reads chain.jsonl the same way 63 v4 does, recomputes the bulk
# planet composition with NCFMAS stoichiometry, and plots it against published
# Mars bulk models.
#
# Why the old panel in 111 had to be redone:
#   it computed the bulk composition as if the whole silicate were olivine, so
#   Mg/Si was identically 2*Mg# and sat at 1.55-1.66, never overlapping any
#   meteorite group. That separation was an artefact of the analysis script,
#   not a result. The forward model uses NCFMAS, where (Mg+Fe)/Si is about 1.35
#   instead of 2, so Mg/Si lands near 1.08 and the diagram becomes meaningful.
#
# Mass balance check on the reference side:
#   Yoshizaki & McDonough 2020 BSM + their core (f = 0.18, Fe = 79.5 wt%)
#   gives Fe/Si = 1.358 by weight; their published value is 1.36.
#   Their Table 10 elemental bulk Mars gives Fe/Si = 0.685 molar; the oxide
#   route used here gives 0.683.

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

# ══════════════════════════════════════════════════════════════════════════
# config
# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = MCMC_DIR
PREFIX    = 'chain_a53B'
EXCLUDE   = ['chain_a52B_04']      # frozen chain

BURNIN_FRAC = 0.3
DROP_LAST   = 1                    # last record can be half written
MIN_STEPS   = 5
H_STATE_TOL = 1.0                  # km, matches 63 v4 and 70's H_MIN_LAYER

SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
FS   = 14
TICK = 14

MARS_RADIUS = 3389.5               # km
M_MARS      = 6.4171e23            # kg

# rho_core_mean is stored in g/cm3 (see panel E of 63 v4), convert to kg/m3
RHO_CORE_SCALE = 1000.0

# Mean BML density, kg/m3. Only used to split the silicate mass between the
# mantle and the BML, which carry different Mg#. A +-10% error moves about
# 0.5% of the planet mass between the two reservoirs.
RHO_BML = 4200.0

# Fe/Si constraint inside the likelihood (script 84). WEIGHT ratio.
FESI_OBS_WT = 1.71
FESI_SIG_WT = 0.17

# Ordinary chondrite Fe/Si range, weight ratio.
# Wasson & Kallemeyn 1988, as quoted by Yoshizaki & McDonough 2020.
OC_FESI_WT = (1.0, 1.6)

# Yoshizaki & McDonough 2020 Table 11, four core models, whole-planet Fe/Si
# by weight: H,O-bearing 1.36 / H,O-free 1.29 / H-free 1.35 / O-free 1.34
YM20_FESI_WT = (1.29, 1.36)

STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
STATE_ORDER = ['molten', 'layered', 'solid']
plt.rcParams['font.family'] = 'DejaVu Sans'
TAG = PREFIX + '_'

PANEL_S = True                     # second panel: Fe/Si vs bulk S

# molar masses, g/mol
M_SI, M_MG, M_FE = 28.0855, 24.305, 55.845
M_SIO2, M_TIO2, M_AL2O3, M_CR2O3 = 60.084, 79.866, 101.961, 151.990
M_FEO, M_MNO, M_MGO, M_CAO = 71.844, 70.937, 40.304, 56.077
M_NA2O, M_K2O, M_P2O5, M_NIO = 61.979, 94.196, 141.944, 74.692

OXIDE_MASS = {'SiO2': M_SIO2, 'TiO2': M_TIO2, 'Al2O3': M_AL2O3,
              'Cr2O3': M_CR2O3, 'FeO': M_FEO, 'MnO': M_MNO, 'MgO': M_MGO,
              'CaO': M_CAO, 'Na2O': M_NA2O, 'K2O': M_K2O, 'P2O5': M_P2O5,
              'NiO': M_NIO}

# ── Mantle base composition fed to nGibbs, oxide wt% ──────────────────────
# ****  CONFIRM AGAINST script 70  ****
# Placeholder = Wanke & Dreibus 1994. Mg# is a sampled parameter, so MgO and
# FeO are rescaled per sample at fixed molar (Mg+Fe); the other oxides are
# held fixed. Molar ratios are invariant under renormalisation to 100%, so
# only the mass per mole of Si changes with Mg#.
# The fesi_sigma cross-check at the end of this script will tell you whether
# this composition matches the forward model, so you may not need to look.
MANTLE_BASE = {'SiO2': 44.4, 'TiO2': 0.14, 'Al2O3': 3.02, 'Cr2O3': 0.76,
               'FeO': 17.9, 'MnO': 0.46, 'MgO': 30.2, 'CaO': 2.45,
               'Na2O': 0.50, 'K2O': 0.04, 'P2O5': 0.16}

# BML is Fo-Fa olivine (Mg,Fe)2SiO4, so (Mg+Fe)/Si = 2 exactly
R_MGFE_SI_BML = 2.0

# ── reference models ──────────────────────────────────────────────────────
# Oxide wt% is bulk SILICATE. f_core and fe_core turn that into a whole-planet
# Fe/Si; where the source publishes no core mass fraction, f_core is None and
# only Mg/Si is drawn. S_bulk is whole-planet S in wt%.
#
#   YM20   Yoshizaki & McDonough 2020, Tables 3, 10, 11
#   W&D94  Wanke & Dreibus 1994, oxides via Taylor 2013 Table 2
#   T13    Taylor 2013 Table 5; core 78.6 wt% Fe+Ni / 21.4 wt% S, section 6.6.
#          f_core is NOT stated by Taylor, 0.21 assumed here
#   MA79   Morgan & Anders 1979
#   LF97   Lodders & Fegley 1997
#   S99    Sanloup et al. 1999, EH45:H55
#   KC08   Khan & Connolly 2008; core 22-25 wt% S, r = 1680 km, no f_core
REF = {
    'YM20':  {'SiO2': 45.5, 'MgO': 31.0, 'FeO': 14.7,
              'f_core': 0.18,  'fe_core': 0.795, 'S_bulk': 1.2},
    'W&D94': {'SiO2': 44.4, 'MgO': 30.2, 'FeO': 17.9,
              'f_core': 0.217, 'fe_core': 0.778, 'S_bulk': 3.1},
    'T13':   {'SiO2': 43.7, 'MgO': 30.5, 'FeO': 18.1,
              'f_core': 0.21,  'fe_core': 0.786, 'S_bulk': None},
    'MA79':  {'SiO2': 41.6, 'MgO': 29.8, 'FeO': 15.9,
              'f_core': None,  'fe_core': None,  'S_bulk': None},
    'LF97':  {'SiO2': 45.4, 'MgO': 29.7, 'FeO': 17.2,
              'f_core': None,  'fe_core': None,  'S_bulk': None},
    'S99':   {'SiO2': 47.5, 'MgO': 27.3, 'FeO': 17.7,
              'f_core': None,  'fe_core': None,  'S_bulk': None},
    'KC08':  {'SiO2': 44.0, 'MgO': 33.0, 'FeO': 17.0,
              'f_core': None,  'fe_core': None,  'S_bulk': None},
}

# Bulk silicate Earth, Mg/Si axis ONLY. Its Fe/Si is about 0.15 molar because
# most Fe is already in the core, so it must not go on a whole-planet x axis.
PYROLITE = {'SiO2': 45.0, 'MgO': 37.8, 'FeO': 8.05}

# Chondrite groups, molar ratios. FILL FROM THE ORIGINAL TABLES BEFORE USE.
# Entries left as None are skipped silently.
CHONDRITE = {'CI': {'MgSi': None, 'FeSi': None},
             'CM': {'MgSi': None, 'FeSi': None},
             'CV': {'MgSi': None, 'FeSi': None},
             'H':  {'MgSi': None, 'FeSi': None},
             'L':  {'MgSi': None, 'FeSi': None},
             'LL': {'MgSi': None, 'FeSi': None},
             'EH': {'MgSi': None, 'FeSi': None},
             'EL': {'MgSi': None, 'FeSi': None}}

# ══════════════════════════════════════════════════════════════════════════
# state classification, same rule as 63 v4
#   solid   : h_liquid < tol  and h_solid  >= tol
#   molten  : h_solid  < tol  and h_liquid >= tol
#   layered : both >= tol
#   fail    : both < tol, or NaN
# ══════════════════════════════════════════════════════════════════════════
def classify_state(h_sol, h_liq, tol=H_STATE_TOL):
    h_sol = np.atleast_1d(np.asarray(h_sol, float))
    h_liq = np.atleast_1d(np.asarray(h_liq, float))
    st = np.full(h_sol.shape, 'fail', dtype='<U8')
    ok = np.isfinite(h_sol) & np.isfinite(h_liq)
    s_y = ok & (h_sol >= tol)
    l_y = ok & (h_liq >= tol)
    st[s_y & l_y]  = 'layered'
    st[~s_y & l_y] = 'molten'
    st[s_y & ~l_y] = 'solid'
    return st

# ══════════════════════════════════════════════════════════════════════════
# read chains
# chain.jsonl stores chain STATE, not proposals. Rejected steps repeat the
# previous record; keep every line including duplicates.
# ══════════════════════════════════════════════════════════════════════════
all_chains, chain_names = [], []
print(f"{'chain':<22s} {'n_raw':>7s}  action")
for dd in sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '*'))):
    name = os.path.basename(dd)
    if name in EXCLUDE:
        print(f'{name:<22s} {0:>7d}  EXCLUDED')
        continue
    path = os.path.join(dd, 'chain.jsonl')
    if not os.path.exists(path):
        print(f'{name:<22s} {0:>7d}  DROPPED (no chain.jsonl)')
        continue
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break                      # truncated last line, stop here
    n_raw = len(recs)
    recs = recs[:-DROP_LAST] if len(recs) > DROP_LAST else []
    if len(recs) < MIN_STEPS:
        print(f'{name:<22s} {n_raw:>7d}  DROPPED (too short)')
        continue
    all_chains.append(recs)
    chain_names.append(name)
    print(f'{name:<22s} {n_raw:>7d}  kept {len(recs)}')

if not all_chains:
    raise SystemExit('no usable chains')

BURN = [int(len(ch) * BURNIN_FRAC) for ch in all_chains]
samples = []
for c in range(len(all_chains)):
    samples.extend(all_chains[c][BURN[c]:])
print(f'\npost burn-in samples: {len(samples)}')
print('params keys:', sorted(samples[0]['params'].keys()))
print('record keys:', sorted(samples[0].keys()))
print('')

R_cmb  = np.array([float(s['params']['R_cmb'])        for s in samples])
w_S    = np.array([float(s['params']['w_S'])          for s in samples])
mg_man = np.array([float(s['params']['Mg#'])          for s in samples])
mg_bml = np.array([float(s['params']['Mg#_bulk_bml']) for s in samples])
D_bml  = np.array([float(s['params']['BML_thickness'])for s in samples])

h_sol    = np.array([np.nan if s.get('h_solid_km') is None
                     else float(s['h_solid_km'])    for s in samples])
h_liq    = np.array([np.nan if s.get('h_liquid_km') is None
                     else float(s['h_liquid_km'])   for s in samples])
rho_core = np.array([np.nan if s.get('rho_core_mean') is None
                     else float(s['rho_core_mean']) for s in samples])
fesi_sig_rec = np.array([np.nan if s.get('fesi_sigma') is None
                         else float(s['fesi_sigma']) for s in samples])

rho_core = rho_core * RHO_CORE_SCALE
state = classify_state(h_sol, h_liq)
print('state counts:', {s: int((state == s).sum())
                        for s in ['solid', 'layered', 'molten', 'fail']})
if not np.isfinite(fesi_sig_rec).any():
    print('NOTE: fesi_sigma absent in these chains, cross-check skipped')
print('')

# ══════════════════════════════════════════════════════════════════════════
# reservoir mass fractions
# ══════════════════════════════════════════════════════════════════════════
# core:  M = 4/3 * pi * R^3 * rho        (R in km -> m)
M_core = (4.0 / 3.0) * np.pi * (R_cmb * 1e3) ** 3 * rho_core
f_core = M_core / M_MARS

# BML: spherical shell from R_cmb to R_cmb + BML_thickness.
# BML_thickness is the sampled parameter; h_solid + h_liquid is its split.
R_bml_top = R_cmb + D_bml
M_bml = ((4.0 / 3.0) * np.pi * ((R_bml_top * 1e3) ** 3 - (R_cmb * 1e3) ** 3)
         * RHO_BML)
f_bml = M_bml / M_MARS

f_man = 1.0 - f_core - f_bml

# consistency: the sampled thickness should equal the two sublayers
d_split = np.nanmax(np.abs(D_bml - (np.nan_to_num(h_sol) + np.nan_to_num(h_liq))))
print(f'max |BML_thickness - (h_solid + h_liquid)| = {d_split:.3f} km')

# bulk planet sulfur, wt%. All light element in the core is taken to be S.
bulk_S = 100.0 * f_core * w_S

# ══════════════════════════════════════════════════════════════════════════
# stoichiometry of the mantle silicate
# ══════════════════════════════════════════════════════════════════════════
n_si_base = MANTLE_BASE['SiO2'] / M_SIO2
n_mg_base = MANTLE_BASE['MgO']  / M_MGO
n_fe_base = MANTLE_BASE['FeO']  / M_FEO

# (Mg+Fe)/Si of the mantle silicate. The old code forced this to 2, which is
# what pinned Mg/Si to 2*Mg#.
R_MGFE_SI_MAN = (n_mg_base + n_fe_base) / n_si_base

# mass of everything except MgO and FeO, per mole of Si, g/mol Si
m_other_per_si = M_SIO2
for ox in MANTLE_BASE:
    if ox in ['MgO', 'FeO', 'SiO2']:
        continue
    m_other_per_si += MANTLE_BASE[ox] / n_si_base

print('\nmantle base composition:')
print(f'  (Mg+Fe)/Si molar  = {R_MGFE_SI_MAN:.4f}   (pure olivine = 2.0000)')
print(f'  base Mg#          = {n_mg_base/(n_mg_base+n_fe_base):.4f}')
print(f'  Mg/Si at Mg#=0.80 = {0.80*R_MGFE_SI_MAN:.4f}   '
      f'(old olivine value {0.80*2.0:.4f})')
print('')

# ══════════════════════════════════════════════════════════════════════════
# bulk planet ratios, per posterior sample
# ══════════════════════════════════════════════════════════════════════════
# mass of mantle silicate per mole of its Si, g/mol Si
#   m = m_other + R * (Mg# * M_MgO + (1 - Mg#) * M_FeO)
m_man_per_si = m_other_per_si + R_MGFE_SI_MAN * (mg_man * M_MGO +
                                                 (1.0 - mg_man) * M_FEO)

# BML, (Mg,Fe)2SiO4, mass per mole Si
m_bml_per_si = M_SIO2 + R_MGFE_SI_BML * (mg_bml * M_MGO +
                                         (1.0 - mg_bml) * M_FEO)

# moles per unit planet mass
n_si     = f_man / m_man_per_si + f_bml / m_bml_per_si
n_mg     = (mg_man * R_MGFE_SI_MAN * f_man / m_man_per_si +
            mg_bml * R_MGFE_SI_BML * f_bml / m_bml_per_si)
n_fe_sil = ((1.0 - mg_man) * R_MGFE_SI_MAN * f_man / m_man_per_si +
            (1.0 - mg_bml) * R_MGFE_SI_BML * f_bml / m_bml_per_si)
n_fe_core = f_core * (1.0 - w_S) / M_FE

MgSi = n_mg / n_si
FeSi = (n_fe_sil + n_fe_core) / n_si

# ══════════════════════════════════════════════════════════════════════════
# reference compositions
# ══════════════════════════════════════════════════════════════════════════
ref_mgsi, ref_fesi = {}, {}
for k in REF:
    d = REF[k]
    ns = d['SiO2'] / M_SIO2
    nm = d['MgO']  / M_MGO
    nf = d['FeO']  / M_FEO
    # Mg/Si of a bulk silicate model IS the whole-planet Mg/Si, because
    # neither Mg nor Si enters the core
    ref_mgsi[k] = nm / ns
    if d['f_core'] is None:
        ref_fesi[k] = None
        continue
    # per 100 g of silicate the planet carries 100*f/(1-f) g of core
    m_core_per_100 = 100.0 * d['f_core'] / (1.0 - d['f_core'])
    ref_fesi[k] = (nf + m_core_per_100 * d['fe_core'] / M_FE) / ns

pyro_mgsi = (PYROLITE['MgO'] / M_MGO) / (PYROLITE['SiO2'] / M_SIO2)

# weight ratio -> molar ratio:  (Fe/Si)_molar = (Fe/Si)_wt * M_Si / M_Fe
fesi_obs = FESI_OBS_WT * M_SI / M_FE
fesi_sig = FESI_SIG_WT * M_SI / M_FE
oc_lo, oc_hi = [v * M_SI / M_FE for v in OC_FESI_WT]
ym_lo, ym_hi = [v * M_SI / M_FE for v in YM20_FESI_WT]

# ══════════════════════════════════════════════════════════════════════════
# printed tables
# ══════════════════════════════════════════════════════════════════════════
print('=' * 72)
print('reference models, molar ratios (weight ratios in brackets)')
print('=' * 72)
print(f"  {'model':<7s}{'Mg/Si':>17s}{'Fe/Si':>18s}")
for k in REF:
    mg_wt = ref_mgsi[k] * M_MG / M_SI
    if ref_fesi[k] is None:
        print(f'  {k:<7s}{ref_mgsi[k]:8.3f} ({mg_wt:6.3f}){"no f_core":>18s}')
    else:
        fe_wt = ref_fesi[k] * M_FE / M_SI
        print(f'  {k:<7s}{ref_mgsi[k]:8.3f} ({mg_wt:6.3f})'
              f'{ref_fesi[k]:9.3f} ({fe_wt:6.3f})')
print(f'  {"pyrolite":<7s}{pyro_mgsi:8.3f} ({pyro_mgsi*M_MG/M_SI:6.3f})'
      f'{"silicate only":>18s}')
print('')
print(f'  likelihood Fe/Si   = {fesi_obs:.3f} +- {fesi_sig:.3f} molar'
      f'  ({FESI_OBS_WT:.2f} +- {FESI_SIG_WT:.2f} wt)')
print(f'  YM20 core variants = {ym_lo:.3f} to {ym_hi:.3f} molar'
      f'  ({YM20_FESI_WT[0]:.2f} to {YM20_FESI_WT[1]:.2f} wt)')
print(f'  ord. chondrites    = {oc_lo:.3f} to {oc_hi:.3f} molar'
      f'  ({OC_FESI_WT[0]:.2f} to {OC_FESI_WT[1]:.2f} wt)')
print('')

print('=' * 72)
print('reservoir mass fractions, 5/50/95')
print('=' * 72)
for nm_, arr in [('f_core', f_core), ('f_bml', f_bml), ('f_mantle', f_man)]:
    q = np.nanpercentile(arr, [5, 50, 95])
    print(f'  {nm_:<9s}{q[0]:8.4f}{q[1]:8.4f}{q[2]:8.4f}')
print('  (f_core outside 0.15-0.25 means RHO_CORE_SCALE is wrong)')
print('')

print('=' * 72)
print('posterior, 5/50/95 percentiles')
print('=' * 72)
print(f"  {'state':<8s}{'n':>7s}{'Mg/Si':>24s}{'Fe/Si':>24s}"
      f"{'bulk S (wt%)':>24s}")
for s in STATE_ORDER + ['all']:
    m = np.isfinite(MgSi) if s == 'all' else ((state == s) & np.isfinite(MgSi))
    if m.sum() < 20:
        continue
    a = np.percentile(MgSi[m],   [5, 50, 95])
    b = np.percentile(FeSi[m],   [5, 50, 95])
    c = np.percentile(bulk_S[m], [5, 50, 95])
    print(f'  {s:<8s}{m.sum():7d}'
          f'{a[0]:8.3f}{a[1]:8.3f}{a[2]:8.3f}'
          f'{b[0]:8.3f}{b[1]:8.3f}{b[2]:8.3f}'
          f'{c[0]:8.2f}{c[1]:8.2f}{c[2]:8.2f}')
print('')

print('=' * 72)
print('fraction of posterior inside external Fe/Si windows')
print('=' * 72)
fin = np.isfinite(FeSi)
print(f'  likelihood 1sigma  '
      f'{np.mean((FeSi[fin] > fesi_obs-fesi_sig) & (FeSi[fin] < fesi_obs+fesi_sig)):.3f}')
print(f'  YM20 core variants '
      f'{np.mean((FeSi[fin] > ym_lo) & (FeSi[fin] < ym_hi)):.3f}')
print(f'  ordinary chondrite '
      f'{np.mean((FeSi[fin] > oc_lo) & (FeSi[fin] < oc_hi)):.3f}')
print('')

# ── cross-check against the forward model ────────────────────────────────
# fesi_sigma is the likelihood residual in sigma units. If it is signed,
#   fesi_pred = FESI_OBS + fesi_sigma * FESI_SIG
# If the forward model stores |residual| instead, only the magnitude below
# is meaningful; in that case compare the two |residual| columns, not the
# reconstructed prediction.
if np.isfinite(fesi_sig_rec).any():
    fesi_pred_wt = FESI_OBS_WT + fesi_sig_rec * FESI_SIG_WT
    here_wt = FeSi * M_FE / M_SI
    m = np.isfinite(fesi_pred_wt) & np.isfinite(here_wt)
    print('=' * 72)
    print('cross-check: forward model Fe/Si vs this script (WEIGHT ratio)')
    print('=' * 72)
    print(f'  fesi_sigma range        {np.nanmin(fesi_sig_rec):+.3f} to '
          f'{np.nanmax(fesi_sig_rec):+.3f}')
    print('  (all positive => stored as |residual|, reconstruction below is'
          ' one-sided)')
    q1 = np.percentile(fesi_pred_wt[m], [5, 50, 95])
    q2 = np.percentile(here_wt[m],      [5, 50, 95])
    print(f'  forward model  5/50/95  {q1[0]:.3f} {q1[1]:.3f} {q1[2]:.3f}')
    print(f'  this script    5/50/95  {q2[0]:.3f} {q2[1]:.3f} {q2[2]:.3f}')
    print(f'  median difference       {np.median(here_wt[m]-fesi_pred_wt[m]):+.4f}')
    print('  a difference under ~0.02 means MANTLE_BASE matches script 70')
    print('')

# ══════════════════════════════════════════════════════════════════════════
# figure
# ══════════════════════════════════════════════════════════════════════════
ncol = 2 if PANEL_S else 1
fig, ax = plt.subplots(1, ncol, figsize=(13 if PANEL_S else 8, 6))
if not PANEL_S:
    ax = [ax]

# external Fe/Si bands, behind everything
for a in np.atleast_1d(ax):
    a.axvspan(oc_lo, oc_hi, color='0.92', zorder=0)
    a.axvspan(ym_lo, ym_hi, color='#35838D', alpha=0.20, zorder=0)
    a.axvspan(fesi_obs - fesi_sig, fesi_obs + fesi_sig, color='0.78', zorder=0)
    a.axvline(fesi_obs, color='0.4', lw=1.5, zorder=0)

for s in STATE_ORDER:
    m = (state == s)
    if m.sum():
        ax[0].plot(FeSi[m], MgSi[m], '.', ms=2, alpha=0.25,
                   color=STATE_COLOR[s], label=s)

# freeze the limits before placing the left-edge tick marks
x0, x1 = ax[0].get_xlim()
ax[0].set_xlim(x0, x1)

ax[0].axhline(pyro_mgsi, color='#97795D', ls='--', lw=1.5, zorder=1)
ax[0].text(x0, pyro_mgsi, ' pyrolite (Mg/Si only)', fontsize=FS - 4,
           va='bottom', ha='left', color='#97795D')

for k in CHONDRITE:
    if CHONDRITE[k]['MgSi'] is None or CHONDRITE[k]['FeSi'] is None:
        continue
    ax[0].plot(CHONDRITE[k]['FeSi'], CHONDRITE[k]['MgSi'], 's', ms=7,
               color='#7B3FA0')
    ax[0].text(CHONDRITE[k]['FeSi'] + 0.012, CHONDRITE[k]['MgSi'], k,
               fontsize=FS - 3, va='center', color='#7B3FA0')

for k in REF:
    if ref_fesi[k] is None:
        # Mg/Si is valid, Fe/Si is not, so mark the y value on the left edge
        ax[0].plot(x0, ref_mgsi[k], '<', ms=7, color='0.35', clip_on=False)
        ax[0].text(x0, ref_mgsi[k], k + ' ', fontsize=FS - 4, va='center',
                   ha='right', color='0.35')
        continue
    ax[0].plot(ref_fesi[k], ref_mgsi[k], 'D', ms=8, color='k',
               markerfacecolor='none', markeredgewidth=1.8)
    ax[0].text(ref_fesi[k] + 0.012, ref_mgsi[k], k, fontsize=FS - 3,
               va='center', color='k')

ax[0].set_xlabel('bulk Fe/Si (molar)', fontsize=FS)
ax[0].set_ylabel('bulk Mg/Si (molar)', fontsize=FS)
ax[0].tick_params(labelsize=TICK)
leg = ax[0].legend(fontsize=FS - 3, loc='lower right', markerscale=6)
for lh in leg.legend_handles:
    lh.set_alpha(1.0)

if PANEL_S:
    for s in STATE_ORDER:
        m = (state == s)
        if m.sum():
            ax[1].plot(FeSi[m], bulk_S[m], '.', ms=2, alpha=0.25,
                       color=STATE_COLOR[s], label=s)
    for k in REF:
        if REF[k]['S_bulk'] is None or ref_fesi[k] is None:
            continue
        ax[1].plot(ref_fesi[k], REF[k]['S_bulk'], 'D', ms=8, color='k',
                   markerfacecolor='none', markeredgewidth=1.8)
        ax[1].text(ref_fesi[k] + 0.012, REF[k]['S_bulk'], k, fontsize=FS - 3,
                   va='center', color='k')
    ax[1].set_xlabel('bulk Fe/Si (molar)', fontsize=FS)
    ax[1].set_ylabel('bulk S (wt%)', fontsize=FS)
    ax[1].tick_params(labelsize=TICK)
    leg = ax[1].legend(fontsize=FS - 3, loc='lower right', markerscale=6)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

fig.tight_layout()
if SAVE_FIG:
    fig.savefig(os.path.join(FIG_DIR, TAG + 'I_bulk_MgSi_FeSi.png'), dpi=150)
plt.show()