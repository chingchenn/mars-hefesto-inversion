#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 132  bulk Mg/Si - Fe/Si discrimination diagram   (weight ratios)
#
# Standalone. Reads chain.jsonl the same way 63 v4 does.
#
# Why the old panel in 111 had to be redone:
#   it computed the bulk composition as if the whole silicate were olivine, so
#   Mg/Si was identically 2*Mg#. The forward model uses NCFMAS, where
#   (Mg+Fe)/Si is about 1.35 instead of 2.
#
# Axes are WEIGHT ratios, matching how the literature reports them.
# fesi_pred is already a weight ratio, so the x axis needs no conversion.
# Molar values are still printed in brackets in the tables.
#   molar -> weight :  Mg/Si * M_Mg/M_Si ,  Fe/Si * M_Fe/M_Si
#
# Reservoirs: core, BML (Fo-Fa olivine), mantle (NCFMAS), fixed crust.
#
# Mass balance check on the reference side:
#   Yoshizaki & McDonough 2020 BSM + their core (f = 0.18, Fe = 79.5 wt%)
#   gives Fe/Si = 1.358 by weight; their published value is 1.36.

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR

# ══════════════════════════════════════════════════════════════════════════
# config
# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = MCMC_DIR
PREFIX    = 'chain_a52B'
EXCLUDE   = ['chain_a52B_04']      # frozen chain

BURNIN_FRAC = 0.3
DROP_LAST   = 1
MIN_STEPS   = 5
H_STATE_TOL = 1.0                  # km, matches 63 v4 and 70's H_MIN_LAYER

SAVE = False
FS   = 14
TICK = 14

PANEL_S       = True               # second panel: Fe/Si vs bulk S
SHOW_PYROLITE = False              # its Mg/Si = 1.08 wt stretches the y axis
INCLUDE_CRUST = True

MARS_RADIUS = 3389.5               # km
M_MARS      = 6.4171e23            # kg
RHO_SCALE   = 1000.0               # recorded densities are g/cm3 -> kg/m3

# ── fixed crust (Samuel 2023, 0-100 km, not sampled) ──────────────────────
# ****  CONFIRM THICKNESS, DENSITY AND COMPOSITION AGAINST script 70  ****
# Generic Martian basaltic crust; the printed f_crust is the sanity handle
# (100 km at 2900 kg/m3 gives 0.063).
CRUST_THICK_KM = 100.0
CRUST_RHO      = 2900.0            # kg/m3
CRUST_COMP = {'SiO2': 49.3, 'TiO2': 0.98, 'Al2O3': 10.5, 'Cr2O3': 0.30,
              'FeO': 18.2, 'MnO': 0.36, 'MgO': 9.1, 'CaO': 6.5,
              'Na2O': 2.9, 'K2O': 0.45, 'P2O5': 0.90}

# ── external Fe/Si values, all WEIGHT ratios ─────────────────────────────
FESI_OBS_WT = 1.71                 # likelihood centre, Wanke & Dreibus
FESI_SIG_WT = 0.17
OC_FESI_WT   = (1.0, 1.6)          # ordinary chondrites, Wasson & Kallemeyn 1988
YM20_FESI_WT = (1.29, 1.36)        # YM20 Table 11, four core models

STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
STATE_ORDER = ['molten', 'solid', 'layered']   # layered drawn last, it is 2.5%
plt.rcParams['font.family'] = 'DejaVu Sans'
TAG = PREFIX + '_'

# molar masses, g/mol
M_SI, M_MG, M_FE = 28.0855, 24.305, 55.845
M_SIO2, M_TIO2, M_AL2O3, M_CR2O3 = 60.084, 79.866, 101.961, 151.990
M_FEO, M_MNO, M_MGO, M_CAO = 71.844, 70.937, 40.304, 56.077
M_NA2O, M_K2O, M_P2O5, M_NIO = 61.979, 94.196, 141.944, 74.692

# molar ratio -> weight ratio
W_MGSI = M_MG / M_SI               # 0.8654
W_FESI = M_FE / M_SI               # 1.9884

OXIDE_MASS = {'SiO2': M_SIO2, 'TiO2': M_TIO2, 'Al2O3': M_AL2O3,
              'Cr2O3': M_CR2O3, 'FeO': M_FEO, 'MnO': M_MNO, 'MgO': M_MGO,
              'CaO': M_CAO, 'Na2O': M_NA2O, 'K2O': M_K2O, 'P2O5': M_P2O5,
              'NiO': M_NIO}

# ── mantle base composition ───────────────────────────────────────────────
# ****  CONFIRM AGAINST composition_from_params() in script 70  ****
# The only number needed is (Mg+Fe)/Si, molar:
#   (Mg+Fe)/Si = sum(x_i * nMgFe_i) / sum(x_i * nSi_i)
#   olivine (Mg,Fe)2SiO4  1 Si, 2 (Mg+Fe)   opx (Mg,Fe)SiO3   1 Si, 1
#   diopside CaMgSi2O6    2 Si, 1           pyrope Mg3Al2Si3O12  3 Si, 3
# Set R_MGFE_SI_MAN_OVERRIDE if you know it; otherwise it comes from the oxide
# dict below, currently Wanke & Dreibus 1994 as a placeholder (gives 1.3512).
# The fesi_pred vs this-script diagnostic suggests the true value is nearer
# 1.22, which would move bulk Mg/Si down by roughly 0.10 in weight ratio.
MANTLE_BASE = {'SiO2': 44.4, 'TiO2': 0.14, 'Al2O3': 3.02, 'Cr2O3': 0.76,
               'FeO': 17.9, 'MnO': 0.46, 'MgO': 30.2, 'CaO': 2.45,
               'Na2O': 0.50, 'K2O': 0.04, 'P2O5': 0.16}
R_MGFE_SI_MAN_OVERRIDE = None

R_MGFE_SI_BML = 2.0                # BML is Fo-Fa olivine, exactly 2

# ── reference models ──────────────────────────────────────────────────────
# Oxide wt% is bulk SILICATE. f_core and fe_core turn that into a whole-planet
# Fe/Si; no f_core means only Mg/Si is drawn. S_bulk is whole-planet S wt%.
#   YM20  Yoshizaki & McDonough 2020, Tables 3, 10, 11
#   W&D94 Wanke & Dreibus 1994, oxides via Taylor 2013 Table 2
#   T13   Taylor 2013 Table 5; core 78.6 wt% Fe+Ni, section 6.6.
#         f_core NOT stated by Taylor, 0.21 assumed here
#   MA79  Morgan & Anders 1979      LF97  Lodders & Fegley 1997
#   S99   Sanloup et al. 1999       KC08  Khan & Connolly 2008
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

# Bulk silicate Earth, Mg/Si axis ONLY (its Fe/Si is not a whole-planet value)
PYROLITE = {'SiO2': 45.0, 'MgO': 37.8, 'FeO': 8.05}

# Chondrite groups, WEIGHT ratios. FILL FROM THE ORIGINAL TABLES BEFORE USE.
# If your source gives molar, multiply Mg/Si by 0.8654 and Fe/Si by 1.9884.
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
            break                      # truncated last line
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
print(f'\npost burn-in samples: {len(samples)}\n')

def grab(key):
    return np.array([np.nan if s.get(key) is None else float(s[key])
                     for s in samples])

R_cmb  = np.array([float(s['params']['R_cmb'])         for s in samples])
w_S    = np.array([float(s['params']['w_S'])           for s in samples])
mg_man = np.array([float(s['params']['Mg#'])           for s in samples])
mg_bml = np.array([float(s['params']['Mg#_bulk_bml'])  for s in samples])
D_bml  = np.array([float(s['params']['BML_thickness']) for s in samples])

h_sol     = grab('h_solid_km')
h_liq     = grab('h_liquid_km')
rho_core  = grab('rho_core_mean')   * RHO_SCALE
rho_l_bml = grab('rho_liquid_bml')  * RHO_SCALE
rho_s_bml = grab('rho_solid_bml')   * RHO_SCALE
fesi_pred = grab('fesi_pred')       # WEIGHT ratio, straight from the model

state = classify_state(h_sol, h_liq)
print('state counts:', {s: int((state == s).sum())
                        for s in ['solid', 'layered', 'molten', 'fail']})
print('')

# ══════════════════════════════════════════════════════════════════════════
# reservoir mass fractions
# ══════════════════════════════════════════════════════════════════════════
# core:  M = 4/3 * pi * R^3 * rho        (R in km -> m)
M_core = (4.0 / 3.0) * np.pi * (R_cmb * 1e3) ** 3 * rho_core
f_core = M_core / M_MARS

# BML: liquid sublayer at the BOTTOM, because R_ScS = R_cmb + h_liquid
R_liq_top = R_cmb + np.nan_to_num(h_liq)
R_bml_top = R_cmb + D_bml
V_liq = (4.0/3.0)*np.pi*((R_liq_top*1e3)**3 - (R_cmb*1e3)**3)
V_sol = (4.0/3.0)*np.pi*((R_bml_top*1e3)**3 - (R_liq_top*1e3)**3)
M_bml = V_liq * np.nan_to_num(rho_l_bml) + V_sol * np.nan_to_num(rho_s_bml)
f_bml = M_bml / M_MARS

if INCLUDE_CRUST:
    V_crust = ((4.0/3.0)*np.pi*((MARS_RADIUS*1e3)**3 -
                                ((MARS_RADIUS - CRUST_THICK_KM)*1e3)**3))
    f_crust = V_crust * CRUST_RHO / M_MARS
else:
    f_crust = 0.0

f_man = 1.0 - f_core - f_bml - f_crust

# bulk planet sulfur, wt%. All light element in the core is taken to be S.
bulk_S = 100.0 * f_core * w_S

d_split = np.nanmax(np.abs(D_bml - (np.nan_to_num(h_sol) + np.nan_to_num(h_liq))))
print(f'max |BML_thickness - (h_solid + h_liquid)| = {d_split:.3f} km')
print(f'f_crust (fixed) = {f_crust:.4f}\n')

# ══════════════════════════════════════════════════════════════════════════
# stoichiometry
# ══════════════════════════════════════════════════════════════════════════
n_si_base = MANTLE_BASE['SiO2'] / M_SIO2
n_mg_base = MANTLE_BASE['MgO']  / M_MGO
n_fe_base = MANTLE_BASE['FeO']  / M_FEO

if R_MGFE_SI_MAN_OVERRIDE is None:
    R_MGFE_SI_MAN = (n_mg_base + n_fe_base) / n_si_base
else:
    R_MGFE_SI_MAN = R_MGFE_SI_MAN_OVERRIDE

# mass of everything except MgO and FeO, per mole of Si, g/mol Si
m_other_per_si = M_SIO2
for ox in MANTLE_BASE:
    if ox in ['MgO', 'FeO', 'SiO2']:
        continue
    m_other_per_si += MANTLE_BASE[ox] / n_si_base

# crust: moles per gram straight from its oxide wt%
c_tot = sum(CRUST_COMP.values())
c_si = CRUST_COMP['SiO2'] / M_SIO2 / c_tot
c_mg = CRUST_COMP['MgO']  / M_MGO  / c_tot
c_fe = CRUST_COMP['FeO']  / M_FEO  / c_tot

print('mantle base composition:')
print(f'  (Mg+Fe)/Si molar      = {R_MGFE_SI_MAN:.4f}   (pure olivine = 2.0000)')
print(f'  base Mg#              = {n_mg_base/(n_mg_base+n_fe_base):.4f}')
print(f'  Mg/Si at Mg#=0.80     = {0.80*R_MGFE_SI_MAN*W_MGSI:.4f} wt'
      f'   ({0.80*R_MGFE_SI_MAN:.4f} molar)')
print(f'  crust Mg/Si           = {c_mg/c_si*W_MGSI:.4f} wt'
      f'   Fe/Si = {c_fe/c_si*W_FESI:.4f} wt')
print('')

# ══════════════════════════════════════════════════════════════════════════
# bulk planet ratios
# ══════════════════════════════════════════════════════════════════════════
# mantle mass per mole of its Si:
#   m = m_other + R * (Mg# * M_MgO + (1 - Mg#) * M_FeO)
m_man_per_si = m_other_per_si + R_MGFE_SI_MAN * (mg_man * M_MGO +
                                                 (1.0 - mg_man) * M_FEO)
# BML, (Mg,Fe)2SiO4
m_bml_per_si = M_SIO2 + R_MGFE_SI_BML * (mg_bml * M_MGO +
                                         (1.0 - mg_bml) * M_FEO)

# moles per unit planet mass
n_si = f_man / m_man_per_si + f_bml / m_bml_per_si + f_crust * c_si
n_mg = (mg_man * R_MGFE_SI_MAN * f_man / m_man_per_si +
        mg_bml * R_MGFE_SI_BML * f_bml / m_bml_per_si +
        f_crust * c_mg)
n_fe_sil = ((1.0 - mg_man) * R_MGFE_SI_MAN * f_man / m_man_per_si +
            (1.0 - mg_bml) * R_MGFE_SI_BML * f_bml / m_bml_per_si +
            f_crust * c_fe)
n_fe_core = f_core * (1.0 - w_S) / M_FE

# plotted quantities, WEIGHT ratios
MgSi = (n_mg / n_si) * W_MGSI
FeSi = fesi_pred                                   # already a weight ratio
FeSi_own = ((n_fe_sil + n_fe_core) / n_si) * W_FESI  # diagnostic only

# ══════════════════════════════════════════════════════════════════════════
# reference compositions, converted to WEIGHT
# ══════════════════════════════════════════════════════════════════════════
ref_mgsi, ref_fesi = {}, {}
for k in REF:
    d = REF[k]
    ns, nm, nf = d['SiO2']/M_SIO2, d['MgO']/M_MGO, d['FeO']/M_FEO
    # Mg/Si of a bulk silicate model IS the whole-planet Mg/Si, because
    # neither Mg nor Si enters the core
    ref_mgsi[k] = (nm / ns) * W_MGSI
    if d['f_core'] is None:
        ref_fesi[k] = None
        continue
    m_core_per_100 = 100.0 * d['f_core'] / (1.0 - d['f_core'])
    ref_fesi[k] = ((nf + m_core_per_100 * d['fe_core'] / M_FE) / ns) * W_FESI

pyro_mgsi = ((PYROLITE['MgO']/M_MGO) / (PYROLITE['SiO2']/M_SIO2)) * W_MGSI

# all the Fe/Si bands are already weight ratios, no conversion needed
fesi_obs, fesi_sig = FESI_OBS_WT, FESI_SIG_WT
oc_lo, oc_hi = OC_FESI_WT
ym_lo, ym_hi = YM20_FESI_WT

# ══════════════════════════════════════════════════════════════════════════
# printed tables
# ══════════════════════════════════════════════════════════════════════════
print('=' * 72)
print('reference models, WEIGHT ratios (molar in brackets)')
print('=' * 72)
print(f"  {'model':<7s}{'Mg/Si':>17s}{'Fe/Si':>18s}")
for k in REF:
    mg_mo = ref_mgsi[k] / W_MGSI
    if ref_fesi[k] is None:
        print(f'  {k:<7s}{ref_mgsi[k]:8.3f} ({mg_mo:6.3f}){"no f_core":>18s}')
    else:
        print(f'  {k:<7s}{ref_mgsi[k]:8.3f} ({mg_mo:6.3f})'
              f'{ref_fesi[k]:9.3f} ({ref_fesi[k]/W_FESI:6.3f})')
print(f'  {"pyrolite":<7s}{pyro_mgsi:8.3f} ({pyro_mgsi/W_MGSI:6.3f})'
      f'{"silicate only":>18s}')
print('')
print(f'  likelihood Fe/Si   = {fesi_obs:.3f} +- {fesi_sig:.3f} wt')
print(f'  YM20 core variants = {ym_lo:.3f} to {ym_hi:.3f} wt')
print(f'  ord. chondrites    = {oc_lo:.3f} to {oc_hi:.3f} wt')
print('')

print('=' * 72)
print('reservoir mass fractions, 5/50/95')
print('=' * 72)
for nm_, arr in [('f_core', f_core), ('f_bml', f_bml), ('f_mantle', f_man)]:
    q = np.nanpercentile(arr, [5, 50, 95])
    print(f'  {nm_:<9s}{q[0]:8.4f}{q[1]:8.4f}{q[2]:8.4f}')
print(f'  {"f_crust":<9s}{f_crust:8.4f}  (fixed)')
print('')

print('=' * 72)
print('posterior, 5/50/95 percentiles   (Mg/Si and Fe/Si are WEIGHT ratios)')
print('=' * 72)
print(f"  {'state':<8s}{'n':>7s}{'Mg/Si':>24s}{'Fe/Si':>24s}"
      f"{'bulk S (wt%)':>24s}")
for s in ['molten', 'layered', 'solid', 'all']:
    m = np.isfinite(FeSi) if s == 'all' else ((state == s) & np.isfinite(FeSi))
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
print('Fe/Si from the forward model, WEIGHT ratio')
print('=' * 72)
q = np.nanpercentile(fesi_pred, [5, 50, 95])
print(f'  fesi_pred 5/50/95   {q[0]:.3f} {q[1]:.3f} {q[2]:.3f}')
print(f'  prior centre        {FESI_OBS_WT:.3f} +- {FESI_SIG_WT:.3f}')
print(f'  median offset       {(q[1]-FESI_OBS_WT)/FESI_SIG_WT:+.2f} sigma')
qo = np.nanpercentile(FeSi_own, [5, 50, 95])
print(f'  this script 5/50/95 {qo[0]:.3f} {qo[1]:.3f} {qo[2]:.3f}'
      f'   (diagnostic; gap => MANTLE_BASE or crust is off)')
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

# ══════════════════════════════════════════════════════════════════════════
# figure
# ══════════════════════════════════════════════════════════════════════════
ncol = 2 if PANEL_S else 1
fig, ax = plt.subplots(1, ncol, figsize=(13 if PANEL_S else 8, 6))
ax = np.atleast_1d(ax)

for a in ax:
    a.axvspan(oc_lo, oc_hi, color='0.92', zorder=0)
    a.axvspan(ym_lo, ym_hi, color='#35838D', alpha=0.20, zorder=0)
    a.axvspan(fesi_obs - fesi_sig, fesi_obs + fesi_sig, color='0.78', zorder=0)
    a.axvline(fesi_obs, color='0.4', lw=1.5, zorder=0)

for s in STATE_ORDER:
    m = (state == s)
    if m.sum():
        ax[0].plot(FeSi[m], MgSi[m], '.', ms=2, alpha=0.25,
                   color=STATE_COLOR[s], label=s)

# freeze the limits before drawing the in-plot Mg/Si markers
x0, x1 = ax[0].get_xlim()
ax[0].set_xlim(x0, x1)
seg = x0 + 0.035 * (x1 - x0)

if SHOW_PYROLITE:
    ax[0].axhline(pyro_mgsi, color='#97795D', ls='--', lw=1.5, zorder=1)
    ax[0].text(seg + 0.004*(x1-x0), pyro_mgsi, 'pyrolite (Mg/Si only)',
               fontsize=FS - 4, va='center', color='#97795D')

for k in CHONDRITE:
    if CHONDRITE[k]['MgSi'] is None or CHONDRITE[k]['FeSi'] is None:
        continue
    ax[0].plot(CHONDRITE[k]['FeSi'], CHONDRITE[k]['MgSi'], 's', ms=7,
               color='#7B3FA0', zorder=3)
    ax[0].text(CHONDRITE[k]['FeSi'] + 0.006*(x1-x0), CHONDRITE[k]['MgSi'], k,
               fontsize=FS - 3, va='center', color='#7B3FA0', zorder=3)

for k in REF:
    if ref_fesi[k] is None:
        # Mg/Si is valid, Fe/Si is not: short segment inside the axes
        ax[0].plot([x0, seg], [ref_mgsi[k]] * 2, '-', lw=2, color='0.35',
                   zorder=1)
        ax[0].text(seg + 0.004*(x1-x0), ref_mgsi[k], k, fontsize=FS - 4,
                   va='center', ha='left', color='0.35')
        continue
    ax[0].plot(ref_fesi[k], ref_mgsi[k], 'D', ms=8, color='k',
               markerfacecolor='none', markeredgewidth=1.8, zorder=4)
    ax[0].text(ref_fesi[k] + 0.006*(x1-x0), ref_mgsi[k], k, fontsize=FS - 3,
               va='center', color='k', zorder=4)

ax[0].set_xlabel('bulk Fe/Si (weight)', fontsize=FS)
ax[0].set_ylabel('bulk Mg/Si (weight)', fontsize=FS)
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
                   markerfacecolor='none', markeredgewidth=1.8, zorder=4)
        ax[1].text(ref_fesi[k], REF[k]['S_bulk'], '  ' + k, fontsize=FS - 3,
                   va='center', ha='left', color='k', zorder=4)
    ax[1].set_xlabel('bulk Fe/Si (weight)', fontsize=FS)
    ax[1].set_ylabel('bulk S (wt%)', fontsize=FS)
    ax[1].tick_params(labelsize=TICK)
    leg = ax[1].legend(fontsize=FS - 3, loc='lower right', markerscale=6)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

fig.tight_layout()
if SAVE:
    fig.savefig(TAG + 'I_bulk_MgSi_FeSi.png', dpi=150)
plt.show()