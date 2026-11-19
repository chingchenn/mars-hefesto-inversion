#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 131: phase assemblage (mass fraction) vs pressure along the self-consistent
# areotherm of ONE parameter set, temperature overlaid on a second x-axis.
# Forward model comes straight from 86 (sc_build_profile), so P(z), T(z) and
# the mantle/BML compositions are exactly what the MCMC sees.
#
# Runs on beno3 (needs nGibbs). Module name starts with a digit, so import via
# spec_from_file_location.
#
# nGibbs phase-fraction call, validated against ForwardMB(outputs=['phase_tables']):
#   phase_tables is a 2-tuple; phase_tables[1] is phaseMassNorm, shape (n, n_phase).
#   Phase name order = EM.isothermal_emulator.ml_indexer.all_phases.
# These are MASS fractions, not volume fractions (nGibbs has no per-phase molar
# volume in this API path) -- axis label says so explicitly.

import os, sys, glob, json, importlib.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ══════════════════════════════════════════════════════════════════════════
# config
# ══════════════════════════════════════════════════════════════════════════
MAIN_PY = os.path.expanduser('~/mars-hefesto-inversion/main_MCMC_TA.py')
ROUTE   = 'B'

# representative model: pick from the actual a52B posterior chain, same logic
# as 130 (post burn-in, WANT_STATE, the accepted step whose h_liquid is
# closest to that state's median -- a representative sample, not best-misfit).
# Set CHAIN_STEP_OVERRIDE = ('chain_a53B_03', 4821) to force a specific one.
PREFIX             = 'chain_a53B'
EXCLUDE            = []
WANT_STATE         = 'layered'        # 'solid' / 'layered' / 'molten'
BURNIN_FRAC        = 0.3
H_STATE_TOL        = 1.0
CHAIN_STEP_OVERRIDE = None

# extra overrides applied on top of whatever params were picked (usually leave empty)
PARAMS_OVERRIDE = {}

N_P      = 120          # pressure nodes for the phase scan (mantle + solid BML)
P_MAX    = None         # GPa; None -> down to P_cmb
INCLUDE_SOLID_BML = True
LABEL_THRESH = 6.0       # %, minimum stable width to get an inline label

SAVE = True
FS   = 14
TICK = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# final colour set: olivine family green (light->dark with pressure), pyroxene
# family blue-gray (same hue as the mantle band in 129/130), garnet warm,
# minor phases neutral. liquid BML tint of 130's C_LIQ.
PHASE_STYLE = [
    ('Olivine',                    '#8FBF8F'),
    ('Wadsleyite',                 '#5E9E7A'),
    ('Ringwoodite',                '#3E7A66'),
    ('Opx',                        '#A9C3D9'),
    ('Cpx',                        '#6C8EAD'),
    ('HP-Cpx',                     '#4E6E8E'),
    ('Garnet',                     '#C98A5B'),
    ('Ferropericlase',             '#8E6B8E'),
    ('sp',                         '#D9B8A0'),
    ('plg',                        '#CFCFCF'),
    ('ky',                         '#E3D3BF'),
    ('qtz',                        '#B5B5B5'),
    ('st',                         '#7F7F7F'),
    ('Ca-pv',                      '#E6D690'),
    ('ak',                         '#9E9E7A'),
    ('bdg',                        '#B89B6E'),
    ('coe',                        '#C9C9B0'),
    ('cf',                         '#D8C08A'),
    ('nal',                        '#BFAF9A'),
    ('neph',                       '#C7B8D6'),
    ('feg',                        '#5A5A5A'),
]
# nGibbs internal phase name -> display name (from EM.isothermal_emulator.ml_indexer.all_phases)
PHASE_ALIAS = {
    'olivine': 'Olivine', 'wadsleyite': 'Wadsleyite', 'ringwoodite': 'Ringwoodite',
    'orthopyroxene': 'Opx', 'clinopyroxene': 'Cpx',
    'hp-clinopyroxene': 'HP-Cpx',
    'garnet': 'Garnet', 'ferropericlase': 'Ferropericlase',
    'plagioclase': 'plg', 'spinel': 'sp', 'kyanite': 'ky', 'quartz': 'qtz',
    'alpha-iron': 'feg', 'gamma-iron': 'feg', 'epsilon-iron': 'feg',
    'ca-perovskite': 'Ca-pv', 'akimotoite': 'ak', 'bridgmanite': 'bdg',
    'coesite': 'coe', 'stishovite': 'st', 'ca-ferrite': 'cf', 'nal-phase': 'nal',
    'nepheline': 'neph',
}
C_LIQ  = '#EFC9C9'   # liquid BML, tint of 130's #CD5C5C
C_LIT  = '#8C8C8C'   # fixed 0-100 km column, matches 130's C_LIT
C_MAN  = '#4E6E8E'
C_SOL  = '#7B9E87'
C_VP   = '#274472'   # velocity panel: Vp
C_VS   = '#B23A48'   # velocity panel: Vs

# ══════════════════════════════════════════════════════════════════════════
# import 86
# ══════════════════════════════════════════════════════════════════════════
spec = importlib.util.spec_from_file_location('m86', MAIN_PY)
M = importlib.util.module_from_spec(spec)
sys.modules['m86'] = M
spec.loader.exec_module(M)
M.CF.configure(route=ROUTE, verbose=False)

# MCMC_DIR should come from 86's `from mars_config import *`; fall back to the
# known beno3 path if that constant is missing or points somewhere else.
MCMC_DIR = getattr(M, 'MCMC_DIR', None) or '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'
print('MCMC_DIR =', MCMC_DIR)

def classify_state(h_sol, h_liq, tol=H_STATE_TOL):
    s_y, l_y = h_sol >= tol, h_liq >= tol
    if s_y and l_y: return 'layered'
    if l_y:         return 'molten'
    if s_y:         return 'solid'
    return 'fail'

if CHAIN_STEP_OVERRIDE is not None:
    chain_name, want_step = CHAIN_STEP_OVERRIDE
    cf = os.path.join(MCMC_DIR, chain_name, 'chain.jsonl')
    chosen_params = None
    for line in open(cf):
        if not line.strip():
            continue
        r = json.loads(line)
        if r['step'] == want_step:
            chosen_params = r['params']; break
    if chosen_params is None:
        raise SystemExit('step %d not found in %s' % (want_step, cf))
    print('using %s step %d (explicit override)' % (chain_name, want_step))
else:
    cand = []   # (chain_file, step, params, h_liq)
    for dd in sorted(glob.glob(os.path.join(MCMC_DIR, PREFIX + '*'))):
        name = os.path.basename(dd)
        if name in EXCLUDE:
            continue
        cf = os.path.join(dd, 'chain.jsonl')
        if not os.path.exists(cf):
            continue
        recs = [json.loads(l) for l in open(cf) if l.strip()]
        if not recs:
            continue
        max_step = recs[-1]['step']
        for r in recs:
            if not r.get('accepted', False):
                continue
            if r['step'] < max_step * BURNIN_FRAC:
                continue
            hs, hl = r.get('h_solid_km'), r.get('h_liquid_km')
            if hs is None or hl is None:
                continue
            if classify_state(hs, hl) == WANT_STATE:
                cand.append((cf, r['step'], r['params'], hl))
    if not cand:
        raise SystemExit('no accepted %s step found under %s' % (WANT_STATE, MCMC_DIR))
    hl_all = np.array([c[3] for c in cand])
    sel = cand[int(np.argmin(np.abs(hl_all - np.median(hl_all))))]
    cf, step, chosen_params, _ = sel
    print('%d accepted %s steps; picked step %d from %s (median h_liq %.0f km)'
          % (len(cand), WANT_STATE, step, cf, np.median(hl_all)))

params = {**chosen_params, **PARAMS_OVERRIDE}
print('params:', {k: round(v, 3) for k, v in params.items()})

prof = M.sc_build_profile(params, M._samuel_cache, verbose=False)
if prof is None:
    raise SystemExit('forward model rejected these params')
bml = prof['bml']
print('P_bml_top %.2f  P_cmb %.2f  h_sol %.0f  h_liq %.0f  T_int %.0f'
      % (prof['P_bml_top'], prof['P_cmb'], bml['h_solid_km'],
         bml['h_liquid_km'], bml['T_interface']))

# ── crust/mantle boundary check ─────────────────────────────────────────────
# distinguishes a real Moho density/velocity contrast (expected, harmless)
# from an artificial jump (unit error, wrong mask, wrong reference table).
_mfx, _mman = prof['m_crust'], prof['m_mantle']
_rc, _rm = prof['rho'][_mfx][-1], prof['rho'][_mman][0]
_pc, _pm = prof['Vp'][_mfx][-1],  prof['Vp'][_mman][0]
_sc, _sm = prof['Vs'][_mfx][-1],  prof['Vs'][_mman][0]
print('boundary at z=100km (P=%.2f GPa):' % prof['P_GPa'][_mman][0])
print('  crust  rho=%.4f  Vp=%.4f  Vs=%.4f' % (_rc, _pc, _sc))
print('  mantle rho=%.4f  Vp=%.4f  Vs=%.4f' % (_rm, _pm, _sm))
print('  Δrho=%+.1f%%  ΔVp=%+.1f%%  ΔVs=%+.1f%%'
      % (100*(_rm-_rc)/_rc, 100*(_pm-_pc)/_pc, 100*(_sm-_sc)/_sc))

# ══════════════════════════════════════════════════════════════════════════
# nGibbs phase-fraction call
# ══════════════════════════════════════════════════════════════════════════
def ng_phase_fractions(P, T, comp_values):
    """P, T: 1-D arrays (GPa, K). comp_values: 8-element list as in 86._comp_values.
    Returns (names, F), F.shape == (len(P), n_phase), MASS fractions."""
    import torch
    x = np.zeros((len(P), 2 + len(comp_values)), dtype=np.float32)
    x[:, 0] = P; x[:, 1] = T; x[:, 2:] = comp_values
    with torch.no_grad():
        out = M.EM.ForwardMB(x, headers=M.NG_T_HEADERS, outputs=['phase_tables'])
    F = np.asarray(out['phase_tables'][1].detach().cpu(), dtype=np.float64)
    names = list(M.EM.isothermal_emulator.ml_indexer.all_phases)
    return names, F

# ══════════════════════════════════════════════════════════════════════════
# scan along the areotherm
# ══════════════════════════════════════════════════════════════════════════
z, P, T, lay = prof['z_km'], prof['P_GPa'], prof['T_K'], prof['layer']
m_fx  = prof['m_crust']
m_man = prof['m_mantle']
m_bml = prof['m_bml']
m_sol_bml = m_bml & (prof['Vs'] > 1e-6)
m_liq_bml = m_bml & (prof['Vs'] <= 1e-6)

P_top = float(P[m_man][0])          # base of the fixed 0-100 km column
P_bot = float(P[m_bml][-1]) if P_MAX is None else P_MAX
P_scan = np.linspace(P_top, P_bot, N_P)
T_scan = np.interp(P_scan, P[m_man | m_bml], T[m_man | m_bml])

# which composition applies at each P: mantle Mg#, or solid-BML Mg_solid
P_bml_top = prof['P_bml_top']
P_int     = bml['P_interface']
in_man  = P_scan <= P_bml_top
in_sbml = (P_scan > P_bml_top) & (P_scan <= P_int) & INCLUDE_SOLID_BML
in_liq  = P_scan > P_int

p_m = M.composition_from_params(params)
cv_m = M._comp_values(p_m, M.compute_oxygen(p_m))
Mg_s = bml['Mg_solid'] if np.isfinite(bml['Mg_solid']) else params['Mg#_bulk_bml']
p_b = M.composition_from_params({'Mg#': Mg_s, 'T_lit': 0.0, 'P_lit': 0.0})
cv_b = M._comp_values(p_b, M.compute_oxygen(p_b))

names = None
F = np.zeros((N_P, 0))
for mask, cv in [(in_man, cv_m), (in_sbml, cv_b)]:
    if not mask.any():
        continue
    n_i, F_i = ng_phase_fractions(P_scan[mask], T_scan[mask], cv)
    if names is None:
        names = list(n_i); F = np.zeros((N_P, len(names)))
    for j, nm in enumerate(n_i):
        if nm not in names:
            names.append(nm); F = np.hstack([F, np.zeros((N_P, 1))])
        F[mask, names.index(nm)] = F_i[:, j]
F = F / np.clip(F.sum(axis=1, keepdims=True), 1e-12, None)
F[in_liq, :] = 0.0

disp = [PHASE_ALIAS.get(n.lower(), n) for n in names]

# merge phases that share a display name (e.g. alpha/gamma/epsilon-iron -> feg)
uniq, cols = [], []
for j, dn in enumerate(disp):
    if dn in uniq:
        cols[uniq.index(dn)] = cols[uniq.index(dn)] + F[:, j]
    else:
        uniq.append(dn); cols.append(F[:, j].copy())
disp, F = uniq, np.array(cols).T
print('phases:', disp)

order = [d for d, _ in PHASE_STYLE if d in disp] + [d for d in disp if d not in dict(PHASE_STYLE)]
col   = dict(PHASE_STYLE)

def label_position(f, frac=0.2, thresh=LABEL_THRESH):
    """Index near the TOP of the first stable (>thresh) run of this phase,
    so labels sit high in each field instead of drifting to wherever the
    field happens to be widest (which crowds the bottom for phases like
    Garnet that grow with depth). `frac` places it deeper into the run for
    consecutive phases (staggered) so two narrow adjacent fields that start
    at the same pressure don't get labelled on the same row."""
    ok = f > thresh
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    start = idx[0]
    end = start
    for k in idx[1:]:
        if k == end + 1:
            end = k
        else:
            break
    return start + max(1, int(frac * (end - start)))

def draw_assemblage(ax, show_crust=True):
    """Fills the phase-fraction stack on ax, from P=0 (surface) to P_bot."""
    if show_crust:
        ax.axhspan(0.0, P_top, color='0.85', lw=0)
        ax.text(50, 0.5 * P_top, 'fixed crust\n(0\u2013100 km, Samuel 2023)',
                ha='center', va='center', fontsize=FS - 3, color='0.35')
    left = np.zeros(N_P)
    _FRACS = [0.15, 0.35, 0.55]
    for k_ord, d in enumerate(order):
        j = disp.index(d)
        f = 100.0 * F[:, j]
        if f.max() < 0.5:
            continue
        ax.fill_betweenx(P_scan, left, left + f, color=col.get(d, '0.5'),
                         ec='white', lw=0.6)
        pos = label_position(f, frac=_FRACS[k_ord % len(_FRACS)])
        if pos is not None:
            x_mid = left[pos] + f[pos] / 2.0
            ax.text(x_mid, P_scan[pos], d, fontsize=FS - 2, ha='center', va='center',
                    color='white', fontweight='bold',
                    path_effects=[pe.Stroke(linewidth=2, foreground='0.2'), pe.Normal()])
        left = left + f
    if in_liq.any():
        ax.axhspan(P_int, P_bot, color=C_LIQ, alpha=0.9, lw=0)
        ax.text(50, 0.5 * (P_int + P_bot), 'liquid BML', fontsize=FS - 2,
                ha='center', va='center', color='#8B3A3A')
    ax.axhline(P_top, color='k', lw=1.2)
    if in_sbml.any():
        ax.axhline(P_bml_top, color='k', lw=0.8, ls='--')
    ax.axhline(P_bot, color='k', lw=1.2)

# ══════════════════════════════════════════════════════════════════════════
# figure 1: assemblage + temperature (top axis)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 9))
draw_assemblage(ax, show_crust=True)

ax.set_ylim(P_bot, 0.0)
ax.set_xlim(0, 100)
ax.set_xlabel('Mass fraction (%)', fontsize=FS)
ax.set_ylabel('Pressure (GPa)', fontsize=FS)
ax.tick_params(labelsize=TICK)

axT = ax.twiny()
axT.plot(T_scan, P_scan, color='k', lw=2.2, ls='--',
         path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()],
         label='Temperature (K)', zorder=10)
axT.set_xlim(0, 3000)
axT.set_ylim(P_bot, 0.0)
axT.set_xlabel('Temperature (K)', fontsize=FS)
axT.tick_params(axis='x', labelsize=TICK)

h2, l2 = axT.get_legend_handles_labels()
ax.legend(h2, l2, loc='lower right', fontsize=FS - 3, framealpha=0.9)

fig.tight_layout()
if SAVE:
    fig.savefig('131_phase_assemblage_T_TA.png', dpi=200, bbox_inches='tight')
print('saved 131_phase_assemblage_T_TA.png')

# ══════════════════════════════════════════════════════════════════════════
# figure 2: assemblage | rho | velocity (Vp & Vs together, different colours)
# ══════════════════════════════════════════════════════════════════════════
win  = (P >= 0.0) & (P <= P_bot)
m_c  = m_fx & win
m_m2 = m_man & win
m_b2 = m_bml & win

fig2, axs = plt.subplots(1, 3, figsize=(15, 9), sharey=True,
                         gridspec_kw={'width_ratios': [1.5, 1, 1], 'wspace': 0.10})

# (a) phase assemblage, same data + crust band as figure 1
draw_assemblage(axs[0], show_crust=True)
axs[0].set_xlim(0, 100)
axs[0].set_xlabel('Mass fraction (%)', fontsize=FS)
axs[0].set_ylabel('Pressure (GPa)', fontsize=FS)
#axs[0].set_title('phase assemblage', fontsize=FS)
axT0 = axs[0].twiny()
axT0.plot(prof['T_K'][win], prof['P_GPa'][win], color='k', lw=2.2, ls='--',
          path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()],
          zorder=10, label='Temperature (K)')
axT0.set_xlim(0, 3000)
axT0.set_ylim(P_bot, 0.0)
axT0.set_xlabel('Temperature (K)', fontsize=FS)
axT0.tick_params(axis='x', labelsize=TICK)
#axT0.legend(loc='lower right', fontsize=FS - 4, framealpha=0.9)
# (b) density: crust + mantle + BML, one continuous colour by layer
a1 = axs[1]
a1.plot(prof['rho'][m_c],  P[m_c],  color=C_LIT, lw=2.6, label='crust (Samuel 2023)')
a1.plot(prof['rho'][m_m2], P[m_m2], color=C_MAN, lw=2.6, label='mantle (nGibbs)')
if m_sol_bml.any():
    a1.plot(prof['rho'][m_sol_bml & win], P[m_sol_bml & win], color=C_SOL, lw=2.6,
            label='solid BML (nGibbs)')
if m_liq_bml.any():
    a1.plot(prof['rho'][m_liq_bml & win], P[m_liq_bml & win], color='#CD5C5C', lw=2.6,
            label='liquid BML (Thomas EoS)')
a1.set_xlabel(r'$\rho$ (g cm$^{-3}$)', fontsize=FS)
a1.set_title(r'$\rho$', fontsize=FS)

# (c) Vp and Vs together, different colours, same layer boundaries
a2 = axs[2]
for m_, ls_ in [(m_c, '-'), (m_m2, '-'), (m_sol_bml & win, '-'), (m_liq_bml & win, '-')]:
    if m_.any():
        a2.plot(prof['Vp'][m_], P[m_], color=C_VP, lw=2.6, ls=ls_)
        a2.plot(prof['Vs'][m_], P[m_], color=C_VS, lw=2.6, ls='--')
a2.plot([], [], color=C_VP, lw=2.6, ls='-',  label='$V_P$')
a2.plot([], [], color=C_VS, lw=2.6, ls='--', label='$V_S$')
a2.set_xlabel('velocity (km s$^{-1}$)', fontsize=FS)
a2.set_title('$V_P$, $V_S$', fontsize=FS)

for a in axs[1:]:
    a.axhspan(0.0, P_top, color='0.85', lw=0, zorder=0)
    a.axhline(P_top, color='k', lw=1.2)
    if in_sbml.any():
        a.axhline(P_bml_top, color='k', lw=0.8, ls='--')
    if in_liq.any():
        a.axhspan(P_int, P_bot, color=C_LIQ, alpha=0.5, lw=0)
    a.axhline(P_bot, color='k', lw=1.2)
    a.grid(True, ls='--', alpha=0.3)

a2.legend(loc='upper left', fontsize=FS -2, framealpha=0.9)

for a in axs:
    a.set_ylim(P_bot, 0.0)
    a.tick_params(labelsize=TICK)

fig2.tight_layout()
if SAVE:
    fig2.savefig('131_assemblage_rho_velocity_TA.png', dpi=200, bbox_inches='tight')
print('saved 131_assemblage_rho_velocity_TA.png')
plt.show()