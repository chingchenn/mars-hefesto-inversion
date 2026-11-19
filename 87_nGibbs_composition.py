#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 132: phase assemblage (mass fraction) vs pressure for several well-fitting
# posterior models, split into cold / mid / hot CMB by T_core.
# Forward model and nGibbs phase call are identical to 131.
# Runs on beno3 (needs nGibbs).
#
# Selection: accepted steps after burn-in -> lowest GOOD_FRAC in total misfit
#            -> T_core tertiles of that good set (or fixed T_CUTS)
#            -> N_PER_BIN random picks per bin, different chains first.

import os, sys, glob, json, importlib.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch
from mars_config import MCMC_DIR, FIG_DIR

# ══════════════════════════════════════════════════════════════════════════
# config
# ══════════════════════════════════════════════════════════════════════════
TAG         = 'YM'
MAIN_PY     = os.path.expanduser('~/mars-hefesto-inversion/main_MCMC_YM.py')
ROUTE       = 'B'
PREFIX      = 'chain_a52B'
EXCLUDE     = ['chain_a52B_04', 'chain_a54B_00', 'chain_a54B_09']           # stuck chains, e.g. ['chain_a54B_05', 'chain_a54B_11']

BURNIN_FRAC = 0.3
GOOD_FRAC   = 0.10           # keep the lowest 10% total misfit
N_PER_BIN   = 2              # picks per T_core bin
SEED        = 1
T_CUTS      = None           # None -> tertiles of the good set; or e.g. (2000.0, 2800.0)
H_STATE_TOL = 1.0            # km, same as 131
PARAMS_OVERRIDE = {}

N_P               = 120      # pressure nodes for the phase scan (mantle + solid BML)
INCLUDE_SOLID_BML = True
LABEL_THRESH      = 6.0      # %, minimum stable width to get an inline label
OL_WAD_THRESH     = 0.01     # mass fraction, wadsleyite onset

SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
FS   = 14
TICK = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

PHASE_STYLE = [
    ('Olivine',        '#8FBF8F'),
    ('Wadsleyite',     '#5E9E7A'),
    ('Ringwoodite',    '#3E7A66'),
    ('Opx',            '#A9C3D9'),
    ('Cpx',            '#6C8EAD'),
    ('HP-Cpx',         '#4E6E8E'),
    ('Garnet',         '#C98A5B'),
    ('Ferropericlase', '#8E6B8E'),
    ('sp',             '#D9B8A0'),
    ('plg',            '#CFCFCF'),
    ('ky',             '#E3D3BF'),
    ('qtz',            '#B5B5B5'),
    ('st',             '#7F7F7F'),
    ('Ca-pv',          '#E6D690'),
    ('ak',             '#9E9E7A'),
    ('bdg',            '#B89B6E'),
    ('coe',            '#C9C9B0'),
    ('cf',             '#D8C08A'),
    ('nal',            '#BFAF9A'),
    ('neph',           '#C7B8D6'),
    ('feg',            '#5A5A5A'),
]
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
C_LIQ = '#EFC9C9'

# ══════════════════════════════════════════════════════════════════════════
# import main program (composition comes from here)
# ══════════════════════════════════════════════════════════════════════════
spec = importlib.util.spec_from_file_location('mmain', MAIN_PY)
M = importlib.util.module_from_spec(spec)
sys.modules['mmain'] = M
spec.loader.exec_module(M)
M.CF.configure(route=ROUTE, verbose=False)
print('MAIN_PY  =', MAIN_PY)
print('MCMC_DIR =', MCMC_DIR)
print('_MG_FE_TOTAL =', M._MG_FE_TOTAL, '  _COMP_FIXED =', M._COMP_FIXED)

def classify_state(h_sol, h_liq, tol=H_STATE_TOL):
    s_y, l_y = h_sol >= tol, h_liq >= tol
    if s_y and l_y: return 'layered'
    if l_y:         return 'molten'
    if s_y:         return 'solid'
    return 'fail'

# ══════════════════════════════════════════════════════════════════════════
# collect accepted post-burn-in states
# ══════════════════════════════════════════════════════════════════════════
rows = []   # (chain, step, misfit, T_core, params, h_sol, h_liq)
for dd in sorted(glob.glob(os.path.join(MCMC_DIR, PREFIX + '_*'))):
    name = os.path.basename(dd)
    if name in EXCLUDE:
        continue
    cf = os.path.join(dd, 'chain.jsonl')
    if not os.path.exists(cf):
        continue
    recs = []
    for l in open(cf):
        if not l.strip():
            continue
        try:
            recs.append(json.loads(l))
        except json.JSONDecodeError:      # half-written last line after a kill
            break
    if not recs:
        continue
    max_step = recs[-1]['step']
    n_acc = 0
    for r in recs:
        if not r.get('accepted', False):
            continue
        if r['step'] < max_step * BURNIN_FRAC:
            continue
        n_acc += 1
        rows.append((name, r['step'], r['misfit'], r['params']['T_core'], r['params'],
                     r.get('h_solid_km', np.nan), r.get('h_liquid_km', np.nan)))
    print(f'{name}: {len(recs):5d} steps, {n_acc:4d} accepted after burn-in')

mis = np.array([r[2] for r in rows])
mg  = np.array([r[4]['Mg#'] for r in rows])
tl  = np.array([r[4]['T_lit'] for r in rows])
pl  = np.array([r[4]['P_lit'] for r in rows])
print(f'\n{len(rows)} accepted states in total')
print(f'posterior median  Mg# = {np.median(mg):.3f}  T_lit = {np.median(tl):.0f} K  '
      f'P_lit = {np.median(pl):.2f} GPa  misfit = {np.median(mis):.2f}')

# ── good fits ────────────────────────────────────────────────────────────
cut  = np.quantile(mis, GOOD_FRAC)
good = [r for r in rows if r[2] <= cut]
print(f'misfit cut ({GOOD_FRAC*100:.0f}%) = {cut:.3f}  ->  {len(good)} good states '
      f'from {len(set(r[0] for r in good))} chains')

# ── cold / mid / hot by T_core ───────────────────────────────────────────
tc_good = np.array([r[3] for r in good])
if T_CUTS is None:
    t1, t2 = np.quantile(tc_good, [1/3, 2/3])
else:
    t1, t2 = T_CUTS
print(f'T_core in good fits: min {tc_good.min():.0f}  cuts {t1:.0f} / {t2:.0f}  '
      f'max {tc_good.max():.0f} K')
bins = {'cold': [r for r in good if r[3] <= t1],
        'mid':  [r for r in good if t1 < r[3] <= t2],
        'hot':  [r for r in good if r[3] > t2]}

rng   = np.random.default_rng(SEED)
picks = []      # (bin label, row)
for lab in ['cold', 'mid', 'hot']:
    b = bins[lab]
    chains = sorted(set(r[0] for r in b))
    rng.shuffle(chains)
    got = []
    for c in chains[:N_PER_BIN]:
        cand = [r for r in b if r[0] == c]
        got.append(cand[rng.integers(len(cand))])
    rest = [r for r in b if r not in got]
    while len(got) < N_PER_BIN and rest:
        got.append(rest.pop(rng.integers(len(rest))))
    print(f'{lab:>4}: {len(b)} good states from {len(chains)} chains, picked {len(got)}')
    picks += [(lab, r) for r in got]
if not picks:
    raise SystemExit('no picks')

# ══════════════════════════════════════════════════════════════════════════
# nGibbs phase-fraction call (same as 131)
# ══════════════════════════════════════════════════════════════════════════
def ng_phase_fractions(P, T, comp_values):
    x = np.zeros((len(P), 2 + len(comp_values)), dtype=np.float32)
    x[:, 0] = P; x[:, 1] = T; x[:, 2:] = comp_values
    with torch.no_grad():
        out = M.EM.ForwardMB(x, headers=M.NG_T_HEADERS, outputs=['phase_tables'])
    F = np.asarray(out['phase_tables'][1].detach().cpu(), dtype=np.float64)
    names = list(M.EM.isothermal_emulator.ml_indexer.all_phases)
    return names, F

def label_position(f, frac=0.2, thresh=LABEL_THRESH):
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

def draw_assemblage(ax, fs_lab):
    # uses the module-level P_scan, F, disp, order, col, P_top, P_bot, ... of the current pick
    ax.axhspan(0.0, P_top, color='0.85', lw=0)
    ax.text(50, 0.5 * P_top, 'fixed crust\n(0\u2013100 km, Samuel 2023)',
            ha='center', va='center', fontsize=fs_lab - 1, color='0.35')
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
            ax.text(left[pos] + f[pos] / 2.0, P_scan[pos], d, fontsize=fs_lab,
                    ha='center', va='center', color='white', fontweight='bold',
                    path_effects=[pe.Stroke(linewidth=2, foreground='0.2'), pe.Normal()])
        left = left + f
    if in_liq.any():
        ax.axhspan(P_int, P_bot, color=C_LIQ, alpha=0.9, lw=0)
        ax.text(50, 0.5 * (P_int + P_bot), 'liquid BML', fontsize=fs_lab,
                ha='center', va='center', color='#8B3A3A')
    ax.axhline(P_top, color='k', lw=1.2)
    if in_sbml.any():
        ax.axhline(P_bml_top, color='k', lw=0.8, ls='--')
    ax.axhline(P_bot, color='k', lw=1.2)

# ══════════════════════════════════════════════════════════════════════════
# loop over picks
# ══════════════════════════════════════════════════════════════════════════
fig_all, ax_all = plt.subplots(1, len(picks), figsize=(4.6 * len(picks), 9), squeeze=False)
ax_all = ax_all[0]
summary = []

for i_pick, (lab, r) in enumerate(picks):
    chain_name, step = r[0], r[1]
    params = {**r[4], **PARAMS_OVERRIDE}
    print(f'\n[{i_pick+1}/{len(picks)}] {lab} CMB  {chain_name} step {step}  '
          f'misfit {r[2]:.3f}  T_core {params["T_core"]:.0f} K')

    prof = M.sc_build_profile(params, M._samuel_cache, verbose=False)
    if prof is None:
        print('  forward model rejected these params, skipped')
        ax_all[i_pick].set_visible(False)
        continue
    bml = prof['bml']
    state = classify_state(bml['h_solid_km'], bml['h_liquid_km'])

    # ── scan grid: mantle + BML (same as 131) ────────────────────────────
    P, T = prof['P_GPa'], prof['T_K']
    m_man, m_bml = prof['m_mantle'], prof['m_bml']
    P_top = float(P[m_man][0])
    P_bot = float(P[m_bml][-1])
    P_scan = np.linspace(P_top, P_bot, N_P)
    T_scan = np.interp(P_scan, P[m_man | m_bml], T[m_man | m_bml])

    P_bml_top = prof['P_bml_top']
    P_int     = bml['P_interface']
    in_man  = P_scan <= P_bml_top
    in_sbml = (P_scan > P_bml_top) & (P_scan <= P_int) & INCLUDE_SOLID_BML
    in_liq  = P_scan > P_int

    # mantle composition: mantle Mg#; solid BML: Mg_solid (same as 131)
    p_m  = M.composition_from_params(params)
    cv_m = M._comp_values(p_m, M.compute_oxygen(p_m))
    Mg_s = bml['Mg_solid'] if np.isfinite(bml['Mg_solid']) else params['Mg#_bulk_bml']
    p_b  = M.composition_from_params({'Mg#': Mg_s, 'T_lit': 0.0, 'P_lit': 0.0})
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
    uniq, cols = [], []
    for j, dn in enumerate(disp):
        if dn in uniq:
            cols[uniq.index(dn)] = cols[uniq.index(dn)] + F[:, j]
        else:
            uniq.append(dn); cols.append(F[:, j].copy())
    disp, F = uniq, np.array(cols).T
    order = [d for d, _ in PHASE_STYLE if d in disp] + [d for d in disp if d not in dict(PHASE_STYLE)]
    col   = dict(PHASE_STYLE)

    # ── numbers for the terminal (mantle part only) ─────────────────────
    # olivine at 5 GPa; Ol->Wad onset (wad > OL_WAD_THRESH) and midpoint (wad >= ol)
    z_m, P_m = prof['z_km'][m_man], P[m_man]
    f_ol5 = np.interp(5.0, P_scan, 100 * F[:, disp.index('Olivine')]) if 'Olivine' in disp else np.nan
    P_on = P_mid = z_on = z_mid = np.nan
    if 'Wadsleyite' in disp and 'Olivine' in disp:
        f_wa = np.where(in_man, F[:, disp.index('Wadsleyite')], 0.0)
        f_ol = F[:, disp.index('Olivine')]
        if (f_wa > OL_WAD_THRESH).any():
            P_on = P_scan[np.argmax(f_wa > OL_WAD_THRESH)]
            z_on = np.interp(P_on, P_m, z_m)
        if ((f_wa >= f_ol) & (f_wa > 0)).any():
            P_mid = P_scan[np.argmax((f_wa >= f_ol) & (f_wa > 0))]
            z_mid = np.interp(P_mid, P_m, z_m)
    print(f'  state {state}  h_sol {bml["h_solid_km"]:.0f}  h_liq {bml["h_liquid_km"]:.0f} km  '
          f'Mg# {params["Mg#"]:.3f}  T_lit {params["T_lit"]:.0f} K  P_lit {params["P_lit"]:.2f} GPa')
    print(f'  olivine @5 GPa {f_ol5:.1f}%   Ol->Wad onset {P_on:.2f} GPa ({z_on:.0f} km)   '
          f'midpoint {P_mid:.2f} GPa ({z_mid:.0f} km)')
    summary.append((lab, chain_name, step, params['T_core'], params['Mg#'],
                    params['T_lit'], state, f_ol5, z_on, z_mid))

    # temperature over the whole silicate column, crust included
    m_sil = prof['m_crust'] | m_man | m_bml

    # ── single figure + one panel of the combined figure ────────────────
    fig, ax = plt.subplots(figsize=(8, 9))
    for a, fs_lab, is_single in [(ax, FS - 2, True), (ax_all[i_pick], FS - 4, False)]:
        draw_assemblage(a, fs_lab)
        a.set_ylim(P_bot, 0.0)
        a.set_xlim(0, 100)
        a.set_xlabel('Mass fraction (%)', fontsize=FS)
        if is_single or i_pick == 0:
            a.set_ylabel('Pressure (GPa)', fontsize=FS)
        a.tick_params(labelsize=TICK)
        aT = a.twiny()
        aT.plot(T[m_sil], P[m_sil], color='k', lw=2.2, ls='--',
                path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()],
                zorder=10)
        aT.set_xlim(0, 3000)
        aT.set_ylim(P_bot, 0.0)
        aT.set_xlabel('Temperature (K)', fontsize=FS)
        aT.tick_params(axis='x', labelsize=TICK)
        aT.set_title(f'{TAG}, {lab} CMB', fontsize=FS)

    fig.tight_layout()
    fname = f'132_assemblage_{TAG}_{i_pick+1}_{lab}_{chain_name[6:]}_s{step}.png'
    if SAVE_FIG:
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=200, bbox_inches='tight')
    print('  saved', fname)
    plt.close(fig)

fig_all.tight_layout()
if SAVE_FIG:
    fig_all.savefig(os.path.join(FIG_DIR, f'132_assemblage_{TAG}_all.png'), dpi=200, bbox_inches='tight')
print(f'\nsaved 132_assemblage_{TAG}_all.png')

# ── summary table ────────────────────────────────────────────────────────
print(f'\n{"bin":>4} {"chain":>14} {"step":>6} {"T_core":>7} {"Mg#":>6} {"T_lit":>6} '
      f'{"state":>8} {"ol@5":>6} {"z_on":>6} {"z_mid":>6}')
for s in summary:
    print(f'{s[0]:>4} {s[1]:>14} {s[2]:6d} {s[3]:7.0f} {s[4]:6.3f} {s[5]:6.0f} '
          f'{s[6]:>8} {s[7]:6.1f} {s[8]:6.0f} {s[9]:6.0f}')
print('Huang et al. 2022 discontinuity: 1006 +- 40 km (taken as the transition midpoint)')