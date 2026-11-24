#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 69: core composition and CMB properties of the light-element runs, on the axes of
#     RoG (Khan et al., subm.) Fig. 31 and Samuel 2023 SM Fig. 10
#
#   python 69.core_composition_vs_RoG.py          (SAVE_FIG=1 to save to FIG_DIR)
#
# Panels
#   (a) core Vp at the CMB vs core density at the CMB, per run
#       rho_CMB is not stored in chain.jsonl: recomputed with core_fes_x at (P_cmb, T_core, w_S, w_X)
#       (T_core is the CMB temperature in the main program). Vp_CMB is the stored Vp_core_cmb.
#       Boxes: Samuel 2023 SM Fig. 10 (models compatible with Fe-O-S(-C-H)), RoG Fig. 31a,b
#       at the top of the core (K23 seismic, GPR EoS). All boxes are digitized by eye: approximate.
#   (b)-(d) C, H, S vs Fe wt% (RoG Fig. 31 c, d, f). Boxes = rough extent of the RoG GPR
#       (red circles) and AIMD (yellow stars) solutions, digitized by eye.
#       Caveat: our core has no Ni and no O, so Fe = 100 (1 - w_S - w_X); RoG's Fe excludes the
#       O they also fit. Compare trends and ranges, not exact positions.
#   (e) w_C vs w_S of the C runs, with the Tsuno et al. 2018 (GCA 238, eq. 3) C-solubility limit
#       ln C = -33.59 - 2726/T - 38.7 P/T + 7.51 ln(100 - S) + 0.48 ln(100 - Ni)   (wt%, Ni = 0)
#       evaluated at TSUNO_PT. Experiments span 6-13 GPa, 1800-2000 C; 20 GPa is an extrapolation.
#       It is a graphite-saturation ceiling, used here only as a post-hoc filter, not as a prior.
#   (f) BML thickness of the C runs: all samples vs samples below the reference Tsuno limit
#       The terminal prints, per C run and per (P, T): fraction above the limit, and state
#       fractions / BML thickness / R_ScS for the samples that pass.
# Burn-in 0.4 per chain (same as 63v2/68/81). a72-a74 are not converged (11/23).
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR
import core_fes as CF
import core_fes_x as CX

# (label, prefix, light, colour, marker)
RUNS = [('a64', 'chain_a64_YM20', 'none', '#333333', 'o'),
        ('a70', 'chain_a70_YM20', 'C',    '#CD5C5C', 'o'),
        ('a74', 'chain_a74_YM20', 'C',    '#FF8C00', 'o'),
        ('a71', 'chain_a71_YM20', 'H',    '#7FA7D9', 'o'),
        ('a73', 'chain_a73_YM20', 'H',    '#1F4E9E', 'o')]
EXCLUDE   = ['chain_a71_YM20_10', 'chain_a71_YM20_12']   # stuck without SKS (155)
BURN_FRAC = 0.4
MIN_STEPS = 300
N_PLOT    = 2500        # random samples per run (rho_CMB is recomputed for these)
SEED      = 0
SAVE_FIG  = os.environ.get('SAVE_FIG') == '1'
FS        = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# digitized boxes (approximate): (x0, x1, y0, y1)
BOX_RV = {'Samuel 2023 SM Fig. 10 (compatible models)': ((6.10, 6.40), (5.00, 5.26), '#2CA02C'),
          'RoG Fig. 31 K23 seismic (core top)':          ((6.30, 6.65), (4.95, 5.35), '#9467BD'),
          'RoG Fig. 31 GPR EoS (core top)':              ((6.42, 6.56), (5.05, 5.22), '#D62728')}
BOX_FE = {'C': {'GPR': ((88, 95), (1.2, 4.0)), 'AIMD': ((84, 91), (0.3, 2.0))},
          'H': {'GPR': ((87, 95), (0.8, 1.3)), 'AIMD': ((84, 91), (0.7, 2.0))},
          'S': {'GPR': ((87, 95), (1.0, 10.0)), 'AIMD': ((84, 91), (5.0, 12.0))}}
BOX_COL = {'GPR': '#D62728', 'AIMD': '#E6B800'}

# Tsuno 2018 eq. 3 evaluation points (P [GPa], T [K], line style); TSUNO_REF is used in (f)
TSUNO_PT  = [(6.0, 2073.0, ':'), (13.0, 2273.0, '-'), (20.0, 2300.0, '--')]
TSUNO_REF = 1


def c_max_tsuno(wS, P, T):
    # returns the C solubility limit as a mass fraction; wS is a mass fraction
    lnC = -33.59 - 2726.0/T - 38.7*P/T + 7.51*np.log(100.0 - 100.0*wS) + 0.48*np.log(100.0)
    return np.exp(lnC)/100.0


def state(r):
    if r['params']['BML_thickness'] < 1.0:
        return 'none'
    hs, hl = r.get('h_solid_km'), r.get('h_liquid_km')
    if hs is None or hl is None:
        return 'fail'
    if hl < 1:
        return 'solid'
    if hs < 1:
        return 'molten'
    return 'layered'


def load_jsonl(path):
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return recs


rng = np.random.default_rng(SEED)
data = {}
for lab, prefix, light, col, mk in RUNS:
    recs = []
    for d in sorted(glob.glob(os.path.join(MCMC_DIR, prefix + '_[0-9][0-9]'))):
        fp = os.path.join(d, 'chain.jsonl')
        if os.path.basename(d) in EXCLUDE or not os.path.exists(fp):
            continue
        R = load_jsonl(fp)
        if len(R) >= MIN_STEPS:
            recs += R[int(len(R)*BURN_FRAC):]
    if not recs:
        print(f'{lab}: no chains')
        continue
    recs = [recs[i] for i in rng.choice(len(recs), size=min(N_PLOT, len(recs)), replace=False)]
    key = {'C': 'w_C', 'H': 'w_H'}.get(light)
    CX.configure(light, verbose=False)
    wS, wX, rho, vp, st, D, rscs = [], [], [], [], [], [], []
    for r in recs:
        p = r['params']
        ws, wx = p['w_S'], (p[key] if key else 0.0)
        out = CX.core_properties(np.array([r['P_cmb']]), np.array([p['T_core']]), ws, wx)
        wS.append(ws); wX.append(wx); rho.append(float(out['rho'][0])); vp.append(r['Vp_core_cmb'])
        st.append(state(r)); D.append(p['BML_thickness'])
        rscs.append(p['R_cmb'] + (r.get('h_liquid_km') or 0.0))
    data[lab] = dict(wS=np.array(wS), wX=np.array(wX), rho=np.array(rho), vp=np.array(vp),
                     st=np.array(st), D=np.array(D), rscs=np.array(rscs),
                     light=light, col=col, mk=mk)
    v = data[lab]
    fe = 100*(1 - v['wS'] - v['wX'])
    print(f"{lab} ({light:>4s}) n={len(rho)}  rho_CMB {np.median(v['rho']):.3f} "
          f"[{np.percentile(v['rho'], 5):.3f}, {np.percentile(v['rho'], 95):.3f}]  "
          f"Vp_CMB {np.median(v['vp']):.3f} [{np.percentile(v['vp'], 5):.3f}, {np.percentile(v['vp'], 95):.3f}]  "
          f"Fe {np.median(fe):.1f}  S {100*np.median(v['wS']):.1f}"
          + (f"  {light} {100*np.median(v['wX']):.2f}" if key else '') + '  (wt%)')

# Tsuno filter, C runs only
for lab, v in data.items():
    if v['light'] != 'C':
        continue
    print(f'\n{lab}: Tsuno 2018 filter (all: molten {np.mean(v["st"] == "molten"):.3f}  '
          f'solid {np.mean(v["st"] == "solid"):.3f}  D {np.median(v["D"]):.0f} km  '
          f'R_ScS {np.median(v["rscs"]):.0f} km)')
    for P, T, ls in TSUNO_PT:
        ok = v['wX'] <= c_max_tsuno(v['wS'], P, T)
        print(f'  {P:4.0f} GPa {T:5.0f} K  above limit {1 - np.mean(ok):.3f}  pass n={ok.sum():5d}  '
              f'molten {np.mean(v["st"][ok] == "molten"):.3f}  solid {np.mean(v["st"][ok] == "solid"):.3f}  '
              f'layered {np.mean(v["st"][ok] == "layered"):.3f}  '
              f'D {np.median(v["D"][ok]):.0f} [{np.percentile(v["D"][ok], 5):.0f}, {np.percentile(v["D"][ok], 95):.0f}] km  '
              f'R_ScS {np.median(v["rscs"][ok]):.0f} km')

fig, axs = plt.subplots(2, 3, figsize=(19, 11))
ax = axs[0, 0]
for nm, ((x0, x1), (y0, y1), c) in BOX_RV.items():
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=c, alpha=0.15, ec=c, lw=1.5, label=nm))
for lab, v in data.items():
    ax.scatter(v['rho'], v['vp'], s=5, alpha=0.35, color=v['col'], label=f"{lab} ({v['light']})",
               rasterized=True)
ax.set_xlabel(r'core $\rho$ at CMB [g/cm$^3$]', fontsize=FS)
ax.set_ylabel('core Vp at CMB [km/s]', fontsize=FS)
ax.set_xlim(5.8, 7.6)
ax.set_ylim(4.3, 5.8)
ax.legend(loc='lower right', fontsize=FS - 5, markerscale=3)
ax.set_title('(a) CMB properties', fontsize=FS)

for ax, (el, ylab, title) in zip([axs[0, 1], axs[0, 2], axs[1, 0]],
                                 [('C', 'C [wt%]', '(b) C vs Fe'), ('H', 'H [wt%]', '(c) H vs Fe'),
                                  ('S', 'S [wt%]', '(d) S vs Fe')]):
    for src, ((x0, x1), (y0, y1)) in BOX_FE[el].items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=BOX_COL[src], alpha=0.15,
                                   ec=BOX_COL[src], lw=1.5, label=f'RoG {src} (approx.)'))
    for lab, v in data.items():
        if el != 'S' and v['light'] != el:
            continue
        fe = 100*(1 - v['wS'] - v['wX'])
        y  = 100*(v['wS'] if el == 'S' else v['wX'])
        ax.scatter(fe, y, s=5, alpha=0.35, color=v['col'], label=lab, rasterized=True)
    ax.set_xlim(70, 100)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Fe [wt%]', fontsize=FS)
    ax.set_ylabel(ylab, fontsize=FS)
    ax.set_title(title, fontsize=FS)
    ax.legend(loc='lower right', fontsize=FS - 5, markerscale=3)
ax = axs[1, 1]
ws_grid = np.linspace(0.0, 0.35, 200)
for lab, v in data.items():
    if v['light'] != 'C':
        continue
    ax.scatter(100*v['wS'], 100*v['wX'], s=5, alpha=0.35, color=v['col'], label=lab, rasterized=True)
for P, T, ls in TSUNO_PT:
    ax.plot(100*ws_grid, 100*c_max_tsuno(ws_grid, P, T), color='k', ls=ls, lw=1.5,
            label=f'Tsuno 2018, {P:.0f} GPa {T:.0f} K')
ax.set_xlim(0, 35)
ax.set_ylim(0, 6)
ax.set_xlabel('S [wt%]', fontsize=FS)
ax.set_ylabel('C [wt%]', fontsize=FS)
ax.set_title('(e) C vs S with C-solubility limit', fontsize=FS)
ax.legend(loc='lower right', fontsize=FS - 5, markerscale=3)

ax = axs[1, 2]
P, T, _ = TSUNO_PT[TSUNO_REF]
bins = np.linspace(0, 400, 41)
for lab, v in data.items():
    if v['light'] != 'C':
        continue
    ok = v['wX'] <= c_max_tsuno(v['wS'], P, T)
    ax.hist(v['D'], bins=bins, density=True, histtype='step', color=v['col'], lw=1.5, ls='--',
            label=f'{lab} all')
    ax.hist(v['D'][ok], bins=bins, density=True, histtype='step', color=v['col'], lw=2,
            label=f'{lab} below Tsuno ({P:.0f} GPa)')
ax.set_xlabel('BML thickness [km]', fontsize=FS)
ax.set_ylabel('density', fontsize=FS)
ax.set_title('(f) BML thickness, Tsuno filter', fontsize=FS)
ax.legend(loc='lower right', fontsize=FS - 5)

for ax in axs.ravel():
    ax.tick_params(labelsize=14)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(os.path.join(FIG_DIR, '69_core_composition_vs_RoG.png'), dpi=200)
plt.show()
