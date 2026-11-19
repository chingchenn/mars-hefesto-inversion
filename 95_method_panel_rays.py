#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 95: method figure ray-path panel through the representative model (90_rep_model.json),
#     drawn like the slide figure: half Mars with InSight on the left, a few representative
#     events labelled by distance, crust / mantle / BML / core shells, and a radius bar that
#     marks R_cmb (metal core) and R_ScS (top of the fluid, where ScS reflects).
#
# The .nd file follows build_taup in the MCMC: the fluid boundary is where Vs -> 0
# (top of the liquid BML, or the CMB if the BML is fully solid).
# Phases per event come from SAMUEL_DATA in the forward script ('A-B' gives A and B);
# SKS and Pdiff are not in the likelihood and are not drawn. Depth phases are skipped for clarity.
# Geometry: angle from InSight = delta - dist (TauP dist runs from the source),
#           x = r cos(pi - angle), y = r sin(pi - angle)  -> InSight at (-R, 0)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script
# ── config ────────────────────────────────────────────────────────────────────
FORWARD_FILE = f'{HERE}/26fall_main_mcmc.py'
REP_JSON     = '90_rep_model.json'
ND_DIR       = '95_nd'
OUT_PDF      = '95_panel_rays.pdf'
R            = 3389.5
EVENTS       = ['S0918a', 'S0173a', 'S1094b', 'S1000a', 'S0976a']   # representative distances
SKIP         = ['pP', 'sP', 'sS']                                     # depth phases not drawn
PHASE_COLORS = {'P': '#d6604d', 'S': '#4393c3', 'PP': '#f4a582', 'PPP': '#fbd0bd',
                'SS': '#92c5de', 'SSS': '#cfe3f0', 'ScS': '#17becf'}
C_CRUST, C_MANTLE, C_SOL, C_LIQ, C_CORE = '#e2ddd5', '#f8f6f2', '#c9e2b3', '#f0bfb1', '#dfe3ea'
plt.rcParams['font.family'] = 'DejaVu Sans'

src = open(FORWARD_FILE).read()
dat = {}
exec(src[src.index('# ── seismic data'):src.index('# ── gravity and pressure')], dat)
DATA = dat['SAMUEL_DATA']

rep = json.load(open(REP_JSON))
d = np.load(rep['profile'])
z, Vp, Vs, rho, lay = d['z_km'], d['Vp'], d['Vs'], d['rho'], d['layer']
z_top, z_cmb = float(d['z_bml_top'][0]), float(d['z_cmb'][0])
z_int = z_top + rep['h_solid_km']

# ── .nd file and TauP model ───────────────────────────────────────────────────
os.makedirs(ND_DIR, exist_ok=True)
name = f"{rep['chain']}_s{rep['step']:05d}"
nd_path = os.path.join(ND_DIR, name + '.nd')
m_fx, m_man, m_bml, m_cor = lay == 0, lay == 1, lay == 2, lay == 3
d_b, vp_b, vs_b, rho_b = z[m_bml], Vp[m_bml], Vs[m_bml], rho[m_bml]
zero = np.flatnonzero(vs_b <= 1e-6)
i0 = int(zero[0]) if len(zero) else len(d_b)
with open(nd_path, 'w') as f:
    for k in np.flatnonzero(m_fx):
        f.write(f"{z[k]:.3f}  {Vp[k]:.4f}  {Vs[k]:.4f}  {rho[k]:.4f}\n")
    f.write("mantle\n")
    for k in np.flatnonzero(m_man):
        f.write(f"{z[k]:.3f}  {Vp[k]:.4f}  {Vs[k]:.4f}  {rho[k]:.4f}\n")
    for k in range(i0):
        f.write(f"{d_b[k]:.3f}  {vp_b[k]:.4f}  {vs_b[k]:.4f}  {rho_b[k]:.4f}\n")
    if i0 < len(d_b):
        if i0 > 0:
            f.write(f"{d_b[i0]:.3f}  {vp_b[i0-1]:.4f}  {vs_b[i0-1]:.4f}  {rho_b[i0-1]:.4f}\n")
        f.write("outer-core\n")
        for k in range(i0, len(d_b)):
            f.write(f"{d_b[k]:.3f}  {vp_b[k]:.4f}  0.0000  {rho_b[k]:.4f}\n")
    else:
        f.write("outer-core\n")
    for k in np.flatnonzero(m_cor):
        f.write(f"{z[k]:.3f}  {Vp[k]:.4f}  0.0000  {rho[k]:.4f}\n")
build_taup_model(nd_path, output_folder=ND_DIR)
model = TauPyModel(model=os.path.join(ND_DIR, name + '.npz'))
z_fluid = d_b[i0] if i0 < len(d_b) else z_cmb
R_cmb, R_scs = R - z_cmb, R - z_fluid
print(f"{name}: R_cmb {R_cmb:.0f} km, fluid top (R_ScS) {R_scs:.0f} km")

# ── shells ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.8))
for r_out, r_in, c in [(R, R - 100, C_CRUST), (R - 100, R - z_top, C_MANTLE),
                       (R - z_top, R - z_int, C_SOL), (R - z_int, R_cmb, C_LIQ), (R_cmb, 0, C_CORE)]:
    ax.add_patch(Wedge((0, 0), r_out, 0, 180, width=r_out - r_in, fc=c, ec='none', zorder=1))
ax.add_patch(Wedge((0, 0), R, 0, 180, fc='none', ec='k', lw=1.2, zorder=4))
ax.add_patch(Wedge((0, 0), R_cmb, 0, 180, fc='none', ec='#a03c2c', lw=1.0, zorder=4))
ax.add_patch(Wedge((0, 0), R_scs, 0, 180, fc='none', ec='#a03c2c', lw=0.6, ls='--', zorder=4))
ax.plot([-R, R], [0, 0], color='k', lw=1.0, zorder=4)

# all event distances as small ticks on the surface
for ev, obs in DATA.items():
    a = np.pi - np.radians(obs['delta'])
    ax.plot([R * np.cos(a), 1.025 * R * np.cos(a)], [R * np.sin(a), 1.025 * R * np.sin(a)],
            color='0.3', lw=0.8, zorder=5)

# ── rays of the representative events ─────────────────────────────────────────
used = set()
for ev in EVENTS:
    obs = DATA[ev]
    phases = sorted({ph for key in obs if key not in ('delta', 'depth') for ph in key.split('-')} - set(SKIP))
    arrivals = model.get_ray_paths(source_depth_in_km=obs['depth'], distance_in_degree=obs['delta'],
                                   phase_list=phases)
    seen = set()
    for arr in arrivals:
        if arr.name in seen or arr.name not in PHASE_COLORS:
            continue
        seen.add(arr.name); used.add(arr.name)
        ang = np.radians(obs['delta']) - arr.path['dist']
        r = R - arr.path['depth']
        lw = 1.6 if arr.name == 'ScS' else 1.0
        ax.plot(r * np.cos(np.pi - ang), r * np.sin(np.pi - ang), color=PHASE_COLORS[arr.name],
                lw=lw, zorder=6 if arr.name == 'ScS' else 5)
    a = np.pi - np.radians(obs['delta'])
    ax.plot(1.06 * R * np.cos(a), 1.06 * R * np.sin(a), 'v', color='#e08a1e', mec='k', ms=10,
            zorder=8, clip_on=False)
    ax.text(1.14 * R * np.cos(a), 1.12 * R * np.sin(a), f"{obs['delta']:.0f}°", fontsize=12,
            ha='center', va='center')
    print(f"  {ev}: {obs['delta']:.1f} deg  {sorted(seen)}")
ax.plot(-R, 0, '*', color='gold', mec='k', ms=16, zorder=9, clip_on=False)
ax.text(-R, -0.05 * R, 'InSight', fontsize=12, ha='center', va='top')

# ── radius bar ────────────────────────────────────────────────────────────────
yb = -0.09 * R
ax.plot([0, R], [yb, yb], color='k', lw=1)
for rr, lab, dy in [(0, '0', 0), (R_cmb, r'$R_{cmb}$' + f' {R_cmb:.0f}', 0),
                    (R_scs, r'$R_{ScS}$' + f' {R_scs:.0f}', -0.08 * R), (R, f'{R:.0f}', 0)]:
    ax.plot([rr, rr], [yb, yb - 0.025 * R], color='k', lw=1)
    ax.text(rr, yb - 0.05 * R + dy, lab, fontsize=12, ha='center', va='top')
ax.text(R / 2, yb - 0.26 * R, 'radius (km)', fontsize=12, ha='center', va='top')

handles = [Line2D([0], [0], color=PHASE_COLORS[p], lw=2, label=p) for p in PHASE_COLORS if p in used]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=len(handles),
          fontsize=12, frameon=False)
ax.set_xlim(-1.18 * R, 1.12 * R); ax.set_ylim(-0.38 * R, 1.2 * R)
ax.set_aspect('equal'); ax.axis('off')
ax.tick_params(labelsize=14)
#fig.savefig(OUT_PDF, bbox_inches='tight')
#fig.savefig(OUT_PDF.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
print(f"saved {OUT_PDF}")
