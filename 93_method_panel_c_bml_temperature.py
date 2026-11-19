#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 93: method figure panel (c) = temperature in and around the BML of the representative
#     model (90_rep_model.json), x = T, y = depth, with
#       mantle solidus  : Duncan 2018 + Fe correction at mantle Mg# (as in the likelihood)
#       solid sublayer : solidus of its own composition X_S  (should stay above T)
#       liquidus of the liquid composition X_L, drawn through the whole BML (T must stay above it in the liquid)
#       both from the MCMC's Fo-Fa code along the model's own P(z)
#     solid sublayer = conductive (linear T), liquid sublayer = adiabat dT/dP = gamma T / K_S

import json
import types
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

# ── config ────────────────────────────────────────────────────────────────────
FORWARD_FILE = f'{HERE}/26fall_main_mcmc.py'
REP_JSON     = '90_rep_model.json'
OUT_PDF      = '93_panel_c.pdf'
Z_ABOVE      = 60.0          # km of mantle shown above the BML
Z_BELOW      = 40.0          # km of core shown below the CMB
NODE_STRIDE  = 3             # evaluate the BML solidus/liquidus on every Nth BML node
SHOW_LIQUIDUS = False        # True: also draw the liquidus of X_L through the BML (liquid-layer check)
C_SOL, C_LIQ, C_CORE = '#c9e2b3', '#f3c4c4', '#ead9b0'
RED = '#9b2d2d'
plt.rcParams['font.family'] = 'DejaVu Sans'
BOX_DER = dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', lw=1.0, ls='--')

src = open(FORWARD_FILE).read()
fwd = types.ModuleType('fwd')
fwd.__dict__.update(np=np, brentq=brentq, fsolve=fsolve)
exec(src[src.index('# ── physics: solidus'):src.index('# ── composition')], fwd.__dict__)

rep = json.load(open(REP_JSON))
d = np.load(rep['profile'])
p = rep['params']
z, T, P, lay = d['z_km'], d['T_K'], d['P_GPa'], d['layer']
z_top, z_cmb = float(d['z_bml_top'][0]), float(d['z_cmb'][0])
z_int = z_top + rep['h_solid_km']
win = (z >= z_top - Z_ABOVE) & (z <= z_cmb + Z_BELOW)

# solidus of X_S on the solid nodes, liquidus of X_L on the liquid nodes, model's own P(z)
ms = (lay == 2) & (d['Vs'] > 1e-6)
ml = (lay == 2) & (d['Vs'] <= 1e-6)
zs_, Tsol_s, zl_, Tliq_l = [], [], [], []
for i in np.flatnonzero(ms)[::NODE_STRIDE]:
    zs_.append(z[i]); Tsol_s.append(fwd.bml_phase_diagram(2000.0, float(P[i]), rep['Mg_solid'])['T_solidus'])
for i in np.flatnonzero(lay == 2)[::NODE_STRIDE]:        # liquidus of X_L over the whole BML (solid + liquid)
    zl_.append(z[i]); Tliq_l.append(fwd.bml_phase_diagram(2000.0, float(P[i]), rep['Mg_liquid'])['T_liquidus'])
print(f"solid sublayer: min (T_sol(X_S) - T) = {np.min(np.array(Tsol_s) - np.interp(zs_, z[ms], T[ms])):+.0f} K")
zl_a, Tliq_a = np.array(zl_), np.array(Tliq_l)
inl = zl_a >= z_int
print(f"liquid sublayer: min (T - T_liq(X_L)) = {np.min(np.interp(zl_a[inl], z[ml], T[ml]) - Tliq_a[inl]):+.0f} K")
# mantle solidus (Duncan 2018 + Fe), mantle nodes in the window
mm = (lay == 1) & win
Tsol_m = np.array([fwd.solidus_duncan_Mg(pp, p['Mg#']) for pp in P[mm]])

# conductive gradient in the solid sublayer and adiabatic gradient in the liquid
gs = (T[ms][-1] - T[ms][0]) / (z[ms][-1] - z[ms][0])
gl = (T[ml][-1] - T[ml][0]) / (z[ml][-1] - z[ml][0])
print(f"solid sublayer {rep['h_solid_km']:.0f} km: {gs:.2f} K/km;  liquid sublayer "
      f"{rep['h_liquid_km']:.0f} km: {gl:.3f} K/km;  thermal_state {rep['thermal_state']}")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.axhspan(z_top, z_int, color=C_SOL, lw=0)
ax.axhspan(z_int, z_cmb, color=C_LIQ, lw=0)
ax.axhspan(z_cmb, z_cmb + Z_BELOW, color=C_CORE, lw=0)
ax.plot(Tsol_m, z[mm], color='darkgreen', lw=1.2, ls='--', label='mantle solidus')
ax.plot(Tsol_s, zs_, color='#3b6d11', lw=1.5, ls='--', label=r'solidus of $X_S$')
# liquidus of X_L is not drawn here: in this panel only the interface matters,
# where T_int = T_sol(X_S) = T_liq(X_L). The full liquid-layer check is the SI figure from 132.
if SHOW_LIQUIDUS:
    ax.plot(Tliq_l, zl_, color='#a32d2d', lw=1.5, ls=':', label=r'liquidus of $X_L$')
ax.plot(T[win], z[win], color='k', lw=2.5)
ax.plot(rep['T_mantle_bottom'], z_top, 'o', color='0.3', ms=7)
ax.plot(rep['T_interface'], z_int, 'o', color='0.3', ms=7)
ax.plot(p['T_core'], z_cmb, 'o', color=RED, ms=7)
ax.text(rep['T_mantle_bottom'] + 20, z_top - 8, 'mantle bottom', fontsize=11, va='bottom')
ax.text(rep['T_interface'] - 15, z_int + 4, r'$T_{int} = T_{sol}(X_S) = T_{liq}(X_L)$', fontsize=11,
        ha='right', va='top')
ax.text(p['T_core'] - 15, z_cmb + 4, r'$T_{core}$', fontsize=12, ha='right', va='top', color=RED)
ax.text(0.03, (z_top + z_int) / 2, 'solid: conductive', transform=ax.get_yaxis_transform(),
        fontsize=11, va='center', color='#3b6d11')
ax.text(0.03, (z_int + z_cmb) / 2, 'liquid: adiabat', transform=ax.get_yaxis_transform(),
        fontsize=11, va='center', color='#a32d2d')
ax.text(0.97, (z_top + z_int) / 2, r'$h_{solid}$', transform=ax.get_yaxis_transform(),
        fontsize=12, va='center', ha='right', bbox=BOX_DER)
ax.text(0.97, (z_int + z_cmb) / 2, r'$h_{liquid}$', transform=ax.get_yaxis_transform(),
        fontsize=12, va='center', ha='right', bbox=BOX_DER)
ax.set_ylim(z_cmb + Z_BELOW, z_top - Z_ABOVE)
ax.set_xlabel('T (K)', fontsize=14)
ax.set_ylabel('depth (km)', fontsize=14)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=10, frameon=True, framealpha=0.9)
ax.tick_params(labelsize=14)
fig.tight_layout()
#fig.savefig(OUT_PDF)
#fig.savefig(OUT_PDF.replace('.pdf', '.png'), dpi=200)
print(f"saved {OUT_PDF}")
