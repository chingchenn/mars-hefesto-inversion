#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 94: method figure panels (d) Vp, Vs and (e) density of the representative model
#     (90_rep_model.json), one continuous depth axis shared by both panels

import json
import numpy as np
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────────
REP_JSON = '90_rep_model.json'
OUT_PDF  = '94_panel_de.pdf'
C_CRUST, C_SOL, C_LIQ, C_CORE = '#d9d9d9', '#c9e2b3', '#f3c4c4', '#ead9b0'
RED = '#9b2d2d'
plt.rcParams['font.family'] = 'DejaVu Sans'

rep = json.load(open(REP_JSON))
d = np.load(rep['profile'])
z, Vp, Vs, rho = d['z_km'], d['Vp'], d['Vs'], d['rho']
z_top, z_cmb = float(d['z_bml_top'][0]), float(d['z_cmb'][0])
z_int = z_top + rep['h_solid_km']
print(f"{rep['chain']} step {rep['step']}: Vs = 0 from {z[(d['layer'] == 2) & (Vs <= 1e-6)].min():.0f} km; "
      f"rho BML {rho[d['layer'] == 2].min():.2f}-{rho[d['layer'] == 2].max():.2f} g/cm3")

fig, ax = plt.subplots(1, 2, figsize=(8, 6.5), sharey=True)
for a in ax:
    for lo, hi, c in [(0, 100, C_CRUST), (z_top, z_int, C_SOL), (z_int, z_cmb, C_LIQ), (z_cmb, 3389.5, C_CORE)]:
        a.axhspan(lo, hi, color=c, lw=0)
ax[0].plot(Vp, z, color='C0', lw=2, label='Vp')
ax[0].plot(Vs, z, color=RED, lw=2, label='Vs')
ax[0].text(0.3, (z_int + z_cmb) / 2, r'$V_s$ = 0', fontsize=11, color=RED, va='center')
ax[0].set_xlim(-0.3, 10.5)
ax[0].set_xlabel('velocity (km/s)', fontsize=14)
ax[0].set_ylabel('depth (km)', fontsize=14)
ax[0].legend(loc='lower right', fontsize=12, frameon=False)
ax[1].plot(rho, z, color='k', lw=2)
ax[1].set_xlim(2.5, 7.5)
ax[1].set_xlabel(r'density (g/cm$^3$)', fontsize=14)
ax[0].set_ylim(3389.5, 0)
for a in ax:
    a.tick_params(labelsize=14)
fig.tight_layout()
#fig.savefig(OUT_PDF)
#fig.savefig(OUT_PDF.replace('.pdf', '.png'), dpi=200)
print(f"saved {OUT_PDF}")
