#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 91: method figure panel (a) = composition structure + temperature structure of the
#     representative model picked by 90 (reads 90_rep_model.json and its profile npz)
#
# composition is split in two stacked axes because Mg# does not apply to the core:
#   top    (0 to z_cmb)    x = Mg#   mantle Mg#, BML bulk Mg# (dotted), X_S (solid), X_L (liquid)
#   bottom (z_cmb to center) x = w_S
# boxed labels = sampled parameters; dashed boxes = derived by the forward model

import json
import numpy as np
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────────
REP_JSON = '90_rep_model.json'
OUT_PDF  = '91_panel_a.pdf'
GAP      = 0.025                 # gap between the two composition axes (figure fraction)
C_CRUST, C_SOL, C_LIQ, C_CORE = '#d9d9d9', '#c9e2b3', '#f3c4c4', '#ead9b0'
RED      = '#9b2d2d'
plt.rcParams['font.family'] = 'DejaVu Sans'
BOX_IN  = dict(boxstyle='round,pad=0.25', fc='white', ec='k', lw=1.2)
BOX_DER = dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', lw=1.0, ls='--')

rep = json.load(open(REP_JSON))
d = np.load(rep['profile'])
p = rep['params']
z, T, lay, Vs = d['z_km'], d['T_K'], d['layer'], d['Vs']
z_top, z_cmb = float(d['z_bml_top'][0]), float(d['z_cmb'][0])
z_int = z_top + rep['h_solid_km']                       # solid/liquid interface depth
z_lit = rep['z_lit_km']
print(f"{rep['chain']} step {rep['step']}: z_bml_top {z_top:.0f} km, interface {z_int:.0f} km, "
      f"z_cmb {z_cmb:.0f} km, z_lit {z_lit:.0f} km")

fig = plt.figure(figsize=(9, 6.5))
h_sil = 0.78 * z_cmb / 3389.5                           # axes height proportional to depth range
axX = fig.add_axes([0.12, 0.12 + 0.78 - h_sil + GAP / 2, 0.36, h_sil - GAP / 2])
axXc = fig.add_axes([0.12, 0.12, 0.36, 0.78 - h_sil - GAP / 2])
axT = fig.add_axes([0.58, 0.12, 0.36, 0.78])           # temperature: one continuous axis

# ── composition, silicate part ────────────────────────────────────────────────
for lo, hi, c in [(0, 100, C_CRUST), (z_top, z_int, C_SOL), (z_int, z_cmb, C_LIQ)]:
    axX.axhspan(lo, hi, color=c, lw=0)
axX.plot([p['Mg#'], p['Mg#']], [100, z_top], color='k', lw=2)
axX.plot([rep['Mg_solid'], rep['Mg_solid']], [z_top, z_int], color='#3b6d11', lw=3)
axX.plot([rep['Mg_liquid'], rep['Mg_liquid']], [z_int, z_cmb], color='#a32d2d', lw=3)
axX.plot([p['Mg#_bulk_bml'], p['Mg#_bulk_bml']], [z_top, z_cmb], color='k', lw=1, ls=':')
axX.text(p['Mg#'] - 0.03, 0.45 * z_top, 'Mg#', fontsize=12, ha='right', bbox=BOX_IN)
axX.text(p['Mg#_bulk_bml'] - 0.03, z_top - 250, r'Mg#$_{BML}$', fontsize=12, ha='right', bbox=BOX_IN)
axX.text(rep['Mg_solid'] + 0.03, z_top - 60, r'$X_S$', fontsize=12, ha='left', bbox=BOX_DER)
axX.text(rep['Mg_liquid'] - 0.03, z_top - 60, r'$X_L$', fontsize=12, ha='right', bbox=BOX_DER)
axX.text(0.03, 50, 'fixed', fontsize=10, va='center', color='0.3')
axX.set_ylim(z_cmb, 0); axX.set_xlim(0, 1)
axX.xaxis.tick_top(); axX.xaxis.set_label_position('top')
axX.set_xlabel('Mg# (mantle + BML)', fontsize=14)
axX.set_ylabel('depth (km)', fontsize=14)

# ── composition, core ─────────────────────────────────────────────────────────
axXc.axhspan(z_cmb, 3389.5, color=C_CORE, lw=0)
axXc.plot([p['w_S'], p['w_S']], [z_cmb, 3389.5], color='k', lw=2)
axXc.text(p['w_S'] + 0.015, 0.55 * (z_cmb + 3389.5), r'$w_S$', fontsize=12, bbox=BOX_IN)
axXc.text(0.25, z_cmb + 120, r'$R_{cmb}$', fontsize=12, ha='center', va='top', bbox=BOX_IN)
axXc.set_ylim(3389.5, z_cmb); axXc.set_xlim(0, 0.4)
axXc.set_yticks([2000, 2500, 3000])
axXc.set_xlabel(r'$w_S$ (core)', fontsize=14)

# ── temperature: one continuous axis, so the isentrope / adiabats can be followed ──
# T(z) is read from the profile npz (the forward model's own output):
#   0 to z_lit        linear in P from T_surf to T_lit (lithosphere, conductive)
#   z_lit to BML top  isentrope at S_lit from nGibbs (kinks = latent heat at phase transitions)
#   solid BML         linear (conductive);  liquid BML  adiabat dT/dP = gamma T / K_S
#   core              adiabat from T_core (Fe-S EoS)
for lo, hi, c in [(0, 100, C_CRUST), (z_top, z_int, C_SOL), (z_int, z_cmb, C_LIQ), (z_cmb, 3389.5, C_CORE)]:
    axT.axhspan(lo, hi, color=c, lw=0)
axT.plot(T, z, color='k', lw=2)
axT.plot(p['T_lit'], z_lit, 'o', color=RED, ms=7)
axT.plot(p['T_core'], z_cmb, 'o', color=RED, ms=7)
axT.text(p['T_lit'] + 120, z_lit + 150, r'$T_{lit}, P_{lit}$', fontsize=12, bbox=BOX_IN)
axT.text(p['T_core'] - 150, z_cmb + 350, r'$T_{core}$', fontsize=12, ha='right', bbox=BOX_IN)
axT.annotate('', xy=(3150, z_top), xytext=(3150, z_cmb), arrowprops=dict(arrowstyle='<->', lw=1.2))
axT.text(3080, z_top - 150, 'D', fontsize=12, ha='center', bbox=BOX_IN)
axT.set_ylim(3389.5, 0); axT.set_xlim(0, 3300)
axT.set_xlabel('T (K)', fontsize=14)
axT.set_ylabel('depth (km)', fontsize=14)

for a in [axX, axXc, axT]:
    a.tick_params(labelsize=14)
#fig.savefig(OUT_PDF)
#fig.savefig(OUT_PDF.replace('.pdf', '.png'), dpi=200)
print(f"saved {OUT_PDF}")
