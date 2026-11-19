#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 92: method figure panel (b) = Fo-Fa binary loop at the interface pressure of the
#     representative model (90_rep_model.json), with the tie-line at T_interface
#
# The phase diagram is the MCMC's own code: only the block between
# '# ── physics: solidus' and '# ── composition' of the forward script is executed
# (needs numpy + scipy only, so this runs on the Mac without nGibbs).
#   solidus  = X_S(T),  liquidus = X_L(T)  from _scan on the branch the model used
#   the model state is a POINT (Mg#_BML, T_int): bulk composition and temperature together
#   decide where it falls (below solidus = solid, inside the loop = layered, above liquidus = molten)
#   lever rule at T_int:  f_liquid = (X_bulk - X_S) / (X_L - X_S)

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
OUT_PDF      = '92_panel_b.pdf'
RED, C_LOOP  = '#9b2d2d', 'C1'
plt.rcParams['font.family'] = 'DejaVu Sans'

src = open(FORWARD_FILE).read()
fwd = types.ModuleType('fwd')
fwd.__dict__.update(np=np, brentq=brentq, fsolve=fsolve)
exec(src[src.index('# ── physics: solidus'):src.index('# ── composition')], fwd.__dict__)

rep = json.load(open(REP_JSON))
P = rep['P_interface']
T_int = rep['T_interface']
X_bulk = rep['params']['Mg#_bulk_bml']
X_S, X_L = rep['Mg_solid'], rep['Mg_liquid']
branch = rep['phase_branch']
loopf = fwd._beta_XS_XL if 'beta' in branch else fwd._gamma_XS_XL
TmFo, TmFa = fwd._Tm_Fo(P), fwd._Tm_Fa(P)
Tg, XS, XL = fwd._scan(loopf, P, TmFo, TmFa)          # Tg already includes _DELTA_SHIFT
f_liq = (X_bulk - X_S) / (X_L - X_S)
print(f"P_int {P:.2f} GPa, branch {branch}, T_int {T_int:.0f} K, X_bulk {X_bulk:.3f}, "
      f"X_S {X_S:.3f}, X_L {X_L:.3f}, f_liquid (mole) {f_liq:.3f}")

fig, ax = plt.subplots(figsize=(5.5, 5))
o, o2 = np.argsort(XS), np.argsort(XL)
ax.fill_betweenx(Tg, XL, XS, color=C_LOOP, alpha=0.2, lw=0)
ax.plot(XS[o], Tg[o], color=C_LOOP, lw=2, label='solidus')
ax.plot(XL[o2], Tg[o2], color=C_LOOP, lw=2, ls='--', label='liquidus')
ax.plot([X_L, X_S], [T_int, T_int], color=RED, lw=2)
ax.plot(X_L, T_int, 'o', color='#a32d2d', ms=8, zorder=5)
ax.plot(X_S, T_int, 'o', color='#3b6d11', ms=8, zorder=5)
ax.text(X_L - 0.02, T_int + 25, r'$X_L$', fontsize=14, ha='right')
ax.text(X_S + 0.02, T_int - 25, r'$X_S$', fontsize=14, va='top')
ax.plot(X_bulk, T_int, '*', color='k', mfc='gold', ms=18, zorder=6)
ax.text(X_bulk, T_int + 45, r'$X_{bulk}$', fontsize=14, ha='center', va='bottom')
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel('Mg#   (Fa  →  Fo)', fontsize=14)
ax.set_ylabel('T (K)', fontsize=14)
ax.set_title(f'Fo–Fa at $P_{{int}}$ = {P:.1f} GPa', fontsize=14)
ax.legend(loc='lower right', fontsize=12, frameon=False)
ax.tick_params(labelsize=14)
fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PDF.replace('.pdf', '.png'), dpi=200)
print(f"saved {OUT_PDF}")
