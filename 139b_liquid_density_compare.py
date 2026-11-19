#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:26:58 2026

@author: chingchen
"""

import os
import glob
import json
import types
import numpy as np
from scipy.optimize import brentq
import openpyxl
import matplotlib.pyplot as plt
from mars_config import DATA_BASE, KHAN_MODEL_DIR, MCMC_DIR, FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

# ── config ────────────────────────────────────────────────────────────────────
HERE             = os.path.dirname(os.path.abspath(__file__))
FORWARD_FILE     = os.path.join(HERE, '26fall_main_mcmc.py')
XLSX_FILE        = os.path.join(HERE, 'Pierru 2026 data 2.xlsx')
SHEET            = 'Data Fig. 6'
CHAIN_ROOT       = MCMC_DIR
SAMUEL_RHO       = (f'{DATA_BASE}/Mars_Samuel_2023/'
                    'Nature_Samuel/METADATA_BML/DATA_FIG1/PANEL_J/rho_profile.dat')
COMPS            = ['YM20']   # a60 compositions to pool, e.g. ['YM20', 'TA', 'EH45']
EXCLUDE          = []
BURN_FRAC        = 0.40
PROFILE_STRIDE   = 5
P_MIN, P_MAX     = 16.0, 23.0
P_GRID           = np.linspace(P_MIN, P_MAX, 141)
MG_MELT          = 0.50       # Fe# 0.5 (S3631), as in Pierru Fig. 6
G_CONST          = 6.674e-11
RHO_SILICATE_MAX = 5.5        # g/cm3, keep the silicate part only (drop the core) for Khan / Samuel
# Pierru sheet columns (checked from values): melt along the reference T profile
#   0 P (GPa), 1 T (K), 2 empty, 3 Asimow model 1, 5 Asimow model 2, 7 Wakabayashi-Funamori, 9 Jing-Karato
#   (columns 4, 6, 8, 10 = Monte Carlo 1 sigma); solid MA17: 14 T, 15 P, 16 rho, 17 sigma
SOLID_T = [1900, 2000, 2100, 2200]

# ── load ONLY the liquid EoS block of the forward model ───────────────────────
src = open(FORWARD_FILE).read()
i0 = src.index('# ── Liquid BML EoS')
i1 = src.index('# ── composition')
m = types.ModuleType('eos')
m.__dict__.update(np=np, brentq=brentq)
exec(src[i0:i1], m.__dict__)

# ── Pierru Fig. 6 source data ('-' and blanks -> nan) ─────────────────────────
ws = openpyxl.load_workbook(XLSX_FILE, data_only=True, read_only=True)[SHEET]
rows = list(ws.iter_rows(values_only=True))[3:]
width = max(len(r) for r in rows)
D = np.full((len(rows), width), np.nan)
for i, r in enumerate(rows):
    for j, c in enumerate(r):
        if isinstance(c, (int, float)):
            D[i, j] = c
mel = np.isfinite(D[:, 0]) & np.isfinite(D[:, 1]) & np.isfinite(D[:, 3])
P_m, T_m = D[mel, 0], D[mel, 1]

# control line: this EoS at Fe# 0.5 along the same T(P) profile (Pierru data stop at 21 GPa)
P_mod = np.linspace(P_MIN, 21.0, 26)
T_mod = np.interp(P_mod, P_m, T_m)
rho_mod = np.array([m.liquid_bml_properties(P, T, MG_MELT)[0] for P, T in zip(P_mod, T_mod)])

# ── a60 posterior: BML liquid, BML solid, mantle density on the P grid ────────
weights = {}
info = {}
for comp in COMPS:
    for path in sorted(glob.glob(os.path.join(CHAIN_ROOT, f'chain_a60_{comp}_*', 'chain.jsonl'))):
        chain_dir = os.path.dirname(path)
        if os.path.basename(chain_dir) in EXCLUDE:
            continue
        lines = [json.loads(l) for l in open(path) if l.strip().endswith('}')]
        n_burn = int(BURN_FRAC * len(lines))
        last_acc = None
        for i, rec in enumerate(lines):
            if rec['accepted']:
                last_acc = rec['step']
            if i >= n_burn and last_acc is not None:
                key = (chain_dir, last_acc)
                weights[key] = weights.get(key, 0) + 1
                info[key] = rec.get('Mg_liquid', -1.0)
keys = sorted(weights)[::PROFILE_STRIDE]

RL, RS, RM, W, TL, MGL = [], [], [], [], [], []
for key in keys:
    f = os.path.join(key[0], f"profile_s{key[1]:05d}.npz")
    if not os.path.exists(f):
        continue
    d = np.load(f)
    P, T, rho, Vs, lay = d['P_GPa'], d['T_K'], d['rho'], d['Vs'], d['layer']
    liq = (lay == 2) & (Vs <= 1e-6)
    sol = (lay == 2) & (Vs > 1e-6)
    man = (lay == 1)
    RL.append(np.interp(P_GRID, P[liq], rho[liq], left=np.nan, right=np.nan) if liq.sum() > 1
              else np.full(len(P_GRID), np.nan))
    RS.append(np.interp(P_GRID, P[sol], rho[sol], left=np.nan, right=np.nan) if sol.sum() > 1
              else np.full(len(P_GRID), np.nan))
    RM.append(np.interp(P_GRID, P[man], rho[man], left=np.nan, right=np.nan))
    TL.append(np.interp(P_GRID, P[liq], T[liq], left=np.nan, right=np.nan) if liq.sum() > 1
              else np.full(len(P_GRID), np.nan))
    W.append(weights[key])
    MGL.append(info[key])
RL, RS, RM, TL = np.array(RL), np.array(RS), np.array(RM), np.array(TL)
W, MGL = np.array(W, float), np.array(MGL)

# weighted 5 / 50 / 95 percentiles at each P (only profiles that have that layer there)
Q = {}
for name, A in [('liq', RL), ('sol', RS), ('man', RM), ('Tliq', TL)]:
    q = np.full((3, len(P_GRID)), np.nan)
    for j in range(len(P_GRID)):
        v = A[:, j]
        ok = np.isfinite(v)
        if ok.sum() < 10:
            continue
        o = np.argsort(v[ok])
        cw = np.cumsum(W[ok][o]); cw /= cw[-1]
        q[:, j] = np.interp([0.05, 0.50, 0.95], cw, v[ok][o])
    Q[name] = q

has_liq = MGL > 0
mg_med = np.median(np.repeat(MGL[has_liq], W[has_liq].astype(int))) if has_liq.any() else np.nan
print(f"a60 {'+'.join(COMPS)}: {len(W)} profiles; with liquid BML: {W[has_liq].sum()/W.sum()*100:.0f}% "
      f"(weighted); median Mg#_liquid {mg_med:.3f} (Fe# {1 - mg_med:.3f})")

# ── Khan 2023 ensemble: columns 0 depth, 1 Vp, 2 Vs, 3 rho (g/cm3), 4 T, 5 g, 6 P ──
KH, KH_PMAX = [], []
for fpath in sorted(glob.glob(os.path.join(KHAN_MODEL_DIR, 'Model_*.txt'))):
    dk = np.loadtxt(fpath, comments='#')
    sil = dk[:, 3] < RHO_SILICATE_MAX
    o = np.argsort(dk[sil, 6])
    KH.append(np.interp(P_GRID, dk[sil, 6][o] / 1e9, dk[sil, 3][o], left=np.nan, right=np.nan))
    KH_PMAX.append(dk[:, 6].max() / 1e9)
KH = np.array(KH)
KQ = np.nanpercentile(KH, [16, 50, 84], axis=0)
print(f"Khan 2023: {len(KH)} models; P column max {np.median(KH_PMAX):.1f} (should be GPa, ~40 at the centre)")

# ── Samuel 2023: rho (kg/m3) vs radius (km); P from hydrostatic integration ──
#   M(r) = int 4 pi r^2 rho dr,  g = G M / r^2,  dP = rho g dr  (same as the forward model)
s = np.loadtxt(SAMUEL_RHO)
o = np.argsort(s[:, 1])
rho_s, r_s = s[o, 0], s[o, 1] * 1e3                     # centre -> surface, kg/m3, m
M_s = np.concatenate([[0.0], np.cumsum(4 * np.pi * 0.5 * (rho_s[1:] + rho_s[:-1])
                                        * (0.5 * (r_s[1:] + r_s[:-1]))**2 * np.diff(r_s))])
g_s = np.zeros(len(r_s))
g_s[1:] = G_CONST * M_s[1:] / r_s[1:]**2
dP_s = 0.5 * (rho_s[1:] * g_s[1:] + rho_s[:-1] * g_s[:-1]) * np.diff(r_s)   # Pa per segment
P_s = np.zeros(len(r_s))
P_s[:-1] = np.cumsum(dP_s[::-1])[::-1] / 1e9            # integrate from the surface (P = 0) down
sil_s = (rho_s / 1000 < RHO_SILICATE_MAX) & (P_s >= P_MIN)
print(f"Samuel 2023: silicate base at {P_s[sil_s].max():.1f} GPa, "
      f"density there {rho_s[sil_s][np.argmax(P_s[sil_s])]/1000:.2f} g/cm3")

# ── summary table ─────────────────────────────────────────────────────────────
for P in [19.0, 21.0]:
    j = np.argmin(np.abs(P_GRID - P))
    print(f"{P:.0f} GPa: a60 liquid BML {Q['liq'][1, j]:.3f} (5-95% {Q['liq'][0, j]:.3f}-{Q['liq'][2, j]:.3f}, "
          f"T {Q['Tliq'][1, j]:.0f} K) | this EoS Fe# 0.5 {np.interp(P, P_mod, rho_mod):.3f} | "
          f"Pierru A1 {np.interp(P, P_m, D[mel, 3])/1000:.3f}  A2 {np.interp(P, P_m, D[mel, 5])/1000:.3f} | "
          f"Khan {KQ[1, j]:.3f}")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
cmap  = plt.cm.plasma_r
cnorm = plt.Normalize(1900, 2200)
for Ts in SOLID_T:
    s1 = (D[:, 14] == Ts) & (D[:, 15] >= P_MIN)
    ax.fill_betweenx(D[s1, 15], D[s1, 16] / 1000 - D[s1, 17] / 1000, D[s1, 16] / 1000 + D[s1, 17] / 1000,
                     color=cmap(cnorm(Ts)), alpha=0.9, lw=0)

# Pierru melts (Fe# 0.5 along his T profile)
sel = P_m >= P_MIN
ax.plot(D[mel, 3][sel] / 1000, P_m[sel], '--', color='C3', lw=1.8, label='Pierru: Asimow model 1 (with Fo)')
ax.plot(D[mel, 5][sel] / 1000, P_m[sel], ls=(0, (8, 4)), color='0.4', lw=1.8, label='Pierru: Asimow model 2 (Fig. 6)')
ax.plot(D[mel, 9][sel] / 1000, P_m[sel], '-', color='0.4', lw=1.4, label='Pierru: Jing-Karato')
ax.plot(D[mel, 7][sel] / 1000, P_m[sel], ':', color='0.4', lw=1.8, label='Pierru: Wakabayashi-Funamori')
# control: this EoS, same melt and T as Pierru
ax.plot(rho_mod, P_mod, '-', color='k', lw=2.6, label='this EoS, Fe# 0.5 on Pierru T profile')

# a60 posterior envelopes
ax.fill_betweenx(P_GRID, Q['liq'][0], Q['liq'][2], color='C0', alpha=0.20, lw=0)
ax.plot(Q['liq'][1], P_GRID, '-', color='C0', lw=2.2, label='a60 posterior: liquid BML (median, 5-95%)')
ax.fill_betweenx(P_GRID, Q['sol'][0], Q['sol'][2], color='C5', alpha=0.20, lw=0)
ax.plot(Q['sol'][1], P_GRID, '-', color='C5', lw=2.2, label='a60 posterior: solid BML')
ax.fill_betweenx(P_GRID, Q['man'][0], Q['man'][2], color='0.5', alpha=0.25, lw=0)
ax.plot(Q['man'][1], P_GRID, '-', color='0.3', lw=1.6, label='a60 posterior: mantle')

# InSight-based models
ax.fill_betweenx(P_GRID, KQ[0], KQ[2], color='darkgreen', alpha=0.15, lw=0)
ax.plot(KQ[1], P_GRID, '-', color='darkgreen', lw=2.0, label='Khan 2023 (median, 16-84%)')
ax.plot(rho_s[sil_s] / 1000, P_s[sil_s], '--', color='darkgreen', lw=2.0, label='Samuel 2023')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
cb = fig.colorbar(sm, ax=ax, pad=0.02)
cb.set_label('T of Pierru bulk solid Rw + Gt (K)', fontsize=14)
cb.ax.tick_params(labelsize=14)

ax.set_ylim(P_MAX, P_MIN)                # pressure increases downward
ax.set_xlim(3.6, 5.0)
ax.set_xlabel(r'density (g cm$^{-3}$)', fontsize=14)
ax.set_ylabel('P (GPa)', fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(loc='lower left', fontsize=8.5)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '139b_pierru_fig6_with_a60.png'), dpi=200)
print("saved 139b_pierru_fig6_with_a60.png")