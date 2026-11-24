#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:34:48 2026

@author: chingchen
"""

import os
import glob
import json
import types
import numpy as np
import matplotlib.pyplot as plt
from mars_config import KHAN_MODEL_DIR, MCMC_DIR, FIG_DIR, SAMUEL_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

# ── config ────────────────────────────────────────────────────────────────────
RUN            = 'a64'       # 'a60' for the old runs (YM20, TA, EH45)
FORWARD_FILE   = f'{HERE}/26fall_main_mcmc_a63.py'   # a60: 26fall_main_mcmc.py
CHAIN_ROOT     = MCMC_DIR
SAMUEL_BASE    = (f'{SAMUEL_DIR}/METADATA_BML/DATA_FIG1')
COMPS          = ['YM20']    # a64 is YM20 only; a60: ['YM20', 'TA', 'EH45']
EXCLUDE        = []          # frozen chains, e.g. 'chain_a60_TA_05'
BURN_FRAC      = 0.40
PROFILE_STRIDE = 5           # use every Nth unique accepted profile
P_GRID         = np.linspace(0.5, 24.0, 201)
SOL_SIGMA      = 12.0        # K, SOLIDUS_SIGMA in the MCMC
MG_PIERRU      = 0.75        # Pierru starting Mg#
G_CONST        = 6.674e-11
RHO_SIL_MAX    = 5.5         # g/cm3, silicate part only (drop the core) for Khan / Samuel
# Pierru 2026 SI Table 4, corrected solidus temperatures (P GPa, T K)
PIERRU_PTS = np.array([[7.95, 1900.], [9.6, 1960.], [14.0, 2090.], [17.7, 2200.], [18.0, 2223.]])
SHALLOW_P  = (2.0, 8.0)      # GPa, zoom window for the second figure
SHALLOW_T  = (1500, 2100)    # K
# Pierru 2026 has no geotherm of its own: Fig. 5 compiles published profiles and states that
# temperatures near the CMB lie at ~1900-2200 K (box below, over the BML pressure range);
# Fig. S9 uses a melt-temperature profile through the S3631 production point (2223 K, ~18 GPa).
PIERRU_TCMB  = (1900.0, 2200.0)   # K
PIERRU_PBML  = (18.0, 22.0)       # GPa, approximate BML pressure range for the box
PIERRU_MELT  = (2223.0, 18.0)     # (T K, P GPa)

# ── Duncan 2018 + Fe correction, exactly as in the MCMC ───────────────────────
src = open(FORWARD_FILE).read()
i0 = src.index('# ── physics: solidus')
i1 = src.index('# ── BML phase diagram')
m = types.ModuleType('sol')
m.__dict__.update(np=np)
exec(src[i0:i1], m.__dict__)
# Pierru 2026 Simon-Glatzel fit, Mg# 0.75; the same Fe correction as the MCMC (-6 K per Fe%)
#   T_S = 1400 (1 + P/2.117127)^(1/4.967588) - 6 (Fe - Fe_ref),  Fe = 100 (1 - Mg#)

# ── Khan 2023 ensemble: columns 0 depth, 1 Vp, 2 Vs, 3 rho (g/cm3), 4 T, 5 g, 6 P (Pa) ──
KT = []
for fpath in sorted(glob.glob(os.path.join(KHAN_MODEL_DIR, 'Model_*.txt'))):
    dk = np.loadtxt(fpath, comments='#')
    P_k = dk[:, 6] / 1e9
    sil = dk[:, 3] < RHO_SIL_MAX
    o = np.argsort(P_k[sil])
    KT.append(np.interp(P_GRID, P_k[sil][o], dk[sil, 4][o], left=np.nan, right=np.nan))
KT = np.array(KT)
KTQ = np.nanpercentile(KT, [16, 50, 84], axis=0)
print(f"Khan 2023: {len(KT)} models; T at 19 GPa median {np.interp(19, P_GRID, KTQ[1]):.0f} K")

# ── Samuel 2023: P(r) from the density profile, then T(r) -> T(P) ─────────────
#   M(r) = int 4 pi r^2 rho dr,  g = G M / r^2,  dP = rho g dr  (same as the forward model)
s = np.loadtxt(os.path.join(SAMUEL_BASE, 'PANEL_J', 'rho_profile.dat'))     # rho (kg/m3), radius (km)
o = np.argsort(s[:, 1])
rho_s, r_s = s[o, 0], s[o, 1] * 1e3
M_s = np.concatenate([[0.0], np.cumsum(4 * np.pi * 0.5 * (rho_s[1:] + rho_s[:-1])
                                        * (0.5 * (r_s[1:] + r_s[:-1]))**2 * np.diff(r_s))])
g_s = np.zeros(len(r_s))
g_s[1:] = G_CONST * M_s[1:] / r_s[1:]**2
dP_s = 0.5 * (rho_s[1:] * g_s[1:] + rho_s[:-1] * g_s[:-1]) * np.diff(r_s)
P_s = np.zeros(len(r_s))
P_s[:-1] = np.cumsum(dP_s[::-1])[::-1] / 1e9            # integrate from the surface (P = 0) down
r_cmb_s = r_s[rho_s / 1000 >= RHO_SIL_MAX].max()        # top of Samuel's metal core (m)
st = np.loadtxt(os.path.join(SAMUEL_BASE, 'PANEL_I', 'Tprofile.dat'))     # T (K), radius (km)
sil_T = st[:, 1] * 1e3 > r_cmb_s
T_sam = st[sil_T, 0]
P_sam = np.interp(st[sil_T, 1] * 1e3, r_s, P_s)
o = np.argsort(P_sam)
T_sam, P_sam = T_sam[o], P_sam[o]
print(f"Samuel 2023: silicate base at {P_sam.max():.1f} GPa, T there {T_sam[-1]:.0f} K")

fig, ax = plt.subplots(1, len(COMPS), figsize=(6 * len(COMPS), 7.5), sharey=True)
ax = np.atleast_1d(ax)
for k, comp in enumerate(COMPS):
    # read chains: all records after burn-in, each sits on the last accepted state
    weights = {}
    info = {}
    for path in sorted(glob.glob(os.path.join(CHAIN_ROOT, f'chain_{RUN}_{comp}_*', 'chain.jsonl'))):
        chain_dir = os.path.dirname(path)
        if os.path.basename(chain_dir) in EXCLUDE:
            continue
        lines = [json.loads(l) for l in open(path) if l.strip().endswith('}')]   # skip a half-written last line
        n_burn = int(BURN_FRAC * len(lines))
        last_acc = None
        for i, rec in enumerate(lines):
            if rec['accepted']:
                last_acc = rec['step']
            if i >= n_burn and last_acc is not None:
                key = (chain_dir, last_acc)
                weights[key] = weights.get(key, 0) + 1
                info[key] = (rec['params']['Mg#'], rec['params']['P_lit'])
    keys = sorted(weights)[::PROFILE_STRIDE]

    TM, TB, W, EX_P, EX_D, MG, PCMB = [], [], [], [], [], [], []
    for key in keys:
        f = os.path.join(key[0], f"profile_s{key[1]:05d}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        P, T, lay = d['P_GPa'], d['T_K'], d['layer']
        mg, P_lit = info[key]
        mm = lay == 1
        mb = lay == 2
        TM.append(np.interp(P_GRID, P[mm], T[mm], left=np.nan, right=np.nan))
        TB.append(np.interp(P_GRID, P[mb], T[mb], left=np.nan, right=np.nan))
        # solidus excess in the adiabatic mantle (P >= P_lit), as in compute_solidus_penalty
        s_ = mm & (P >= P_lit)
        Tp = 1400.0 * (1 + P[s_] / 2.117127) ** (1 / 4.967588) - 6.0 * (100 * (1 - mg) - 100 * (1 - MG_PIERRU))
        Td = np.array([m.solidus_duncan_Mg(p, mg) for p in P[s_]])
        EX_P.append(np.max(T[s_] - Tp))
        EX_D.append(np.max(T[s_] - Td))
        W.append(weights[key]); MG.append(mg); PCMB.append(P[mb].max())
    TM, TB = np.array(TM), np.array(TB)
    W, EX_P, EX_D = np.array(W, float), np.array(EX_P), np.array(EX_D)
    MG, PCMB = np.array(MG), np.array(PCMB)

    # weighted 5/50/95 percentiles on the P grid
    QM = np.full((3, len(P_GRID)), np.nan)
    QB = np.full((3, len(P_GRID)), np.nan)
    for j in range(len(P_GRID)):
        for Tarr, Q in [(TM, QM), (TB, QB)]:
            v = Tarr[:, j]
            ok = np.isfinite(v)
            if ok.sum() < 10:
                continue
            o = np.argsort(v[ok])
            cw = np.cumsum(W[ok][o]); cw /= cw[-1]
            Q[:, j] = np.interp([0.05, 0.50, 0.95], cw, v[ok][o])

    mg_med = np.median(np.repeat(MG, W.astype(int)))
    print(f"\n== {comp}: {len(W)} profiles ({len(weights)} unique states, stride {PROFILE_STRIDE}), "
          f"median mantle Mg# {mg_med:.3f}, P_cmb {PCMB.min():.1f}-{PCMB.max():.1f} GPa")
    for lab, ex in [('Duncan 2018 + Fe (in likelihood)', EX_D), ('Pierru 2026 + same Fe corr.', EX_P)]:
        print(f"  {lab:34s}: above solidus {np.sum(W[ex > 0])/W.sum()*100:5.1f}%  "
              f"above by > {SOL_SIGMA:.0f} K {np.sum(W[ex > SOL_SIGMA])/W.sum()*100:5.1f}%  "
              f"max excess median {np.median(np.repeat(ex, W.astype(int))):+.0f} K")

    # panel: x = T, y = P (increasing downward)
    a = ax[k]
    a.fill_betweenx(P_GRID, KTQ[0], KTQ[2], color='0.6', alpha=0.25, lw=0)
    a.plot(KTQ[1], P_GRID, '-', color='0.4', lw=1.6, label='Khan 2023 (median, 16-84%)')
    a.plot(T_sam, P_sam, '--', color='0.3', lw=1.6, label='Samuel 2023')
    a.fill_betweenx(P_GRID, QM[0], QM[2], color='C0', alpha=0.25, lw=0)
    a.plot(QM[1], P_GRID, color='C0', lw=2.0, label=f'{RUN} mantle (median, 5-95%)')
    a.fill_betweenx(P_GRID, QB[0], QB[2], color='C3', alpha=0.25, lw=0)
    a.plot(QB[1], P_GRID, color='C3', lw=2.0, label=f'{RUN} BML (median, 5-95%)')
    Pd = np.linspace(1.0, 24.0, 116)
    a.plot([m.solidus_duncan_Mg(p, mg_med) for p in Pd], Pd, '--', color='darkgreen', lw=1.6,
           label=f'Duncan 2018 + Fe (Mg# {mg_med:.2f})')
    Tpd = 1400.0 * (1 + Pd / 2.117127) ** (1 / 4.967588) - 6.0 * (100 * (1 - mg_med) - 100 * (1 - MG_PIERRU))
    a.plot(Tpd[Pd <= 19.0], Pd[Pd <= 19.0], '-', color='limegreen', lw=1.8,
           label=f'Pierru 2026 + Fe (Mg# {mg_med:.2f})')
    a.plot(Tpd[Pd >= 19.0], Pd[Pd >= 19.0], ':', color='limegreen', lw=1.8)
    a.plot(PIERRU_PTS[:, 1], PIERRU_PTS[:, 0], 'D', color='limegreen', mec='k', ms=6,
           label='Pierru solidus data (Mg# 0.75)')
    a.add_patch(plt.Rectangle((PIERRU_TCMB[0], PIERRU_PBML[0]), PIERRU_TCMB[1] - PIERRU_TCMB[0],
                              PIERRU_PBML[1] - PIERRU_PBML[0], fill=False, ec='m', lw=1.5, ls='--',
                              label='Pierru 2026: near-CMB T from published models'))
    a.plot(*PIERRU_MELT, '*', color='m', ms=12, mec='k', label='Pierru S3631 melt production (2223 K)')
    a.set_xlim(1400, 3300)
    a.set_xlabel('T (K)', fontsize=14)
    a.set_title(f'{RUN} {comp}', fontsize=14)
    a.tick_params(labelsize=14)

ax[0].set_ylim(P_GRID[-1], P_GRID[0])    # pressure increases downward (shared y)
ax[0].set_ylabel('P (GPa)', fontsize=14)
ax[0].legend(loc='upper right', fontsize=9)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, f'140_{RUN}_geotherm_solidus.png'), dpi=200)
print(f"\nsaved 140_{RUN}_geotherm_solidus.png")

# second figure: shallow zoom, where the solidus penalty acts
for a in ax:
    a.set_xlim(*SHALLOW_T)
ax[0].set_ylim(SHALLOW_P[1], SHALLOW_P[0])
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, f'140_{RUN}_geotherm_solidus_shallow.png'), dpi=200)
print(f"saved 140_{RUN}_geotherm_solidus_shallow.png")

# restore the full range so the figure shown on screen is the full-depth one
for a in ax:
    a.set_xlim(1400, 3300)
ax[0].set_ylim(P_GRID[-1], P_GRID[0])
plt.show()