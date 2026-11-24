#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 40: first-order test of replacing the fixed Samuel 2023 0-100 km column with a crust of mean
#     thickness h and density rho_c (the gravity/topography route, Wieczorek 2022 ranges)
#
#   python 40.crust_column_mass.py                (SAVE_FIG=1 to save to FIG_DIR)
#
# For N_PROF saved posterior profiles per run (profile_sNNNNN.npz + the chain record of that step):
#   rho(z) above 100 km is replaced by rho_c for z < h and by the profile's uppermost-mantle
#   density for h <= z < 100 km; below 100 km nothing changes.
#   M_new = M_pred * M'/M0,  moi_new = moi_pred * (I'/I0) * (M0/M')   (shell integrals, ratios only,
#   so the integration grid error cancels against the stored M_pred / moi_pred).
# Ignored (first order): the pressure / gravity feedback on deeper layers.
# Output: median residual over profiles, in formal sigma, on the (h, rho_c) grid; the residual
# spread over the Wieczorek box gives a first estimate of a gravity-based column sigma.
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

RUNS = [('a74 Fe-S-C', 'chain_a74_YM20'), ('a73 Fe-S-H', 'chain_a73_YM20')]
N_PROF    = 300
SEED      = 0
R_MARS    = 3389.5          # km
Z_FIX     = 100.0           # km, bottom of the fixed column
MASS_OBS  = 6.4171e23
MOI_OBS   = 0.3634
MASS_FORM = 3.1e-4
MOI_FORM  = 6.0e-5
H_GRID    = np.arange(30.0, 72.1, 2.0)          # Wieczorek 2022 mean thickness 30-72 km
RHO_GRID  = np.arange(2550.0, 3100.1, 25.0)     # Goossens 2017 2582 +- 209; Wieczorek max 2850-3100
# Wieczorek 2022 uniform-density models (their Fig. 3 text, 39-km InSight anchor): average thickness
# rises from ~48 km at rho_c 2550 to ~59 km at ~3050 (max permissible). Anchor 39 +- 8 km scales the
# curve (their full 1-sigma range 39-72 km). Read from the text, not digitized: approximate.
W_RHO  = (2550.0, 3050.0)
W_H39  = (48.0, 59.0)
W_ANCH = (39.0, 8.0)
N_CRUST = 4000
def h_wieczorek(rho_c, anchor):
    return (W_H39[0] + (W_H39[1] - W_H39[0])*(rho_c - W_RHO[0])/(W_RHO[1] - W_RHO[0]))*anchor/W_ANCH[0]
REF = {'Samuel 2023': (66.7, 3025.0), 'Kim 2023': (48.0, 2650.0)}
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'
FS = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

zf = np.linspace(0.0, R_MARS, 20001)            # fine radial grid (depth, km)
rf = (R_MARS - zf)*1e3                            # radius, m
dz = (zf[1] - zf[0])*1e3
w_m = 4*np.pi*rf**2*dz                            # mass weight per (kg/m3)
w_i = (8*np.pi/3)*rf**4*dz                        # moment weight per (kg/m3)
top = zf < Z_FIX


def load_jsonl(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break
        out[r.get('step')] = r
    return out


rng = np.random.default_rng(SEED)
res = {}
for lab, prefix in RUNS:
    items = []
    for d in sorted(glob.glob(os.path.join(MCMC_DIR, prefix + '_[0-9][0-9]'))):
        if not os.path.exists(os.path.join(d, 'chain.jsonl')):
            continue
        recs = load_jsonl(os.path.join(d, 'chain.jsonl'))
        for f in glob.glob(os.path.join(d, 'profile_s*.npz')):
            s = int(os.path.basename(f)[9:14])
            if s in recs and s > 300 and recs[s].get('M_pred') is not None:
                items.append((f, recs[s]))
    items = [items[i] for i in rng.choice(len(items), size=min(N_PROF, len(items)), replace=False)]
    dm = np.full((len(items), len(H_GRID), len(RHO_GRID)), np.nan)
    di = np.full_like(dm, np.nan)
    dm0, di0 = np.zeros(len(items)), np.zeros(len(items))
    for k, (f, rec) in enumerate(items):
        p = np.load(f)
        rho = np.interp(zf, p['z_km'], p['rho'])*1e3
        M0, I0 = np.sum(rho*w_m), np.sum(rho*w_i)
        rho_man = rho[np.argmin(np.abs(zf - Z_FIX + 1.0))]     # uppermost mantle density just above 100 km
        dm0[k] = rec['M_pred']/MASS_OBS - 1
        di0[k] = rec['moi_pred'] - MOI_OBS
        for a, h in enumerate(H_GRID):
            for b, rc in enumerate(RHO_GRID):
                rn = rho.copy()
                rn[top] = np.where(zf[top] < h, rc, rho_man)
                M1, I1 = np.sum(rn*w_m), np.sum(rn*w_i)
                Mn = rec['M_pred']*M1/M0
                mn = rec['moi_pred']*(I1/I0)*(M0/M1)
                dm[k, a, b] = Mn/MASS_OBS - 1
                di[k, a, b] = mn - MOI_OBS
    res[lab] = dict(dm=dm, di=di, dm0=dm0, di0=di0)
    print(f'\n{lab}: {len(items)} profiles')
    print(f'  current (Samuel column)   mass {np.median(dm0)/MASS_FORM:+6.1f}  MoI {np.median(di0)/MOI_FORM:+6.1f}  (formal sigma, median)')
    for nm, (h, rc) in REF.items():
        a, b = np.argmin(np.abs(H_GRID - h)), np.argmin(np.abs(RHO_GRID - rc))
        print(f'  {nm:<12} h={H_GRID[a]:.0f} rho={RHO_GRID[b]:.0f}  mass {np.median(dm[:, a, b])/MASS_FORM:+6.1f}'
              f'  MoI {np.median(di[:, a, b])/MOI_FORM:+6.1f}')
    zm, zi = np.median(dm, axis=0)/MASS_FORM, np.median(di, axis=0)/MOI_FORM
    a, b = np.unravel_index(np.argmin(zm**2 + zi**2), zm.shape)
    print(f'  best (h, rho) for the median model: h={H_GRID[a]:.0f} km rho={RHO_GRID[b]:.0f}  '
          f'mass {zm[a, b]:+.1f}  MoI {zi[a, b]:+.1f}')
    sm = np.std(np.median(dm, axis=0))
    si = np.std(np.median(di, axis=0))
    from scipy.interpolate import RegularGridInterpolator
    rc_s = rng.uniform(*W_RHO, N_CRUST)
    an_s = rng.normal(*W_ANCH, N_CRUST)
    h_s  = np.clip(h_wieczorek(rc_s, an_s), H_GRID[0], H_GRID[-1])
    fm = RegularGridInterpolator((H_GRID, RHO_GRID), np.median(dm, axis=0))
    fi = RegularGridInterpolator((H_GRID, RHO_GRID), np.median(di, axis=0))
    vm, vi = fm(np.c_[h_s, rc_s]), fi(np.c_[h_s, rc_s])
    print(f'  Wieczorek curve (rho_c U[2550,3050], anchor 39+-8): mass {np.median(vm)/MASS_FORM:+.1f} '
          f'sd {np.std(vm)/MASS_FORM:.1f} formal ({100*np.std(vm):.3f} %)   '
          f'MoI {np.median(vi)/MOI_FORM:+.1f} sd {np.std(vi)/MOI_FORM:.1f} formal ({np.std(vi):.2e})')
    for a0 in [31.0, 39.0, 47.0]:
        rr = np.linspace(*W_RHO, 501); hh = np.clip(h_wieczorek(rr, a0), H_GRID[0], H_GRID[-1])
        m_, i_ = fm(np.c_[hh, rr])/MASS_FORM, fi(np.c_[hh, rr])/MOI_FORM
        k = np.argmin(m_**2 + i_**2)
        print(f'    anchor {a0:.0f} km: best rho_c {rr[k]:.0f}, h {hh[k]:.1f} km  mass {m_[k]:+.1f}  MoI {i_[k]:+.1f}'
              f'   (mass=0 at rho {rr[np.argmin(np.abs(m_))]:.0f}, MoI=0 at rho {rr[np.argmin(np.abs(i_))]:.0f})')
    print(f'  spread over the whole (h, rho) box (uniform): mass {100*sm:.3f} %  ({sm/MASS_FORM:.1f} formal)  '
          f'MoI {si:.2e}  ({si/MOI_FORM:.1f} formal)   [current sigma: 0.27 %, 6.4e-4]')

fig, axs = plt.subplots(2, len(res), figsize=(7*len(res), 11), squeeze=False)
for j, (lab, v) in enumerate(res.items()):
    for i, (key, form, nm) in enumerate([('dm', MASS_FORM, 'mass'), ('di', MOI_FORM, 'MoI')]):
        ax = axs[i, j]
        Z = np.median(v[key], axis=0)/form
        lim = np.nanmax(np.abs(Z))
        pc = ax.pcolormesh(RHO_GRID, H_GRID, Z, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
        cs = ax.contour(RHO_GRID, H_GRID, Z, levels=[-4, 0, 4], colors='k', linewidths=[1, 2, 1])
        ax.clabel(cs, fontsize=FS - 4)
        for a0, lsw in [(31.0, ':'), (39.0, '-'), (47.0, ':')]:
            rr = np.linspace(*W_RHO, 50)
            ax.plot(rr, h_wieczorek(rr, a0), color='#2CA02C', ls=lsw, lw=2.5,
                    label=f'Wieczorek 2022, InSight {a0:.0f} km' if (i == 0 and j == 0) else None)
        for nmr, (h, rc) in REF.items():
            ax.plot(rc, h, 'k*', ms=12)
            ax.annotate(nmr, (rc, h), xytext=(5, 5), textcoords='offset points', fontsize=FS - 3)
        cb = fig.colorbar(pc, ax=ax)
        cb.set_label(f'median {nm} residual [formal sigma]', fontsize=FS - 2)
        cb.ax.tick_params(labelsize=14)
        ax.set_xlabel(r'crust density $\rho_c$ [kg/m$^3$]', fontsize=FS)
        ax.set_ylabel('mean crustal thickness h [km]', fontsize=FS)
        ax.set_title(f'{lab}: {nm}', fontsize=FS)
        ax.tick_params(labelsize=14)
        ax.set_ylim(H_GRID[0], H_GRID[-1])
axs[0, 0].legend(loc='lower right', fontsize=FS - 4)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(os.path.join(FIG_DIR, '40_crust_column_mass.png'), dpi=200)
plt.show()
