#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 40: first-order test of replacing the fixed Samuel 2023 0-100 km column with a crust of mean
#     thickness h and density rho_c (the gravity/topography route, Wieczorek 2022)
#
#   python 40.crust_column_mass.py                (SAVE_FIG=1 to save to FIG_DIR)
#
# For N_PROF saved posterior profiles per run (profile_sNNNNN.npz + the chain record of that step):
#   rho(z) above 100 km is replaced by rho_c for z < h and by rho_m for h <= z < 100 km;
#   below 100 km nothing changes. rho_m (uppermost mantle) is a choice, looped over RHO_M_LIST,
#   and the same rho_m picks the Wieczorek curve, so the curve and the column agree.
#   M_new = M_pred * M'/M0,  moi_new = moi_pred * (I'/I0) * (M0/M')   (shell integrals, ratios only,
#   so the integration grid error cancels against the stored M_pred / moi_pred).
#   Only the top 100 km changes: M' = M0 + sum((rho' - rho) w_m) over z < 100 km (same for I).
# Ignored (first order): the pressure / gravity feedback on deeper layers; the core and mantle
#   parameters are those of a chain run with loose sigma (no re-fit), so residuals are upper bounds.
#
# Wieczorek 2022 curve from 97 (wieczorek2022_fit.json, their Fig. 4):
#   h(rho_c; rho_m, a) = c0 + c1*x + c2*x^2,  x = 1000/(rho_m - rho_c), linear in anchor a
#   rho_c in [2550, rho_c,max],  rho_c,max = rho_m - drho0(a)
# Per-profile search (the planned per-forward crust adjustment): along the curve for anchor a,
#   pick rho_c minimizing |dM/sig_M| + |dI/sig_I| (L1, formal sigma), report the residual left.
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mars_config import MCMC_DIR, FIG_DIR

RUNS = [('a74 Fe-S-C', 'chain_a74_YM20'), ('a73 Fe-S-H', 'chain_a73_YM20')]
N_PROF    = 300
SEED      = 0
R_MARS    = 3389.5          # km
Z_FIX     = 100.0           # km, bottom of the fixed column
MASS_OBS  = 6.4171e23
MOI_OBS   = 0.3634
MASS_FORM = 3.1e-4          # mean density 3935 +- 1.2 kg/m3, relative
MOI_FORM  = 6.0e-5          # Konopliv 2020
H_GRID    = np.arange(30.0, 80.1, 2.0)          # km
RHO_GRID  = np.arange(2550.0, 3200.1, 25.0)     # kg/m3; Goossens 2017 2582 +- 209
RHO_M_LIST = [3382.0, 3446.0]                   # Khan2022 (Wieczorek Table 2); Samuel 2023 column 66.7-100 km
ANCH_LIST = [31.0, 39.0, 47.0]                  # InSight 39 +- 8 km (Knapmeyer-Endrun 2021)
W_ANCH    = (39.0, 8.0)
W_RHO_MIN = 2550.0
N_SEARCH  = 201
N_CRUST   = 4000
WF   = json.load(open('wieczorek2022_fit.json'))
W_A  = np.array(WF['anchors'], float)
W_C  = np.array([WF['c_avg'][str(int(a))] for a in W_A])
W_D0 = np.array([WF['drho0'][str(int(a))] for a in W_A])
def w_idx(anchor):                               # bracketing anchors; linear extrapolation outside 20-45
    a = np.asarray(anchor, float)
    i = np.clip(np.searchsorted(W_A, a) - 1, 0, len(W_A) - 2)
    return i, (a - W_A[i])/(W_A[i + 1] - W_A[i])
def h_wieczorek(rho_c, rho_m, anchor):
    x = 1e3/(rho_m - np.asarray(rho_c, float))
    i, f = w_idx(anchor)
    h0 = W_C[i, 0] + W_C[i, 1]*x + W_C[i, 2]*x**2
    h1 = W_C[i + 1, 0] + W_C[i + 1, 1]*x + W_C[i + 1, 2]*x**2
    return h0 + f*(h1 - h0)
def rho_c_max(rho_m, anchor):
    i, f = w_idx(anchor)
    return rho_m - (W_D0[i] + f*(W_D0[i + 1] - W_D0[i]))
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
zt, wmt, wit = zf[top], w_m[top], w_i[top]
HH, RR = np.meshgrid(H_GRID, RHO_GRID, indexing='ij')


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
    nk, nm, na = len(items), len(RHO_M_LIST), len(ANCH_LIST)
    dm = np.full((nm, nk, len(H_GRID), len(RHO_GRID)), np.nan)
    di = np.full_like(dm, np.nan)
    best = np.full((nm, na, nk, 5), np.nan)       # rho_c, h, z_mass, z_moi, at_bound (0 no, -1 low, +1 high)
    dm0, di0, rho99 = np.zeros(nk), np.zeros(nk), np.zeros(nk)
    for k, (f, rec) in enumerate(items):
        p = np.load(f)
        rho = np.interp(zf, p['z_km'], p['rho'])*1e3
        M0, I0 = np.sum(rho*w_m), np.sum(rho*w_i)
        rt = rho[top]
        rho99[k] = rho[np.argmin(np.abs(zf - Z_FIX + 1.0))]     # stored profile just above 100 km
        dm0[k] = rec['M_pred']/MASS_OBS - 1
        di0[k] = rec['moi_pred'] - MOI_OBS
        for j, rm in enumerate(RHO_M_LIST):
            # grid: column = rho_c above h, rho_m from h to 100 km
            cols = np.where(zt[None, :] < HH.ravel()[:, None], RR.ravel()[:, None], rm)
            M1 = M0 + (cols - rt) @ wmt
            I1 = I0 + (cols - rt) @ wit
            dm[j, k] = (rec['M_pred']*M1/M0/MASS_OBS - 1).reshape(HH.shape)
            di[j, k] = (rec['moi_pred']*(I1/I0)*(M0/M1) - MOI_OBS).reshape(HH.shape)
            # search along the Wieczorek curve of each anchor
            for ia, a0 in enumerate(ANCH_LIST):
                rr = np.linspace(W_RHO_MIN, rho_c_max(rm, a0), N_SEARCH)
                hh = h_wieczorek(rr, rm, a0)
                cols = np.where(zt[None, :] < hh[:, None], rr[:, None], rm)
                M1 = M0 + (cols - rt) @ wmt
                I1 = I0 + (cols - rt) @ wit
                zm = (rec['M_pred']*M1/M0/MASS_OBS - 1)/MASS_FORM
                zi = (rec['moi_pred']*(I1/I0)*(M0/M1) - MOI_OBS)/MOI_FORM
                b = np.argmin(np.abs(zm) + np.abs(zi))
                best[j, ia, k] = rr[b], hh[b], zm[b], zi[b], (-1 if b == 0 else (1 if b == N_SEARCH - 1 else 0))
    res[lab] = dict(dm=dm, di=di, dm0=dm0, di0=di0, best=best)

    print(f'\n================ {lab}: {nk} profiles ================')
    print(f'  current (Samuel column)   mass {np.median(dm0)/MASS_FORM:+6.1f}  MoI {np.median(di0)/MOI_FORM:+6.1f}'
          f'  (formal sigma, median);  stored profile rho at 99 km: median {np.median(rho99):.0f} kg/m3')
    for j, rm in enumerate(RHO_M_LIST):
        print(f'\n  --- rho_m = {rm:.0f} (fills h-100 km, picks the Wieczorek curve) ---')
        for nmr, (h, rc) in REF.items():
            a, b = np.argmin(np.abs(H_GRID - h)), np.argmin(np.abs(RHO_GRID - rc))
            print(f'  {nmr:<12} h={H_GRID[a]:.0f} rho={RHO_GRID[b]:.0f}  mass {np.median(dm[j, :, a, b])/MASS_FORM:+6.1f}'
                  f'  MoI {np.median(di[j, :, a, b])/MOI_FORM:+6.1f}')
        fm = RegularGridInterpolator((H_GRID, RHO_GRID), np.median(dm[j], axis=0))
        fi = RegularGridInterpolator((H_GRID, RHO_GRID), np.median(di[j], axis=0))
        print('  median model along the curve (L1 best; where mass = 0 and MoI = 0):')
        for a0 in ANCH_LIST:
            rr = np.linspace(W_RHO_MIN, rho_c_max(rm, a0), 501)
            hh = np.clip(h_wieczorek(rr, rm, a0), H_GRID[0], H_GRID[-1])
            m_, i_ = fm(np.c_[hh, rr])/MASS_FORM, fi(np.c_[hh, rr])/MOI_FORM
            k = np.argmin(np.abs(m_) + np.abs(i_))
            zm0 = rr[np.argmin(np.abs(m_))] if m_.min()*m_.max() < 0 else np.nan
            zi0 = rr[np.argmin(np.abs(i_))] if i_.min()*i_.max() < 0 else np.nan
            print(f'    anchor {a0:.0f}: rho_c {rr[k]:.0f} (max {rr[-1]:.0f}), h {hh[k]:.1f} km  mass {m_[k]:+.1f}  MoI {i_[k]:+.1f}'
                  f'   (mass=0 at {zm0:.0f}, MoI=0 at {zi0:.0f}; nan = no crossing on the curve)')
        print('  per-profile search (formal sigma): rho_c 16/50/84 | h 50 | at bound low/high | '
              'mass 16/50/84 | MoI 16/50/84 | P(both<1,2,4) | P(|MoI|<4)')
        for ia, a0 in enumerate(ANCH_LIST):
            B = best[j, ia]
            q = lambda v: '/'.join(f'{x:+.1f}' for x in np.percentile(v, [16, 50, 84]))
            mx = np.maximum(np.abs(B[:, 2]), np.abs(B[:, 3]))
            print(f'    anchor {a0:.0f}: {"/".join(f"{x:.0f}" for x in np.percentile(B[:, 0], [16, 50, 84]))} | '
                  f'{np.median(B[:, 1]):.1f} | {np.mean(B[:, 4] < 0):.2f}/{np.mean(B[:, 4] > 0):.2f} | '
                  f'{q(B[:, 2])} | {q(B[:, 3])} | '
                  f'{np.mean(mx < 1):.2f} {np.mean(mx < 2):.2f} {np.mean(mx < 4):.2f} | {np.mean(np.abs(B[:, 3]) < 4):.2f}')
        # spread of the median model over the Wieczorek family: anchor N(39, 8) clipped at 2 sigma,
        # rho_c uniform on [2550, rho_c,max(anchor)]
        an_s = np.clip(rng.normal(*W_ANCH, N_CRUST), W_ANCH[0] - 2*W_ANCH[1], W_ANCH[0] + 2*W_ANCH[1])
        rc_s = W_RHO_MIN + rng.uniform(0, 1, N_CRUST)*(rho_c_max(rm, an_s) - W_RHO_MIN)
        h_s = np.clip(h_wieczorek(rc_s, rm, an_s), H_GRID[0], H_GRID[-1])
        vm, vi = fm(np.c_[h_s, rc_s]), fi(np.c_[h_s, rc_s])
        print(f'  spread over the Wieczorek family: mass sd {100*np.std(vm):.3f} % ({np.std(vm)/MASS_FORM:.1f} formal)'
              f'   MoI sd {np.std(vi):.2e} ({np.std(vi)/MOI_FORM:.1f} formal)   [current sigma: 0.27 %, 6.4e-4]')

for j, rm in enumerate(RHO_M_LIST):
    fig, axs = plt.subplots(2, len(res), figsize=(7*len(res), 11), squeeze=False)
    for jj, (lab, v) in enumerate(res.items()):
        for i, (key, form, nm) in enumerate([('dm', MASS_FORM, 'mass'), ('di', MOI_FORM, 'MoI')]):
            ax = axs[i, jj]
            Z = np.median(v[key][j], axis=0)/form
            lim = np.nanmax(np.abs(Z))
            pc = ax.pcolormesh(RHO_GRID, H_GRID, Z, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
            cs = ax.contour(RHO_GRID, H_GRID, Z, levels=[-4, 0, 4], colors='k', linewidths=[1, 2, 1])
            ax.clabel(cs, fontsize=FS - 4)
            for a0, lsw in [(31.0, ':'), (39.0, '-'), (47.0, ':')]:
                rr = np.linspace(W_RHO_MIN, rho_c_max(rm, a0), 50)
                ax.plot(rr, h_wieczorek(rr, rm, a0), color='#2CA02C', ls=lsw, lw=2.5,
                        label=f'Wieczorek 2022, InSight {a0:.0f} km' if (i == 0 and jj == 0) else None)
            ia = ANCH_LIST.index(39.0)
            ax.plot(v['best'][j, ia, :, 0], v['best'][j, ia, :, 1], '.', color='k', ms=2, alpha=0.3,
                    label='per-profile best, 39 km' if (i == 0 and jj == 0) else None)
            for nmr, (h, rc) in REF.items():
                ax.plot(rc, h, 'k*', ms=12)
                ax.annotate(nmr, (rc, h), xytext=(5, 5), textcoords='offset points', fontsize=FS - 3)
            cb = fig.colorbar(pc, ax=ax)
            cb.set_label(f'median {nm} residual [formal sigma]', fontsize=FS - 2)
            cb.ax.tick_params(labelsize=14)
            ax.set_xlabel(r'crust density $\rho_c$ [kg/m$^3$]', fontsize=FS)
            ax.set_ylabel('mean crustal thickness h [km]', fontsize=FS)
            ax.set_title(f'{lab}: {nm}, ' + r'$\rho_m$' + f' = {rm:.0f}', fontsize=FS)
            ax.tick_params(labelsize=14)
            ax.set_ylim(H_GRID[0], H_GRID[-1])
    axs[0, 0].legend(loc='lower right', fontsize=FS - 4)
    fig.tight_layout()
    if SAVE_FIG:
        fig.savefig(os.path.join(FIG_DIR, f'40_crust_column_mass_rm{rm:.0f}.png'), dpi=200)

# residual left after the per-profile search, anchor 39 km
fig, axs = plt.subplots(len(res), 2, figsize=(13, 5*len(res)), squeeze=False)
ia = ANCH_LIST.index(39.0)
for jj, (lab, v) in enumerate(res.items()):
    for i, nm in enumerate(['mass', 'MoI']):
        ax = axs[jj, i]
        for j, rm in enumerate(RHO_M_LIST):
            ax.hist(v['best'][j, ia, :, 2 + i], bins=np.arange(-15, 15.1, 0.5), histtype='step', lw=2,
                    label=r'$\rho_m$' + f' = {rm:.0f}')
        ax.axvspan(-1, 1, color='0.85', zorder=0)
        ax.set_xlabel(f'{nm} residual after rho_c search [formal sigma]', fontsize=FS)
        ax.set_ylabel('profiles', fontsize=FS)
        ax.set_title(f'{lab}, anchor 39 km', fontsize=FS)
        ax.tick_params(labelsize=14)
axs[0, 0].legend(loc='lower right', fontsize=FS - 2)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(os.path.join(FIG_DIR, '40_crust_search_residual.png'), dpi=200)
plt.show()
