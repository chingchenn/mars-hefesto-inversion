#!/usr/bin/env python3
"""
Plot nGibbs best-fit MCMC profile vs Khan et al. 2023 + HeFESTo + Samuel 2023
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
MCMC_BASE_DIR = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'
CHAIN_PREFIX  = 'chain_a10'
MODEL_DIR     = '/net/beno3/data1/jcchen2/Mars_Khan_2023/LSL_Models'
HEFESTO_FILE  = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc/chain_a07_26/step_00076/s3_final/fort.56'
SAMUEL_BASE   = '/net/beno3/data1/jcchen2/Mars_Samuel_2023/Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1'

DEPTH_MAX    = 1800
SPLICE_DEPTH = 200.0
MARS_RADIUS  = 3389.5
labelsize    = 13
bwith        = 2.0

colors = {
    'khan_vp':    '#4198B9',
    'khan_vs':    '#97795D',
    'khan_rho':   '#35838D',
    'ngibbs_vp':  '#C0392B',
    'ngibbs_vs':  '#8E44AD',
    'ngibbs_rho': '#27AE60',
    'ngibbs_T':   '#E67E22',
    'hefesto':    '#2C3E50',
    'samuel':     '#E74C3C',
    'samuel_T':   '#2980B9',
}

# ── LOAD SAMUEL 2023 ──────────────────────────────────────────────────────────
print("Loading Samuel 2023...")
def r2d(r): return MARS_RADIUS - r

samvp,    depvp_sam   = np.loadtxt(f'{SAMUEL_BASE}/PANEL_K/vp_profile.dat').T
samvs,    depvs_sam   = np.loadtxt(f'{SAMUEL_BASE}/PANEL_K/vs_profile.dat').T
rho_sam,  deprho_sam  = np.loadtxt(f'{SAMUEL_BASE}/PANEL_J/rho_profile.dat').T
Tprofile, depTprofile = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tprofile.dat').T
Tliq,     depTliq     = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tliq.dat').T
Tsol,     depTsol     = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tsol.dat').T

sam_vp_depth  = r2d(depvp_sam)
sam_vs_depth  = r2d(depvs_sam)
sam_rho_depth = r2d(deprho_sam)
sam_T_depth   = r2d(depTprofile)
sam_liq_depth = r2d(depTliq)
sam_sol_depth = r2d(depTsol)

samvp_kms   = samvp / 1000.0
samvs_kms   = samvs / 1000.0
sam_rho_gcc = rho_sam / 1000.0

# ── LOAD HeFESTo BEST FIT ─────────────────────────────────────────────────────
print("Loading HeFESTo best fit...")
with open(HEFESTO_FILE) as f:
    f.readline()
    cols = f.readline().split()
print(f"HeFESTo columns: {cols}")

df56 = pd.read_csv(HEFESTO_FILE, sep=r'\s+', skiprows=2, names=cols)
for col in cols:
    df56[col] = pd.to_numeric(df56[col], errors='coerce')
df56 = df56.dropna(subset=['P(GPa)', 'T(K)', 'rho(g/cm^3)', 'VS(km/s)', 'VP(km/s)'])

hef_P   = df56['P(GPa)'].values
hef_rho = df56['rho(g/cm^3)'].values
hef_Vs  = df56['VS(km/s)'].values
hef_Vp  = df56['VP(km/s)'].values
hef_T   = df56['T(K)'].values

g_mars    = 3.45
dP        = np.diff(hef_P) * 1e9
rho_mid   = (hef_rho[:-1] + hef_rho[1:]) / 2 * 1000
dz        = dP / (rho_mid * g_mars) / 1000
hef_depth = np.zeros(len(hef_P))
hef_depth[1:] = np.cumsum(dz)
print(f"HeFESTo: {hef_depth[0]:.0f}–{hef_depth[-1]:.0f} km")

# ── FIND BEST nGibbs PROFILE ACROSS ALL CHAINS ────────────────────────────────
print(f"Searching best model in {CHAIN_PREFIX}...")
all_chain_files = sorted(glob.glob(f'{MCMC_BASE_DIR}/{CHAIN_PREFIX}_*/chain.json'))

best_misfit    = 999
best_step      = None
best_chain_dir = None
best_params    = None
best_chain_obj = None

for fpath in all_chain_files:
    try:
        with open(fpath) as f:
            chain = json.load(f)
        for s in chain:
            if s.get('accepted') and s['misfit'] < best_misfit:
                best_misfit    = s['misfit']
                best_step      = s['step']
                best_chain_dir = os.path.dirname(fpath)
                best_params    = s['params']
                best_chain_obj = chain
    except Exception as e:
        print(f"Error {fpath}: {e}")

print(f"Best: chain={best_chain_dir}")
print(f"      step={best_step}, misfit={best_misfit:.4f}")
print(f"      params={best_params}")

profile_path = f'{best_chain_dir}/profile_s{best_step:05d}.npz'
best     = np.load(profile_path)
ng_depth = best['depth_km']
ng_Vp    = best['Vp']
ng_Vs    = best['Vs']
ng_rho   = best['rho']
ng_T     = best['T_K']
ng_P     = best['P_GPa']

# ── LOAD ALL 1000 KHAN MODELS ─────────────────────────────────────────────────
print("Loading Khan raw models...")
files = sorted(glob.glob(os.path.join(MODEL_DIR, 'Model_*.txt')))
khan_depth_list, khan_vp_list, khan_vs_list, khan_rho_list = [], [], [], []

for fpath in files:
    try:
        data  = np.loadtxt(fpath, comments='#')
        depth = data[:, 0]
        vp    = data[:, 1]
        vs    = data[:, 2]
        rho   = data[:, 3]
        mask  = (vs > 0.01) & (depth <= DEPTH_MAX)
        if mask.sum() < 5:
            continue
        khan_depth_list.append(depth[mask])
        khan_vp_list.append(vp[mask])
        khan_vs_list.append(vs[mask])
        khan_rho_list.append(rho[mask])
    except Exception:
        pass

n_models = len(khan_depth_list)
print(f'Khan: {n_models} models loaded')

Z_full = np.linspace(0, DEPTH_MAX, 500)

def make_stats(lst):
    mat = np.full((n_models, len(Z_full)), np.nan)
    for i, (z, v) in enumerate(zip(khan_depth_list, lst)):
        if z.max() < Z_full.max() * 0.8:
            continue
        mat[i] = np.interp(Z_full, z, v, left=np.nan, right=np.nan)
    return (np.nanmedian(mat, 0),
            np.nanpercentile(mat, 16, 0),
            np.nanpercentile(mat, 84, 0))

vp_med,  vp_p16,  vp_p84  = make_stats(khan_vp_list)
vs_med,  vs_p16,  vs_p84  = make_stats(khan_vs_list)
rho_med, rho_p16, rho_p84 = make_stats(khan_rho_list)

# ── BUILD FULL nGibbs PROFILE (Khan 0-200km + nGibbs 200km+) ─────────────────
mask_shallow = Z_full <= SPLICE_DEPTH
mask_deep    = ng_depth > SPLICE_DEPTH

full_depth = np.concatenate([Z_full[mask_shallow], ng_depth[mask_deep]])
full_Vp    = np.concatenate([vp_med[mask_shallow],  ng_Vp[mask_deep]])
full_Vs    = np.concatenate([vs_med[mask_shallow],  ng_Vs[mask_deep]])
full_rho   = np.concatenate([rho_med[mask_shallow], ng_rho[mask_deep]])

# ── HELPERS ───────────────────────────────────────────────────────────────────
def add_pressure_axis(ax):
    axr = ax.twinx()
    axr.set_ylim(DEPTH_MAX, 0)
    p_ticks = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    depth_at_p = np.interp(p_ticks, ng_P, ng_depth)
    axr.set_yticks(depth_at_p)
    axr.set_yticklabels([f'{p:.0f}' for p in p_ticks], color='gray', fontsize=9)
    axr.tick_params(axis='y', colors='gray', length=4)
    for sp in axr.spines.values():
        sp.set_linewidth(bwith)
    return axr

def setup_ax(ax, xlabel, show_ylabel=True):
    ax.set_ylim(DEPTH_MAX, 0)
    if show_ylabel:
        ax.set_ylabel('Depth (km)', fontsize=labelsize)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.grid(True, ls='--', alpha=0.3, lw=0.8)
    ax.tick_params(labelsize=labelsize)
    for sp in ax.spines.values():
        sp.set_linewidth(bwith)

def plot_khan_ensemble(ax, data_list, med, p16, p84, color):
    for i in range(n_models):
        ax.plot(data_list[i], khan_depth_list[i],
                color=color, alpha=0.04, lw=0.5, rasterized=True)
    ax.fill_betweenx(Z_full, p16, p84, color=color, alpha=0.20,
                     label='Khan 16–84%')
    ax.plot(med, Z_full, color=color, lw=2.0, label='Khan median')

splice_kw  = dict(color='gray', ls=':', lw=1.2, alpha=0.6)
hefesto_kw = dict(color=colors['hefesto'], lw=2.0, ls='-.', zorder=6)
samuel_kw  = dict(color=colors['samuel'],  lw=2.0, ls='-',  zorder=4)
ngibbs_kw  = dict(lw=2.2, ls='--', zorder=5)

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 9),
                          gridspec_kw={'wspace': 0.42})
fig.patch.set_facecolor('#FAFAFA')

# Panel 1: Vp
ax = axes[0]
setup_ax(ax, 'Vp (km/s)', show_ylabel=True)
plot_khan_ensemble(ax, khan_vp_list, vp_med, vp_p16, vp_p84, colors['khan_vp'])
ax.plot(samvp_kms, sam_vp_depth, **samuel_kw, label='Samuel 2023 (BML)')
ax.plot(full_Vp,   full_depth,   color=colors['ngibbs_vp'], **ngibbs_kw,
        label=f'nGibbs (step {best_step}, misfit={best_misfit:.2f})')
ax.plot(hef_Vp,    hef_depth,    **hefesto_kw,
        label='HeFESTo (a07 step76, misfit=0.70)')
ax.axhline(SPLICE_DEPTH, **splice_kw)
ax.legend(fontsize=8, loc='lower left', framealpha=0.85)
axr = add_pressure_axis(ax)
axr.set_ylabel('Pressure (GPa)', fontsize=10, color='gray')

# Panel 2: Vs
ax = axes[1]
setup_ax(ax, 'Vs (km/s)', show_ylabel=False)
plot_khan_ensemble(ax, khan_vs_list, vs_med, vs_p16, vs_p84, colors['khan_vs'])
ax.plot(samvs_kms, sam_vs_depth, **samuel_kw, label='Samuel 2023 (BML)')
ax.plot(full_Vs,   full_depth,   color=colors['ngibbs_vs'], **ngibbs_kw,
        label='nGibbs best')
ax.plot(hef_Vs,    hef_depth,    **hefesto_kw, label='HeFESTo best')
ax.axhline(SPLICE_DEPTH, **splice_kw)
ax.legend(fontsize=8, loc='lower left', framealpha=0.85)
add_pressure_axis(ax)

# Panel 3: density
ax = axes[2]
setup_ax(ax, 'ρ (g/cm³)', show_ylabel=False)
plot_khan_ensemble(ax, khan_rho_list, rho_med, rho_p16, rho_p84, colors['khan_rho'])
ax.plot(sam_rho_gcc, sam_rho_depth, **samuel_kw, label='Samuel 2023 (BML)')
ax.plot(full_rho,    full_depth,    color=colors['ngibbs_rho'], **ngibbs_kw,
        label='nGibbs best')
ax.plot(hef_rho,     hef_depth,     **hefesto_kw, label='HeFESTo best')
ax.axhline(SPLICE_DEPTH, **splice_kw)
ax.legend(fontsize=8, loc='lower right', framealpha=0.85)
add_pressure_axis(ax)

# Panel 4: Temperature
ax = axes[3]
setup_ax(ax, 'T (K)', show_ylabel=False)
ax.plot(Tprofile, sam_T_depth,   color=colors['samuel_T'], lw=2.0,
        label='Samuel 2023 T')
ax.plot(Tsol,     sam_sol_depth, color='orange', lw=1.5, ls='--',
        label='solidus')
ax.plot(Tliq,     sam_liq_depth, color='brown',  lw=1.5, ls='--',
        label='liquidus')
ax.plot(ng_T,  ng_depth,  color=colors['ngibbs_T'], lw=2.2, ls='--',
        zorder=5, label='nGibbs best')
ax.plot(hef_T, hef_depth, **hefesto_kw, label='HeFESTo best')

# 標記 P_lit
p_lit_val = best_params['P_lit']
d_lit_val = float(np.interp(p_lit_val, ng_P, ng_depth))
ax.axhline(d_lit_val, color=colors['ngibbs_T'], ls=':', lw=1.5, alpha=0.7)
ax.text(200, d_lit_val - 35,
        f"P_lit={p_lit_val:.1f} GPa", fontsize=7, color=colors['ngibbs_T'])

ax.axhline(SPLICE_DEPTH, **splice_kw)
ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
axr = add_pressure_axis(ax)
axr.set_ylabel('Pressure (GPa)', fontsize=10, color='gray')

params_str = (f"T_lit={best_params['T_lit']:.0f} K  "
              f"P_lit={best_params['P_lit']:.2f} GPa  "
              f"Mg#={best_params['Mg#']:.3f}")

fig.suptitle(
    f'Mars Mantle: nGibbs vs HeFESTo vs Samuel 2023 vs Khan et al. 2023 (n={n_models})\n'
    f'nGibbs best: {os.path.basename(best_chain_dir)} step {best_step} | '
    f'Misfit={best_misfit:.4f} | {params_str}\n'
    f'HeFESTo best: chain_a07_26 step 76 | Misfit=0.6993 | '
    f'T_lit=1877K  P_lit=8.46GPa  Mg#=0.806\n'
    f'(dotted = {SPLICE_DEPTH:.0f} km splice: Khan above, nGibbs below)',
    fontsize=11, y=1.00
)

out_path = f'{MCMC_BASE_DIR}/best_fit_{CHAIN_PREFIX}_comparison.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.show()