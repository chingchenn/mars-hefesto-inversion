# 20.plot_bml_result.py
# Vp/Vs, density and temperature vs depth: Khan 2023 LSL models (median, 16-84%),
# Samuel 2023, a reference HeFESTo run, and the best MCMC a04 model (mantle + BML).
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mars_config import KHAN_MODEL_DIR, RUN_BASE, SAMUEL_DIR

plt.rcParams['font.family'] = 'DejaVu Sans'

HEFESTO_FILE       = f'{RUN_BASE}/02_YM_bulk_mantle_BML_T/fort.56'
SAMUEL_BASE        = f'{SAMUEL_DIR}/METADATA_BML/DATA_FIG1/'
BEST_MANTLE_FORT56 = f'{RUN_BASE}/best_M04/files/mnt/user-data/outputs/best_run/mantle_step2/fort.56'
BEST_BML_FORT56    = f'{RUN_BASE}/best_M04/files/mnt/user-data/outputs/best_run/bml/fort.56'

g_mars      = 3.35     # m/s^2
MARS_RADIUS = 3389.5   # km
DEPTH_MAX   = 2000     # km
P_BML_TOP   = 17.05    # GPa, BML top pressure; best mantle drawn only above it
BML_TOP_KM  = 1560 - 150   # km, BML top depth = CMB depth (~1560 km, khan_cache) - 150 km
labelsize   = 14
bwith       = 2.5

BEST_COLOR = 'magenta'
BEST_LW    = 2.2
BEST_LABEL = 'Best (MCMC a04)'

# fort.56 reader; depth from hydrostatic integration: dz = dP / (rho * g)
def load_fort56(fpath):
    with open(fpath) as f:
        f.readline()
        cols = f.readline().split()
    df = pd.read_csv(fpath, sep=r'\s+', skiprows=2, names=cols)
    P   = df['P(GPa)'].values
    rho = df['rho(g/cm^3)'].values
    dz  = np.diff(P) * 1e9 / ((rho[:-1] + rho[1:]) / 2 * 1000 * g_mars) / 1000   # km
    depth = np.concatenate([[0.0], np.cumsum(dz)])
    return P, rho, df['VS(km/s)'].values, df['VP(km/s)'].values, df['T(K)'].values, depth

# ── reference HeFESTo run ────────────────────────────────────────────────────
hef_P, hef_rho, hef_vs, hef_vp, hef_T, hef_depth = load_fort56(HEFESTO_FILE)
P_MAX = hef_P.max()

# ── best_run mantle (P <= P_BML_TOP only) ────────────────────────────────────
bm_P, bm_rho, bm_vs, bm_vp, bm_T, bm_depth = load_fort56(BEST_MANTLE_FORT56)
m = bm_P <= P_BML_TOP
bm_P, bm_rho, bm_vs, bm_vp, bm_T, bm_depth = bm_P[m], bm_rho[m], bm_vs[m], bm_vp[m], bm_T[m], bm_depth[m]
print(f"best mantle loaded: {len(bm_P)} points, P={bm_P.min():.1f}~{bm_P.max():.1f} GPa")

# ── best_run BML (depth offset to BML top) ───────────────────────────────────
bb_P, bb_rho, bb_vs, bb_vp, bb_T, bb_depth_rel = load_fort56(BEST_BML_FORT56)
bb_depth = bb_depth_rel + BML_TOP_KM
print(f"best BML loaded: {len(bb_P)} points, P={bb_P.min():.1f}~{bb_P.max():.1f} GPa")

# ── Khan et al. models (solid part, P <= P_MAX + 1 GPa) ──────────────────────
khan_depth_list, khan_vp_list, khan_vs_list, khan_rho_list = [], [], [], []
for fpath in sorted(glob.glob(f'{KHAN_MODEL_DIR}/Model_*.txt')):
    data = np.loadtxt(fpath, comments='#')   # depth km, Vp, Vs km/s, rho g/cc, ..., P Pa (col 6)
    mask = (data[:, 2] > 0.01) & (data[:, 6] / 1e9 <= P_MAX + 1)
    if mask.sum() < 5:
        continue
    khan_depth_list.append(data[mask, 0])
    khan_vp_list.append(data[mask, 1])
    khan_vs_list.append(data[mask, 2])
    khan_rho_list.append(data[mask, 3])
n_models = len(khan_depth_list)
print(f'Khan: {n_models} models loaded')

# median and 16/84 percentiles on a common depth grid (models shallower than 0.8*DEPTH_MAX skipped)
Z_common = np.linspace(0, DEPTH_MAX, 400)
def make_stats(lst):
    mat = np.full((n_models, len(Z_common)), np.nan)
    for i, (z, v) in enumerate(zip(khan_depth_list, lst)):
        if z.max() < Z_common.max() * 0.8:
            continue
        mat[i] = np.interp(Z_common, z, v, left=np.nan, right=np.nan)
    return np.nanmedian(mat, 0), np.nanpercentile(mat, 16, 0), np.nanpercentile(mat, 84, 0)

vp_med,  vp_p16,  vp_p84  = make_stats(khan_vp_list)
vs_med,  vs_p16,  vs_p84  = make_stats(khan_vs_list)
rho_med, rho_p16, rho_p84 = make_stats(khan_rho_list)

# ── Samuel 2023 (value, radius km) ───────────────────────────────────────────
Tliq,     depTliq     = np.loadtxt(SAMUEL_BASE + 'PANEL_I/Tliq.dat').T
Tprofile, depTprofile = np.loadtxt(SAMUEL_BASE + 'PANEL_I/Tprofile.dat').T
Tsol,     depTsol     = np.loadtxt(SAMUEL_BASE + 'PANEL_I/Tsol.dat').T
rho_sam,  deprho_sam  = np.loadtxt(SAMUEL_BASE + 'PANEL_J/rho_profile.dat').T
samvp,    depvp_sam   = np.loadtxt(SAMUEL_BASE + 'PANEL_K/vp_profile.dat').T
samvs,    depvs_sam   = np.loadtxt(SAMUEL_BASE + 'PANEL_K/vs_profile.dat').T

# depth axis on the left, HeFESTo pressure ticks on a twin right axis (all curves share the depth scale)
def setup_ax(ax, xlabel):
    ax.set_ylim(DEPTH_MAX, 0)
    ax.set_ylabel('depth (km)', fontsize=labelsize)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.grid(True, ls='--', alpha=0.35)
    ax.tick_params(labelsize=labelsize)
    for sp in ax.spines.values():
        sp.set_linewidth(bwith)
    axr = ax.twinx()
    axr.set_ylim(DEPTH_MAX, 0)
    p_ticks = np.arange(0, 23, 2)
    axr.set_yticks(np.interp(p_ticks, hef_P, hef_depth))
    axr.set_yticklabels([f'{p:.0f}' for p in p_ticks], color='gray')
    axr.tick_params(labelsize=labelsize)
    for sp in axr.spines.values():
        sp.set_linewidth(bwith)
    return axr

# ── plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
fig.subplots_adjust(wspace=0.35, left=0.10, right=0.93, top=0.92, bottom=0.09)

# Panel 1: Vp and Vs
ax = ax1
for i in range(n_models):
    ax.plot(khan_vp_list[i], khan_depth_list[i], color='#4198B9', alpha=0.05, lw=0.5, rasterized=True)
    ax.plot(khan_vs_list[i], khan_depth_list[i], color='#97795D', alpha=0.05, lw=0.5, rasterized=True)
ax.fill_betweenx(Z_common, vp_p16, vp_p84, color='#4198B9', alpha=0.15)
ax.fill_betweenx(Z_common, vs_p16, vs_p84, color='#97795D', alpha=0.15)
ax.plot(vp_med, Z_common, color='#4198B9', lw=2.0, label='Khan Vp (median)')
ax.plot(vs_med, Z_common, color='#97795D', lw=2.0, label='Khan Vs (median)')
ax.plot(samvp / 1000, MARS_RADIUS - depvp_sam, color='red',  lw=2, label='Samuel Vp')
ax.plot(samvs / 1000, MARS_RADIUS - depvs_sam, color='blue', lw=2, label='Samuel Vs')

axr = setup_ax(ax1, 'velocity (km/s)')
axr.plot(hef_vp, hef_depth, color='#CD5C5C', lw=2.2, ls='--', label='HeFESTo Vp')
axr.plot(hef_vs, hef_depth, color='#282130', lw=2.2, ls='--', label='HeFESTo Vs')
axr.plot(bm_vp, bm_depth, color=BEST_COLOR, lw=BEST_LW, ls='-',  label=f'{BEST_LABEL} Vp')
axr.plot(bm_vs, bm_depth, color=BEST_COLOR, lw=BEST_LW, ls='--', label=f'{BEST_LABEL} Vs')
axr.plot(bb_vp, bb_depth, color=BEST_COLOR, lw=BEST_LW, ls='-')
axr.plot(bb_vs, bb_depth, color=BEST_COLOR, lw=BEST_LW, ls='--')
ax.set_xlim(0, 11)
ax.legend(fontsize=8.5, loc='lower left', framealpha=0.85)

# Panel 2: density
ax = ax2
for i in range(n_models):
    ax.plot(khan_rho_list[i], khan_depth_list[i], color='#35838D', alpha=0.05, lw=0.5, rasterized=True)
ax.fill_betweenx(Z_common, rho_p16, rho_p84, color='#35838D', alpha=0.20)
ax.plot(rho_med, Z_common, color='#35838D', lw=2.0, label='Khan ρ (median)')
ax.plot(rho_sam / 1000, MARS_RADIUS - deprho_sam, color='k', lw=2, label='Samuel 2023 noBML')

axr = setup_ax(ax2, 'rho')
axr.plot(hef_rho, hef_depth, color='#35838D', lw=2.2, ls='--', label='HeFESTo ρ')
axr.plot(bm_rho, bm_depth, color=BEST_COLOR, lw=BEST_LW, ls='-', label=f'{BEST_LABEL} ρ')
axr.plot(bb_rho, bb_depth, color=BEST_COLOR, lw=BEST_LW, ls='-')
ax.set_xlim(1.5, 7)
ax.legend(fontsize=8.5, loc='lower right', framealpha=0.85)
ax.set_ylabel('')

# Panel 3: temperature
ax = ax3
axr = setup_ax(ax3, 'temperature')
axr.plot(hef_T, hef_depth, color='#282130', lw=1.8, ls=':', label='T (HeFESTo)')
axr.plot(bm_T, bm_depth, color=BEST_COLOR, lw=BEST_LW, ls='-', label=f'{BEST_LABEL} T')
axr.plot(bb_T, bb_depth, color=BEST_COLOR, lw=BEST_LW, ls='-')
ax.plot(Tprofile, MARS_RADIUS - depTprofile, color='k',      lw=1.8, label='Samuel 2023 noBML')
ax.plot(Tliq,     MARS_RADIUS - depTliq,     color='brown',  lw=1.8, label='liquidus')
ax.plot(Tsol,     MARS_RADIUS - depTsol,     color='orange', lw=1.8, label='solidus')
axr.set_ylabel('pressure (GPa)\n[HeFESTo]', fontsize=labelsize, color='gray')
ax.legend(fontsize=8.5, loc='upper right', framealpha=0.85)
ax.set_ylabel('')

plt.show()
