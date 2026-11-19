#!/usr/bin/env python3
"""
Plot accepted temperature profiles from profile_s*.npz files
讀真實的 HeFESTo T profile，比參數近似準確

用法（在 server 上跑）:
  python3 plot_T_profiles_npz.py /path/to/mcmc/chain_a21_*/
  python3 plot_T_profiles_npz.py  # 自動找當前目錄下 chain_a21_*/
"""
import json, glob, sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mars_config import MCMC_DIR, FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

# ── 路徑 ─────────────────────────────────────────────────────
chain_dirs = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob('chain_a21_*/'))
if not chain_dirs:
    chain_dirs = sorted(glob.glob(f'{MCMC_DIR}/chain_a21_*/'))

print(f"Found {len(chain_dirs)} chain directories")

# ── solidus / liquidus 參考線 ─────────────────────────────────
def solidus_duncan(P, Mg=0.75):
    if   P <= 10:  TC = -4.877*P**2 + 120.2*P + 1088
    elif P <= 23:  TC = -1.323*(P-10)**2 + 38.18*(P-10) + 1802
    else:          TC = 77.75*(P-23) + 2075
    return TC + 273.15 + (-6.0)*(100*(1-Mg) - 100*(1-0.75))

def liquidus_ruedas(P, Mg=0.75):
    return (2160.6 + 64.7109*P - 3.97463*P**2 + 0.0957894*P**3
            + (-6.0)*(100*(1-Mg) - 100*(1-0.75)))

P_ref  = np.linspace(0, 22, 300)
depth_ref = P_ref / 0.01135   # km（線性近似）
T_sol_075 = np.array([solidus_duncan(P, 0.75) for P in P_ref])
T_sol_065 = np.array([solidus_duncan(P, 0.65) for P in P_ref])
T_liq_075 = np.array([liquidus_ruedas(P, 0.75) for P in P_ref])

TRUE_CMB = 1743.3   # km

# ── 讀所有 chain ──────────────────────────────────────────────
profiles  = []   # list of (depth_km, T_K) arrays
misfits   = []
params_list = []
has_bml   = []

for chain_dir in chain_dirs:
    chain_file = os.path.join(chain_dir, 'chain.json')
    if not os.path.exists(chain_file):
        print(f"  [skip] no chain.json in {chain_dir}")
        continue

    chain = json.load(open(chain_file))
    accepted = [s for s in chain if s.get('accepted')]

    n_loaded = 0
    for s in accepted:
        step = s['step']
        npz_path = os.path.join(chain_dir, f'profile_s{step:05d}.npz')
        if not os.path.exists(npz_path):
            continue

        d = np.load(npz_path)

        # mantle part (HeFESTo)
        depth_m = d['depth_km']
        T_m     = d['T_K']

        # BML part (if exists)
        if 'bml_depth_km' in d:
            depth_b = d['bml_depth_km']
            T_b     = d['bml_T_K']
            full_depth = np.concatenate([depth_m, depth_b])
            full_T     = np.concatenate([T_m,     T_b])
            bml = True
        else:
            full_depth = depth_m
            full_T     = T_m
            bml = False

        # sort by depth
        idx = np.argsort(full_depth)
        profiles.append((full_depth[idx], full_T[idx]))
        misfits.append(s['misfit'])
        params_list.append(s['params'])
        has_bml.append(bml)
        n_loaded += 1

    print(f"  {os.path.basename(chain_dir)}: {len(accepted)} accepted, {n_loaded} npz loaded")

print(f"\nTotal profiles loaded: {len(profiles)}")
if not profiles:
    print("No profiles found. Check paths.")
    import sys; sys.exit(1)

misfits  = np.array(misfits)
has_bml  = np.array(has_bml)

# best models
best_idx  = np.argsort(misfits)
best10    = best_idx[:10]
best50    = best_idx[:50]

# ── 插值到共同深度網格（for percentile calculation）────────────
depth_grid = np.linspace(0, TRUE_CMB, 400)
T_interp   = np.full((len(profiles), len(depth_grid)), np.nan)

for i, (dep, T) in enumerate(profiles):
    # only interpolate within the range of this profile
    mask = (depth_grid >= dep.min()) & (depth_grid <= dep.max())
    T_interp[i, mask] = np.interp(depth_grid[mask], dep, T)

T_med = np.nanmedian(T_interp, axis=0)
T_p10 = np.nanpercentile(T_interp, 10, axis=0)
T_p25 = np.nanpercentile(T_interp, 25, axis=0)
T_p75 = np.nanpercentile(T_interp, 75, axis=0)
T_p90 = np.nanpercentile(T_interp, 90, axis=0)

# ── colour map by misfit ──────────────────────────────────────
m_lo  = np.percentile(misfits, 5)
m_hi  = np.percentile(misfits, 95)
norm  = plt.Normalize(m_lo, m_hi)
cmap  = cm.viridis_r

# ── figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 11), sharey=True)

# ─── left: all profiles ──────────────────────────────────────
ax = axes[0]

# random subsample to avoid overplotting
n_plot = min(300, len(profiles))
idx_plot = np.random.choice(len(profiles), n_plot, replace=False)

for i in idx_plot:
    dep, T = profiles[i]
    col = cmap(norm(misfits[i]))
    ax.plot(T, dep, color=col, lw=0.4, alpha=0.25)

# best 50 on top
for i in best50:
    dep, T = profiles[i]
    col = cmap(norm(misfits[i]))
    lw  = 1.8 if i in best10 else 0.9
    ax.plot(T, dep, color=col, lw=lw, alpha=0.9)

# reference lines
ax.plot(T_sol_075, depth_ref, 'r-',  lw=2.0, label='Solidus Mg#=0.75 (Duncan)')
ax.plot(T_sol_065, depth_ref, 'r--', lw=1.5, label='Solidus Mg#=0.65')
ax.plot(T_liq_075, depth_ref, 'b--', lw=1.5, label='Liquidus Mg#=0.75 (Ruedas)')
ax.axhline(TRUE_CMB, color='k', lw=1.5, ls=':', label=f'CMB {TRUE_CMB:.0f} km')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Misfit', shrink=0.8, pad=0.02)

ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_ylabel('Depth (km)', fontsize=12)
ax.set_title(f'All accepted T profiles (HeFESTo)\n'
             f'n={len(profiles)} total  |  best 50 highlighted\n'
             f'with BML: {has_bml.sum()}  |  without BML: {(~has_bml).sum()}',
             fontsize=10, fontweight='bold')
ax.invert_yaxis()
ax.set_xlim(0, 4500)
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)

# ─── right: median + envelope ─────────────────────────────────
ax = axes[1]

valid = ~np.isnan(T_med)
ax.fill_betweenx(depth_grid[valid], T_p10[valid], T_p90[valid],
                 alpha=0.15, color='steelblue', label='10–90th pct')
ax.fill_betweenx(depth_grid[valid], T_p25[valid], T_p75[valid],
                 alpha=0.25, color='steelblue', label='25–75th pct')
ax.plot(T_med[valid], depth_grid[valid], 'steelblue', lw=2.5, label='Median')

# best 10 in gold
for i in best10:
    dep, T = profiles[i]
    ax.plot(T, dep, color='gold', lw=1.2, alpha=0.85, zorder=5)
ax.plot([], [], color='gold', lw=1.5,
        label=f'Best 10 (misfit={misfits[best10[-1]]:.3f}–{misfits[best10[0]]:.3f})')

# reference lines
ax.plot(T_sol_075, depth_ref, 'r-',  lw=2.0, label='Solidus Mg#=0.75')
ax.plot(T_sol_065, depth_ref, 'r--', lw=1.5, label='Solidus Mg#=0.65')
ax.plot(T_liq_075, depth_ref, 'b--', lw=1.5, label='Liquidus Mg#=0.75')
ax.axhline(TRUE_CMB, color='k', lw=1.5, ls=':')

ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_title('Median ± percentile envelope\n'
             '+ best 10 models (gold)',
             fontsize=10, fontweight='bold')
ax.set_xlim(0, 4500)
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# ── BML zoom inset on right panel ────────────────────────────
axins = ax.inset_axes([0.55, 0.05, 0.42, 0.35])   # x0, y0, width, height (axes fraction)
for i in best50:
    dep, T = profiles[i]
    bml_mask = dep >= 1400
    if bml_mask.sum() > 2:
        col = cmap(norm(misfits[i]))
        lw  = 1.5 if i in best10 else 0.6
        axins.plot(T[bml_mask], dep[bml_mask], color=col, lw=lw, alpha=0.8)

# solidus/liquidus in inset
d_zoom = depth_ref[depth_ref >= 1400]
P_zoom = P_ref[depth_ref >= 1400]
axins.plot([solidus_duncan(P) for P in P_zoom], d_zoom, 'r-',  lw=1.5)
axins.plot([solidus_duncan(P, 0.65) for P in P_zoom], d_zoom, 'r--', lw=1.0)
axins.plot([liquidus_ruedas(P) for P in P_zoom], d_zoom, 'b--', lw=1.0)
axins.axhline(TRUE_CMB, color='k', lw=1.2, ls=':')
axins.set_xlim(1500, 4000)
axins.set_ylim(TRUE_CMB + 20, 1400)
axins.set_title('BML zoom', fontsize=8)
axins.tick_params(labelsize=7)
axins.grid(alpha=0.3)

fig.suptitle(
    f'HeFESTo Temperature Profiles  |  {len(profiles)} accepted steps  |  '
    f'{len(chain_dirs)} chains\n'
    f'Best misfit = {misfits.min():.4f}  |  '
    f'Median misfit = {np.median(misfits):.4f}',
    fontsize=11, fontweight='bold')

plt.tight_layout()
out = f'{FIG_DIR}/T_profiles_npz.png'
if SAVE_FIG: plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")